import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from ultralytics import YOLO
import shutil

def closest_color(requested_color):
    colors = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'purple': (128, 0, 128),
        'brown': (165, 42, 42),
        'orange': (255, 165, 0)
    }

    min_colors = {}
    for name, rgb in colors.items():
        rd = (rgb[0] - requested_color[0]) ** 2
        gd = (rgb[1] - requested_color[1]) ** 2
        bd = (rgb[2] - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    # Define the tolerance
    tolerance = 10000

    # Find the closest color
    min_difference = min(min_colors.keys())
    if min_difference < tolerance:
        return min_colors[min_difference]
    else:
        return None

def rgb_to_color_name(rgb):
    color_name = closest_color(rgb)
    return color_name

def create_color_masks(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image to a PIL image
    image = Image.fromarray(image)

    # Convert the image to CMYK
    image = image.convert('CMYK')

    # Convert the PIL image back to an OpenCV image
    image = np.array(image)

    # Rest of your code...
    
    # Reshape the image to be a simple list of RGB pixels
    pixels = image.reshape(-1, 3)

    # Convert to floating point
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    K = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(image.shape)

    # Get the center of the image
    center_x, center_y = segmented_image.shape[0] // 2, segmented_image.shape[1] // 2

    # Get the colors at the center of the image
    center_colors = segmented_image[center_x-10:center_x+10, center_y-10:center_y+10].reshape(-1, 3)

    # Get the two most common colors
    unique_colors, counts = np.unique(center_colors, return_counts=True, axis=0)
    most_common_colors = unique_colors[np.argsort(-counts)][:3]

    masks = []
    for color in most_common_colors:
        lower = np.array(color, dtype = "uint8")
        upper = np.array(color, dtype = "uint8")
        mask = cv2.inRange(segmented_image, lower, upper)
        masks.append(mask)

    return masks, most_common_colors, image

def is_color_at_edges(mask):
    """Check if a color is present at the sides of a mask."""
    left_edge = mask[:, :5].flatten()
    right_edge = mask[:, -5:].flatten()

    edges = np.concatenate([left_edge, right_edge])
    num_white_pixels = np.sum(edges == 255)
    total_pixels = edges.size

    z = (num_white_pixels / total_pixels) * 100

    if z > 5:
        return True
    else:
        return False

def yolo_detect(image_path, output_path, model):
    # Load the YOLO model
    model = YOLO(model=model)

    # Perform the detection
    results = model(image_path)

    # Save the image with the detections
    results.save(output_path)

def runcns():
    input_directory = "shapes"
    output_directory = "output"

    # Iterate over all images in the directory
    for folder in os.listdir(input_directory):
        os.makedirs(os.path.join(output_directory, folder), exist_ok=True)
        output_directory = os.path.join(output_directory, folder)
        for filename in tqdm(os.listdir(os.path.join(input_directory, folder))):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_directory, folder, filename)
                masks, unique_colors = create_color_masks(image_path)
                # Save each mask to a file
                for mask, color in zip(masks, unique_colors):
                    color_name = rgb_to_color_name(color)  # Convert RGB to color name
                    mask_filename = os.path.splitext(filename)[0]

                    # Count the number of pixels of the color in the mask
                    num_color_pixels = np.sum(mask == 255)

                    # Only save the mask if the color is not the most common color at the edges
                    n = 0
                    if not is_color_at_edges(mask) and num_color_pixels > 500 and n == 0:
                        output_path = os.path.join(output_directory, f"{mask_filename}_{color_name}_shape.png")
                        cv2.imwrite(output_path, mask)
                        n += 1
                    else:
                        output_path = os.path.join(output_directory, f"{mask_filename}_{color_name}_leter.png")
                        cv2.imwrite(output_path, mask)
        output_directory = "output"

def load_data(input_path, output_path):
    for filename in tqdm(os.listdir(input_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_path, filename)
            output_path = os.path.join(output_path, filename)
            shutil.copy(image_path, output_path)



