import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import webcolors

'''

def closest_color(requested_color):
    colors = {
        'white': ([i for i in range(245, 256)], [i for i in range(245, 256)], [i for i in range(245, 256)]),
        'black': ([i for i in range(0, 11)], [i for i in range(0, 11)], [i for i in range(0, 11)]),
        'red': ([255], [i for i in range(0, 11)], [i for i in range(0, 11)]),
        'blue': ([i for i in range(0, 11)], [i for i in range(0, 11)], [255]),
        'green': ([i for i in range(0, 11)], [255], [i for i in range(0, 11)]),
        'purple': ([128], [i for i in range(0, 11)], [128]),
        'brown': ([165], [42], [42]),
        'orange': ([255], [165], [i for i in range(0, 11)])
    }

    min_colors = {}
    for name, rgb in colors.items():
        rd = min((i - requested_color[0]) ** 2 for i in rgb[0])
        gd = min((i - requested_color[1]) ** 2 for i in rgb[1])
        bd = min((i - requested_color[2]) ** 2 for i in rgb[2])
        min_colors[(rd + gd + bd)] = name

    # Define the tolerance
    tolerance = 15000

    # Find the closest color
    min_difference = min(min_colors.keys())
    if min_difference < tolerance:
        return min_colors[min_difference]
    else:
        return None
'''

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def rgb_to_color_name(rgb):
    color_name = closest_color(rgb)
    return color_name

def create_color_masks(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a simple list of RGB pixels
    pixels = image.reshape(-1, 3)

    # Convert to floating point
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    K = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(image.shape)
    # Get the colors at the center of the image
    center_colors = segmented_image.reshape(-1, 3)

    # Get the two most common colors
    unique_colors, counts = np.unique(center_colors, return_counts=True, axis=0)
    most_common_colors = unique_colors[np.argsort(-counts)][:3]

    masks = []
    for color in most_common_colors:
        lower = np.array(color, dtype = "uint8")
        upper = np.array(color, dtype = "uint8")
        mask = cv2.inRange(segmented_image, lower, upper)
        masks.append(mask)

    return masks, most_common_colors

def is_color_at_edges(mask, threshold=5):
    """Check if a color is present at the sides of a mask."""
    left_edge = mask[:, :threshold].flatten()
    right_edge = mask[:, -threshold:].flatten()

    edges = np.concatenate([left_edge, right_edge])
    num_white_pixels = np.sum(edges == 255)
    total_pixels = edges.size
    z = (num_white_pixels / total_pixels) * 100
    if z > 5:
        return True
    else:
        return False

# Directory containing images
#input_directory = "E:\\generated_data\\generated_data\\HR"
input_directory = "shapes"
output_directory = "output"

# Iterate over all images in the directory
for folder in os.listdir(input_directory):
    os.makedirs(os.path.join(output_directory, folder), exist_ok=True)
    os.makedirs(os.path.join(output_directory, str(folder+"l")), exist_ok=True)
    output_directory = os.path.join(output_directory, folder)
    output_directory2 = os.path.join(output_directory, str(folder+"l"))
    for filename in tqdm(os.listdir(os.path.join(input_directory, folder))):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_directory, folder, filename)
            masks, unique_colors = create_color_masks(image_path)
            # Save each mask to a file
            n = 0
            for mask, color in zip(masks, unique_colors):
                num_color_pixels = np.sum(mask == 255)
                if not is_color_at_edges(mask) and num_color_pixels > 500 and n == 0:
                    color_name = rgb_to_color_name(color)  # Convert RGB to color name
                    mask_filename = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_directory, f"{mask_filename}_{color_name}.png")
                    cv2.imwrite(output_path, mask)
                    n += 1
                elif num_color_pixels < 500:
                    color_name = rgb_to_color_name(color)  # Convert RGB to color name
                    mask_filename = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_directory2, f"{mask_filename}_leter_{color_name}.png")
                    cv2.imwrite(output_path, mask)
                else:
                    color_name = rgb_to_color_name(color)
                    mask_filename = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_directory2, f"{mask_filename}_{color_name}.png")
                    cv2.imwrite(output_path, mask)


    output_directory = "output"


