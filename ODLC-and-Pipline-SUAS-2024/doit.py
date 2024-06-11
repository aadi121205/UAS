import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image       
import numpy as np
import webcolors
import random
from ultralytics import YOLO
modle_rpn = YOLO('best.pt')
model=YOLO('last.pt')

file = open("sd.txt", "r")
sd = int(file.read())

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

    return masks, most_common_colors

def is_color_at_edges(mask, threshold=15):
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

def doit(sd):
    # Create a VideoCapture object
    cam = cv2.VideoCapture("x1.mp4")

    # Set the duration to 5 seconds
    duration = 5  # in seconds
    start_time = cv2.getTickCount()

    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
        ret, frame = cam.read()
        # Save 1 second of video
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_count = int(fps)
        output_path = "output.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
        for _ in range(frame_count):
            out.write(frame)
        out.release()

    cam.release()

    modle_rpn.predict("output.mp4", save = True, save_crop = True)

    if sd == 1:
        inputh_path = f"runs/detect/predict/crops/_object_"
    else:
        inputh_path = f"runs/detect/predict{sd}/crops/_object_"

    # Get a list of image files in the input directory
    image_files = [file for file in os.listdir(inputh_path) if file.endswith(".jpg")]

    print(f"Found {len(image_files)} image files")

    # Select a random image file from the list
    image_file = random.choice(image_files)

    # Construct the full path to the image file
    image_path = os.path.join(inputh_path, image_file)

    # Call the function to create color masks for the image
    masks, unique_colors = create_color_masks(image_path)

    # Specify the output directory for saving the masks
    output_directory = "saves/masks1"
    output_directory2 = "saves/masks2"
    model.predict(inputh_path, save = True)

    # Save each mask to a file
    for i, mask in enumerate(masks):
        output_path = os.path.join(output_directory, f"{sd}mask{i}.png")
        output_path2 = os.path.join(output_directory2, f"{sd}mask{i}.png")
        cv2.imwrite(output_path, mask)
    n = 0
    for mask, color in zip(masks, unique_colors):
        num_color_pixels = np.sum(mask == 255)
        if not is_color_at_edges(mask) and num_color_pixels > 250 and n == 0:
            color_name = rgb_to_color_name(color)  # Convert RGB to color name
            mask_filename = os.path.splitext(os.path.basename(image_path))[0]
            # Count the number of pixels of the color in the mask
            output_path =  f"saves/{mask_filename}_{color_name}.png"
            cv2.imwrite(output_path, mask)
            n += 1

doit(sd)