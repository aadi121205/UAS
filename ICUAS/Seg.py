import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import webcolors

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
    
    K = 10  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(image.shape)

    # Get the colors at the center of the image
    center_colors = segmented_image.reshape(-1, 3)

    # Get the most common colors
    unique_colors, counts = np.unique(center_colors, return_counts=True, axis=0)
    most_common_colors = unique_colors[np.argsort(-counts)][:10]

    masks = []
    for color in most_common_colors:
        lower = np.array(color, dtype="uint8")
        upper = np.array(color, dtype="uint8")
        mask = cv2.inRange(segmented_image, lower, upper)
        masks.append(mask)

    return masks, most_common_colors

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


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

def denoise_mask(mask, kernel_size=3):
    # Create a kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Convert mask to 8-bit single-channel image
    mask = mask.astype(np.uint8)
    
    # Apply morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Close small holes
    return mask

def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in tqdm(os.listdir(input_directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_directory, filename)
            masks, unique_colors = create_color_masks(image_path)

            Fmasks = []
            for mask, color in zip(masks, unique_colors):
                if not is_color_at_edges(mask, threshold=2):
                    num_color_pixels = np.sum(mask == 255)
                    if num_color_pixels > 40:
                        Fmasks.append(mask)
            merged_masks = []
            if Fmasks:
                # Sum the masks
                merged_mask = np.sum(Fmasks, axis=0)
                merged_mask[merged_mask > 255] = 255  # Ensure the values are in the valid range
                
                mask_name = f"{filename.split('.')[0]}_merged.png"
                mask_path = os.path.join(output_directory, mask_name)
                merged_mask = denoise_mask(merged_mask, kernel_size=2)
                cv2.imwrite(mask_path, merged_mask)
                merged_masks.append(merged_mask)
    return merged_masks


if __name__ == "__main__":
    input_directory = "output_folder"  # Change this to your input directory
    output_directory = "output_directory_path"  # Change this to your output directory
    x = main(input_directory, output_directory)