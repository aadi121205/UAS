import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load and process each input image
input_images = ["1.png","2.png","3.png","4.png","5.png","6.png","7.png","8.png","10.png","11.png"]  # List of input image filenames 'Pixelated Image.png'


output_images = []
house_counts = []
house_priorities = []
rescue_ratios = []
image_names = []


blue_priority = 2
red_priority = 1


def Detected_Big_Shapes(image):
    gray_image = image
    x = 0
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 1)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=150, threshold2=150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over each contour to detect and draw big shapes
    for contour in contours:
    # Calculate the area of the contour
        area = cv2.contourArea(contour)
        if area > 50:
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            num_sides = len(approx)
            shape = ""
            if num_sides == 3:
                shape = "Triangle"
                # Check if the triangle is black
                #if cv2.mean(blurred_image, mask=contour)[0] < 10:
                shape = "Black Triangle"
                x = x + 1

        
    # Draw the shape name and outline on the image
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    x, y = approx[0][0]
    cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # Display the detected big shapes
    cv2.destroyAllWindows()
    return x


for image_filename in input_images:
    print(f"Loading image: {image_filename}")
    src = cv2.imread(image_filename)
    image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV )
    #image = cv2.GaussianBlur(x, (5, 5), 0)
    if image is None:
        print(f"Error loading image: {image_filename}")
        continue
    
    # Define color ranges for burnt and green grass
    lower_green = np.array([32, 0, 0])
    upper_green = np.array([90, 255, 225])
    lower_burnt = np.array([4, 0, 0])
    upper_burnt = np.array([32, 255, 255])
    lower_blue = np.array([120,0,0])
    upper_blue = np.array([120,255,255])
    lower_red = np.array([0,0,0])
    upper_red = np.array([0,255,255])

    # Create masks for burnt and green grass
    burnt_mask = cv2.inRange(image, lower_burnt, upper_burnt)
    green_mask = cv2.inRange(image, lower_green, upper_green)
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)
    red_mask = cv2.inRange(image, lower_red, upper_red)
    combined_bb_mask = cv2.bitwise_or(burnt_mask, blue_mask)
    combined_br_mask = cv2.bitwise_or(burnt_mask, red_mask)
    combined_gb_mask = cv2.bitwise_or(green_mask, blue_mask)
    combined_gr_mask = cv2.bitwise_or(green_mask, red_mask)
    #combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    # Overlay colors on burnt and green grass areas
    output_image = image.copy()
    output_image[burnt_mask > 0] = [0, 0, 0]  # brown for burnt grass
    output_image[green_mask > 0] = [255, 255, 0]  # Green for green grass
    output_image[blue_mask > 0] = [255, 0, 0]  # Green for green grass
    output_image[red_mask > 0] = [0, 0, 255]  # Green for green grass

    output_images.append(output_image)

    # Count houses and calculate priorities
    burnt_houses = Detected_Big_Shapes(burnt_mask)
    green_houses = Detected_Big_Shapes(green_mask)
    red_houses = Detected_Big_Shapes(red_mask)
    blue_houses = Detected_Big_Shapes(blue_mask)

    total_houses = red_houses + blue_houses
    
    blue_burnt_houses = total_houses - Detected_Big_Shapes(combined_bb_mask)
    blue_green_houses = total_houses - Detected_Big_Shapes(combined_gb_mask)
    red_burnt_houses = total_houses - Detected_Big_Shapes(combined_br_mask)
    red_green_houses = total_houses - Detected_Big_Shapes(combined_gr_mask)


    burnt_priority = blue_priority * blue_burnt_houses + red_priority * red_burnt_houses
    green_priority = blue_priority * blue_green_houses + red_priority * red_green_houses
    
    house_counts.append([burnt_houses, green_houses])
    house_priorities.append([burnt_priority, green_priority])
    
    rescue_ratio = burnt_priority / green_priority
    rescue_ratios.append([rescue_ratio])
    
    image_names.append(image_filename)

# Sort images based on rescue ratio
sorted_indices = sorted(range(len(rescue_ratios)), key=lambda k: rescue_ratios[k], reverse=True)
sorted_images = [output_images[i] for i in sorted_indices]
sorted_image_names = [image_names[i] for i in sorted_indices]

# Display or save the results as needed
for i, output_image in enumerate(sorted_images):
    cv2.imshow(f'Result for {sorted_image_names[i]}', output_image)
cv2.waitKey(0)
print(output_images,house_counts,house_priorities,rescue_ratios,image_names)
cv2.destroyAllWindows()
