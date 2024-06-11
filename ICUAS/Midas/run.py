import cv2
import torch
import urllib.request
import numpy as np

import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

filename = "1.jpeg"

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

print("Input shape:", input_batch.shape)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
output = prediction.cpu().numpy()

# Scale the output array to the [0, 255] range
output_scaled = ((output - output.min()) * (255 / (output.max() - output.min()))).astype(np.uint8)

# Save the scaled output array
cv2.imwrite("output.jpg", output_scaled)

plt.imshow(output)

plt.show()

plt.imshow(cv2.imread("output.jpg"))

plt.show()

print("Done")

def visualize_depth_map(output):
    print("Output shape:", output.shape)
    new_output = np.zeros((output.shape[0], output.shape[1], 3))

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            if output[i][j] < 10:
                new_output[i][j] = [0, 0, 0]
            elif output[i][j] < 20:
                new_output[i][j] = [255, 0, 0]
            elif output[i][j] < 30:
                new_output[i][j] = [0, 255, 0]
            elif output[i][j] < 40:
                new_output[i][j] = [0, 0, 255]
            elif output[i][j] < 50:
                new_output[i][j] = [255, 255, 0]

    new_output = new_output.astype(np.uint8)
    plt.imshow(new_output)
    plt.show()

visualize_depth_map(output)