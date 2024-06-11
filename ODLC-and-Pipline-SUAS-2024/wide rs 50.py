import torch
from torchvision import models, transforms
from PIL import Image
import os

# Initialize the model
model = models.wide_resnet50_2(pretrained=False)

# Load the state dictionary
state_dict = torch.load('model42.pt', map_location=torch.device('cpu'))

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.eval()
# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing the images
image_dir = 'C:\\Users\\aadit\\Documents\\Codes and Projects\\Py\\UAS\\ODLC\\output'

# Loop over all images in the directory
for image_name in os.listdir(image_dir):
    # Open the image file
    with Image.open(os.path.join(image_dir, image_name)) as img:
        # Apply the transformations
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Run the model on the image
        out = model(batch_t)

        # Print the predicted class
        _, predicted = torch.max(out, 1)
        print(f'Image {image_name} is predicted to be in class {predicted.item()}')