import torch
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
import os

# Directory containing the images
image_dir = 'C:\\Users\\aadit\\Documents\\Codes and Projects\\Py\\UAS\\ODLC\\output'

# Loop over all images in the directory
for image_name in os.listdir(image_dir):
    # Load the image
    image_path = os.path.join(image_dir, image_name)
    img = Image.open(image_path)
    img = img.convert('RGB')


    # Apply the transformations and add a batch dimension
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Run the model on the image
    out = model(batch_t)

    # Print the top 5 predicted classes
    _, indices = torch.topk(out, 5)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(f'Image: {image_name}')
    for idx in indices[0]:
        print('Label:', idx.item(), ', Confidence Score:', percentage[idx].item(), '%')
    print('\n')