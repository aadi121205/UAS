import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn

# Define the model architecture (should be the same as during training)
class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier,self).__init__()
        self.res=torch.nn.Sequential(*(list(models.wide_resnet50_2(weights = "DEFAULT").children())[:-1]))
        self.fc=nn.Linear(2048,26)

    def forward(self,x):
        output=self.res(x)
        output=torch.flatten(output,1)
        output=self.fc(output)
        return output

# Load the model
classifier = CustomClassifier()
classifier.load_state_dict(torch.load('model42.pt'))
classifier.eval()

# Define the image transformations (should be the same as during training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Directory containing the images
image_dir = 'C:\\Users\\aadit\\Documents\\Codes and Projects\\Py\\UAS\\ODLC\\output'

output_dir = 'C:\\Users\\aadit\\Documents\\Codes and Projects\\Py\\UAS\\ODLC\\output2'

# Loop over all images in the directory
for image_name in os.listdir(image_dir):
    # Open the image file
    with Image.open(os.path.join(image_dir, image_name)) as img:
        # Convert the image to RGB
        img = img.convert('RGB')

        # Apply the transformations
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Run the model on the image
        out = classifier(batch_t)

        # Get the probabilities by applying the softmax function
        probabilities = torch.nn.functional.softmax(out, dim=1)

        # Get the maximum probability
        max_prob, predicted = torch.max(probabilities, 1)

        # Only make a prediction if the maximum probability is above the threshold
        threshold = 0.99999
        if max_prob.item() >= threshold:
            print(f'Image {image_name} is predicted to be in class {predicted.item()} with probability {max_prob.item()}')
            # Save the image in the output directory
            img.save(os.path.join(output_dir, str(image_name.split('.')[0] + '_predicted_' + str(predicted.item()) + '.png')))
        else:
            print(f'The model is not sure about the class of image {image_name}')