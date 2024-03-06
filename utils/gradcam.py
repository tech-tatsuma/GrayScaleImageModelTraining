import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet50 import CustomResNet50 # Assuming CustomResNet50 is defined in models.resnet50
from torchvision import transforms
from PIL import Image
import argparse

from datasets.dataset import SilhouetteImageDataset

# Class definition for GradCAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model # The model to be inspected
        self.target_layer = target_layer # The specific layer to focus on
        self.gradients = None # Variable to store gradients
        self.forward_output = None # Variable to store forward pass outputs

        # Setting hooks for target layer
        target_layer.register_forward_hook(self.save_forward_output) # Save output after forward pass
        target_layer.register_full_backward_hook(self.save_gradients) # Save gradients after backward pass

    # Method to save the output of the forward pass
    def save_forward_output(self, module, input, output):
        self.forward_output = output

    # Method to save gradients
    def save_gradients(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]

    # Method to generate the Class Activation Map (CAM)
    def generate_cam(self, input_image, target_class=None):
        # Set model to evaluation mode
        self.model.eval()
        output = self.model(input_image) # Forward pass to get prediction
        if target_class is None:
            target_class = output.argmax().item() # If no target class is specified, use the class with highest score
        output[:, target_class].backward() # Compute gradients for the target class

        # Obtain gradients and feature maps, calculate weights
        gradients = self.gradients.detach() 
        forward_output = self.forward_output.detach() 
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True) # Calculate average gradients as weights

        # Compute the Class Activation Map
        cam = torch.mul(forward_output, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam) # Keep only positive influences
        cam = F.interpolate(cam, input_image.shape[2:], mode='bilinear', align_corners=False) # Resize to match input image size
        cam = cam - cam.min() # Normalize by subtracting the minimum value
        cam = cam / cam.max() # Normalize by dividing by the maximum value

        return cam

# Function to visualize CAM overlayed on the input image
def visualize_cam(cam_image, input_image, save_path=None):
    cam_image = cam_image.cpu().squeeze().numpy() # Convert CAM to numpy array
    input_image = TF.to_pil_image(input_image.cpu().squeeze()) # Convert input image to PIL format
    plt.imshow(input_image, alpha=0.5) 
    plt.imshow(cam_image, cmap='jet', alpha=0.5) # Overlay CAM using jet colormap
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image", type=str, required=True)
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--output_path", type=str, required=True)
    args = argparser.parse_args()

    temp_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    all_dataset = SilhouetteImageDataset(directory='archive', transform=temp_transform)

    # Get DataLoader
    temp_loader = DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False)

    # Calculate mean and standard deviation of the dataset
    data = next(iter(temp_loader))[0]
    mean = data.mean([0, 2, 3])
    std = data.std([0, 2, 3])

    # Specify the model and the target layer
    model = CustomResNet50(num_classes=4)
    model_path = args.model
    model.load_state_dict(torch.load(model_path))
    target_layer = model.model.layer4

    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)

    # Initialize GradCAM
    def preprocess_image(image_path, size=(128, 128)):
        image = Image.open(image_path).convert('RGB')
        # transform to grayscale if image is 3-channel RGB
        if image.mode == 'RGB':
            image = image.convert('L')  
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),  # 1-channel mean and std
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    image_path = args.image

    # Preprocess the image
    input_image = preprocess_image(image_path)

    # Generate CAM
    cam = grad_cam.generate_cam(input_image)

    # CAMの可視化
    visualize_cam(cam, input_image, save_path=args.output_path)