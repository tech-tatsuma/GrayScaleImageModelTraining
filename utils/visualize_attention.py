import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image

from models.visiontransformer.vit import VisionTransformer
from datasets.dataset import CustomImageDataset

def visualize_attention_map(model, input_image, input_image_path, output_path):
    # Load the original image
    original_image = Image.open(input_image_path).convert('RGB')
    # Convert to grayscale if in RGB mode
    if original_image.mode == 'RGB':
            original_image = original_image.convert('L')

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to 128x128
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize(mean=[mean], std=[std]) # Normalize using calculated mean and std
    ])
    input_image = transform(original_image).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Execute a forward pass with the model and input image to get attention weights
    _, attn_weights = model(input_image)
    all_attn_weights = torch.stack([attn_weight.mean(0) for attn_weight in attn_weights], dim=0)

    # Calculate the average of attention weights and convert to a numpy array
    attn_map = all_attn_weights.mean(0).detach().cpu().numpy()

    # Resize (upscale) the attention map
    attn_map = Image.fromarray(attn_map).resize((original_image.width, original_image.height), Image.BILINEAR)
    attn_map = np.array(attn_map)
    attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))  # Normalize
    attn_map = Image.fromarray(np.uint8(plt.cm.jet(attn_map) * 255))

    # Overlay the attention map on the original image
    blended_image = Image.blend(original_image, attn_map, alpha=0.5)

    # Display and save the overlaid image
    plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)
    plt.axis('off')  # Hide the axis
    plt.savefig(output_path)
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Temporary transformation to calculate mean and std
    temp_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    all_dataset = CustomImageDataset(directory='archive', transform=temp_transform)

    # Get a data loader
    temp_loader = DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False)

    # Calculate mean and standard deviation from the dataset
    data = next(iter(temp_loader))[0]
    mean = data.mean([0, 2, 3])
    std = data.std([0, 2, 3])

    # Function to load and preprocess image
    def preprocess_image(image_path, size=(128, 128)):
        image = Image.open(image_path).convert('RGB')
        if image.mode == 'RGB':
            image = image.convert('L')  # Convert RGB to grayscale
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),  # Adjust for 1-channel mean and std
        ])
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        return image

    input_image = preprocess_image(args.image_path)

    # Load the model
    model = VisionTransformer(
        image_size=128,
        patch_size=16,
        in_channels=1,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=4,
        hidden_dims=[64, 128, 256]
    ).to(device)

    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path))  # Load pretrained model weights
    model.eval()  # Set model to evaluation mode

    # Visualize the attention map
    visualize_attention_map(model, input_image, args.image_path, args.output_path)  # Visualize attention map