import torch
import torch.nn as nn
from torchvision import models

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomResNet50, self).__init__()
        # Load the ResNet50 model, without using pretrained weights in this case
        self.model = models.resnet50(weights=None)
        
        # Retrieve the original first convolutional layer
        original_first_layer = self.model.conv1
        
        # Create a new convolutional layer for 1-channel input
        self.model.conv1 = nn.Conv2d(1, original_first_layer.out_channels,
                                     kernel_size=original_first_layer.kernel_size,
                                     stride=original_first_layer.stride,
                                     padding=original_first_layer.padding,
                                     bias=False)
        
        # Set the weights of the new convolutional layer to the mean of the original layer's weights
        self.model.conv1.weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)
        
        # Change the number of output units in the final layer to match the desired number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        # Propagate the input x through the model
        return self.model(x)