import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        # First convolution layer: 32 filters, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        # Second convolution layer: 64 filters, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolution layer: 128 filters, 3x3 kernel, stride 1, padding 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 16 * 16, num_classes)  # Fixed dimensions
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)  # Flattening before FC layer
        x = self.fc(x)
        return x
