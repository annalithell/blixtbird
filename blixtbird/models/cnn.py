# fenics/models/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from blixtbird.models.base import ModelBase


class Net(ModelBase):
    """
    Convolutional Neural Network for image classification.
    Default model used in Fenics for FashionMNIST dataset.
    """
    
    def __init__(self):
        super(Net, self).__init__()
        # First Conv2D + MaxPooling layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second Conv2D + MaxPooling layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third Conv2D + MaxPooling layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output