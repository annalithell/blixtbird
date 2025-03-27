# custom_model_example.py
# 
# This file shows how to create and register a custom model with Fenics.
# You can use this as a template for creating your own models.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the necessary base classes from Fenics
from fenics.models import ModelBase, ModelFactory


class MyCustomModel(ModelBase):
    """
    Example of a custom model for Fenics.
    This is a simple CNN with customizable number of filters.
    """
    
    def __init__(self, num_filters=32):
        """
        Initialize the custom model.
        
        Args:
            num_filters: Number of filters in the first convolutional layer
        """
        super().__init__()
        
        # Define your model architecture here
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate the size of the flattened features
        # For FashionMNIST (28x28), after two 2x2 pooling layers: 28/2/2 = 7
        flattened_size = (num_filters*2) * 7 * 7
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for FashionMNIST
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply log softmax to get log probabilities
        return F.log_softmax(x, dim=1)


# Register the model with the factory
# This makes it available to use with --model_type custom_model
ModelFactory.register_model('custom_model', MyCustomModel)


# =======================================================================
# How to use this custom model:
# =======================================================================
#
# 1. Place this file in your project directory
#
# 2. Import this module before running Fenics:
#    ```
#    # In your script or notebook
#    import custom_model_example
#    ```
#
# 3. Run Fenics with the model_type parameter:
#    ```
#    # Command line
#    python fenics.py setup --model_type custom_model
#    python fenics.py run
#    ```
#
#    or in the configuration file (config.yaml):
#    ```yaml
#    simulations:
#      my_simulation:
#        # other parameters...
#        model_type: custom_model
#    ```
# 
# 4. To pass parameters to your model:
#    ```python
#    # This creates a custom model with 64 filters
#    custom_model = ModelFactory.get_model('custom_model', num_filters=64)
#    ```
#
# 5. For multiple custom models, you can create additional files
#    and import them all before running Fenics.