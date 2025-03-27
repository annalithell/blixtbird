# fenics/models/mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from fenics.models.base import ModelBase


class MLP(ModelBase):
    """
    Multi-Layer Perceptron for image classification.
    Example of an alternative model that users might add.
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], output_dim=10):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Input dimension (flattened image size)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        # Flatten the input if it's not already flat
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = self.layers(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)