# fenics/models/base.py

import torch.nn as nn
from abc import ABC, abstractmethod


class ModelBase(nn.Module, ABC):
    """
    Base class for all models in Fenics.
    Users can extend this class to create their own custom models.
    """
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass