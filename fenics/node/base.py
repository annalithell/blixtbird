# fenics/node/base.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
    
class BaseNode(ABC):
    """ A  base node class for all nodes. """    
    
    def __init__(self, node_id: int, data_path: Optional[str] = None, neighbors: Optional[int] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the node
        
        Args:
            node_id: ID of the node
            data_path: Path to training data for node
            logger: Logger instance
        """
        self.node_id = node_id
        self.data_path = data_path
        self.neighbors = neighbors
        self.model_params = None  # Placeholder for model parameters
        self.logger = logger or logging.getLogger()
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execution function for the node.
        
        Returns:
            Result of the node execution, depending on the node type
        """
        pass