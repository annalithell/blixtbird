# fenics/node/base.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from mpi4py import MPI
#from fenics.node.node_type import NodeType

from fenics.models import ModelFactory
    
class AbstractNode(ABC):
    """ A  base node class for all nodes. """    
    
    def __init__(self, node_id: int, data_path: Optional[str] = None, neighbors: Optional[int] = None, model_type: Optional[str] = None, epochs: Optional[int] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the node
        
        Args:
            node_id: ID of the node
            data_path: Path to training data for node
            logger: Logger instance
        """
        self.node_id = node_id
        self.data_path = data_path
        self.model_params = None  # Placeholder for model parameters
        self.neighbors = neighbors

        self.neighbor_models = {}
        self.neighbor_statedicts = {}
        self.comm = MPI.COMM_WORLD

        self.data_sizes = {}
        self.data_sizes[self.node_id] = 0 

        self.model = ModelFactory.get_model(model_type)
        self.metrics_train = []
        self.metrics_test = []
        self.epochs = epochs

        self.logger = logger or logging.getLogger()
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execution function for the node.
        
        Returns:
            Result of the node execution, depending on the node type
        """
        pass