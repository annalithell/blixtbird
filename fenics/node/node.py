# fenics/node/attacks/freerider.py

import torch
import logging
from typing import Optional
from fenics.node.nodetype import NodeType

from fenics.node.base import BaseNode


class Node(BaseNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int, data_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize a standard node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, data_path, logger)
        self.type = NodeType.NORMAL

    def train_model(self):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """


        return
        
    def execute(self):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        self.params = self.train_model()

        #self.logger.info(f"[node_{self.node_id}] is a normal node and do nothing")
        #return model.parameters()
