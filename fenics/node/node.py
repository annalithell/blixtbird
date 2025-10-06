# fenics/node/attacks/freerider.py

import torch
import logging
from typing import Optional

from fenics.node.base import BaseNode


class Node(BaseNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize a standard node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, logger)
        
    def execute(self, model: torch.nn.Module):
        """
        Execution function for a standard node.
        
        Returns:
            Model parameters of the node
        """
        self.logger.info(f"[node_{self.node_id}] is a normal node and do nothing")
        return model.parameters()
