# fenics/node/attacks/poison.py

import torch
import logging
from typing import Optional

from fenics.node.base import BaseNode


class PoisonAttack(BaseNode):
    """Model poisoning attack that adds significant noise to model parameters."""
    
    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the poison attack.
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, logger)
    
    def execute(self, model: torch.nn.Module) -> None:
        """
        Execute the poison attack by adding noise to model parameters.
        
        Args:
            model: Model to poison
        """
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.5)  # Add significant noise
        
        self.logger.info(f"[node_{self.node_id}] Model poisoned.")
