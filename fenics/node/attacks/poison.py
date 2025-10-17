# fenics/node/attacks/poison.py

import torch
import logging
from typing import Optional, override

from fenics.node.attacks.attack_node import AttackNode
from fenics.node.attacks.attack_registry import register_attack

@register_attack("poison")
class PoisonAttack(AttackNode):
    """Model poisoning attack that adds significant noise to model parameters."""

    def __init__(self, node_id: int, neighbors: Optional[int], data_path: str, attack_type: str = "poison", logger: Optional[logging.Logger] = None):
        """
        Initialize the poison attack.
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, neighbors, data_path, logger)
        self.attack_round = 0 # Placeholder for potential future use
        self.__attack_type__ = attack_type
        self.logger = logger or logging.getLogger()
    
    #@override
    def execute(self) -> None:
        """
        Execute the poison attack by adding noise to model parameters.
        
        Args:
            model: Model to poison
        """
        #with torch.no_grad():
        #    for param in model.parameters():
        #        param.add_(torch.randn(param.size()) * 0.5)  # Add significant noise
        
        #self.logger.info(f"[node_{self.node_id}] Model poisoned.")
        return