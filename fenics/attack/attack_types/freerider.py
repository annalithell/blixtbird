# fenics/attack/attack_types/freerider.py

import torch
import logging
from typing import Optional

from fenics.attack.attack_types.base import Attack
#from fenics.attack.attack_factory import AttackFactory

class FreeRiderAttack(Attack):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the free-rider attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, logger)
        
    def execute(self, model: torch.nn.Module):
        """
        Execute the free-rider attack by learning model parameters while doing no work. 
        
        Args:
            model: Model to intercept
        """
        self.logger.info(f"[node_{self.node_id}] is a freer-rider: intercepts {model.parameters()}")
        return model.state_dict()


## This is done explicitly in attack_factory.py
# Register the attack
#AttackFactory.register_attack('freerider', FreeRiderAttack)