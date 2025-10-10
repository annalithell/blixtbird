# fenics/node/attacks/freerider.py

import torch
import logging
from typing import Optional, override


from fenics.node.attacks.base import BaseAttack
from fenics.node.attacks.attackregistry import register_attack


@register_attack("freerider")
class FreeRiderAttack(BaseAttack):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the free-rider attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
            self.attack_type = FREERIDER
        """
        super().__init__(node_id)
        self.attack_round = 0 # Placeholder for potential future use
        self.attack_type = self.__class__.__attack_type__ 
        
    @override
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