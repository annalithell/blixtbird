# fenics/node/attacks/freerider.py

import torch
import logging
from typing import Optional, override


from fenics.node.attacks.attack_node import AttackNode
from fenics.node.attacks.attack_registry import register_attack


@register_attack("freerider")
class FreeRiderAttack(AttackNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """

    def __init__(self, node_id: int, neighbors: Optional[int], data_path: str, attack_type: str = "freerider", logger: Optional[logging.Logger] = None):
        """
        Initialize the free-rider attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
            self.attack_type = freerider
        """
        super().__init__(node_id, neighbors, data_path, logger)
        self.attack_round = 0 # Placeholder for potential future use
        self.__attack_type__ = attack_type # TODO redundant?
        self.logger = logger or logging.getLogger()

    #@override
    def execute(self,epochs):
        """
        Execute the free-rider attack by learning model parameters while doing no work. 
        
        Args:
            model: Model to intercept
        """
        #self.logger.info(f"[node_{self.node_id}] is a freer-rider: intercepts {model.parameters()}")
        return 


## This is done explicitly in attack_factory.py
# Register the attack
#AttackFactory.register_attack('freerider', FreeRiderAttack)