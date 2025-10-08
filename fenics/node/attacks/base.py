# fenics/node/attacks/baseattack.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from fenics.node.attacktype import AttackType
from fenics.node.nodetype import NodeType

class BaseAttack(ABC):
    """ A  base attack class for all attacks. """    
    
    def __init__(self, node_id: int, node: NodeType = NodeType.ATTACK,logger: Optional[logging.Logger] = None):
        """
        Initialize the attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        self.node_id = node_id
        self.node_type = node
        self.training_data = None  # Placeholder for training data
        self.model_params = None  # Placeholder for model parameters
        self.logger = logger or logging.getLogger()
        self.attack_type = AttackType.NONE

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execution function for the node.
        
        Returns:
            Result of the node execution, depending on the node type
        """
        pass