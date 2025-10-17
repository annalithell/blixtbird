from fenics.node.abstract import AbstractNode
from fenics.node.node_type import NodeType
from fenics.node.attacks.attack_registry import get_attack 
from typing import Optional, Dict, Type, Callable, override
import logging

##__attack_registry__: Dict[str, Type] = {}

class AttackNode(AbstractNode):
    """ Base class for all attack nodes. """
    def __init__(self, node_id:int, attack_name : str ="none", logger:Optional[logging.Logger]=None):
        """
        Initialize an attack node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id)
        self.logger = logger or logging.getLogger()
        self.attack = get_attack(attack_name, node_id=node_id)
        self.attack_type = self.attack.__attack_type__
        self.type = NodeType.ATTACK 
    

    @override
    def execute(self, *args, **kwargs):
        """
        Execution function for an attack node.
        
        Returns:
            Result of the attack node execution, depending on the specific attack strategy
        """
        self.logger.info(f"[node_{self.node_id}] is an attack node and do nothing")
        return None 