from fenics.node.base import BaseNode, NodeType
from fenics.node.attacks import delay, poison, freerider
from typing import Optional, Dict, Type, Callable, override
import logging

##__attack_registry__: Dict[str, Type] = {}

class AttackNode(BaseNode):
    """ Base class for all attack nodes. """
    def __init__(self, node_id:int, logger:Optional[logging.Logger]=None):
        """
        Initialize an attack node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, NodeType.ATTACK, logger)
    
    ##def register_attack(name: str = None) -> Callable[[Type], Type]:
        
    @override
    def execute(self, *args, **kwargs):
        """
        Execution function for an attack node.
        
        Returns:
            Result of the attack node execution, depending on the specific attack strategy
        """
        self.logger.info(f"[node_{self.node_id}] is an attack node and do nothing")
        return None 