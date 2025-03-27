# fenics/attack/attack_types/base.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional


class Attack(ABC):
    """Base class for all attack types."""
    
    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the attack.
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        self.node_id = node_id
        self.logger = logger or logging.getLogger()
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the attack.
        
        Returns:
            Result of the attack, depending on the attack type
        """
        pass