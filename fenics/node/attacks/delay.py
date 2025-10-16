# fenics/node/attacks/delay.py

import random
import time
import logging
from typing import Optional, override

from fenics.node.attacks.base import BaseAttack
from fenics.node.attacks.attackregistry import register_attack

@register_attack("delay")
class DelayAttack(BaseAttack):
    """Delay attack that simulates network delay by sleeping."""

    def __init__(self, node_id: int, attack_type: str = "DELAY", logger: Optional[logging.Logger] = None):
        """
        Initialize the delay attack.
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """

        super().__init__(node_id)
        self.attack_round = 0 # Placeholder for potential future use
        self.__attack_type__ = attack_type
        self.logger = logger or logging.getLogger()
        
    #@override
    def execute(self) -> float:
        """
        Execute the delay attack by sleeping for a random amount of time.
        
        Returns:
            Total time spent including the delay
        """
        start_time = time.time()
        
        # Using the exact same delay range as the original code
        delay_duration = random.uniform(500, 700)
        self.logger.info(f"[node_{self.node_id}] Delaying sending updates by {delay_duration:.2f} seconds.")
        
        # Simulate network delay
        time.sleep(delay_duration)
        
        # Calculate total time spent
        end_time = time.time()
        sending_time = end_time - start_time
        
        return sending_time
