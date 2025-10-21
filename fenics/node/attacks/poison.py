# fenics/node/attacks/poison.py

import torch
import logging
from typing import Optional, override

from fenics.node.attacks.attack_node import AttackNode
from fenics.node.attacks.attack_registry import register_attack

@register_attack("poison")
class PoisonAttack(AttackNode):
    """Model poisoning attack that adds significant noise to model parameters."""

    def __init__(self, node_id: int, data_path: str, neighbors: Optional[int], model_type, epochs: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the poison attack.
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, data_path, neighbors, model_type, epochs, logger)
        self.attack_round = 0 # Placeholder for potential future use
        self.logger = logger or logging.getLogger()
    
    def _poison_state_dict(self, state_dict):
        for _, param in state_dict.items():
            
            if param is None:
                continue

            if not isinstance(param, torch.Tensor):
                continue

            noise = torch.rand_like(param) * (0.5 * param.std())

            with torch.no_grad():
                param.add_(noise)

        return state_dict

    def execute(self) -> None:
        """
        Execute the poison attack by adding noise to model parameters.
        
        Args:
            model: Model to poison
        """
        self.model_params, self.training_time = self.train_model()
        self.logger.info(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")

        if not isinstance(self.model_params, dict):
            self.logger.error("Unexpected model_params type: %s", type(self.model_params))
            return self.model_params, self.training_time
        
        poisoned_params = self._poison_state_dict(self.model_params)

        try:
            self.model_params = poisoned_params
        except Exception as e:
            self.logger.debug(f"Could not load poisoned params into model: {e}")