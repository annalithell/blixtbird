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
    
    def _poison_state_dict(self):
        for _, param in self.model.state_dict().items():
            
            if param is None:
                continue

            if not isinstance(param, torch.Tensor):
                continue

            noise = torch.rand_like(param) * (3 * param.std())

            with torch.no_grad():
                param.add_(noise)


    def execute(self) -> None:
        """
        Execute the poison attack by adding noise to model parameters.
        
        Args:
            model: Model to poison
        """
        self.train_model()
        print(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")

        if not isinstance(self.model.state_dict(), dict):
            print(f"Unexpected model_state_dict type: %s", type(self.model.state_dict()))
            return self.model.state_dictx(), self.training_time
        
        self._poison_state_dict()
        """
        try:
            self. = poisoned_params
        except Exception as e:
            self.logger.debug(f"Could not load poisoned params into model: {e}")
        """

        print(f"[Node {self.node_id}] will now start sending data")
        self.send()
        print(f"[Node {self.node_id}] has completed sending, now stating the recieve operation")
        self.recv()
        print(f"[Node {self.node_id}] Communication completed, starting aggregation .....")
        self.aggregate()
        #evaluate model after aggregation
        self.append_test_metrics_after_aggregation()
