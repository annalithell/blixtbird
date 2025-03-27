# fenics/attack/attack_types/poison.py

import torch
import logging
from typing import Optional

from fenics.attack.attack_types.base import Attack


class PoisonAttack(Attack):
    """Model poisoning attack that adds significant noise to model parameters."""
    
    def __init__(self, node_id: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the poison attack.
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, logger)
    
    def execute(self, model: torch.nn.Module) -> None:
        """
        Execute the poison attack by adding noise to model parameters.
        
        Args:
            model: Model to poison
        """
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.5)  # Add significant noise
        
        self.logger.info(f"[node_{self.node_id}] Model poisoned.")


# import torch
# import logging
# from typing import Dict, Optional

# from fenics.attack.attack_types.base import Attack


# class PoisonAttack(Attack):
#     """Model poisoning attack that adds significant noise to model parameters."""
    
#     def __init__(self, node_id: int, noise_scale: float = 0.5, logger: Optional[logging.Logger] = None):
#         """
#         Initialize the poison attack.
        
#         Args:
#             node_id: ID of the attacker node
#             noise_scale: Scale of the noise to add
#             logger: Logger instance
#         """
#         super().__init__(node_id, logger)
#         self.noise_scale = noise_scale
    
#     def execute(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
#         """
#         Execute the poison attack by adding noise to model parameters.
        
#         Args:
#             model: Model to poison
            
#         Returns:
#             Poisoned model state dictionary
#         """
#         self.logger.info(f"[node_{self.node_id}] Executing model poisoning attack with noise scale {self.noise_scale}")
        
#         # Make a copy of the state dict to avoid modifying the original
#         poisoned_state_dict = {}
        
#         with torch.no_grad():
#             for name, param in model.state_dict().items():
#                 # Add random noise to the parameter
#                 noise = torch.randn_like(param) * self.noise_scale
#                 poisoned_param = param + noise
#                 poisoned_state_dict[name] = poisoned_param
        
#         self.logger.info(f"[node_{self.node_id}] Model successfully poisoned")
#         return poisoned_state_dict