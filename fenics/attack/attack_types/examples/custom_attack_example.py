# fenics/attack/attack_types/examples/custom_attack_example.py
# 
# This file shows how to create a custom attack type for Fenics.
# You can use this as a template for creating your own attacks.

import torch
import logging
from typing import Dict, Optional

# Import the necessary base class from Fenics
from fenics.attack.attack_types.base import Attack
# Import the AttackFactory for registration
from fenics.attack.attack_factory import AttackFactory


class LabelFlippingAttack(Attack):
    """
    Example of a custom attack for Fenics that simulates a label flipping attack.
    This attack modifies the last layer of the model to flip predictions.
    """
    
    def __init__(self, node_id: int, flip_intensity: float = 0.8, logger: Optional[logging.Logger] = None):
        """
        Initialize the label flipping attack.
        
        Args:
            node_id: ID of the attacker node
            flip_intensity: Intensity of the label flipping (0.0 to 1.0)
            logger: Logger instance
        """
        super().__init__(node_id, logger)
        self.flip_intensity = flip_intensity
    
    def execute(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Execute the label flipping attack by modifying the last layer weights.
        
        Args:
            model: The model to attack
            
        Returns:
            Modified model state dictionary
        """
        self.logger.info(f"[node_{self.node_id}] Executing label flipping attack with intensity {self.flip_intensity}")
        
        # Make a copy of the state dict
        state_dict = model.state_dict().copy()
        
        # Find the last layer weights and bias
        last_layer_weight = None
        last_layer_bias = None
        
        for name, param in state_dict.items():
            if 'weight' in name and len(param.shape) == 2:
                # This is likely a fully connected layer weight
                if last_layer_weight is None or param.shape[0] < last_layer_weight.shape[0]:
                    last_layer_weight = param
                    last_layer_name = name
            
            if 'bias' in name and len(param.shape) == 1:
                # This is likely a bias vector
                if last_layer_bias is None or param.shape[0] < last_layer_bias.shape[0]:
                    last_layer_bias = param
                    last_layer_bias_name = name
        
        if last_layer_weight is not None:
            # Modify the weights to flip predictions
            # For demonstration, we'll switch the weights of the first two classes
            num_classes = last_layer_weight.shape[0]
            if num_classes >= 2:
                # Create a permutation matrix that swaps classes
                with torch.no_grad():
                    # Get original weights
                    original_weights = last_layer_weight.clone()
                    
                    # Create permutation indices (swap adjacent classes)
                    idx = torch.arange(num_classes)
                    idx[::2] = idx[1::2].clone()
                    idx[1::2] = idx[::2].clone()
                    
                    # Apply permutation with intensity factor
                    new_weights = original_weights[idx]
                    mixed_weights = (1 - self.flip_intensity) * original_weights + self.flip_intensity * new_weights
                    
                    # Update the state dictionary
                    state_dict[last_layer_name] = mixed_weights
                    
                    # Also swap biases if available
                    if last_layer_bias is not None:
                        original_bias = last_layer_bias.clone()
                        new_bias = original_bias[idx]
                        mixed_bias = (1 - self.flip_intensity) * original_bias + self.flip_intensity * new_bias
                        state_dict[last_layer_bias_name] = mixed_bias
        
        self.logger.info(f"[node_{self.node_id}] Label flipping attack completed")
        return state_dict
    
# Register the attack with the AttackFactory
# This makes it available to use with the 'label_flipping' attack name
AttackFactory.register_attack('label_flipping', LabelFlippingAttack)


# =======================================================================
# How to use this custom attack:
# =======================================================================
#
# 1. Place this file in your project directory
#
# 2. Import this module before running Fenics:
#    ```
#    # In your script or notebook
#    import custom_attack_example
#    from fenics.attack.attack_types import Attack
#    
#    # Register the attack with the AttackManager
#    from fenics.attack import AttackManager
#    
#    # Assuming you're using a Simulation class:
#    def get_attack_type(node_id, round_num):
#        if node_id in attacker_node_ids and round_num in attacker_attack_rounds.get(node_id, set()):
#            # Use the custom attack type
#            return 'label_flipping'
#        return None
#    
#    # Create the attack when needed
#    def create_attack(node_id, attack_type):
#        if attack_type == 'label_flipping':
#            return LabelFlippingAttack(node_id, flip_intensity=0.8)
#        # Handle other attack types
#        return None
#    ```
#
# 3. Run Fenics with attackers and specify your attack:
#    ```
#    # Command line
#    python fenics.py setup --use_attackers --attacker_nodes 1 3 5 --attacks label_flipping
#    python fenics.py run
#    ```
#
#    or in the configuration file (config.yaml):
#    ```yaml
#    simulations:
#      my_simulation:
#        # other parameters...
#        use_attackers: true
#        attacker_nodes: [1, 3, 5]
#        attacks: [label_flipping]
#    ```
#
# 4. To create the attack manually with custom parameters:
#    ```python
#    from fenics.attack.attack_factory import AttackFactory
#    
#    # Create a label flipping attack with 0.5 intensity
#    attack = AttackFactory.get_attack('label_flipping', node_id=1, flip_intensity=0.5)
#    ```
#
# 5. For multiple custom attacks, you can create additional files
#    and import them all before running Fenics.
#
# Note: For integrating custom attacks more deeply, you may need to modify
# the AttackManager to recognize and handle your new attack type. The example
# above simplifies this process but might require adjustments based on your
# specific implementation.