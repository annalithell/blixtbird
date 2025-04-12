# Fenics Attack Models

This directory contains the attack implementations for the Fenics simulator.

## Available Attacks

1. **Poison Attack (poison)**: Model poisoning attack that adds noise to model parameters, defined in `attack_types/poison.py`
2. **Delay Attack (delay)**: Attack that simulates network delay, defined in `attack_types/delay.py`

## How to Use Different Attacks

You can specify which attacks to use by setting the `attacks` parameter when running Fenics:

```bash
# Command line
python fenics.py setup --use_attackers --attacks poison delay
python fenics.py run
```

Or in your YAML configuration file:

```yaml
simulations:
  my_simulation:
    # other parameters...
    use_attackers: true
    attacks: [poison, delay]
```

## Adding Custom Attacks

To add your own custom attack:

1. Create a new Python file in the `attack_types` directory with your attack class that inherits from `Attack`
2. Register your attack type in the AttackManager
3. Import your custom attack file before running Fenics

See the `examples/custom_attack_example.py` file for a complete example of how to create a custom attack.

### Example Code

```python
# my_attack.py
from fenics.attack.attack_types.base import Attack
import torch

class MyCustomAttack(Attack):
    def __init__(self, node_id, intensity=0.3, logger=None):
        super().__init__(node_id, logger)
        self.intensity = intensity
    
    def execute(self, model):
        # Implement your attack logic here
        self.logger.info(f"[node_{self.node_id}] Executing custom attack with intensity {self.intensity}")
        
        # Example: Flip the signs of all parameters
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(-self.intensity)
        
        self.logger.info(f"[node_{self.node_id}] Custom attack completed")
        return model.state_dict()
```

Then, when starting Fenics, you'll need to make sure your attack is registered and specified:

```bash
python fenics.py setup --use_attackers --attacks my_custom
python fenics.py run
```

## The Attack System

The attack system consists of:

1. `AttackManager`: Handles selecting attackers and planning when attacks occur
2. `Attack` base class: Defines the interface for all attack types
3. Specific attack implementations that implement the `execute` method

This modular design allows for easy extension with new attack types.