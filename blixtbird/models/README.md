# Fenics Models

This directory contains the model implementations for the Fenics simulator.

## Available Models

1. **CNN (cnn)**: Default convolutional neural network model defined in `cnn.py`
2. **MLP (mlp)**: Multi-layer perceptron model defined in `mlp.py`

## How to Use Different Models

You can specify which model to use by setting the `model_type` parameter when running Fenics:

```bash
# Command line
python fenics.py setup --model_type mlp
python fenics.py run
```

Or in your YAML configuration file:

```yaml
simulations:
  my_simulation:
    # other parameters...
    model_type: mlp
```

## Adding Custom Models

To add your own custom model:

1. Create a new Python file with your model class that inherits from `ModelBase`
2. Register your model with `ModelFactory`
3. Import your custom model file before running Fenics

See the `examples/custom_model_example.py` file for a complete example of how to create and register a custom model.

### Example Code

```python
# my_model.py
from fenics.models import ModelBase, ModelFactory
import torch.nn as nn
import torch.nn.functional as F

class MyModel(ModelBase):
    def __init__(self):
        super().__init__()
        # Define your model architecture
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Implement forward pass
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)

# Register the model
ModelFactory.register_model('my_model', MyModel)
```

Then import your file before running Fenics:

```python
import my_model  # This registers your model
```

And use it:

```bash
python fenics.py setup --model_type my_model
python fenics.py run
```

## The ModelFactory

The `ModelFactory` class in `factory.py` is responsible for managing the available models. It provides three main methods:

1. `register_model(name, model_class)`: Register a new model
2. `get_model(model_name, **kwargs)`: Get an instance of a model by name
3. `list_available_models()`: Get a dictionary of all registered models

You can use these methods to dynamically register and retrieve models at runtime.