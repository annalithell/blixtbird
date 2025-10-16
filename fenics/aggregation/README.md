# Fenics Aggregation Module

This directory contains the model aggregation functionality for the Fenics simulator.

## Components

1. **base.py**: Contains the base class for aggregation strategies.
2. **fedavg.py**: Contains the implementation of Federated Averaging algorithm.
3. **factory.py**: Contains the factory for creating aggregation strategy instances.

## Available Strategies

1. **Federated Averaging (fedavg)**: Weighted average of model parameters based on dataset sizes.

## Using Different Strategies

You can specify which aggregation strategy to use programmatically:

```python
from fenics.aggregation import AggregationFactory

# Get a FedAvg strategy instance
strategy = AggregationFactory.get_strategy('fedavg')

# Use the strategy to aggregate models
aggregated_model = strategy.aggregate(models_state_dicts, data_sizes)
```

For backward compatibility, you can also use the traditional function:

```python
from fenics.aggregation import aggregate

# Aggregate models using FedAvg
aggregated_model = aggregate(models_state_dicts, data_sizes, logger)
```

## Creating Custom Aggregation Strategies

To add your own custom aggregation strategy:

1. Create a new Python file with your strategy class that inherits from `AggregationStrategy`
2. Implement the required methods, especially `aggregate()`
3. Register your strategy with `AggregationFactory`
4. Import your custom strategy file before using it

### Example Code

```python
# my_strategy.py
import torch
from fenics.aggregation import AggregationStrategy, AggregationFactory

class MedianStrategy(AggregationStrategy):
    """
    Aggregation strategy that takes the median of each parameter.
    """
    
    def aggregate(self, models_state_dicts, data_sizes):
        if not models_state_dicts:
            self.logger.warning("No models to aggregate.")
            return None
            
        # Initialize an empty state dict for the aggregated model
        aggregated_state_dict = {}
        
        # Get the list of all parameter keys
        param_keys = list(models_state_dicts[0].keys())
        
        for key in param_keys:
            # Stack parameters from all models
            stacked_params = torch.stack([state_dict[key] for state_dict in models_state_dicts])
            # Take the median along the first dimension (across models)
            aggregated_state_dict[key] = torch.median(stacked_params, dim=0).values
            
        self.logger.info("Aggregation using Median Strategy completed successfully.")
        return aggregated_state_dict

# Register the strategy
AggregationFactory.register_strategy('median', MedianStrategy)
```

Then import and use your custom strategy:

```python
import my_strategy  # This registers your strategy

# Get your custom strategy
strategy = AggregationFactory.get_strategy('median')
```

## The AggregationFactory

The `AggregationFactory` class in `factory.py` is responsible for managing the available strategies. It provides methods for:

1. Registering new strategy types
2. Creating strategy instances by name
3. Listing all available strategies

This allows for easy extensibility and runtime selection of aggregation strategies.