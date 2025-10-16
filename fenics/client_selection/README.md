# Fenics Client Selection Module

This directory contains the client selection functionality for the Fenics simulator.

## Components

1. **selector.py**: Contains the `ClientSelector` class that manages client selection.
2. **strategies/**: Directory containing various client selection strategy implementations:
   - **uniform.py**: Uniform random selection strategy
   - **md_sampling.py**: Multinomial distribution sampling strategy
3. **factory.py**: Contains the factory for creating selection strategy instances.

## Available Strategies

1. **Uniform Selection (uniform)**: Selects clients uniformly at random.
2. **MD Sampling (md_sampling)**: Selects clients based on multinomial distribution using predefined probabilities.

## Using Different Selection Strategies

In your code, you can use the `ClientSelector` class:

```python
from fenics.client_selection import ClientSelector

# Create a client selector with a 60% participation rate
selector = ClientSelector(nodes=list(range(10)), participation_rate=0.6)

# Select clients uniformly
selected_nodes = selector.select_clients_uniform()

# Select clients based on probabilities
probabilities = [0.1, 0.2, 0.1, 0.05, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]
selected_nodes = selector.select_clients_md_sampling(probabilities)

# Precompute nodes for all rounds
nodes_per_round = selector.precompute_participating_nodes(num_rounds=5, probabilities=probabilities)
```

Or you can use the factory to get specific strategies:

```python
from fenics.client_selection.factory import SelectionFactory

# Get a specific strategy
uniform_strategy = SelectionFactory.get_strategy('uniform')
selected_nodes = uniform_strategy(nodes=list(range(10)), num_participants=5)
```

## Creating Custom Selection Strategies

To add your own custom selection strategy:

1. Create a new Python file in the `strategies` directory (e.g., `weighted.py`)
2. Implement your selection strategy function in this file
3. Register it with the `SelectionFactory`

### Example Code

```python
# fenics/client_selection/strategies/weighted.py

import numpy as np
import logging
from typing import List, Optional

def select_clients_weighted(nodes: List[int], 
                           num_participants: int, 
                           weights: List[float],
                           logger: Optional[logging.Logger] = None) -> List[int]:
    """
    Select clients with higher preference for nodes with higher weights.
    
    Args:
        nodes: List of all node IDs
        num_participants: Number of nodes to select
        weights: List of weights for each node
        logger: Logger instance
        
    Returns:
        List of selected node IDs
    """
    logger = logger or logging.getLogger()
    
    # Normalize weights to probabilities
    probabilities = [w / sum(weights) for w in weights]
    
    # Select nodes based on probabilities
    selected_indices = np.random.choice(
        len(nodes), 
        size=num_participants, 
        replace=False, 
        p=probabilities
    )
    
    selected_nodes = [nodes[i] for i in selected_indices]
    logger.info(f"Selected nodes using weighted strategy: {selected_nodes}")
    return selected_nodes
```

Then register the strategy in your main code:

```python
# Import your custom strategy
from fenics.client_selection.strategies.weighted import select_clients_weighted

# Register the strategy
SelectionFactory.register_strategy('weighted', select_clients_weighted)
```

Then use it:

```python
weighted_strategy = SelectionFactory.get_strategy('weighted')
selected_nodes = weighted_strategy(
    nodes=list(range(10)), 
    num_participants=5, 
    weights=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)
```

## The SelectionFactory

The `SelectionFactory` class in `factory.py` is responsible for managing the available selection strategies. It provides methods for:

1. Registering new strategy types
2. Creating strategy instances by name
3. Listing all available strategies

This allows for easy extensibility and runtime selection of client selection strategies.