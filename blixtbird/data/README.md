# Blixtbird Data Module

This directory contains the data handling functionality for the Fenics simulator.

## Components

1. **handler.py**: Contains functions for loading and distributing datasets.
2. **module.py**: Contains the `DataModule` class that encapsulates data operations.

## Functions

### load_datasets_dirichlet

```python
def load_datasets_dirichlet(num_nodes, alpha):
    """
    Load and distribute the FashionMNIST dataset among nodes using Dirichlet distribution.
    
    Args:
        num_nodes: Number of nodes
        alpha: Dirichlet distribution parameter
        
    Returns:
        Tuple of (train_datasets, test_dataset, labels)
    """
```

### print_class_distribution

```python
def print_class_distribution(train_datasets, logger):
    """
    Print the class distribution for each node's dataset.
    
    Args:
        train_datasets: List of training datasets for each node
        logger: Logger instance
    """
```

### distribute_data_dirichlet

```python
def distribute_data_dirichlet(labels, num_nodes, alpha):
    """
    Distribute data indices among nodes according to a Dirichlet distribution.
    
    Args:
        labels: Array of data labels
        num_nodes: Number of nodes
        alpha: Dirichlet distribution parameter
        
    Returns:
        Dictionary mapping node indices to lists of data indices
    """
```

## DataModule

The `DataModule` class provides a high-level interface for data operations:

```python
class DataModule:
    """
    A module to handle all data loading and preprocessing operations.
    """
    
    def __init__(self, num_nodes, alpha, topology, topology_file=None, output_dir="results",
                 logger=None, batch_size=32, random_seed=0):
        # Initialize the module
        
    def setup(self):
        # Load datasets and create topology
        
    def get_train_loader(self, node_id):
        # Get training loader for a node
        
    def get_test_loader(self, node_id):
        # Get test loader for a node
        
    def get_data_sizes(self):
        # Get dataset sizes for each node
        
    def calculate_selection_probabilities(self):
        # Calculate selection probabilities
```

## Usage

```python
# Create and setup the data module
data_module = DataModule(
    num_nodes=10,
    alpha=0.5,
    topology='fully_connected',
    output_dir='results'
)
data_module.setup()

# Get training and test loaders
train_loader = data_module.get_train_loader(node_id=0)
test_loader = data_module.get_test_loader(node_id=0)

# Get data sizes and selection probabilities
data_sizes = data_module.get_data_sizes()
probabilities = data_module.calculate_selection_probabilities()
```