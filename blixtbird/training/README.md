# Fenics Training Module

**OBSERVE: This functionality has been inherited from Fenics and not been thoroughly tested for Blixtbird**

**TODO: Refactor training methods utilized in the Blixtbird framework to this class**

This directory contains the training and evaluation functionality for the Fenics simulator.

## Components

1. **trainer.py**: Contains the `local_train` function for training models on local data.
2. **evaluator.py**: Contains the `evaluate` function for evaluating models on test data.
3. **utils.py**: Contains utility functions like `summarize_model_parameters`.

## Functions

### local_train

```python
def local_train(node_id, local_model, train_dataset, epochs, attacker_type):
    """
    Train a local model for a specific node.
    
    Args:
        node_id: ID of the node
        local_model: Model to train
        train_dataset: Training dataset
        epochs: Number of training epochs
        attacker_type: Type of attack to simulate (if any)
        
    Returns:
        Tuple of (model state dictionary, training time)
    """
```

### evaluate

```python
def evaluate(model, test_loader):
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        
    Returns:
        Tuple of (test loss, accuracy, f1 score, precision, recall)
    """
```

### summarize_model_parameters

```python
def summarize_model_parameters(node_name, model_state_dict, logger):
    """
    Summarize model parameters for a node after local training.
    
    Args:
        node_name: Name of the node
        model_state_dict: Model state dictionary
        logger: Logger instance
    """
```

## Customization

To customize the training process, you can modify the hyperparameters in `trainer.py`:

- Learning rate
- Weight decay
- Batch size
- Optimizer type
- Loss function

For different evaluation metrics, you can modify `evaluator.py` to add or change metrics.