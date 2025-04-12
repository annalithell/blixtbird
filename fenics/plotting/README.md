# Fenics Plotting Module

This directory contains the plotting functionality for the Fenics simulator.

## Components

1. **metrics.py**: Contains functions for visualizing data distributions and metrics.

## Functions

### visualize_data_distribution

```python
def visualize_data_distribution(train_datasets, num_nodes, class_names, output_dir, logger):
    """
    Visualize the data distribution across nodes.
    
    Args:
        train_datasets: List of training datasets for each node
        num_nodes: Number of nodes in the network
        class_names: List of class names
        output_dir: Directory to save the visualization
        logger: Logger instance
    """
```

### plot_metrics_with_convergence

```python
def plot_metrics_with_convergence(metrics, rounds_range, total_execution_time, output_dir, logger):
    """
    Plot metrics with convergence detection.
    
    Args:
        metrics: Dictionary of metrics for each node
        rounds_range: Range of rounds to plot
        total_execution_time: Total execution time of the simulation
        output_dir: Directory to save the plots
        logger: Logger instance
    """
```

### plot_loss_line

```python
def plot_loss_line(metrics, rounds_range, output_dir, logger):
    """
    Plot loss lines over rounds.
    
    Args:
        metrics: Dictionary of metrics for each node
        rounds_range: Range of rounds to plot
        output_dir: Directory to save the plots
        logger: Logger instance
    """
```

### plot_training_aggregation_times

```python
def plot_training_aggregation_times(rounds_range, total_training_times, total_aggregation_times, total_execution_time, output_dir, logger):
    """
    Plot training and aggregation times over rounds.
    
    Args:
        rounds_range: Range of rounds to plot
        total_training_times: List of total training times for each round
        total_aggregation_times: List of total aggregation times for each round
        total_execution_time: Total execution time of the simulation
        output_dir: Directory to save the plots
        logger: Logger instance
    """
```

### plot_additional_metrics

```python
def plot_additional_metrics(rounds_range, cpu_usages, round_times, output_dir, logger):
    """
    Plot additional metrics such as CPU usage and round times.
    
    Args:
        rounds_range: Range of rounds to plot
        cpu_usages: List of CPU usage percentages for each round
        round_times: List of times taken for each round
        output_dir: Directory to save the plots
        logger: Logger instance
    """
```

## Usage

```python
# Visualize data distribution
visualize_data_distribution(train_datasets, num_nodes, class_names, output_dir, logger)

# Plot metrics with convergence detection
plot_metrics_with_convergence(metrics, rounds_range, total_execution_time, output_dir, logger)

# Plot loss lines
plot_loss_line(metrics, rounds_range, output_dir, logger)

# Plot training and aggregation times
plot_training_aggregation_times(rounds_range, total_training_times, total_aggregation_times, total_execution_time, output_dir, logger)

# Plot additional metrics
plot_additional_metrics(rounds_range, cpu_usages, round_times, output_dir, logger)
```