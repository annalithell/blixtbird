# utils.py

import logging
import os
import numpy as np
from collections import defaultdict

def setup_logging(output_dir):
    """
    Configures logging to write to both console and a file within output_dir.
    Clears existing handlers to prevent duplication.
    
    Args:
        output_dir: Directory to save log file
    """
    logger = logging.getLogger()
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "simulation_log.txt"))
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    logger.setLevel(logging.INFO)
    logger.info("Logging is configured.")


def calculate_selection_probabilities(node_datasets):
    """
    Calculate selection probabilities based on the size of each node's dataset.
    
    Args:
        node_datasets: Dictionary mapping node IDs to their datasets
    
    Returns:
        List of selection probabilities for each node
    """
    data_sizes = [len(dataset) for dataset in node_datasets.values()]
    total_size = sum(data_sizes)
    probabilities = [size / total_size for size in data_sizes]
    return probabilities


def detect_convergence(metric_values, threshold=0.01, patience=3):
    """
    Detect the round where the metric (e.g., accuracy) converges.
    Convergence is defined as the improvement over the last `patience` rounds being below `threshold`.
    
    Args:
        metric_values: List of metric values over rounds
        threshold: Minimum improvement to continue training
        patience: Number of consecutive rounds to wait for improvement
    
    Returns:
        int or None: The round number where convergence occurred, or None if not converged
    """
    for i in range(len(metric_values) - patience):
        improvements = [metric_values[i + j + 1] - metric_values[i + j] for j in range(patience)]
        if all(impr < threshold for impr in improvements):
            return i + patience + 1  # +1 for 1-based indexing
    return None