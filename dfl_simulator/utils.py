# utils.py

import logging
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Prevent GUI-related errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import psutil

'''def setup_logging(output_dir):
    # Configure logging to write to both console and a file within output_dir
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "simulation_log.txt")),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is configured.")'''
def setup_logging(output_dir):
    """
    Configures logging to write to both console and a file within output_dir.
    Clears existing handlers to prevent duplication.
    """
    logger = logging.getLogger()
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "simulation_log.txt"))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    #console_handler = logging.StreamHandler()
    #console_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    #console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    #logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    logger.info("Logging is configured.")


def calculate_selection_probabilities(node_datasets):
    """
    Calculate selection probabilities based on the size of each node's dataset.
    
    Args:
        node_datasets (dict): Dictionary mapping node IDs to their datasets.
    
    Returns:
        list: List of selection probabilities for each node.
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
        metric_values (list): List of metric values over rounds.
        threshold (float): Minimum improvement to continue training.
        patience (int): Number of consecutive rounds to wait for improvement.
    
    Returns:
        int or None: The round number where convergence occurred, or None if not converged.
    """
    for i in range(len(metric_values) - patience):
        improvements = [metric_values[i + j + 1] - metric_values[i + j] for j in range(patience)]
        if all(impr < threshold for impr in improvements):
            return i + patience + 1 # +1 for 1-based indexing
    return None
