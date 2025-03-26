# dfl_simulator/aggregation.py

import torch
import logging

def aggregate(models_state_dicts, data_sizes, logger=None):
    """
    Aggregates model parameters using Weighted Federated Averaging (FedAvg).

    Args:
        models_state_dicts (list of dict): List of state dictionaries from each participating node.
        data_sizes (list of int): List of data sizes corresponding to each node.
        logger (logging.Logger, optional): Logger for logging aggregation details.

    Returns:
        dict or None: Aggregated state dictionary if models are provided, else None.
    """
    if not models_state_dicts:
        if logger:
            logger.warning("No models to aggregate.")
        return None

    # Initialize an empty state dict for the aggregated model
    aggregated_state_dict = {}
    total_data = sum(data_sizes)

    # Get the list of all parameter keys
    param_keys = list(models_state_dicts[0].keys())

    for key in param_keys:
        # Initialize a tensor for the weighted sum
        weighted_sum = torch.zeros_like(models_state_dicts[0][key])
        for state_dict, size in zip(models_state_dicts, data_sizes):
            weighted_sum += state_dict[key] * size
        # Compute the weighted average
        aggregated_state_dict[key] = weighted_sum / total_data

    if logger:
        logger.info("Aggregation using Weighted Federated Averaging (FedAvg) completed successfully.")

    return aggregated_state_dict
