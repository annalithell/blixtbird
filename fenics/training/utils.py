# fenics/training/utils.py

import logging


def summarize_model_parameters(node_name, model_state_dict, logger):
    """
    Summarize model parameters for a node after local training.
    
    Args:
        node_name: Name of the node
        model_state_dict: Model state dictionary
        logger: Logger instance
    """
    logger.info(f"\n[Summary] Model parameters for node_{node_name} after local training:")
    for key, param in model_state_dict.items():
        param_np = param.cpu().numpy()
        mean = param_np.mean()
        std = param_np.std()
        logger.info(f" Layer: {key:<20} | Mean: {mean:.6f} | Std: {std:.6f}")