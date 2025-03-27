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
        weights: List of weights for each node (representing importance)
        logger: Logger instance
        
    Returns:
        List of selected node IDs
    """
    logger = logger or logging.getLogger()
    
    # Validate input
    if len(nodes) != len(weights):
        raise ValueError("Number of nodes must match number of weights")
    
    if num_participants > len(nodes):
        raise ValueError("Cannot select more participants than available nodes")
    
    # Normalize weights to probabilities
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are zero, default to uniform distribution
        probabilities = [1.0 / len(nodes)] * len(nodes)
    else:
        probabilities = [w / total_weight for w in weights]
    
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


# Example of registering this strategy (uncommenting if using directly)
# from fenics.client_selection.factory import SelectionFactory
# SelectionFactory.register_strategy('weighted', select_clients_weighted)