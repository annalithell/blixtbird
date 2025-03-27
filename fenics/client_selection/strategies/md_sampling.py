# fenics/client_selection/strategies/md_sampling.py

import numpy as np
import logging
from typing import List, Optional


def select_clients_md_sampling(nodes: List[int], 
                              num_participants: int, 
                              probabilities: List[float], 
                              logger: Optional[logging.Logger] = None) -> List[int]:
    """
    Select clients based on predefined probabilities using multinomial sampling.
    
    Args:
        nodes: List of all node IDs
        num_participants: Number of nodes to select
        probabilities: Selection probabilities for each node
        logger: Logger instance
        
    Returns:
        List of selected node IDs
    """
    logger = logger or logging.getLogger()
    selected_indices = np.random.choice(
        len(nodes), 
        size=num_participants, 
        replace=False, 
        p=probabilities
    )
    selected_nodes = [nodes[i] for i in selected_indices]
    logger.info(f"Selected nodes for participation: {selected_nodes}")
    return selected_nodes