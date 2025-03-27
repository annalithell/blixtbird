# fenics/client_selection/strategies/uniform.py

import random
import logging
from typing import List, Optional


def select_clients_uniform(nodes: List[int], 
                          num_participants: int, 
                          logger: Optional[logging.Logger] = None) -> List[int]:
    """
    Select clients uniformly at random.
    
    Args:
        nodes: List of all node IDs
        num_participants: Number of nodes to select
        logger: Logger instance
        
    Returns:
        List of selected node IDs
    """
    logger = logger or logging.getLogger()
    selected_nodes = random.sample(nodes, num_participants)
    logger.info(f"Selected nodes for participation: {selected_nodes}")
    return selected_nodes