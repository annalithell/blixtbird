# fenics/client_selection/selector.py

import numpy as np
import random
import logging
from typing import List, Dict, Optional

from fenics.client_selection.strategies.uniform import select_clients_uniform
from fenics.client_selection.strategies.md_sampling import select_clients_md_sampling


class ClientSelector:
    """
    A module to handle client selection strategies in federated learning.
    """
    
    def __init__(self, 
                 nodes: List[int],
                 participation_rate: float,
                 random_seed: int = 0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the client selector.
        
        Args:
            nodes: List of all node IDs
            participation_rate: Fraction of nodes to select in each round
            random_seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.nodes = nodes
        self.participation_rate = participation_rate
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Calculate number of participants
        self.num_participants = max(1, int(len(nodes) * participation_rate))
    
    def select_clients_uniform(self) -> List[int]:
        """
        Select clients uniformly at random.
        
        Returns:
            List of selected node IDs
        """
        return select_clients_uniform(self.nodes, self.num_participants, self.logger)
    
    def select_clients_md_sampling(self, probabilities: List[float]) -> List[int]:
        """
        Select clients based on predefined probabilities using multinomial sampling.
        
        Args:
            probabilities: Selection probabilities for each node
            
        Returns:
            List of selected node IDs
        """
        return select_clients_md_sampling(
            self.nodes, 
            self.num_participants, 
            probabilities, 
            self.logger
        )
    
    def precompute_participating_nodes(self, 
                                      num_rounds: int, 
                                      probabilities: Optional[List[float]] = None) -> List[List[int]]:
        """
        Precompute participating nodes for all rounds.
        
        Args:
            num_rounds: Number of rounds
            probabilities: Selection probabilities for each node
            
        Returns:
            List of lists of selected node IDs for each round
        """
        participating_nodes_per_round = []
        
        for rnd in range(num_rounds):
            if probabilities:
                selected_nodes = self.select_clients_md_sampling(probabilities)
            else:
                selected_nodes = self.select_clients_uniform()
            
            participating_nodes_per_round.append(selected_nodes)
            
        return participating_nodes_per_round