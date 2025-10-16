# fenics/client_selection/base.py

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class ClientSelector(ABC):
    """
    Base class for all client selection strategies.
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
        
        # Calculate number of participants
        self.num_participants = max(1, int(len(nodes) * participation_rate))
    
    @abstractmethod
    def select_clients(self, **kwargs) -> List[int]:
        """
        Select clients for participation in a round.
        
        Args:
            **kwargs: Additional arguments for specific selection strategies
            
        Returns:
            List of selected node IDs
        """
        pass
    
    def precompute_participating_nodes(self, num_rounds: int, **kwargs) -> List[List[int]]:
        """
        Precompute participating nodes for all rounds.
        
        Args:
            num_rounds: Number of rounds
            **kwargs: Additional arguments for specific selection strategies
            
        Returns:
            List of lists of selected node IDs for each round
        """
        participating_nodes_per_round = []
        
        for rnd in range(num_rounds):
            selected_nodes = self.select_clients(**kwargs)
            participating_nodes_per_round.append(selected_nodes)
            self.logger.info(f"Round {rnd+1}: Selected {len(selected_nodes)} nodes for participation")
        
        return participating_nodes_per_round