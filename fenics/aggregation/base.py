# fenics/aggregation/base.py

import torch
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class AggregationStrategy(ABC):
    """
    Base class for all aggregation strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the aggregation strategy.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger()
    
    @abstractmethod
    def aggregate(self, models_state_dicts: List[Dict[str, torch.Tensor]], 
                  data_sizes: List[int]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate model parameters.
        
        Args:
            models_state_dicts: List of state dictionaries from each participating node
            data_sizes: List of data sizes corresponding to each node
            
        Returns:
            Aggregated state dictionary if models are provided, else None
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of the aggregation strategy.
        
        Returns:
            Name of the strategy
        """
        return self.__class__.__name__.replace('Strategy', '')