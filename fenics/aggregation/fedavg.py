# fenics/aggregation/fedavg.py

import torch
import logging
from typing import List, Dict, Optional

from fenics.aggregation.base import AggregationStrategy


class FedAvgStrategy(AggregationStrategy):
    """
    Federated Averaging (FedAvg) aggregation strategy.
    
    This strategy computes a weighted average of model parameters based on the
    size of each node's dataset.
    """
    
    def aggregate(self, models_state_dicts: List[Dict[str, torch.Tensor]], 
                  data_sizes: List[int]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate model parameters using Weighted Federated Averaging (FedAvg).
        
        Args:
            models_state_dicts: List of state dictionaries from each participating node
            data_sizes: List of data sizes corresponding to each node
            
        Returns:
            Aggregated state dictionary if models are provided, else None
        """
        if not models_state_dicts:
            self.logger.warning("No models to aggregate.")
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
        
        self.logger.info("Aggregation using Weighted Federated Averaging (FedAvg) completed successfully.")
        return aggregated_state_dict