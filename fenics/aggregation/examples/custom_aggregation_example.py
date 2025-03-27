# custom_aggregation_example.py
# 
# This file shows how to create and register a custom aggregation strategy with Fenics.
# You can use this as a template for creating your own strategies.

import torch
from typing import List, Dict, Optional

# Import the necessary base classes from Fenics
from fenics.aggregation import AggregationStrategy, AggregationFactory


class TrimmedMeanStrategy(AggregationStrategy):
    """
    Example of a custom aggregation strategy for Fenics.
    This implements a trimmed mean, which removes a percentage of extreme values
    before averaging, to provide some robustness against attacks.
    """
    
    def __init__(self, trim_percentage: float = 0.2, logger: Optional = None):
        """
        Initialize the trimmed mean strategy.
        
        Args:
            trim_percentage: Percentage of extreme values to trim (0.0 to 0.5)
            logger: Logger instance
        """
        super().__init__(logger)
        if not 0 <= trim_percentage < 0.5:
            raise ValueError("Trim percentage must be between 0.0 and 0.5")
        self.trim_percentage = trim_percentage
    
    def aggregate(self, models_state_dicts: List[Dict[str, torch.Tensor]], 
                 data_sizes: List[int]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate model parameters using trimmed mean.
        
        Args:
            models_state_dicts: List of state dictionaries from each participating node
            data_sizes: List of data sizes corresponding to each node
            
        Returns:
            Aggregated state dictionary if models are provided, else None
        """
        if not models_state_dicts:
            self.logger.warning("No models to aggregate.")
            return None
        
        # Calculate how many models to trim from each end
        num_models = len(models_state_dicts)
        if num_models < 3:
            self.logger.warning("Not enough models for trimmed mean. Using standard mean instead.")
            # Fall back to simple averaging if there are too few models
            return self._simple_average(models_state_dicts)
        
        trim_count = int(num_models * self.trim_percentage)
        self.logger.info(f"Trimming {trim_count} models from each end out of {num_models} total models.")
        
        # Initialize an empty state dict for the aggregated model
        aggregated_state_dict = {}
        
        # Get the list of all parameter keys
        param_keys = list(models_state_dicts[0].keys())
        
        for key in param_keys:
            # Stack parameters from all models along a new dimension
            stacked_params = torch.stack([state_dict[key] for state_dict in models_state_dicts])
            
            # Sort values along the model dimension
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            
            # Remove the specified percentage from both ends
            trimmed_params = sorted_params[trim_count:num_models-trim_count]
            
            # Average the remaining values
            aggregated_state_dict[key] = torch.mean(trimmed_params, dim=0)
        
        self.logger.info("Aggregation using Trimmed Mean completed successfully.")
        return aggregated_state_dict
    
    def _simple_average(self, models_state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute a simple average of model parameters.
        
        Args:
            models_state_dicts: List of state dictionaries from each participating node
            
        Returns:
            Averaged state dictionary
        """
        # Initialize an empty state dict for the aggregated model
        aggregated_state_dict = {}
        
        # Get the list of all parameter keys
        param_keys = list(models_state_dicts[0].keys())
        
        for key in param_keys:
            # Stack parameters from all models
            stacked_params = torch.stack([state_dict[key] for state_dict in models_state_dicts])
            # Take the mean along the first dimension (across models)
            aggregated_state_dict[key] = torch.mean(stacked_params, dim=0)
        
        return aggregated_state_dict


# Register the strategy with the factory
# This makes it available to use with the name 'trimmed_mean'
AggregationFactory.register_strategy('trimmed_mean', TrimmedMeanStrategy)


# =======================================================================
# How to use this custom aggregation strategy:
# =======================================================================
#
# 1. Place this file in your project directory
#
# 2. Import this module before using the strategy:
#    ```
#    # In your script or notebook
#    import custom_aggregation_example
#    ```
#
# 3. Create and use the strategy:
#    ```python
#    from fenics.aggregation import AggregationFactory
#    
#    # Get a trimmed mean strategy with 20% trimming
#    strategy = AggregationFactory.get_strategy('trimmed_mean', trim_percentage=0.2)
#    
#    # Use it to aggregate models
#    aggregated_model = strategy.aggregate(models_state_dicts, data_sizes)
#    ```
#
# 4. For multiple custom strategies, you can create additional files
#    and import them all before using Fenics.