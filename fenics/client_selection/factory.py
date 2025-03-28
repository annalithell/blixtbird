# fenics/client_selection/factory.py

from typing import Dict, Callable, List, Optional
import logging

# from fenics.client_selection.strategies.uniform import select_clients_uniform
from fenics.client_selection.strategies.md_sampling import select_clients_md_sampling


class SelectionFactory:
    """
    Factory class for creating client selection strategy functions.
    This makes it easy for users to select different selection strategies.
    """
    
    # Registry of available strategies
    _strategies: Dict[str, Callable] = {
        'uniform': select_clients_uniform,
        'md_sampling': select_clients_md_sampling,
    }
    
    @classmethod
    def register_strategy(cls, name: str, strategy_function: Callable) -> None:
        """
        Register a new selection strategy.
        
        Args:
            name: Name of the strategy
            strategy_function: Function implementing the strategy
            
        Example:
            >>> from fenics.client_selection.strategies.weighted import select_clients_weighted
            >>> SelectionFactory.register_strategy('weighted', select_clients_weighted)
        """
        cls._strategies[name] = strategy_function
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> Callable:
        """
        Get a strategy function by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Function implementing the requested strategy
            
        Raises:
            ValueError: If the strategy name is not recognized
        """
        if strategy_name not in cls._strategies:
            available_strategies = list(cls._strategies.keys())
            raise ValueError(f"Unknown selection strategy: '{strategy_name}'. Available strategies: {available_strategies}")
        
        return cls._strategies[strategy_name]
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, Callable]:
        """
        List all available strategies.
        
        Returns:
            Dictionary mapping strategy names to strategy functions
        """
        return cls._strategies.copy()
