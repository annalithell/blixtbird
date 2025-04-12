# fenics/aggregation/factory.py

from typing import Dict, Type, Optional
import logging

from fenics.aggregation.base import AggregationStrategy
from fenics.aggregation.fedavg import FedAvgStrategy


class AggregationFactory:
    """
    Factory class for creating aggregation strategy instances.
    This makes it easy for users to select different aggregation strategies.
    """
    
    # Registry of available strategies
    _strategies: Dict[str, Type[AggregationStrategy]] = {
        'fedavg': FedAvgStrategy,
    }
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[AggregationStrategy]) -> None:
        """
        Register a new aggregation strategy.
        
        Args:
            name: Name of the strategy
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get_strategy(cls, strategy_name: str, logger: Optional[logging.Logger] = None, **kwargs) -> AggregationStrategy:
        """
        Get a strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy
            logger: Logger instance
            **kwargs: Additional arguments to pass to the strategy constructor
            
        Returns:
            Instance of the requested strategy
            
        Raises:
            ValueError: If the strategy name is not recognized
        """
        if strategy_name not in cls._strategies:
            available_strategies = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: '{strategy_name}'. Available strategies: {available_strategies}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(logger=logger, **kwargs)
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, Type[AggregationStrategy]]:
        """
        List all available strategies.
        
        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return cls._strategies.copy()