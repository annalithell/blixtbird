# fenics/aggregation/__init__.py

from fenics.aggregation.fedavg import FedAvgStrategy
from fenics.aggregation.base import AggregationStrategy
from fenics.aggregation.factory import AggregationFactory

__all__ = [
    'FedAvgStrategy',
    'AggregationStrategy',
    'AggregationFactory'
]