# fenics/models/__init__.py

from fenics.models.cnn import Net
from fenics.models.base import ModelBase
# from fenics.models.mlp import MLP
from fenics.models.factory import ModelFactory

__all__ = [
    'Net',
    'ModelBase',
    # 'MLP',
    'ModelFactory'
]
