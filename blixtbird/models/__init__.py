# fenics/models/__init__.py

from blixtbird.models.cnn import Net
from blixtbird.models.base import ModelBase
# from fenics.models.mlp import MLP
from blixtbird.models.factory import ModelFactory

__all__ = [
    'Net',
    'ModelBase',
    # 'MLP',
    'ModelFactory'
]
