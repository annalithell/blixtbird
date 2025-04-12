# fenics/training/__init__.py

from fenics.training.trainer import local_train
from fenics.training.evaluator import evaluate
from fenics.training.utils import summarize_model_parameters

__all__ = [
    'local_train',
    'evaluate',
    'summarize_model_parameters'
]