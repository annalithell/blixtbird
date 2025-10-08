# fenics/training/__init__.py

from fenics.training.trainer import local_train, load_datasets
from fenics.training.evaluator import evaluate
from fenics.training.utils import summarize_model_parameters

__all__ = [
    'local_train',
    'load_datasets',
    'evaluate',
    'summarize_model_parameters'
]