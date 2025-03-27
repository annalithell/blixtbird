# fenics/data/__init__.py

from fenics.data.handler import load_datasets_dirichlet, print_class_distribution, distribute_data_dirichlet
from fenics.data.module import DataModule

__all__ = [
    'load_datasets_dirichlet',
    'print_class_distribution',
    'distribute_data_dirichlet',
    'DataModule'
]