# fenics/client_selection/__init__.py

from fenics.client_selection.selector import ClientSelector
# from fenics.client_selection.strategies.uniform import select_clients_uniform
from fenics.client_selection.strategies.md_sampling import select_clients_md_sampling
from fenics.client_selection.factory import SelectionFactory

__all__ = [
    'ClientSelector',
    # 'select_clients_uniform',
    'select_clients_md_sampling',
    'SelectionFactory'
]
