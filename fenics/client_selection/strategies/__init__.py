# fenics/client_selection/strategies/__init__.py
from fenics.client_selection.strategies.uniform import select_clients_uniform
from fenics.client_selection.strategies.md_sampling import select_clients_md_sampling

__all__ = [
    'select_clients_uniform',
    'select_clients_md_sampling'
]