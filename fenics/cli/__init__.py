# fenics/cli/__init__.py

from fenics.cli.shell import FenicsShell
from fenics.cli.runner import run_fenics_shell

__all__ = [
    'FenicsShell',
    'run_fenics_shell'
]