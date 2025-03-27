# fenics/cli/runner.py

from colorama import init
from fenics.cli.shell import FenicsShell


def run_fenics_shell():
    """
    Run the Fenics Shell.
    """
    # Initialize colorama
    init(autoreset=True)
    
    # Create and run the shell
    shell = FenicsShell()
    shell.cmdloop()


if __name__ == "__main__":
    run_fenics_shell()