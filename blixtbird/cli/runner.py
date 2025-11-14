# fenics/cli/runner.py

from colorama import init
from blixtbird.cli.shell import BlixtbirdShell


def run_blixtbird_shell():
    """
    Run the Blixtbird Shell.
    """
    # Initialize colorama
    init(autoreset=True)
    
    # Create and run the shell
    shell = BlixtbirdShell()
    shell.cmdloop()


if __name__ == "__main__":
    run_blixtbird_shell()