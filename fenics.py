# fenics/fenics.py

"""
Main entry point for the Fenics Simulator.

This script initializes and runs the Fenics Shell, which provides
an interactive command-line interface for running decentralized
federated learning simulations.
"""

from fenics.cli import run_fenics_shell

def main():
    """Run the Fenics Shell."""
    run_fenics_shell()

if __name__ == "__main__":
    main()