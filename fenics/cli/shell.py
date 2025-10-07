# fenics/cli/shell.py

import cmd
import os
import shlex
import logging
import numpy as np
import yaml
import pyfiglet
from colorama import Fore

from fenics.config import parse_arguments
from fenics.utils import setup_logging
from fenics.cli.utils import display_parameters
from fenics.cli.commands import (
    setup_environment,
    run_simulation_command,
    list_simulations
)


class FenicsShell(cmd.Cmd):
    """
    Interactive command-line interface for Fenics.
    """
    
    intro = pyfiglet.figlet_format("Fenics", font="slant") + "\nWelcome to Fenics Shell! Type 'help' to see available commands.\n"
    prompt = "Fenics> "
    
    def __init__(self):
        """Initialize the Fenics Shell."""
        super().__init__()
        self.output_dir = "results"
        self.logger = None
        self.simulation_args = None  # To store simulation arguments
    
    def do_parameters(self, arg):
        """Display all available simulation parameters and their options."""
        display_parameters()
    
    def do_setup(self, arg):
        """
        Initialize the simulation environment with desired configurations.
        Usage: setup --rounds 3 --epochs 1 --topology fully_connected --participation_rate 0.6 --protocol gossip
        """
        try:
            args = parse_arguments(shlex.split(arg))
            
            # Print parsed arguments
            print(Fore.BLUE + "Parsed Arguments:")
            for arg_name, arg_value in vars(args).items():
                print(Fore.BLUE + f"{arg_name.replace('_', ' ').capitalize()}: {arg_value}")
                
                # Initialize logging if not already done
                if not self.logger:
                    self.logger = logging.getLogger()
                self.logger.info(f"{arg_name}: {arg_value}")

            # Create output directory and setup logging
            self.output_dir = setup_environment(self.logger)
            
            # Setup logging
            setup_logging(self.output_dir)
            self.logger = logging.getLogger()
            self.logger.info("Logging is configured.")
            
            self.simulation_args = args
            print(Fore.GREEN + "Setup completed successfully.")
        except Exception as e:
            print(Fore.RED + f"An error occurred during setup: {e}")
    
    def do_run(self, arg):
        """
        Execute the simulation with the provided options.
        Usage: run --rounds 3 --epochs 1 --topology fully_connected --participation_rate 0.6 --protocol gossip
        """
        if not self.simulation_args:
            print(Fore.RED + "Please run the 'setup' command first with desired configurations.")
            return
        
        try:
            # Run the simulation command
            run_simulation_command(arg, self.simulation_args, self.output_dir, self.logger)
        except Exception as e:
            self.logger.error(f"An error occurred during execution: {e}")
            print(Fore.RED + f"An error occurred during execution: {e}")
    
    def do_list_simulations(self, arg):
        """List all available simulation configurations."""
        config_file = "config.yaml"  # Default config file
        if self.simulation_args and hasattr(self.simulation_args, 'config') and self.simulation_args.config:
            config_file = self.simulation_args.config
        
        list_simulations(config_file)
    
    def do_help_custom(self, arg):
        """Show available commands and their descriptions."""
        commands = {
            'setup': 'Initialize the simulation environment with desired configurations. For ex. use: setup --config config.yaml --simulation_name sim1',
            'run': 'Execute the simulation with the provided options.',
            'list_simulations': 'List all available simulation configurations.',
            'parameters': 'Display all available simulation parameters and their options.',
            'help': 'Show this help message.',
            'exit': 'Exit the Fenics shell.',
            'quit': 'Exit the Fenics shell.'
        }
        
        if arg:
            # Show help for a specific command
            if arg in commands:
                print(f"{arg}: {commands[arg]}")
            else:
                print(f"No help available for '{arg}'.")
        else:
            print("Available commands:")
            for cmd_name, desc in commands.items():
                print(f"  {cmd_name:<20} {desc}")
            print("\nType 'help [command]' to get more information about a specific command.")
    
    def complete_setup(self, text, line, begidx, endidx):
        """Provide tab completion for setup command."""
        # TODO: Change options
        options = ['--rounds', '--epochs', '--topology', '--participation_rate', '--gossip_steps', 
                   '--protocol', '--use_attackers', '--num_attackers', '--attacks', '--alpha', 
                   '--config', '--simulation_name', '--model_type']
        
        if not text:
            completions = options
        else:
            completions = [option for option in options if option.startswith(text)]
        
        return completions
    
    def complete_run(self, text, line, begidx, endidx):
        """Provide tab completion for run command."""
        return self.complete_setup(text, line, begidx, endidx)
    
    def do_exit(self, arg):
        """Exit the Fenics shell."""
        print(Fore.GREEN + "Exiting Fenics. Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the Fenics shell."""
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """Show this help message."""
        self.do_help_custom(arg)
    
    def emptyline(self):
        """Do nothing on empty input line."""
        pass
    
    def default(self, line):
        """Handle unrecognized commands."""
        print(Fore.RED + f"Unrecognized command: '{line}'. Type 'help' to see available commands.")