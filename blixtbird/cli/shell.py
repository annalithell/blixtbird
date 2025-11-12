# blixtbird/cli/shell.py

import cmd
import shlex
import logging
import pyfiglet
from colorama import Fore

from blixtbird.config import parse_arguments
from blixtbird.utils import setup_logging
from blixtbird.cli.utils import display_parameters
from blixtbird.cli.commands import (
    setup_environment,
    run_simulation_command,
    list_simulations
)


class BlixtbirdShell(cmd.Cmd):
    """
    Interactive command-line interface for Blixtbird.
    """
    
    intro = pyfiglet.figlet_format("BLIXTBIRD", font="doom") + "\nWelcome to BLIXTBIRD Shell! Type 'help' to see available commands.\n"
    prompt = "BLIXTBIRD> "
    
    def __init__(self):
        """Initialize the Blixtbird Shell."""
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
            ## TAKEN FROM do_list_simulations TO LOAD CONFIG IF PROVIDED

            if self.simulation_args and hasattr(self.simulation_args, 'config') and self.simulation_args.config:
                if args.config and args.simulation_name:
                    from blixtbird.config import load_config_from_file
                    yaml_args = load_config_from_file(args.config, args.simulation_name)
                    print(Fore.BLUE + f"Loaded configuration '{args.simulation_name}' from '{args.config}'")
                    for key, value in vars(yaml_args).items():
                        print(Fore.BLUE + f"{key.replace('_', ' ').capitalize()}: {value}")
                        # Initialize logging if not already done
                        if not self.logger:
                            self.logger = logging.getLogger()
                        self.logger.info(f"{key}: {value}")
            print(Fore.BLUE + "Parsed Arguments:")
            
            for arg_name, arg_value in vars(args).items():
                print(Fore.BLUE + f"{arg_name.replace('_', ' ').capitalize()}: {arg_value}")
            
            if not self.logger:
                    self.logger = logging.getLogger()
                    self.logger.info("Simulation arguments loaded successfully.")
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
            'exit': 'Exit the Blixtbird shell.',
            'quit': 'Exit the Blixtbird shell.'
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
        options = ['--rounds', '--epochs', '--num_nodes', '--node_type_map',  
                   '--use_attackers', '--topology', '--topology_file', '--max_attacks', 
                   '--gossip_steps', '--protocol',   '--alpha', '--model_type',
                   '--config', '--simulation_name']
        
        if not text:
            completions = options
        else:
            completions = [option for option in options if option.startswith(text)]
        
        return completions
    
    def complete_run(self, text, line, begidx, endidx):
        """Provide tab completion for run command."""
        return self.complete_setup(text, line, begidx, endidx)
    
    def do_exit(self, arg):
        """Exit the Blixtbird shell."""
        print(Fore.GREEN + "Exiting Blixtbird. Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the Blixtbird shell."""
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