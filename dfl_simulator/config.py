# config.py

import argparse
import yaml

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Decentralized Federated Learning Simulation")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Number of local epochs")
    parser.add_argument("--num_nodes", type=int, default=5, help="Number of nodes in the network")
    parser.add_argument("--num_attackers", type=int, default=0, help="Number of attacker nodes")
    parser.add_argument("--attacker_nodes", nargs='+', type=int, default=None, help="List of attacker node indices")
    parser.add_argument("--attacks", nargs='+', default=[], help="Types of attacks: delay, poison")
    parser.add_argument("--use_attackers", action='store_true', help="Include attacker nodes")
    parser.add_argument("--participation_rate", type=float, default=0.5, help="Fraction of nodes participating in each round (0 < rate <= 1)")
    parser.add_argument("--topology", type=str, default="fully_connected", help="Network topology: fully_connected, ring, random, custom")
    parser.add_argument("--topology_file", type=str, default=None, help="Path to the custom topology file (edge list). Required if topology is 'custom'.")
    parser.add_argument("--max_attacks", type=int, default=None, help="Maximum number of times an attacker can perform an attack")
    parser.add_argument("--gossip_steps", type=int, default=3, help="Number of gossip iterations per round")
    parser.add_argument("--protocol", type=str, default="gossip", choices=['gossip', 'neighboring'], help="Communication protocol to use: gossip, neighboring")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet distribution parameter for data distribution")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file (e.g., YAML or JSON)")
    # Argument to specify which simulation configuration to use
    parser.add_argument("--simulation_name", type=str, help="Name of the simulation configuration to use")

    # Parse known arguments first
    parsed_args, remaining_args = parser.parse_known_args(args)

    config_data = {}
    # Load configurations from the config file if simulation_name is provided
    if parsed_args.simulation_name:
        config_file = parsed_args.config or "config.yaml"
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e  # Re-raise the exception to be handled in main.py

        # Select the specified simulation configuration
        simulations = config_data.get('simulations', {})
        simulation_config = simulations.get(parsed_args.simulation_name)
        if not simulation_config:
            raise ValueError(f"Simulation configuration '{parsed_args.simulation_name}' not found in {config_file}.")

        # Override simulation parameters with config file
        for key, value in simulation_config.items():
            if hasattr(parsed_args, key):
                setattr(parsed_args, key, value)

    # Finally, parse remaining_args to allow command-line overrides
    final_args = parser.parse_args(remaining_args, namespace=parsed_args)

    return final_args

'''def validate_parameters(args):
    if not (0 < args.participation_rate <= 1):
        raise ValueError("participation_rate must be between 0 and 1.")
    if args.num_attackers > args.num_nodes:
        raise ValueError("num_attackers cannot exceed num_nodes.")
    if args.topology == "custom" and not args.topology_file:
        raise ValueError("topology_file must be provided when topology is 'custom'.")
    # Add more validations as needed'''
