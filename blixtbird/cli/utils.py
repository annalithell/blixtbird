#fenics/cli/utils.py

from colorama import Fore


def display_parameters():
    """Display all available simulation parameters and their options."""
    parameters_info = {
        "rounds": {
            "description": "Number of federated learning rounds",
            "type": "int",
            "default": 5,
            "options": "Any positive integer"
        },
        "epochs": {
            "description": "Number of local epochs",
            "type": "int",
            "default": 1,
            "options": "Any positive integer"
        },
        "num_nodes": {
            "description": "Number of nodes in the network",
            "type": "int",
            "default": 5,
            "options": "Any positive integer"
        },
        "model_type": {
            "description": "Type of model to use",
            "type": "str",
            "default": "cnn",
            "options": "cnn, mlp, or custom models registered with ModelFactory"
        },
        "use_attackers": {
            "description": "Include attacker nodes",
            "type": "bool",
            "default": False,
            "options": "True or False"
        },
        "topology": {
            "description": "Network topology type",
            "type": "str",
            "default": "fully_connected",
            "options": "fully_connected, ring, random, custom"
        },
        "topology_file": {
            "description": "Path to the custom topology file (edge list)",
            "type": "str",
            "default": "None",
            "options": "Required if topology is 'custom'"
        },
        "max_attacks": {
            "description": "Maximum number of times an attacker can perform an attack",
            "type": "int",
            "default": "None",
            "options": "Any positive integer or None for unlimited"
        },
        "gossip_steps": {
            "description": "Number of gossip iterations per round",
            "type": "int",
            "default": 3,
            "options": "Any positive integer"
        },
        "protocol": {
            "description": "Communication protocol to use",
            "type": "str",
            "default": "gossip",
            "options": "gossip, neighboring"
        },
        "alpha": {
            "description": "Dirichlet distribution parameter for data distribution",
            "type": "float",
            "default": 0.5,
            "options": "Any positive float"
        },
        "node_type_map": {
            "description": "Map node IDs to their node types (e.g. base, freerider, poison)",
            "type": "dict[int, str]",
            "default": "None",
            "options": "Any mapping of integers to strings"
        },
        "config": {
            "description": "Configuration file for placing all the parameters",
            "type": "yaml",
            "default": "config.yaml",
            "options": "Specific configuration file path"
        },
        "simulation_name": {
            "description": "Name of the simulation configuration to use",
            "type": "str",
            "default": "None",
            "options": "Specific simulation name"
        }
    }

    print(Fore.CYAN + "Available Simulation Parameters:\n")
    for param, details in parameters_info.items():
        print(Fore.YELLOW + f"{param}:")
        print(Fore.WHITE + f"  Description : {details['description']}")
        print(Fore.WHITE + f"  Type        : {details['type']}")
        print(Fore.WHITE + f"  Default     : {details['default']}")
        print(Fore.WHITE + f"  Options     : {details['options']}\n")