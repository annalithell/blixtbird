# fenics/config.py

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
import yaml
import os
import argparse

class SimulationConfig(BaseModel):
    """Pydantic model for simulation configuration parameters."""
    
    # Basic parameters
    rounds: int = Field(5, description="Number of federated learning rounds")
    epochs: int = Field(1, description="Number of local epochs")
    num_nodes: int = Field(5, description="Number of nodes in the network")
    
    # Attacker parameters
    num_attackers: int = Field(0, description="Number of attacker nodes")
    attacker_nodes: Optional[List[int]] = Field(None, description="List of attacker node indices")
    attacks: List[str] = Field([], description="Types of attacks: delay, poison")
    use_attackers: bool = Field(False, description="Include attacker nodes")
    max_attacks: Optional[int] = Field(None, description="Maximum number of times an attacker can perform an attack")
    
    # Network parameters
    participation_rate: float = Field(0.5, description="Fraction of nodes participating in each round (0 < rate <= 1)")
    topology: str = Field("fully_connected", description="Network topology: fully_connected, ring, random, custom")
    topology_file: Optional[str] = Field(None, description="Path to the custom topology file (edge list)")
    
    # Communication parameters
    gossip_steps: int = Field(3, description="Number of gossip iterations per round")
    protocol: str = Field("gossip", description="Communication protocol to use: gossip, neighboring")
    
    # Data distribution parameters
    alpha: float = Field(0.5, description="Dirichlet distribution parameter for data distribution")
    
    # Model parameters
    model_type: str = Field("cnn", description="Model type to use: cnn, mlp, custom")
    
    @validator('participation_rate')
    def validate_participation_rate(cls, v):
        if not (0 < v <= 1):
            raise ValueError("participation_rate must be between 0 and 1")
        return v
    
    @validator('num_attackers')
    def validate_num_attackers(cls, v, values):
        if 'num_nodes' in values and v > values['num_nodes']:
            raise ValueError("num_attackers cannot exceed num_nodes")
        return v
    
    @validator('topology_file')
    def validate_topology_file(cls, v, values):
        if values.get('topology') == 'custom' and not v:
            raise ValueError("topology_file must be provided when topology is 'custom'")
        return v
    
    @validator('protocol')
    def validate_protocol(cls, v):
        if v not in ['gossip', 'neighboring']:
            raise ValueError("protocol must be either 'gossip' or 'neighboring'")
        return v


def load_config_from_file(config_file: str, simulation_name: Optional[str] = None) -> SimulationConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to the configuration file
        simulation_name: Name of the simulation configuration to use
        
    Returns:
        SimulationConfig object with the loaded parameters
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if simulation_name:
        simulations = config_data.get('simulations', {})
        if simulation_name not in simulations:
            raise ValueError(f"Simulation configuration '{simulation_name}' not found in {config_file}")
        
        simulation_config = simulations[simulation_name]
        return SimulationConfig(**simulation_config)
    
    return SimulationConfig(**config_data)


def parse_arguments_pydantic(args=None, parser=None) -> SimulationConfig:
    """
    Parse command-line arguments and create a SimulationConfig object.
    
    Args:
        args: Command-line arguments
        parser: Existing ArgumentParser instance
        
    Returns:
        SimulationConfig object with the parsed parameters
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Decentralized Federated Learning Simulation")
    
    # Add all arguments from the Pydantic model
    parser.add_argument("--rounds", type=int, help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, help="Number of local epochs")
    parser.add_argument("--num_nodes", type=int, help="Number of nodes in the network")
    parser.add_argument("--num_attackers", type=int, help="Number of attacker nodes")
    parser.add_argument("--attacker_nodes", nargs='+', type=int, help="List of attacker node indices")
    parser.add_argument("--attacks", nargs='+', help="Types of attacks: delay, poison")
    parser.add_argument("--use_attackers", action='store_true', help="Include attacker nodes")
    parser.add_argument("--participation_rate", type=float, help="Fraction of nodes participating in each round (0 < rate <= 1)")
    parser.add_argument("--topology", type=str, help="Network topology: fully_connected, ring, random, custom")
    parser.add_argument("--topology_file", type=str, help="Path to the custom topology file (edge list)")
    parser.add_argument("--max_attacks", type=int, help="Maximum number of times an attacker can perform an attack")
    parser.add_argument("--gossip_steps", type=int, help="Number of gossip iterations per round")
    parser.add_argument("--protocol", type=str, help="Communication protocol to use: gossip, neighboring")
    parser.add_argument("--alpha", type=float, help="Dirichlet distribution parameter for data distribution")
    parser.add_argument("--model_type", type=str, help="Model type to use: cnn, mlp, custom")
    parser.add_argument("--config", type=str, help="Path to configuration file (e.g., YAML)")
    parser.add_argument("--simulation_name", type=str, help="Name of the simulation configuration to use")
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Convert to dictionary, filtering out None values
    config_dict = {k: v for k, v in vars(parsed_args).items() 
                   if v is not None and k not in ['config', 'simulation_name']}
    
    # Load from config file if specified
    if parsed_args.config and parsed_args.simulation_name:
        config = load_config_from_file(parsed_args.config, parsed_args.simulation_name)
        
        # Override with command-line arguments
        for key, value in config_dict.items():
            setattr(config, key, value)
            
        return config
    
    # Create config from command-line arguments
    return SimulationConfig(**config_dict)


# For backward compatibility with the original parse_arguments function
def parse_arguments(args=None):
    """
    Legacy function for parsing command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Namespace with the parsed arguments
    """
    # Use the pydantic parser but convert the config to a simple namespace
    config = parse_arguments_pydantic(args)
    
    # Convert to namespace
    parser = argparse.ArgumentParser()
    for key, value in config.dict().items():
        parser.add_argument(f"--{key}", default=value)
    
    return parser.parse_args([])  # Parse empty args but use the defaults we just set