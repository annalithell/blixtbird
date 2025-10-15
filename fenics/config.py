# fenics/config.py

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml
import os
import argparse

class SimulationConfig(BaseModel):
    """Pydantic model for simulation configuration parameters."""
    
    # Basic parameters
    rounds: int = Field(5, description="Number of federated learning rounds")
    epochs: int = Field(1, description="Number of local epochs")
    num_nodes: int = Field(5, description="Number of nodes in the network")
    node_type_map: dict = Field({0:'base', 1:'base', 2:'base', 3:'base', 4:'base'}, description="Node type of each node in the network")   # TODO check if this actually works

    # Attacker parameters
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
    
    # @field_validator('participation_rate')
    # def validate_participation_rate(cls, v):
    #     if not (0 < v <= 1):
    #         raise ValueError("participation_rate must be between 0 and 1")
    #     return v
    
    # @field_validator('node_type_map')
    # def validate_num_attackers(cls, v, values):
    #     if 'num_nodes' in values and len(v) > values.get['num_nodes']:
    #         raise ValueError("Attacker nodes cannot exceed num_nodes")
    #     return v
    
    # @field_validator('topology_file')
    # def validate_topology_file(cls, v, values):
    #     if values.get('topology') == 'custom' and not v:
    #         raise ValueError("topology_file must be provided when topology is 'custom'")
    #     return v
    
    # @field_validator('protocol')
    # def validate_protocol(cls, v):
    #     if v not in ['gossip', 'neighboring']:
    #         raise ValueError("protocol must be either 'gossip' or 'neighboring'")
    #     return v
    
    class Config:
        # Allow extra fields for flexibility
        extra = "ignore"
        # Make sure we validate upon assignment
        validate_assignment = True


def load_config_from_file(config_file: Union[str, Path], simulation_name: Optional[str] = None) -> SimulationConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to the configuration file
        simulation_name: Name of the simulation configuration to use
        
    Returns:
        SimulationConfig object with the loaded parameters
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if simulation_name:
        simulations = config_data.get('simulations', {})
        if simulation_name not in simulations:
            raise ValueError(f"Simulation configuration '{simulation_name}' not found in {config_file}")
        
        simulation_config = simulations[simulation_name]
        return SimulationConfig(**simulation_config)
    # If no simulation name is provided, try to load the config directly
    return SimulationConfig(**config_data)

def parse_cli_args(args=None) -> Dict[str, Any]:
    """
    Parse command-line arguments into a dictionary.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description="Fenics Decentralized Federated Learning Simulator")
    # Add only the essential arguments for loading config
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--simulation_name", type=str, help="Name of the simulation configuration to use")
    
    # Parse known arguments first
    parsed_args, _ = parser.parse_known_args(args)
    return vars(parsed_args)


def parse_arguments(args=None) -> SimulationConfig:
    """
    Get the configuration from command-line arguments and/or config file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        SimulationConfig object
    """
    # Parse CLI arguments
    cli_args = parse_cli_args(args)
    
    # If config file is provided, load it
    if cli_args.get('config') and cli_args.get('simulation_name'):
        config = load_config_from_file(
            cli_args['config'], 
            cli_args['simulation_name']
        )
        return config
    
    # If no config file, create a default config
    return SimulationConfig()

def list_simulations(config_file: Union[str, Path]) -> List[str]:
    """
    List all simulation configurations in a config file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        List of simulation names
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    simulations = config_data.get('simulations', {})
    return list(simulations.keys())
