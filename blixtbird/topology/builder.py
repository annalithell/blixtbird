# fenics/topology/builder.py

import networkx as nx
from typing import Optional

from blixtbird.topology.factory import TopologyFactory


def build_topology(num_nodes: int, topology_type: str, topology_file: Optional[str] = None) -> nx.Graph:
    """
    Build a network topology of the specified type.
    
    Args:
        num_nodes: Number of nodes in the network
        topology_type: Type of topology to build ('fully_connected', 'ring', 'random', 'custom')
        topology_file: Path to the topology file (required for 'custom' topology)
        
    Returns:
        NetworkX graph representing the built topology
        
    Raises:
        ValueError: If an invalid topology type is specified or if a topology file is missing for custom topology
    """
    return TopologyFactory.build_topology(topology_type, num_nodes, topology_file)