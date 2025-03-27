# fenics/topology/factory.py

import networkx as nx
from typing import Dict, Type, Optional, Any

from fenics.topology.base import TopologyBase
from fenics.topology.types import (
    FullyConnectedTopology,
    RingTopology,
    RandomTopology,
    CustomTopology
)


class TopologyFactory:
    """
    Factory class for creating network topology instances.
    This makes it easy for users to select different topology types.
    """
    
    # Registry of available topologies
    _topologies: Dict[str, Type[TopologyBase]] = {
        'fully_connected': FullyConnectedTopology,
        'ring': RingTopology,
        'random': RandomTopology,
        'custom': CustomTopology,
    }
    
    @classmethod
    def register_topology(cls, name: str, topology_class: Type[TopologyBase]) -> None:
        """
        Register a new topology type.
        
        Args:
            name: Name of the topology
            topology_class: Topology class that inherits from TopologyBase
        """
        cls._topologies[name] = topology_class
    
    @classmethod
    def build_topology(cls, topology_type: str, num_nodes: int, topology_file: Optional[str] = None, **kwargs) -> nx.Graph:
        """
        Build a network topology of the specified type.
        
        Args:
            topology_type: Type of topology to build
            num_nodes: Number of nodes in the network
            topology_file: Path to the topology file (for custom topologies)
            **kwargs: Additional arguments to pass to the topology constructor
            
        Returns:
            NetworkX graph representing the built topology
            
        Raises:
            ValueError: If the topology type is not recognized
        """
        if topology_type not in cls._topologies:
            available_topologies = list(cls._topologies.keys())
            raise ValueError(f"Unknown topology: '{topology_type}'. Available topologies: {available_topologies}")
        
        topology_class = cls._topologies[topology_type]
        
        # Handle special case for custom topology
        if topology_type == 'custom':
            if topology_file is None:
                raise ValueError("Custom topology requires a topology file.")
            topology = topology_class(num_nodes=num_nodes, topology_file=topology_file, **kwargs)
        else:
            topology = topology_class(num_nodes=num_nodes, **kwargs)
        
        return topology.build()
    
    @classmethod
    def list_available_topologies(cls) -> Dict[str, Type[TopologyBase]]:
        """
        List all available topologies.
        
        Returns:
            Dictionary mapping topology names to topology classes
        """
        return cls._topologies.copy()