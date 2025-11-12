# blixtbird/topology/types.py

import networkx as nx
from typing import Optional

from blixtbird.topology.base import TopologyBase


class FullyConnectedTopology(TopologyBase):
    """
    Fully connected (complete) network topology where every node is connected to every other node.
    """
    
    def build(self) -> nx.Graph:
        """
        Build a fully connected network.
        
        Returns:
            Complete graph with num_nodes nodes
        """
        return nx.complete_graph(self.num_nodes)


class RingTopology(TopologyBase):
    """
    Ring network topology where nodes form a single cycle.
    """
    
    def build(self) -> nx.Graph:
        """
        Build a ring network.
        
        Returns:
            Cycle graph with num_nodes nodes
        """
        return nx.cycle_graph(self.num_nodes)


class RandomTopology(TopologyBase):
    """
    Random network topology with probabilistic edge creation.
    """
    
    def __init__(self, num_nodes: int, p_connect: float = 0.3, seed: int = 0):
        """
        Initialize the random topology.
        
        Args:
            num_nodes: Number of nodes in the network
            p_connect: Probability of edge creation between any two nodes
            seed: Random seed for reproducibility
        """
        super().__init__(num_nodes)
        self.p_connect = p_connect
        self.seed = seed
    
    def build(self) -> nx.Graph:
        """
        Build a random network.
        
        Returns:
            Erdos-Renyi random graph with num_nodes nodes
        """
        return nx.erdos_renyi_graph(self.num_nodes, self.p_connect, seed=self.seed)


class CustomTopology(TopologyBase):
    """
    Custom network topology loaded from an edge list file.
    """
    
    def __init__(self, num_nodes: int, topology_file: str):
        """
        Initialize the custom topology.
        
        Args:
            num_nodes: Number of nodes in the network
            topology_file: Path to the edge list file
        """
        super().__init__(num_nodes)
        self.topology_file = topology_file
    
    def build(self) -> nx.Graph:
        """
        Build a custom network from an edge list file.
        
        Returns:
            Graph loaded from the edge list file
            
        Raises:
            ValueError: If the topology file is not provided or cannot be read
        """
        if self.topology_file is None:
            raise ValueError("Custom topology requires a topology file.")
        
        try:
            G = nx.read_edgelist(self.topology_file, nodetype=int)
            # Ensure the graph has the correct number of nodes
            G.add_nodes_from(range(self.num_nodes))
            return G
        except Exception as e:
            raise ValueError(f"Failed to load topology from file: {str(e)}")