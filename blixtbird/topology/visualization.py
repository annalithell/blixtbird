# fenics/topology/visualization.py

import networkx as nx
import matplotlib.pyplot as plt
import os
import logging
from typing import Optional


def visualize_and_save_topology(G: nx.Graph, topology_type: str, output_dir: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Visualize the network topology and save it as an image.
    
    Args:
        G: NetworkX graph to visualize
        topology_type: Type of topology for the title
        output_dir: Directory to save the visualization
        logger: Logger instance
    """
    logger = logger or logging.getLogger()
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=0)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(f"Network Topology: {topology_type}")
    
    topology_path = os.path.join(output_dir, 'network_topology.png')
    plt.savefig(topology_path)
    plt.close()
    
    logger.info(f"Network topology plot saved as '{topology_path}'.")