# topology.py

import networkx as nx
import matplotlib.pyplot as plt
import logging
import os

def create_nodes(num_nodes):
    return list(range(num_nodes))  # Nodes are represented by integer indices

def build_topology(num_nodes, topology_type, topology_file=None):
    if topology_type == 'fully_connected':
        G = nx.complete_graph(num_nodes)
    elif topology_type == 'ring':
        G = nx.cycle_graph(num_nodes)
    elif topology_type == 'random':
        p_connect = 0.3  # Probability of edge creation
        G = nx.erdos_renyi_graph(num_nodes, p_connect, seed=0)
    elif topology_type == 'custom':
        if topology_file is None:
            raise ValueError("Custom topology requires a topology file.")
        G = nx.read_edgelist(topology_file, nodetype=int)
        # Ensure the graph has the correct number of nodes
        G.add_nodes_from(range(num_nodes))
    else:
        raise ValueError("Invalid topology type. Choose from 'fully_connected', 'ring', 'random', 'custom'.")
    return G

def visualize_and_save_topology(G, topology_type, output_dir):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=0)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(f"Network Topology: {topology_type}")
    topology_path = os.path.join(output_dir, 'network_topology.png')
    plt.savefig(topology_path)
    plt.close()
    logging.info(f"Network topology plot saved as '{topology_path}'.")

