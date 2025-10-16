# module.py

import torch
import random
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import os

from fenics.data.handler import load_datasets_dirichlet, print_class_distribution
from fenics.topology import create_nodes, build_topology, visualize_and_save_topology
from fenics.plotting import visualize_data_distribution
from fenics.utils import calculate_selection_probabilities
from fenics.node.node import Node
from fenics.node.attacknode import AttackNode
from fenics.node.attacks.attackregistry import ATTACK_REGISTRY


class DataModule:
    """
    A module to handle all data loading and preprocessing operations.
    """
    
    def __init__(self, 
                 num_nodes: int, 
                 node_type_map: Dict[int, str], ### TO BE ADDED IN THE COMMAND.PY isnide run_simulation_command() ### TODO PRESUMED STRUCTURE {0:"normal", 1:"attack", 2:"mitigation"}
                 alpha: float, 
                 topology: str,
                 topology_file: Optional[str] = None,
                 output_dir: str = "results",
                 logger: Optional[logging.Logger] = None,
                 batch_size: int = 32,
                 random_seed: int = 0):
        """
        Initialize the data module.
        
        Args:
            num_nodes: Number of nodes in the network
            alpha: Dirichlet distribution parameter
            topology: Network topology type
            topology_file: Path to the custom topology file
            output_dir: Directory to save outputs
            logger: Logger instance
            batch_size: Batch size for data loaders
            random_seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.node_type_map = node_type_map  ### TODO PRESUMED STRUCTURE {0:"normal", 1:"attack", 2:"mitigation"}
        self.alpha = alpha
        self.topology = topology
        self.topology_file = topology_file
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger()
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize attributes that will be set later
        self.train_datasets = None
        self.test_dataset = None
        self.labels = None
        self.nodes = None
        self.G = None
        self.node_datasets = None
        self.test_loaders_per_node = None
        
    def setup(self) -> None:
        # TODO think about this when distributing data_training sets
        """
        Set up the data module by loading datasets and creating network topology.
        """
        # Load datasets with Dirichlet distribution
        self.train_datasets, self.test_dataset, self.labels = load_datasets_dirichlet(
            self.num_nodes, alpha=self.alpha, save_to_file=True, output_dir=self.output_dir
        )
        
        # Print and visualize class distribution
        print_class_distribution(self.train_datasets, self.logger)
        
        # Define class names for FashionMNIST
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
        visualize_data_distribution(
            self.train_datasets, 
            self.num_nodes, 
            class_names,
            self.output_dir, 
            self.logger
        )

        #def factory(node_id: int):
        #    for i in range(len(self.node_type_map)):
        #        if self.node_type_map[i].lower() in ATTACK_REGISTRY: ## SHOULD I CHECK keys() ??
        #            return AttackNode(node_id, attack_name=self.node_type_map[i].lower())
        #        else:
        #            return Node(node_id)
        #TODO INCLUDE ONE FOR MITIGATION NODE

        # Create nodes and build topology
        self.nodes = create_nodes(self.num_nodes)

        #TODO idk it should be temporary only to start code working correctly
        self.nodes_ids = [i for i in self.nodes]

        self.G = build_topology(self.num_nodes, self.topology, self.topology_file)
        visualize_and_save_topology(self.G, self.topology, self.output_dir)
        
        # Map datasets to nodes
        #self.node_datasets = {node_id: self.train_datasets[node_id] for node_id in self.nodes_ids}
        
        # Create test loaders for each node
        self._create_test_loaders()
        
        self.logger.info("Data module setup complete")
    
    def _create_test_loaders(self) -> None:
        """
        Split the global test dataset into per-node test datasets and create test loaders.
        """
        # Shuffle and split test dataset equally among nodes
        test_indices = list(range(len(self.test_dataset)))
        random.shuffle(test_indices)
        split_size = len(self.test_dataset) // self.num_nodes
        
        self.test_loaders_per_node = {}
        for i in range(self.num_nodes):
            start = i * split_size
            end = (i + 1) * split_size if i < self.num_nodes - 1 else len(self.test_dataset)
            node_test_indices = test_indices[start:end]
            node_test_subset = torch.utils.data.Subset(self.test_dataset, node_test_indices)
            node_test_loader = torch.utils.data.DataLoader(
                node_test_subset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            self.test_loaders_per_node[self.nodes[i]] = node_test_loader
    
    def get_train_loader(self, node_id: int) -> torch.utils.data.DataLoader:
        """
        Get training data loader for a specific node.
        
        Args:
            node_id: Node ID
            
        Returns:
            DataLoader for the node's training data
        """
        return torch.utils.data.DataLoader(
            self.node_datasets[node_id], 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def get_test_loader(self, node_id: int) -> torch.utils.data.DataLoader:
        """
        Get test data loader for a specific node.
        
        Args:
            node_id: Node ID
            
        Returns:
            DataLoader for the node's test data
        """
        return self.test_loaders_per_node[node_id]
    
    def get_data_sizes(self) -> Dict[int, int]:
        """
        Get the size of the dataset for each node.
        
        Returns:
            Dictionary mapping node IDs to dataset sizes
        """
        return {node: len(dataset) for node, dataset in self.node_datasets.items()}
    
    def calculate_selection_probabilities(self) -> List[float]:
        """
        Calculate selection probabilities based on the size of each node's dataset.
        
        Returns:
            List of selection probabilities for each node
        """
        return calculate_selection_probabilities(self.node_datasets)