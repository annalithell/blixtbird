# blixtbird/node/attacks/freerider.py

import torch
import torch.nn as nn
import logging
from typing import Optional
import time

from torchvision import datasets, transforms
from blixtbird.node.attacks.attack_node import AttackNode
from blixtbird.node.attacks.attack_registry import register_attack

@register_attack("freerider")
class FreeRiderAttack(AttackNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """

    def __init__(self, node_id: int, data_path: str, neighbors: Optional[int], model_type, epochs, logger: Optional[logging.Logger] = None):
        """
        Initialize the free-rider attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
            self.attack_type = freerider
        """
        super().__init__(node_id, data_path, neighbors, model_type, epochs, logger)
        self.attack_round = 0 # Placeholder for potential future use


    def train_model(self):
        """
        Free-rider attack:
            - When a node participates in the network without training a model.  

        Returns:
            Model parameters of the node
        
        """
        train_dataset = torch.load(self.data_path, weights_only=False)
        self.data_sizes[self.node_id] = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        start_time = time.time()
        print((f"[Free-rider node {self.node_id}] fakes training..."))

        # Create test DataLoader
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        #  Use random metrics
        self.append_training_metrics(train_loader)
        self.append_test_metrics(test_loader)

        # TODO: manipulate training time in a more clever way to simulate free-rider behavior
        self.training_time = time.time() - start_time


    def execute(self):
        """
        Execute the free-rider attack by learning model parameters while doing no work. 
        
        Args:
            model: Model to intercept
        """
        self.train_model()
        print((f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s"))

        print(f"[Node {self.node_id}] will now start sending data")
        self.send()
        print(f"[Node {self.node_id}] has completed sending, now starting the recieve operation")
        self.recv()
        print(f"[Node {self.node_id}] Communication completed, starting aggregation .....")
        self.aggregate()
        #evaluate model after aggregation
        self.append_test_metrics_after_aggregation()
