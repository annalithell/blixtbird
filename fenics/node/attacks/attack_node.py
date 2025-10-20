# fenics/node/baseattack.py

import torch
import torch.nn as nn
import logging
import pickle
import time

from mpi4py import MPI
from typing import Optional
from fenics.node.node_type import NodeType
from fenics.node.attacks.attack_registry import get_attack 
from fenics.training.trainer import local_train
from fenics.training.evaluator import evaluate
from fenics.node.abstract import AbstractNode


class AttackNode(AbstractNode):
    """ A  base attack class for all attacks. """    
    
    def __init__(self, node_id: int, data_path: str,  neighbors: Optional[int], model_type: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, data_path, neighbors, model_type, logger)
        self.node_type = NodeType.ATTACK
    

    def train_model(self, train_dataset, epochs):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """
        device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        start_time = time.time()
        self.model.train()
        self.logger.info(f"[Node {self.node_id}] Training for {epochs} epochs...")

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            self.logger.info(f"[Node {self.node_id}] Epoch {epoch+1}/{epochs}")

            #  After each epoch append training metrics
            self.append_training_metrics(self.model, train_loader)

        # after each epoch evaluate test
        #TODO
        #self.append_test_metrics()

        training_time = time.time() - start_time
        return self.model.state_dict(), training_time # NOT NEEDED??
    

    def append_training_metrics(self, model, train_loader):
        # Evaluation phase: training data

        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader)

        self.metrics_train.append({'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'train_f1_score': train_f1,
                                'train_precision': train_precision,
                                'train_recall':train_recall})
    

    def execute(self, epochs):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        train_dataset = torch.load(self.data_path, weights_only=False)
        self.model_params, self.training_time = self.train_model(train_dataset, epochs)
        self.logger.info(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")
