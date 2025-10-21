# fenics/node/baseattack.py

import torch
import torch.nn as nn
import logging
import pickle
import time

from mpi4py import MPI
from typing import Optional
from torchvision import datasets, transforms

from fenics.node.node_type import NodeType
from fenics.training.evaluator import evaluate
from fenics.node.abstract import AbstractNode


class AttackNode(AbstractNode):
    """ A  base attack class for all attacks. """    
    
    def __init__(self, node_id: int, data_path: str,  neighbors: Optional[int], model_type: str, epochs, logger: Optional[logging.Logger] = None):
        """
        Initialize the attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, data_path, neighbors, model_type, epochs, logger)
        self.node_type = NodeType.ATTACK
    

    def train_model(self):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """
        train_dataset = torch.load(self.data_path, weights_only=False)
        device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create test DataLoader
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        start_time = time.time()
        self.model.train()
        #self.logger.info(f"[Node {self.node_id}] Training for {self.epochs} epochs...")
        print(f"[Node {self.node_id}] Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            #self.logger.info(f"[Node {self.node_id}] Epoch {epoch+1}/{self.epochs}")
            print(f"[Node {self.node_id}] Epoch {epoch+1}/{self.epochs}")
            #  After each epoch append training metrics
            self.append_training_metrics(train_loader)  

        # after each epoch evaluate test
        self.append_test_metrics(test_loader)

        training_time = time.time() - start_time
        return self.model.state_dict(), training_time # TODO training time NEEDED??
    

    def append_training_metrics(self, train_loader):
        # Evaluation phase: training data

        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(self.model, train_loader)

        self.metrics_train.append({'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'train_f1_score': train_f1,
                                'train_precision': train_precision,
                                'train_recall':train_recall})
    

    def append_test_metrics(self, test_loader):
        # Evaluation phase: testing data
        for _ in range(0, self.epochs):

            loss, accuracy, f1, precision, recall = evaluate(self.model, test_loader)

            self.metrics_test.append({'test_loss': loss,
                                    'test_accuracy': accuracy,
                                    'test_f1_score': f1,
                                    'test_precision': precision,
                                    'test_recall': recall})


    def execute(self):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        #train_dataset = torch.load(self.data_path, weights_only=False)
        self.model_params, self.training_time = self.train_model()
        #self.logger.info(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")
        print((f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s"))
