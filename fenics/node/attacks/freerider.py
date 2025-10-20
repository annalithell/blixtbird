# fenics/node/attacks/freerider.py

import torch
import torch.nn as nn
import logging
from typing import Optional, override
import time

from fenics.node.attacks.attack_node import AttackNode
from fenics.node.attacks.attack_registry import register_attack
from fenics.training.evaluator import evaluate
from fenics.models import ModelFactory

@register_attack("freerider")
class FreeRiderAttack(AttackNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """

    def __init__(self, node_id: int, neighbors: Optional[int], data_path: str, model_type, attack_type: str = "freerider", logger: Optional[logging.Logger] = None):
        """
        Initialize the free-rider attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
            self.attack_type = freerider
        """
        super().__init__(node_id, neighbors, data_path, logger)
        self.attack_round = 0 # Placeholder for potential future use
        self.__attack_type__ = attack_type # TODO redundant?


    def train_model(self, train_dataset, epochs=5):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        start_time = time.time()
        model.train()
        self.logger.info(f"[Node {self.node_id}] Training for {epochs} epochs...")

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            self.logger.info(f"[Node {self.node_id}] Epoch {epoch+1}/{epochs}")

            #  After each epoch append training metrics
            self.append_training_metrics(model, train_loader)

        # after each epoch evaluate test
        #TODO
        #self.append_test_metrics()

        training_time = time.time() - start_time
        return model.state_dict(), training_time # NOT NEEDED??
    
    def append_training_metrics(self, model, train_loader):
        # Evaluation phase: training data

        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader)

        self.metrics_train.append({'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'train_f1_score': train_f1,
                                'train_precision': train_precision,
                                'train_recall':train_recall})
        
    """  
    def append_test_metrics(self, epochs):
        # Evaluation phase: testing data
        for _ in range(0, epochs):
            #TODO change evaluate function - adapt to new data
            #loss, accuracy, f1, precision, recall = evaluate(model, test_loader)

            loss = np.random.random(1)[0]
            accuracy = np.random.random(1)[0]
            f1 = np.random.random(1)[0]
            precision = np.random.random(1)[0]
            recall = np.random.random(1)[0]

            self.metrics_test.append({'test_loss': loss,
                                    'test_accuracy': accuracy,
                                    'test_f1_score': f1,
                                    'test_precision': precision,
                                    'test_recall': recall})
    """ 

    def execute(self, epochs=5):
        """
        Execute the free-rider attack by learning model parameters while doing no work. 
        
        Args:
            model: Model to intercept
        """
        # TODO fix this so no training happens
        train_dataset = torch.load(self.data_path, weights_only=False)
        self.model_params, self.training_time = self.train_model(train_dataset, epochs)
        self.logger.info(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")
        

## This is done explicitly in attack_factory.py
# Register the attack
#AttackFactory.register_attack('freerider', FreeRiderAttack)