# fenics/node/attacks/freerider.py

import torch
import torch.nn as nn
import logging
from typing import Optional, override
import time
import numpy as np

from fenics.node.attacks.attack_node import AttackNode
from fenics.node.attacks.attack_registry import register_attack

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
        #self.__attack_type__ = attack_type


    def train_model(self):
        """
        Free-rider attack:
            - When a node participates in the network without training a model.  

        Returns:
            Model parameters of the node
        
        """
        train_dataset = torch.load(self.data_path, weights_only=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        start_time = time.time()
        self.logger.info(f"[Free-rider node {self.node_id}] fakes training...")

        #  Use random metrics
        self.append_training_metrics()
        self.append_test_metrics()

        # TODO: manipulate training time
        training_time = time.time() - start_time
        return self.model.state_dict(), training_time # NOT NEEDED??

    def append_training_metrics(self):
        # TODO maybe not generate random data here. (when data has been aggregated)
        # Evaluation phase: training data
        train_loss = np.random.random(1)[0]
        train_accuracy = np.random.random(1)[0]
        train_f1 = np.random.random(1)[0]
        train_precision = np.random.random(1)[0]
        train_recall = np.random.random(1)[0]

        self.metrics_train.append({'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'train_f1_score': train_f1,
                                'train_precision': train_precision,
                                'train_recall':train_recall})
    
    def append_test_metrics(self):
        # Evaluation phase: testing data
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


    def execute(self):
        """
        Execute the free-rider attack by learning model parameters while doing no work. 
        
        Args:
            model: Model to intercept
        """
        self.model_params, self.training_time = self.train_model()
        self.logger.info(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")
        

## This is done explicitly in attack_factory.py
# Register the attack
#AttackFactory.register_attack('freerider', FreeRiderAttack)