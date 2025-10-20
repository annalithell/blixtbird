# fenics/node/baseattack.py

import torch
import logging
from mpi4py import MPI
import pickle
from typing import Optional
from fenics.node.node_type import NodeType
from fenics.node.attacks.attack_registry import get_attack 
from fenics.training.trainer import local_train

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
    

    def train_model(self):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """

        return
    

    def execute(self):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        self.model_params = self.train_model()

        #self.logger.info(f"[node_{self.node_id}] is a normal node and do nothing")
        #return model.parameters()
