# fenics/node/baseattack.py

import torch
import logging
from mpi4py import MPI
import pickle
from typing import Optional
from fenics.node.node_type import NodeType
from fenics.node.attacks.attack_registry import get_attack 

from fenics.node.abstract import AbstractNode


class AttackNode(AbstractNode):
    """ A  base attack class for all attacks. """    
    
    def __init__(self, node_id: int, neighbors: Optional[int], data_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the attack
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, neighbors, data_path, logger)
        self.node_type = NodeType.ATTACK
        
        #self.attack = get_attack(attack_name, node_id=node_id)
        #self.attack_type = self.attack.__attack_type__
        self.comm = MPI.COMM_WORLD


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
