from typing import List, Optional
import logging

from fenics.node.node import Node
from fenics.node.nodetype import NodeType
from fenics.node.attacknode import AttackNode
from fenics.node.attacks.attackregistry import autodiscover_attack_modules, get_attack, ATTACK_REGISTRY

class Simulator_MPI:
    
    def __init__(self,
                 node_id: int,
                 node_dataset_path: str,
                 type: str,
                 neighbors: List[int], 
                 attack_type: Optional[str] = None):
        
        self.node_id = node_id
        self.node_dataset_path = node_dataset_path
        self.type = type
        self.neighbors = neighbors
        self.attack_type = attack_type

        # TODO
        # self.simulation_rounds = simulation_rounds # TODO add in config.py

        # create the correct node instance 
        # either attack node or base node
        # TODO: CHANGE WHEN TO CALL THIS, SHOULD HAPPEN ONLY ONCE??
        autodiscover_attack_modules()

        # TODO: future implementation include mitigation type
        self.node = self.make_node()
    
    def get_own_info(self):
        print(f'Node: {self.node_id} with negighbors:{self.neighbors}, type: {self.type} and data_path: {self.node_dataset_path}')

    def make_node(self):
        """ 
        Create a node instance based on the node type. 
        """
        logger = logging.getLogger(f"Node_{self.node_id}")

        if self.type in ATTACK_REGISTRY.keys():
            node_instance = get_attack(self.type, node_id=self.node_id)
            # TODO fix to include this in logger
            print(f'Node: {self.node_id} created node instance:{node_instance}')

        else: 
            node_instance = Node(
                node_id=self.node_id,
                neighbors=self.neighbors,
                data_path=self.node_dataset_path,
                logger=logger
            )
            # TODO fix to include this in logger
            print(f'Node: {self.node_id} created node instance:{node_instance}')

        return node_instance
    
    def run_simulator(self):
        """
        This method will initialize the model training and any eventual attacks. 
        """
        # STEP 0: Iterate for self.simulation_rounds 
        # for round in range(self.simulation_rounds):

        # STEP 1: Call execute for node instance. 
        self.node.execute()

        # STEP 2: AGGREGATION
        # Wait until params from neighbors have been collected

