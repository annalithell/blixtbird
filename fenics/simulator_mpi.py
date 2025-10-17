from typing import List, Optional
import logging

from fenics.node.normal_node import NormalNode
from fenics.node.node_type import NodeType
from fenics.node.attacks.old.attacknode import AttackNode
from fenics.node.attacks.attack_registry import autodiscover_attack_modules, get_attack, ATTACK_REGISTRY

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
        self.attack_type = attack_type # TODO: This is never used, fix?

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
            attack_type = get_attack(self.type)
            node_instance=attack_type (
                node_id=self.node_id,
                neighbors=self.neighbors,
                data_path=self.node_dataset_path,
                logger=logger    
            )
            print(f'Node: {self.node_id} created node instance:{self.type}')
 
        elif self.type == "base": # TODO fix proper elif statement (config.yaml)
            node_instance = NormalNode(
                node_id=self.node_id,
                neighbors=self.neighbors,
                data_path=self.node_dataset_path,
                logger=logger
            )
            # TODO fix to include this in logger
            print(f'Node: {self.node_id} created node instance:{self.type}')
        else:
            # TODO add proper error handling
            print(f"{self.type} not implemented.")
        return node_instance
    
    def run_simulator(self):
        """
        This method will initialize the model training and any eventual attacks. 
        """
        # STEP 0: Iterate for self.simulation_rounds 
        # for round in range(self.simulation_rounds):

        # STEP 1: Call execute for node instance. 
        self.node.execute()

        # self.params = Training model (i nod)
        # aggregation(Self.node.params)
        # 


        # STEP 2: AGGREGATION
        # Wait until params from neighbors have been collected

