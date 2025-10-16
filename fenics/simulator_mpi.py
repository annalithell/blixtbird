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

        # create the correct node instance 
        # either attack node or base node
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

        else: 
            node_instance = Node(
                node_id=self.node_id,
                #dataset_path=self.node_dataset_path,
                logger=logger
            )

        return node_instance

