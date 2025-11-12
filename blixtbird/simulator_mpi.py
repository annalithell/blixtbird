from typing import List
import logging

from blixtbird.node.normal_node import NormalNode
from blixtbird.node.attacks.attack_registry import autodiscover_attack_modules, get_attack, ATTACK_REGISTRY
from blixtbird.training.trainer import load_dataset
from blixtbird.local_data_manipulation.csv_metric import make_pandas_df, make_csv, concat_pandas_df
from blixtbird.training.trainer import load_dataset

class Simulator_MPI:
    
    def __init__(self,
                 node_id: int,
                 node_dataset_path: str,
                 type: str,
                 neighbors: List[int], 
                 epochs: int, 
                 rounds: int, 
                 model: str):
        
        self.node_id = node_id
        self.node_dataset_path = node_dataset_path
        self.type = type
        self.neighbors = neighbors
        self.epochs = epochs
        self.simulation_rounds = rounds
        self.model = model
        #self.attack_type = attack_type # TODO: This is never used, fix?

        # create the correct node instance 
        # either attack node or base node
        # TODO: CHANGE WHEN TO CALL THIS, SHOULD HAPPEN ONLY ONCE
        autodiscover_attack_modules()

        # TODO: future implementation include mitigation type
        self.node = self.make_node()
        self.node_dataset = load_dataset(node_id)
        self.metrics_train = []
        self.metrics_test = []
    
    def get_own_info(self):
        print(f'Node: {self.node_id} with negighbors:{self.neighbors}, type: {self.type}, data_path: {self.node_dataset_path} and epochs: {self.epochs}')

    def make_node(self):
        """ 
        Create a node instance based on the node type. 
        """
        logger = logging.getLogger(f"Node_{self.node_id}")

        if self.type in ATTACK_REGISTRY.keys():
            attack_type = get_attack(self.type)
            node_instance=attack_type (
                node_id=self.node_id,
                data_path=self.node_dataset_path,
                neighbors=self.neighbors,
                model_type=self.model,
                epochs = self.epochs, 
                logger=logger    
            )
            print(f'Node: {self.node_id} created node instance:{self.type}')
 
        elif self.type == "base": # TODO fix proper elif statement (config.yaml)
            node_instance = NormalNode( 
                node_id=self.node_id,
                data_path=self.node_dataset_path,
                neighbors=self.neighbors,
                model_type=self.model,
                epochs = self.epochs,
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
        # Iterate for self.simulation_rounds 
        for round in range(self.simulation_rounds):

            # execute includes training + potential attack.
            self.node.execute()


    def make_local_metrics(self):

        train_df = make_pandas_df(self.node.metrics_train)
        test_df = make_pandas_df(self.node.metrics_test)
        test_df_aa = make_pandas_df(self.node.metrics_test_after_aggregation)

        df = concat_pandas_df(concat_pandas_df(train_df, test_df), test_df_aa)

        make_csv(df, self.node_id)

        return
