from typing import List
import numpy as np

from fenics.training.trainer import load_dataset
from fenics.local_data_manipulation.csv_metric import make_pandas_df, make_csv, concat_pandas_df
from fenics.training.evaluator import evaluate
from fenics.training.trainer import load_dataset

class Simulator_MPI:
    
    def __init__(self,
                 node_id: int,
                 node_dataset_path: str,
                 type: str,
                 neighbors: List[int]):
        
        self.node_id = node_id
        self.node_dataset_path = node_dataset_path
        self.type = type
        self.neighbors = neighbors
        self.node_dataset = load_dataset(node_id)
        self.dataset = load_dataset(node_id)
        self.metrics_train = []
        self.metrics_test = []
    
    def get_own_info(self):
        print(f'Node: {self.node_id} with negighbors:{self.neighbors}, type: {self.type} and data_path: {self.node_dataset_path}')


    def run_simulation(self):
        """
        This function is for TEST METRICS!!!
        similar layout as in old simulation
        """

        epochs = 5

        # Evaluation phase: training data

        for _ in range(0, epochs):
            #TODO change evaluate function - adapt to new data
            #train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader)

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
            
        return
        

    def make_local_metrics(self):

        train_df = make_pandas_df(self.metrics_train)
        test_df = make_pandas_df(self.metrics_test)

        df = concat_pandas_df(train_df, test_df)

        make_csv(df, self.node_id)

        return
