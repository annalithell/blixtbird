# fenics/simulator_2.0.py

import torch
import random
import time
import logging
import multiprocessing
import psutil
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Set

from fenics.models import ModelFactory
from fenics.training import local_train, evaluate, summarize_model_parameters
from fenics.communication import send_update
from fenics.aggregation import AggregationFactory
from fenics.node.node import Node

class Simulator2:
    """
    Handles the simulation of the federated learning process.
    """
    
    def __init__(self,
                 nodes: List[int],
                 node_datasets: Dict[int, Any],
                 test_loaders_per_node: Dict[int, Any],
                 participating_nodes_per_round: List[List[int]],
                 attacker_node_ids: List[int],
                 attacker_attack_rounds: Dict[int, Set[int]],
                 num_rounds: int,
                 epochs: int,
                 attacks: List[str],
                 model_type: str = "cnn",
                 logger: Optional[logging.Logger] = None):
        
        """
        Initialize the simulator.
        
        Args:
            nodes: List of all node IDs
            node_datasets: Dictionary mapping node IDs to their datasets
            test_loaders_per_node: Dictionary mapping node IDs to their test loaders
            participating_nodes_per_round: List of lists of participating nodes for each round
            attacker_node_ids: List of attacker node IDs
            attacker_attack_rounds: Dictionary mapping attacker node IDs to sets of rounds when they will attack
            num_rounds: Number of training rounds
            epochs: Number of local epochs
            attacks: List of attack types
            model_type: Type of model to use (e.g., 'cnn', 'mlp')
            logger: Logger instance
        """
        
        self.nodes = nodes
        self.node_datasets = node_datasets
        self.test_loaders_per_node = test_loaders_per_node
        self.participating_nodes_per_round = participating_nodes_per_round
        self.attacker_node_ids = attacker_node_ids
        self.attacker_attack_rounds = attacker_attack_rounds
        self.num_rounds = num_rounds
        self.epochs = epochs
        self.attacks = attacks
        self.model_type = model_type
        self.logger = logger or logging.getLogger()
        self.real_nodes = self.make_nodes()

    def make_nodes(self):
        """ Create a list of "real" nodes. """
        real_nodes = []

        # dict = read .yaml file and parse 

        #for node_id in max_id:
        for node_id in self.nodes:
            # if node_id in yaml.nodes:
                # node = Node(node_id=node_id, node_type = yaml.nodes.val)
                # real_nodes.append(node)

            if node_id in self.attacker_node_ids: 
                #attacker_type = random.choice(self.attacks)
                node = Node(node_id=node_id, logger=self.logger)
                real_nodes.append(node)
        
        return real_nodes
        
    def run_simulation(self) -> Tuple[Dict, List[float], List[float], List[float], List[float], float]:
        """
        Run the simulation.
        
        Returns:
            Tuple containing:
                - metrics: Dictionary of metrics for each node
                - cpu_usages: List of CPU usage percentages for each round
                - round_times: List of times taken for each round
                - total_training_time_per_round: List of total training times for each round
                - total_aggregation_time_per_round: List of total aggregation times for each round
                - total_execution_time: Total execution time for the entire simulation
        """
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        metrics = defaultdict(lambda: defaultdict(list))
        cpu_usages = []
        round_times = []
        node_training_times = defaultdict(list)
        total_training_time_per_round = []
        total_aggregation_time_per_round = []

        # Initialize global model and distribute

        # Initialize local models for each node
        
        # Store node stats for parameter evolution
        node_stats = {}
        rounds_list = []  # Initialize rounds list, replace with counter in node?
        # TODO init parameter in node class "when_attack"

        # Initialize total execution time
        simulation_start_time = time.time()

        # Use ProcessPoolExecutor for CPU-bound tasks like training
        with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(self.real_nodes))) as training_executor:
            # Initialize tqdm progress bar
            with tqdm(total=self.num_rounds, desc="Simulation Progress", unit="round") as pbar:
                for rnd in range(1, self.num_rounds + 1):
                    # TODO look into what self.num_rounds
                    self.logger.info(f"\n=== Round {rnd}/{self.num_rounds} ===")
                    rounds_list.append(rnd)  # Append current round number, remove and replace with counter
                    start_time_round = time.time()
                    cpu_usage = psutil.cpu_percent(interval=None)  # Record CPU usage at the start

                    # Use precomputed participating nodes
                    participating_nodes = self.participating_nodes_per_round[rnd - 1]
                    self.logger.info(f"Participating nodes: {participating_nodes}")

                    # Training phase: Local training for each participating node

                    # Collect training results, aka extract training data and preprocess it, record training time
                    # TODO: Add model_parameters + training_data inside node class?
                    # TODO: Summarize model parameteres
                    # Think about training times?

                    # Sending phase: Nodes send updates (with possible delays)
                    # This is where the funky send_update function was previously located


                    # Aggregation phase: Perform FedAvg aggregation after all updates are sent
                    # TODO this is where they did the centralized approach 
                    # TODO fix aggregation function
                    # TODO update global model

                    # Evaluation phase: training data
                    # TODO replace old parameters with new ones
                    # TODO refactor this into a separate logger function
                    self.logger.info("\n=== Evaluation Phase of training data ===")
                    for node in self.real_nodes:
                        train_loader = torch.utils.data.DataLoader(self.node_datasets[node.node_id], batch_size=32, shuffle=False)
                        model = local_models[node.node_id]
                        self.logger.info(f"\nEvaluating model for node_{node.node_id} on training data...")
                        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader)
                        self.logger.info(f"[Round {rnd}] Training Data Evaluation of node_{node.node_id} -> Loss: {train_loss:.4f}, "
                                    f"Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, "
                                    f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                        # Record training data metrics per node
                        metrics[node.node_id]['train_loss'].append(train_loss)
                        metrics[node.node_id]['train_accuracy'].append(train_accuracy)
                        metrics[node.node_id]['train_f1_score'].append(train_f1)
                        metrics[node.node_id]['train_precision'].append(train_precision)
                        metrics[node.node_id]['train_recall'].append(train_recall)

                    # Evaluation phase: testing data
                    self.logger.info("\n=== Evaluation Phase of testing data ===")
                    for node in self.real_nodes:
                        test_loader = self.test_loaders_per_node[node.node_id]
                        model = local_models[node.node_id]
                        self.logger.info(f"\nEvaluating model for node_{node.node_id} on testing data...")
                        loss, accuracy, f1, precision, recall = evaluate(model, test_loader)
                        self.logger.info(f"[Round {rnd}] Evaluation of node_{node.node_id} -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                                    f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                        # Record metrics per node
                        metrics[node.node_id]['loss'].append(loss)
                        metrics[node.node_id]['accuracy'].append(accuracy)
                        metrics[node.node_id]['f1_score'].append(f1)
                        metrics[node.node_id]['precision'].append(precision)
                        metrics[node.node_id]['recall'].append(recall)

                    # Finalize round timing
                    end_time_round = time.time()
                    round_time = end_time_round - start_time_round
                    round_times.append(round_time)
                    cpu_usages.append(cpu_usage)
                    self.logger.info(f"[Round {rnd}] Time taken: {round_time:.2f} seconds")

                    # Update the tqdm progress bar
                    pbar.update(1)
                    pbar.set_postfix({"CPU Usage": f"{cpu_usage}%"})  # Optional: Add more metrics if desired

        simulation_end_time = time.time()
        total_execution_time = simulation_end_time - simulation_start_time

        return metrics, cpu_usages, round_times, total_training_time_per_round, total_aggregation_time_per_round, total_execution_time
