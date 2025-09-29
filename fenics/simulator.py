# fenics/simulator.py

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

class Simulator:
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

        for node_id in self.nodes:
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

        # Initialize local models for each node
        local_models = {}
        for node in self.real_nodes:
            # Use the model factory to create the appropriate model type
            local_models[node.node_id] = ModelFactory.get_model(self.model_type)
        local_params = {node.node_id: local_models[node.node_id].state_dict() for node in self.real_nodes}

        # Initialize global model
        global_model = ModelFactory.get_model(self.model_type)
        global_model_state = global_model.state_dict()
        
        # Store node stats for parameter evolution
        node_stats = {}
        rounds_list = []  # Initialize rounds list

        # Initialize total execution time
        simulation_start_time = time.time()

        # Use ProcessPoolExecutor for CPU-bound tasks like training
        with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(self.real_nodes))) as training_executor:
            # Initialize tqdm progress bar
            with tqdm(total=self.num_rounds, desc="Simulation Progress", unit="round") as pbar:
                for rnd in range(1, self.num_rounds + 1):
                    self.logger.info(f"\n=== Round {rnd}/{self.num_rounds} ===")
                    rounds_list.append(rnd)  # Append current round number
                    start_time_round = time.time()
                    cpu_usage = psutil.cpu_percent(interval=None)  # Record CPU usage at the start

                    # Use precomputed participating nodes
                    participating_nodes = self.participating_nodes_per_round[rnd - 1]
                    self.logger.info(f"Participating nodes: {participating_nodes}")

                    # Training phase: Local training for each participating node
                    future_to_node = {}
                    for node in participating_nodes:
                        attacker_type = None
                        if node in self.attacker_node_ids:
                            if rnd in self.attacker_attack_rounds.get(node, set()):
                                # The node will perform an attack in this round
                                if self.attacks:
                                    attacker_type = random.choice(self.attacks)
                                    self.logger.info(f"Node {node} is performing '{attacker_type}' attack in round {rnd}.")
                            else:
                                self.logger.info(f"Node {node} is an attacker but will not perform attack in round {rnd}.")
                        # Load the local model
                        local_model = local_models[node]
                        # Submit local training task
                        future = training_executor.submit(local_train, node, local_model, self.node_datasets[node], self.epochs, attacker_type)
                        future_to_node[future] = node

                    # Collect training results
                    successful_nodes = []
                    training_times = []
                    models_state_dicts = []
                    data_sizes = []
                    for future in as_completed(future_to_node):
                        node = future_to_node[future]
                        try:
                            updated_params, training_time = future.result()  # Receive tuple
                            # Update the local model parameters
                            local_models[node].load_state_dict(updated_params)
                            successful_nodes.append(node)
                            models_state_dicts.append(updated_params)
                            data_sizes.append(len(self.node_datasets[node]))
                            self.logger.info(f"Node {node} completed training in {training_time:.2f} seconds.")

                            summarize_model_parameters(node, updated_params, self.logger)

                            # Record training time
                            node_training_times[node].append(training_time)  # Record training time
                            training_times.append(training_time)  # Collect training times for this round

                        except Exception as exc:
                            self.logger.error(f"Node {node} generated an exception during training: {exc}")

                    # Calculate total training time for this round
                    total_training_time = sum(training_times)
                    total_training_time_per_round.append(total_training_time)

                    # Sending phase: Nodes send updates (with possible delays)
                    if successful_nodes:
                        send_futures = {}
                        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(self.real_nodes))) as send_executor:
                            for node in successful_nodes:
                                attacker_type = None
                                if node in self.attacker_node_ids and rnd in self.attacker_attack_rounds.get(node, set()):
                                    if 'delay' in self.attacks:
                                        attacker_type = 'delay'
                                # Submit send_update
                                future = send_executor.submit(send_update, node, attacker_type)
                                send_futures[future] = node
                        # Collect sending times
                        sending_times = []
                        for future in as_completed(send_futures):
                            node = send_futures[future]
                            try:
                                sending_time = future.result()
                                sending_times.append(sending_time)
                                self.logger.info(f"Node {node} finished sending updates in {sending_time:.2f} seconds.")
                            except Exception as exc:
                                self.logger.error(f"Node {node} generated an exception during sending updates: {exc}")
                        sending_time_total = max(sending_times) if sending_times else 0  # Max sending time
                        self.logger.info(f"Total Sending Time for round {rnd}: {sending_time_total:.2f} seconds.")
                    else:
                        sending_time_total = 0
                        self.logger.info("No participating nodes to send updates.")

                    # Aggregation phase: Perform FedAvg aggregation after all updates are sent
                    if successful_nodes and models_state_dicts and data_sizes:
                        # aggregated_state_dict = aggregate(models_state_dicts, data_sizes, self.logger)
                        fedavg_strategy = AggregationFactory.get_strategy('fedavg', logger=self.logger)
                        aggregated_state_dict = fedavg_strategy.aggregate(models_state_dicts, data_sizes)
                        if aggregated_state_dict:
                            # Update the global model
                            global_model.load_state_dict(aggregated_state_dict)
                            self.logger.info("Global model updated using Weighted Federated Averaging (FedAvg).")

                            # Distribute the updated global model to all nodes
                            for node in self.real_nodes:
                                local_models[node.node_id].load_state_dict(aggregated_state_dict)
                            self.logger.info("Global model distributed to all nodes.")
                        else:
                            self.logger.warning("Aggregation returned None. Global model not updated.")
                        # Record aggregation time as sending time
                        aggregation_time = sending_time_total  # Simplistic approach
                        total_aggregation_time_per_round.append(aggregation_time)
                    else:
                        aggregation_time = 0
                        total_aggregation_time_per_round.append(aggregation_time)
                        self.logger.info("No participating nodes to perform aggregation.")

                    # Evaluation phase: training data
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
