# main.py

import cmd
import shlex
import sys
import os
import pyfiglet
import random
import time
from collections import defaultdict
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import torch
import numpy as np
import psutil
import logging
from colorama import init, Fore, Style
import yaml
from tqdm import tqdm


'''from .config import parse_arguments
from .model import Net
from .data_handler import load_datasets_dirichlet, print_class_distribution
from .topology import create_nodes, build_topology, visualize_and_save_topology
from .training import local_train, evaluate, summarize_model_parameters
from .communication import send_update, gossip_step, neighboring_step
from .utils import setup_logging, calculate_selection_probabilities
from .plotting import (visualize_data_distribution, plot_metrics_with_convergence, 
                      plot_loss_line, plot_training_aggregation_times, plot_additional_metrics)'''

from dfl_simulator.config import parse_arguments
from dfl_simulator.model import Net
from dfl_simulator.aggregation import aggregate
from dfl_simulator.data_handler import load_datasets_dirichlet, print_class_distribution
from dfl_simulator.topology import create_nodes, build_topology, visualize_and_save_topology
from dfl_simulator.training import local_train, evaluate, summarize_model_parameters
from dfl_simulator.communication import send_update, gossip_step, neighboring_step
from dfl_simulator.utils import setup_logging, calculate_selection_probabilities
from dfl_simulator.plotting import (
    visualize_data_distribution, 
    plot_metrics_with_convergence, 
    plot_loss_line, 
    plot_training_aggregation_times, 
    plot_additional_metrics
)

# Initialize colorama
init(autoreset=True)

def run_simulation(args, output_dir, logger):
    # Log and print all simulation parameters
    logger.info("Simulation Parameters:")
    print(Fore.MAGENTA + "Simulation Parameters:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
        print(Fore.MAGENTA + f"{arg}: {value}")

    metrics = defaultdict(lambda: defaultdict(list))
    cpu_usages = []
    round_times = []
    node_training_times = defaultdict(list)  # To store training times per node
    aggregation_times = defaultdict(list)     # To store aggregation times per round
    num_participants_per_round = [] # To store number of participants per round

    # Ensure reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    num_nodes = args.num_nodes
    train_datasets, test_dataset, labels = load_datasets_dirichlet(num_nodes, alpha=args.alpha)  
    print_class_distribution(train_datasets, logger)

    visualize_data_distribution(train_datasets, num_nodes, 
                                ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
                                output_dir, logger)
    nodes = create_nodes(num_nodes)

    # Build network topology
    G = build_topology(num_nodes, args.topology, args.topology_file)
    visualize_and_save_topology(G, args.topology, output_dir)

    # Map datasets to nodes
    node_datasets = {node: train_datasets[node] for node in nodes}

    # Split the global test_dataset into per-node test datasets
    # Shuffle and split test dataset equally among nodes
    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)
    split_size = len(test_dataset) // num_nodes
    test_loaders_per_node = {}
    for i in range(num_nodes):
        start = i * split_size
        end = (i + 1) * split_size if i < num_nodes -1 else len(test_dataset)
        node_test_indices = test_indices[start:end]
        node_test_subset = torch.utils.data.Subset(test_dataset, node_test_indices)
        node_test_loader = torch.utils.data.DataLoader(node_test_subset, batch_size=32, shuffle=False)
        test_loaders_per_node[nodes[i]] = node_test_loader

    # Implement MD Client Sampling based on data size
    selection_probabilities = calculate_selection_probabilities(node_datasets)
    logger.info(f"Client selection probabilities based on data size: {selection_probabilities}")

    # Precompute participating nodes for each round using MD sampling
    num_rounds = args.rounds
    participating_nodes_per_round = []
    num_participants = max(1, int(num_nodes * args.participation_rate))

    # Function to select clients based on MD sampling
    def select_clients_md_sampling(nodes, probabilities, m):
        """
        Select m clients based on predefined probabilities using multinomial sampling.
        """
        selected_indices = np.random.choice(len(nodes), size=m, replace=False, p=probabilities)
        selected_nodes = [nodes[i] for i in selected_indices]
        return selected_nodes

    for rnd in range(1, num_rounds + 1):
        participating_nodes = select_clients_md_sampling(nodes, selection_probabilities, num_participants)
        participating_nodes_per_round.append(participating_nodes)
        num_participants_per_round.append(len(participating_nodes))  # Track number of participants

    # Identify attacker nodes
    attacker_node_ids = []
    attack_counts = {}
    if args.use_attackers:
        if args.attacker_nodes is not None:
            attacker_node_ids = args.attacker_nodes
        else:
            # Ensure num_attackers does not exceed num_nodes
            num_attackers = min(args.num_attackers, num_nodes)
            # Use fixed seed random generator for attacker selection
            attacker_rng = random.Random(12345)
            attacker_node_ids = attacker_rng.sample(range(num_nodes), num_attackers)
        logger.info(f"Attacker nodes: {attacker_node_ids} with attacks: {args.attacks}")
        
        # Precompute the rounds in which each attacker participates
        attacker_participation_rounds = {attacker_node_id: [] for attacker_node_id in attacker_node_ids}
        for rnd, participating_nodes in enumerate(participating_nodes_per_round, start=1):
            for attacker_node_id in attacker_node_ids:
                if attacker_node_id in participating_nodes:
                    attacker_participation_rounds[attacker_node_id].append(rnd)

        # For each attacker node, randomly select rounds to perform attacks
        attacker_attack_rounds = {}
        for attacker_node_id in attacker_node_ids:
            participation_rounds = attacker_participation_rounds[attacker_node_id]
            max_attacks = args.max_attacks
            if max_attacks is None or max_attacks >= len(participation_rounds):
                # Attacker will perform attack in all participation rounds
                attack_rounds = participation_rounds
            else:
                attack_rounds = random.sample(participation_rounds, max_attacks)
            attacker_attack_rounds[attacker_node_id] = set(attack_rounds)

    # Initialize local models for each node
    local_models = {}
    for node in nodes:
        local_models[node] = Net()
    local_params = {node: local_models[node].state_dict() for node in nodes}

    # Initialize global model
    global_model = Net()
    global_model_state = global_model.state_dict()
    # Store node stats for parameter evolution
    node_stats = {}
    rounds_list = []  # Initialize rounds list

    # Initialize total execution time
    simulation_start_time = time.time()

    # Initialize lists to store total and aggregation times per round
    total_training_time_per_round = []
    total_aggregation_time_per_round = []

    # Use ProcessPoolExecutor for CPU-bound tasks like training
    with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), num_nodes)) as training_executor:
        # Initialize tqdm progress bar
        with tqdm(total=num_rounds, desc="Simulation Progress", unit="round") as pbar:
            for rnd in range(1, num_rounds + 1):
                logger.info(f"\n=== Round {rnd}/{num_rounds} ===")
                rounds_list.append(rnd)  # Append current round number
                start_time_round = time.time()
                cpu_usage = psutil.cpu_percent(interval=None)  # Record CPU usage at the start

                # Use precomputed participating nodes
                participating_nodes = participating_nodes_per_round[rnd - 1]
                logger.info(f"Participating nodes: {participating_nodes}")

                # Training phase: Local training for each participating node
                future_to_node = {}
                for node in participating_nodes:
                    attacker_type = None
                    if node in attacker_node_ids:
                        if rnd in attacker_attack_rounds[node]:
                            # The node will perform an attack in this round
                            if args.attacks:
                                attacker_type = random.choice(args.attacks)
                                logger.info(f"Node {node} is performing '{attacker_type}' attack in round {rnd}.")
                        else:
                            logger.info(f"Node {node} is an attacker but will not perform attack in round {rnd}.")
                    # Load the local model
                    local_model = local_models[node]
                    # Submit local training task
                    future = training_executor.submit(local_train, node, local_model, node_datasets[node], args.epochs, attacker_type)
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
                        data_sizes.append(len(node_datasets[node]))
                        logger.info(f"Node {node} completed training in {training_time:.2f} seconds.")

                        summarize_model_parameters(node, updated_params, logger)

                        # Record training time
                        node_training_times[node].append(training_time)  # Record training time
                        training_times.append(training_time)  # Collect training times for this round

                    except Exception as exc:
                        logger.error(f"Node {node} generated an exception during training: {exc}")

                # Calculate total training time for this round
                total_training_time = sum(training_times)
                total_training_time_per_round.append(total_training_time)

                # Sending phase: Nodes send updates (with possible delays)
                if successful_nodes:
                    send_futures = {}
                    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), num_nodes)) as send_executor:
                        for node in successful_nodes:
                            attacker_type = None
                            if node in attacker_node_ids and rnd in attacker_attack_rounds[node]:
                                if 'delay' in args.attacks:
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
                            logger.info(f"Node {node} finished sending updates in {sending_time:.2f} seconds.")
                        except Exception as exc:
                            logger.error(f"Node {node} generated an exception during sending updates: {exc}")
                    sending_time_total = max(sending_times) if sending_times else 0  # Max sending time
                    logger.info(f"Total Sending Time for round {rnd}: {sending_time_total:.2f} seconds.")
                else:
                    sending_time_total = 0
                    logger.info("No participating nodes to send updates.")

                # Aggregation phase: Perform FedAvg aggregation after all updates are sent
                if successful_nodes and models_state_dicts and data_sizes:
                    aggregated_state_dict = aggregate(models_state_dicts, data_sizes, logger)
                    if aggregated_state_dict:
                        # Update the global model
                        global_model.load_state_dict(aggregated_state_dict)
                        logger.info("Global model updated using Weighted Federated Averaging (FedAvg).")

                        # Distribute the updated global model to all nodes
                        for node in nodes:
                            local_models[node].load_state_dict(aggregated_state_dict)
                        logger.info("Global model distributed to all nodes.")
                    else:
                        logger.warning("Aggregation returned None. Global model not updated.")
                    # Record aggregation time as sending time
                    aggregation_time = sending_time_total  # Simplistic approach
                    total_aggregation_time_per_round.append(aggregation_time)
                    #logger.info(f"Gossiping for round {rnd} completed.")
                else:
                    aggregation_time = 0
                    total_aggregation_time_per_round.append(aggregation_time)
                    logger.info("No participating nodes to perform gossiping.")

                # Evaluation phase: training data
                logger.info("\n=== Evaluation Phase of training data ===")
                for node in nodes:
                    train_loader = torch.utils.data.DataLoader(node_datasets[node], batch_size=32, shuffle=False)
                    model = local_models[node]
                    logger.info(f"\nEvaluating model for node_{node} on training data...")
                    train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader)
                    logger.info(f"[Round {rnd}] Training Data Evaluation of node_{node} -> Loss: {train_loss:.4f}, "
                                f"Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, "
                                f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                    # Record training data metrics per node
                    metrics[node]['train_loss'].append(train_loss)
                    metrics[node]['train_accuracy'].append(train_accuracy)
                    metrics[node]['train_f1_score'].append(train_f1)
                    metrics[node]['train_precision'].append(train_precision)
                    metrics[node]['train_recall'].append(train_recall)

                # Evaluation phase: testing data
                logger.info("\n=== Evaluation Phase of testing data ===")
                for node in nodes:
                    test_loader = test_loaders_per_node[node]
                    model = local_models[node]
                    logger.info(f"\nEvaluating model for node_{node} on testing data...")
                    loss, accuracy, f1, precision, recall = evaluate(model, test_loader)
                    logger.info(f"[Round {rnd}] Evaluation of node_{node} -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                                f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                    # Record metrics per node
                    metrics[node]['loss'].append(loss)
                    metrics[node]['accuracy'].append(accuracy)
                    metrics[node]['f1_score'].append(f1)
                    metrics[node]['precision'].append(precision)
                    metrics[node]['recall'].append(recall)

                # Finalize round timing
                end_time_round = time.time()
                round_time = end_time_round - start_time_round
                round_times.append(round_time)
                cpu_usages.append(cpu_usage)
                logger.info(f"[Round {rnd}] Time taken: {round_time:.2f} seconds")

                # Update the tqdm progress bar
                pbar.update(1)
                pbar.set_postfix({"CPU Usage": f"{cpu_usage}%"})  # Optional: Add more metrics if desired

    simulation_end_time = time.time()
    total_execution_time = simulation_end_time - simulation_start_time

    return metrics, cpu_usages, round_times, total_training_time_per_round, total_aggregation_time_per_round, total_execution_time

class PhoenixShell(cmd.Cmd):
    intro = pyfiglet.figlet_format("Phoenix", font="slant") + "\nWelcome to Phoenix Shell! Type 'help' to see available commands.\n"
    prompt = "Phoenix> "
    
    def __init__(self):
        super().__init__()
        self.output_dir = "results"
        self.logger = None
        #self.setup_done = False
        self.simulation_args = None  # To store simulation arguments
    
    def do_parameters(self, arg):
        """Display all available simulation parameters and their options."""
        parameters_info = {
            "rounds": {
                "description": "Number of federated learning rounds",
                "type": "int",
                "default": 5,
                "options": "Any positive integer"
            },
            "epochs": {
                "description": "Number of local epochs",
                "type": "int",
                "default": 1,
                "options": "Any positive integer"
            },
            "num_nodes": {
                "description": "Number of nodes in the network",
                "type": "int",
                "default": 5,
                "options": "Any positive integer"
            },
            "num_attackers": {
                "description": "Number of attacker nodes",
                "type": "int",
                "default": 0,
                "options": "0 to num_nodes"
            },
            "attacker_nodes": {
                "description": "List of attacker node indices",
                "type": "list of ints",
                "default": "None",
                "options": "Specific node indices (e.g., 0 1 2)"
            },
            "attacks": {
                "description": "Types of attacks",
                "type": "list of strings",
                "default": "[]",
                "options": "delay, poison"
            },
            "use_attackers": {
                "description": "Include attacker nodes",
                "type": "bool",
                "default": False,
                "options": "True or False"
            },
            "participation_rate": {
                "description": "Fraction of nodes participating in each round",
                "type": "float",
                "default": 0.5,
                "options": "0 < rate <= 1"
            },
            "topology": {
                "description": "Network topology type",
                "type": "str",
                "default": "fully_connected",
                "options": "fully_connected, ring, random, custom"
            },
            "topology_file": {
                "description": "Path to the custom topology file (edge list)",
                "type": "str",
                "default": "None",
                "options": "Required if topology is 'custom'"
            },
            "max_attacks": {
                "description": "Maximum number of times an attacker can perform an attack",
                "type": "int",
                "default": "None",
                "options": "Any positive integer or None for unlimited"
            },
            "gossip_steps": {
                "description": "Number of gossip iterations per round",
                "type": "int",
                "default": 3,
                "options": "Any positive integer"
            },
            "protocol": {
                "description": "Communication protocol to use",
                "type": "str",
                "default": "gossip",
                "options": "gossip, neighboring"
            },
            "alpha": {
                "description": "Dirichlet distribution parameter for data distribution",
                "type": "float",
                "default": 0.5,
                "options": "Any positive float"
            },
            "config": {
                "description": "Configuration file for placing all the parameters",
                "type": "yaml",
                "default": "config.yaml",
                "options": "Specific configuration file path"
            },
            "simulation_name": {
                "description": "Name of the simulation configuration to use",
                "type": "str",
                "default": "None",
                "options": "Specific simulation name"
            }
        }

        print(Fore.CYAN + "Available Simulation Parameters:\n")
        for param, details in parameters_info.items():
            print(Fore.YELLOW + f"{param}:")
            print(Fore.WHITE + f"  Description : {details['description']}")
            print(Fore.WHITE + f"  Type        : {details['type']}")
            print(Fore.WHITE + f"  Default     : {details['default']}")
            print(Fore.WHITE + f"  Options     : {details['options']}\n")

    def do_setup(self, arg):
        """Initialize the simulation environment with desired configurations.
        Usage: setup --rounds 3 --epochs 1 --topology fully_connected --participation_rate 0.6 --gossip_steps 3 --protocol gossip
        """
        '''if self.setup_done:
            print(Fore.YELLOW + "Setup has already been completed.")
            return'''
        try:
            args = parse_arguments(shlex.split(arg))
            #validate_parameters(args)
            # Debug: Print parsed arguments
            print(Fore.BLUE + "Parsed Arguments:")
            for arg_name, arg_value in vars(args).items():
                print(Fore.BLUE + f"{arg_name.replace('_', ' ').capitalize()}: {arg_value}")
                # Initialize logging if not already done
                if not self.logger:
                    #logging.basicConfig(level=logging.INFO)
                    self.logger = logging.getLogger()
                self.logger.info(f"{arg_name}: {arg_value}")

            # Reset the setup_done flag to allow re-setup
            #self.setup_done = False

            # Step 1: Create 'results' directory or a new unique one if it exists
            base_dir = "results"
            if not os.path.exists(base_dir):
                self.output_dir = base_dir
                os.makedirs(self.output_dir)
                # Temporary logger before setting up full logging
                #logging.basicConfig(level=logging.INFO)
                self.logger.info(f"Created directory: {self.output_dir}")
            else:
                i = 1
                while True:
                    new_dir = f"{base_dir}_{i}"
                    if not os.path.exists(new_dir):
                        self.output_dir = new_dir
                        os.makedirs(self.output_dir)
                        #logging.basicConfig(level=logging.INFO)
                        self.logger.info(f"Created directory: {self.output_dir}")
                        break
                    i += 1

            # Step 2: Setup logging
            setup_logging(self.output_dir)
            self.logger = logging.getLogger()
            self.logger.info("Logging is configured.")
            #self.setup_done = True
            self.simulation_args = args
            print(Fore.GREEN + "Setup completed successfully.")
        except Exception as e:
            print(Fore.RED + f"An error occurred during setup: {e}")

    def do_run(self, arg):
        """Execute the simulation with the provided options.
        Usage: run --rounds 3 --epochs 1 --topology fully_connected --participation_rate 0.6 --gossip_steps 3 --protocol gossip
        """
        if not self.simulation_args:
            print(Fore.RED + "Please run the 'setup' command first with desired configurations.")
            return
        try:
            # Split the argument string into a list
            arg_list = shlex.split(arg)
        
            if arg_list:
                # Parse the new arguments provided during 'run'
                run_args = parse_arguments(arg_list)
                # Override simulation_args with run_args
                for key, value in vars(run_args).items():
                    if value is not None:
                        setattr(self.simulation_args, key, value)
            else:
                # No new arguments provided; retain simulation_args as set during 'setup'
                run_args = self.simulation_args
            
            # Debug: Print final simulation arguments after override
            '''print(Fore.BLUE + "Final Simulation Arguments:")
            print(Fore.BLUE + f"Rounds: {self.simulation_args.rounds}")
            print(Fore.BLUE + f"Epochs: {self.simulation_args.epochs}")
            print(Fore.BLUE + f"Topology: {self.simulation_args.topology}")
            print(Fore.BLUE + f"Participation Rate: {self.simulation_args.participation_rate}")
            print(Fore.BLUE + f"Number of Nodes: {simulation_args.num_nodes}")
            print(Fore.BLUE + f"Protocol: {simulation_args.protocol}")
            print(Fore.BLUE + f"Gossip Steps: {simulation_args.gossip_steps}")
            print(Fore.BLUE + f"Use Attackers: {simulation_args.use_attackers}")
            print(Fore.BLUE + f"Number of Attackers: {simulation_args.num_attackers}")
            print(Fore.BLUE + f"Attacks: {simulation_args.attacks}")
            print(Fore.BLUE + f"Alpha: {simulation_args.alpha}")
            print(Fore.BLUE + f"Config File: {simulation_args.config}")'''
            # Debug: Print final simulation arguments after override using a loop
            print(Fore.BLUE + "Final Simulation Arguments:")
            for arg_name, arg_value in vars(self.simulation_args).items():
                # Capitalize and format argument names
                formatted_arg_name = arg_name.replace('_', ' ').capitalize()
                print(Fore.BLUE + f"{formatted_arg_name}: {arg_value}")
                # Log the argument
                self.logger.info(f"{arg_name}: {arg_value}")

            print(Fore.CYAN + "Starting simulation...")
            self.logger.info("Starting simulation...")

            # Run the simulation
            metrics, cpu_usages, round_times, total_training_time_per_round, total_aggregation_time_per_round, total_execution_time = run_simulation(self.simulation_args, self.output_dir, self.logger)
            
            # After all rounds, compute training and aggregation times per round
            rounds_range = range(1, self.simulation_args.rounds + 1)

            # Log total training and aggregation times per round
            for rnd, (train_time, agg_time) in enumerate(zip(total_training_time_per_round, total_aggregation_time_per_round), start=1):
                self.logger.info(f"Round {rnd}: Total Training Time = {train_time:.2f} seconds")
                self.logger.info(f"Round {rnd}: Total Aggregation Time = {agg_time:.2f} seconds")

            # Plot the metrics with convergence and execution time annotations
            plot_metrics_with_convergence(metrics, rounds_range, total_execution_time, self.output_dir, self.logger)
            plot_loss_line(metrics, rounds_range, self.output_dir, self.logger)
            plot_training_aggregation_times(rounds_range, total_training_time_per_round, total_aggregation_time_per_round, total_execution_time, self.output_dir, self.logger)
            plot_additional_metrics(rounds_range, cpu_usages, round_times, self.output_dir, self.logger)

            # Calculate and log detailed statistics
            if cpu_usages and round_times:
                avg_cpu_usage = np.mean(cpu_usages)
                avg_round_time = np.mean(round_times)
                self.logger.info(f"\nAverage CPU Usage per Round: {avg_cpu_usage:.2f}%")
                self.logger.info(f"Average Time Taken per Round: {avg_round_time:.2f} seconds")

                # Calculate total metrics
                total_test_losses = []
                total_accuracies = []
                total_f1_scores = []
                total_precisions = []
                total_recalls = []

                for node in metrics:
                    total_test_losses.extend(metrics[node]['loss'])
                    total_accuracies.extend(metrics[node]['accuracy'])
                    total_f1_scores.extend(metrics[node]['f1_score'])
                    total_precisions.extend(metrics[node]['precision'])
                    total_recalls.extend(metrics[node]['recall'])

                # Now compute the averages
                avg_test_loss = np.mean(total_test_losses)
                avg_accuracy = np.mean(total_accuracies)
                avg_f1_score = np.mean(total_f1_scores)
                avg_precision = np.mean(total_precisions)
                avg_recall = np.mean(total_recalls)

                # Compute average training metrics over all nodes and rounds
                total_train_losses = []
                total_train_accuracies = []
                total_train_f1_scores = []
                total_train_precisions = []
                total_train_recalls = []

                for node in metrics:
                    total_train_losses.extend(metrics[node]['train_loss'])
                    total_train_accuracies.extend(metrics[node]['train_accuracy'])
                    total_train_f1_scores.extend(metrics[node]['train_f1_score'])
                    total_train_precisions.extend(metrics[node]['train_precision'])
                    total_train_recalls.extend(metrics[node]['train_recall'])

                # Now compute the averages
                avg_train_loss = np.mean(total_train_losses)
                avg_train_accuracy = np.mean(total_train_accuracies)
                avg_train_f1_score = np.mean(total_train_f1_scores)
                avg_train_precision = np.mean(total_train_precisions)
                avg_train_recall = np.mean(total_train_recalls)

                # Log the averages
                self.logger.info("\nAverage Evaluation Metrics over all nodes and rounds:")
                self.logger.info(f"Average Test Loss: {avg_test_loss:.4f}")
                self.logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
                self.logger.info(f"Average F1 Score: {avg_f1_score:.4f}")
                self.logger.info(f"Average Precision: {avg_precision:.4f}")
                self.logger.info(f"Average Recall: {avg_recall:.4f}")

                # Log the averaged training metrics
                self.logger.info("\nAverage Training Metrics over all nodes and rounds:")
                self.logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
                self.logger.info(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
                self.logger.info(f"Average Training F1 Score: {avg_train_f1_score:.4f}")
                self.logger.info(f"Average Training Precision: {avg_train_precision:.4f}")
                self.logger.info(f"Average Training Recall: {avg_train_recall:.4f}")

                self.logger.info("\nSimulation complete. Plots have been saved as PNG files.")
                print(Fore.CYAN + "\nSimulation completed successfully. Check the 'results' directory for outputs.")
            else:
                self.logger.info("\nNo metrics recorded to compute averages.")
                print(Fore.YELLOW + "\nSimulation completed, but no metrics were recorded.")
        except Exception as e:
            self.logger.error(f"An error occurred during execution: {e}")
            print(Fore.RED + f"An error occurred during execution: {e}")

    def do_help_custom(self, arg):
        """Show available commands and their descriptions."""
        commands = {
            'setup': 'Initialize the simulation environment with desired configurations. For ex. use: setup --config config.yaml --simulation_name sim1',
            'run': 'Execute the simulation with the provided options.',
            'list_simulations': 'List all available simulation configurations.',
            'parameters': 'Display all available simulation parameters and their options.',
            'help': 'Show this help message.',
            'exit': 'Exit the Phoenix shell.',
            'quit': 'Exit the Phoenix shell.'
        }
        if arg:
            # Show help for a specific command
            if arg in commands:
                print(f"{arg}: {commands[arg]}")
            else:
                print(f"No help available for '{arg}'.")
        else:
            print("Available commands:")
            for cmd_name, desc in commands.items():
                print(f"  {cmd_name:<20} {desc}")
            print("\nType 'help [command]' to get more information about a specific command.")

    def complete_setup(self, text, line, begidx, endidx):
        options = ['--rounds', '--epochs', '--topology', '--participation_rate', '--gossip_steps', '--protocol', '--use_attackers', '--num_attackers', '--attacks', '--alpha', '--config', '--simulation_name']
        if not text:
            completions = options
        else:
            completions = [option for option in options if option.startswith(text)]
        return completions
    
    def do_list_simulations(self, arg):
        """List all available simulation configurations."""
        config_file = "config.yaml"  # Default config file
        try:
            if self.simulation_args and self.simulation_args.config:
                config_file = self.simulation_args.config
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            simulations = config_data.get('simulations', {})
            if not simulations:
                print(Fore.YELLOW + "No simulations found in the config file.")
            else:
                print(Fore.GREEN + "Available simulations:")
                for sim in simulations:
                    print(Fore.GREEN + f" - {sim}")
        except FileNotFoundError:
            print(Fore.RED + f"Config file '{config_file}' not found.")
        except yaml.YAMLError as ye:
            print(Fore.RED + f"Error parsing YAML file: {ye}")
        except Exception as e:
            print(Fore.RED + f"Error loading config file: {e}")


    def complete_run(self, text, line, begidx, endidx):
        return self.complete_setup(text, line, begidx, endidx)

    def do_exit(self, arg):
        """Exit the Phoenix shell."""
        print(Fore.GREEN + "Exiting Phoenix. Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the Phoenix shell."""
        return self.do_exit(arg)

    def do_help(self, arg):
        """Show this help message."""
        self.do_help_custom(arg)

    def emptyline(self):
        """Do nothing on empty input line."""
        pass

    def default(self, line):
        """Handle unrecognized commands."""
        print(Fore.RED + f"Unrecognized command: '{line}'. Type 'help' to see available commands.")

def run_phoenix_shell():
    shell = PhoenixShell()
    shell.cmdloop()

def main():
    run_phoenix_shell()

if __name__ == "__main__":
    main()
