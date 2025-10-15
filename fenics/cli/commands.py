# fenics/cli/commands.py

import os
import shlex
import logging
import numpy as np
import yaml
from colorama import Fore

from fenics.config import parse_arguments, SimulationConfig, load_config_from_file
from fenics.data import DataModule
from fenics.client_selection import ClientSelector
from fenics.attack import AttackManager
from fenics.simulator import Simulator
from fenics.utils import setup_logging
from fenics.plotting import plot_metrics_with_convergence, plot_loss_line, plot_training_aggregation_times, plot_additional_metrics


def setup_environment(logger=None):
    """
    Create the output directory for simulation results.
    
    Args:
        logger: Logger instance
        
    Returns:
        Path to the output directory
    """
    logger = logger or logging.getLogger()
    
    # Step 1: Create 'results' directory or a new unique one if it exists
    base_dir = "results"
    if not os.path.exists(base_dir):
        output_dir = base_dir
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    else:
        i = 1
        while True:
            new_dir = f"{base_dir}_{i}"
            if not os.path.exists(new_dir):
                output_dir = new_dir
                os.makedirs(output_dir)
                logger.info(f"Created directory: {output_dir}")
                break
            i += 1
    
    return output_dir


def run_simulation_command(arg, simulation_args, output_dir, logger):
    """
    Run the simulation with the provided options.
    
    Args:
        arg: Command-line arguments
        simulation_args: Simulation arguments from setup
        output_dir: Output directory
        logger: Logger instance
    """
    # Parse any additional arguments provided during run
    if arg:
        arg_list = shlex.split(arg)
        run_args = parse_arguments(arg_list)
        
        # Override simulation_args with run_args
        for key, value in vars(run_args).items():
            if value is not None:
                setattr(simulation_args, key, value)
    
    # Print final simulation arguments
    print(Fore.BLUE + "Final Simulation Arguments:")
    for arg_name, arg_value in vars(simulation_args).items():
        formatted_arg_name = arg_name.replace('_', ' ').capitalize()
        print(Fore.BLUE + f"{formatted_arg_name}: {arg_value}")
        logger.info(f"{arg_name}: {arg_value}")
    
    print(Fore.CYAN + "Starting simulation...")
    logger.info("Starting simulation...")
    
    # Set up data module
    data_module = DataModule(
        num_nodes=simulation_args.num_nodes,
        alpha=simulation_args.alpha,
        topology=simulation_args.topology,
        topology_file=simulation_args.topology_file,
        output_dir=output_dir,
        logger=logger
    )
    data_module.setup()
    
    # Set up client selector
    client_selector = ClientSelector(
        nodes=data_module.nodes,
        participation_rate=simulation_args.participation_rate,
        logger=logger
    )
    
    # Calculate selection probabilities based on data size
    selection_probabilities = data_module.calculate_selection_probabilities()
    
    # Precompute participating nodes for all rounds
    participating_nodes_per_round = client_selector.precompute_participating_nodes(
        num_rounds=simulation_args.rounds,
        probabilities=selection_probabilities
    )
    
    # Set up attack manager
    attack_manager = AttackManager(
        num_nodes=simulation_args.num_nodes,
        use_attackers=simulation_args.use_attackers,
        num_attackers=simulation_args.num_attackers,
        attacker_nodes=simulation_args.attacker_nodes,
        attacks=simulation_args.attacks,
        max_attacks=simulation_args.max_attacks,
        logger=logger
    )
    
    # Identify attackers
    attacker_node_ids = attack_manager.identify_attackers()
    
    # Plan attacks
    attacker_attack_rounds = attack_manager.plan_attacks(participating_nodes_per_round)
    
    # Create and run the simulator
    simulator = Simulator(
        nodes=data_module.nodes,
        node_datasets=data_module.node_datasets,
        test_loaders_per_node=data_module.test_loaders_per_node,
        participating_nodes_per_round=participating_nodes_per_round,
        attacker_node_ids=attacker_node_ids,
        attacker_attack_rounds=attacker_attack_rounds,
        num_rounds=simulation_args.rounds,
        epochs=simulation_args.epochs,
        attacks=simulation_args.attacks,
        model_type=getattr(simulation_args, 'model_type', 'cnn'),
        output_dir=output_dir,
        logger=logger
    )
    
    # Run the simulation
    metrics, cpu_usages, round_times, total_training_time_per_round, total_aggregation_time_per_round, total_execution_time = simulator.run_simulation()
    
    # After all rounds, compute training and aggregation times per round
    rounds_range = range(1, simulation_args.rounds + 1)
    
    # Log total training and aggregation times per round
    for rnd, (train_time, agg_time) in enumerate(zip(total_training_time_per_round, total_aggregation_time_per_round), start=1):
        logger.info(f"Round {rnd}: Total Training Time = {train_time:.2f} seconds")
        logger.info(f"Round {rnd}: Total Aggregation Time = {agg_time:.2f} seconds")
    
    # Plot the metrics with convergence and execution time annotations
    plot_metrics_with_convergence(metrics, rounds_range, total_execution_time, output_dir, logger)
    plot_loss_line(metrics, rounds_range, output_dir, logger)
    plot_training_aggregation_times(rounds_range, total_training_time_per_round, total_aggregation_time_per_round, total_execution_time, output_dir, logger)
    plot_additional_metrics(rounds_range, cpu_usages, round_times, output_dir, logger)
    
    # Calculate and log detailed statistics
    if cpu_usages and round_times:
        avg_cpu_usage = np.mean(cpu_usages)
        avg_round_time = np.mean(round_times)
        logger.info(f"\nAverage CPU Usage per Round: {avg_cpu_usage:.2f}%")
        logger.info(f"Average Time Taken per Round: {avg_round_time:.2f} seconds")
        
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
        logger.info("\nAverage Evaluation Metrics over all nodes and rounds:")
        logger.info(f"Average Test Loss: {avg_test_loss:.4f}")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average F1 Score: {avg_f1_score:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Recall: {avg_recall:.4f}")
        
        # Log the averaged training metrics
        logger.info("\nAverage Training Metrics over all nodes and rounds:")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
        logger.info(f"Average Training F1 Score: {avg_train_f1_score:.4f}")
        logger.info(f"Average Training Precision: {avg_train_precision:.4f}")
        logger.info(f"Average Training Recall: {avg_train_recall:.4f}")
        
        logger.info("\nSimulation complete. Plots have been saved as PDF files.")
        print(Fore.CYAN + "\nSimulation completed successfully. Check the output directory for results.")
    else:
        logger.info("\nNo metrics recorded to compute averages.")
        print(Fore.YELLOW + "\nSimulation completed, but no metrics were recorded.")


def list_simulations(config_file):
    """
    List all available simulation configurations in the config file.
    
    Args:
        config_file: Path to the config file
    """
    try:
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