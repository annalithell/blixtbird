# fenics/cli/commands.py

import os
import shlex
import logging
import numpy as np
import yaml
from colorama import Fore
import subprocess
import sys

from blixtbird.config import parse_arguments, SimulationConfig, load_config_from_file
from blixtbird.data import DataModule
from blixtbird.utils import setup_logging
from blixtbird.plotting import plot_metrics_with_convergence, plot_loss_line, plot_metrics_for_data_after_aggregation, plot_training_aggregation_times, plot_additional_metrics
from blixtbird.node.attacks.attack_registry import autodiscover_attack_modules
from blixtbird.local_data_manipulation.yaml_maker import create_yaml
from blixtbird.local_data_manipulation.csv_metric import load_csv

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

    #TODO Call atack register here?
    #autodiscover_attack_modules()

    # Set up data module
    data_module = DataModule(
        num_nodes=simulation_args.num_nodes,
        node_type_map=simulation_args.node_type_map, 
        alpha=simulation_args.alpha,
        topology=simulation_args.topology,
        topology_file=simulation_args.topology_file,
        output_dir=output_dir,
        logger=logger
    )
    
    data_module.setup()
    
    # Create YAML configuration for MPI simulation
    create_yaml(
        G = data_module.G,
        node_type_map=simulation_args.node_type_map,
        output_dir = output_dir,
        epochs = simulation_args.epochs,
        rounds = simulation_args.rounds, 
        model = simulation_args.model_type
        )

    #Some python buffer variable that helps print stdout in realtime
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    command = [
        'mpiexec',
        '-n',
        str(simulation_args.num_nodes),
        'python',
        '-m',
        'fenics.mpi_script'
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    stdout_lines = []
    stderr_lines = []

    print(Fore.GREEN + f"\n--- Standard Output (stdout): Starting MPI---")
    
    #Print stdout in real time
    while process.poll() is None or process.stdout.readable():
        line = process.stdout.readline()
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            stdout_lines.append(line)
        if not line and process.poll() is not None:
            break
    
    process.wait()

    stderr_output = process.stderr.read()
    if stderr_output:
        stderr_lines.append(stderr_output)

    print(Fore.GREEN,"\n--- End of MPI part ---")
    print(f"Exit Code: {process.returncode}")

    full_stderr = "".join(stderr_lines).strip()
    if full_stderr:
        print(Fore.RED, "\n--- Standard Errors (stderr): ---")
        print(Fore.RED, full_stderr)
    
    #TODO For now all metrics are comment!
    # After all rounds, compute training and aggregation times per round
    rounds_range = range(1, simulation_args.rounds + 1)
    
    """
    Local metrics plotting! - This is moment where MPI scripts aren't working anymore.
    """

    metrics = {}
    total_execution_time = 2000 #TODO change to real time

    for node_id, _ in (simulation_args.node_type_map).items():
        metric = load_csv(node_id)
        metrics[node_id] = metric

        plot_metrics_with_convergence(metric, rounds_range, total_execution_time, output_dir, logger, False, node_id)
        plot_metrics_for_data_after_aggregation(metric, rounds_range, output_dir, node_id)
        plot_loss_line(metric, rounds_range, output_dir, logger, False, node_id)
    
    # # Log total training and aggregation times per round
    # for rnd, (train_time, agg_time) in enumerate(zip(total_training_time_per_round, total_aggregation_time_per_round), start=1):
    #     logger.info(f"Round {rnd}: Total Training Time = {train_time:.2f} seconds")
    #     logger.info(f"Round {rnd}: Total Aggregation Time = {agg_time:.2f} seconds")
    
    # Plot the metrics with convergence and execution time annotations
    all_nodes = True
    node_id = None

    plot_metrics_with_convergence(metrics, rounds_range, total_execution_time, output_dir, logger, all_nodes, node_id)
    plot_loss_line(metrics, rounds_range, output_dir, logger, all_nodes, node_id)


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