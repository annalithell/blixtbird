# blixtbird/__init__.py

__version__ = "1.0.1"
__author__ = "Gabriel Bengtsson, Zaid Haj-Ibrhaim, Piotr Krzyczkowski and Anna Lithell"
__email__ = "anna.lithell@gustas.se"

# Import key components from modules
from blixtbird.cli import BlixtbirdShell, run_blixtbird_shell
from blixtbird.config import parse_arguments, SimulationConfig, load_config_from_file
from blixtbird.models import Net, ModelBase, ModelFactory
from blixtbird.data import load_datasets_dirichlet, print_class_distribution, DataModule
from blixtbird.topology import create_nodes, build_topology, visualize_and_save_topology, TopologyFactory
from blixtbird.training import local_train, evaluate, summarize_model_parameters
from blixtbird.utils import setup_logging, calculate_selection_probabilities, detect_convergence
from blixtbird.node.abstract import AbstractNode
from blixtbird.node.normal_node import NormalNode
from blixtbird.node.attacks.delay import DelayAttack
from blixtbird.node.attacks.poison import PoisonAttack
from blixtbird.node.attacks.freerider import FreeRiderAttack
from blixtbird.plotting import visualize_data_distribution, plot_metrics_with_convergence, plot_loss_line, plot_training_aggregation_times, plot_additional_metrics, plot_metrics_for_data_after_aggregation
from blixtbird.simulator_mpi import Simulator_MPI
from blixtbird.local_data_manipulation.csv_metric import make_csv, make_pandas_df
from blixtbird.local_data_manipulation.yaml_maker import create_yaml, get_node_data, get_neighbors, get_output_dir

__all__ = [
    # CLI
    'BlixtbirdShell',
    'run_blixtbird_shell',

    # Config
    'parse_arguments',
    'SimulationConfig',
    'load_config_from_file',
    'parse_arguments_pydantic',
    
    # Models
    'Net',
    'ModelBase',
    # 'MLP',
    'ModelFactory',

    # Data
    'load_datasets_dirichlet',
    'print_class_distribution',
    'DataModule',

    # Topology
    'create_nodes',
    'build_topology',
    'visualize_and_save_topology',
    'TopologyFactory',

    # Training
    'local_train',
    'evaluate',
    'summarize_model_parameters',

    # Communication
    'send_update',
    'gossip_step',
    'neighboring_step',
    'CommunicationProtocol',
    'ProtocolFactory',

    # Aggregation
    'FedAvgStrategy',
    'AggregationStrategy',
    'AggregationFactory',


    # Utils
    'setup_logging',
    'calculate_selection_probabilities',
    'detect_convergence',

    # Simulation
    'Simulator',
    'Simulator_MPI',

    # Plotting
    'visualize_data_distribution',
    'plot_metrics_with_convergence',
    'plot_loss_line',
    'plot_training_aggregation_times',
    'plot_additional_metrics',
    'plot_metrics_for_data_after_aggregation',

    # Local Data Functions
    'make_csv',
    'make_pandas_df',
    'create_yaml',
    'get_node_data',
    'get_neighbors',
    'get_output_dir'
]

# Initialize default logging if not already set up
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
