# fenics/__init__.py

__version__ = "2.0.1"
__author__ = "Shubham Saha, Sifat Nawrin Nova"
__email__ = "shuvsaha7@gmail.com, nawrinnova04@gmail.com"

# Import key components from modules
from fenics.cli import FenicsShell, run_fenics_shell
from fenics.config import parse_arguments, SimulationConfig, load_config_from_file
# from fenics.models import Net, ModelBase, MLP, ModelFactory
from fenics.models import Net, ModelBase, ModelFactory
from fenics.data import load_datasets_dirichlet, print_class_distribution, DataModule
from fenics.topology import create_nodes, build_topology, visualize_and_save_topology, TopologyFactory
from fenics.training import local_train, evaluate, summarize_model_parameters
from fenics.communication import send_update, gossip_step, neighboring_step, CommunicationProtocol, ProtocolFactory
from fenics.aggregation import FedAvgStrategy, AggregationStrategy, AggregationFactory
# from fenics.client_selection import ClientSelector, select_clients_uniform, select_clients_md_sampling, SelectionFactory
from fenics.client_selection import ClientSelector, select_clients_md_sampling, SelectionFactory
from fenics.utils import setup_logging, calculate_selection_probabilities, detect_convergence
from fenics.simulator import Simulator
from fenics.attack import AttackManager, AttackFactory, Attack #, PoisonAttack, DelayAttack
from fenics.node.base import BaseNode
from fenics.node.attacks.delay import DelayAttack
from fenics.node.attacks.poison import PoisonAttack
from fenics.node.attacks.freerider import FreeRiderAttack
from fenics.plotting import visualize_data_distribution, plot_metrics_with_convergence, plot_loss_line, plot_training_aggregation_times, plot_additional_metrics
from fenics.simulator_mpi import Simulator_MPI
from fenics.local_data_manipulation.csv_metric import make_csv, make_pandas_df
from fenics.local_data_manipulation.yaml_maker import create_yaml, get_node_data, get_neighbors, get_output_dir

__all__ = [
    # CLI
    'FenicsShell',
    'run_fenics_shell',

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
    
    # Client Selection
    'ClientSelector',
    # 'select_clients_uniform',
    'select_clients_md_sampling',
    'SelectionFactory',

    # Utils
    'setup_logging',
    'calculate_selection_probabilities',
    'detect_convergence',

    # Simulation
    'Simulator',
    'Simulator_MPI',

    # Attack
    'AttackManager',
    'Attack',
    'PoisonAttack',
    'DelayAttack',
    'AttackFactory',

    # Plotting
    'visualize_data_distribution',
    'plot_metrics_with_convergence',
    'plot_loss_line',
    'plot_training_aggregation_times',
    'plot_additional_metrics',

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
