# fenics/__init__.py

__version__ = "1.3.2"
__author__ = ""
__email__ = ""

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
from fenics.attack import AttackManager, Attack, PoisonAttack, DelayAttack, AttackFactory
from fenics.plotting import visualize_data_distribution, plot_metrics_with_convergence, plot_loss_line, plot_training_aggregation_times, plot_additional_metrics

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
]

# Initialize default logging if not already set up
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
