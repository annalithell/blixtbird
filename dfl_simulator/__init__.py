# dfl_simulator/__init__.py

__version__ = "1.3.2"
__author__ = "Shubham Saha, Sifat Nawrin Nova"
__email__ = "shuvsaha7@gmail.com, nawrinnova04@gmail.com"

from dfl_simulator.main import PhoenixShell, run_phoenix_shell
from dfl_simulator.config import parse_arguments
from dfl_simulator.model import Net
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

__all__ = [
    'PhoenixShell',
    'run_phoenix_shell',
    'parse_arguments',
    'Net',
    'load_datasets_dirichlet',
    'print_class_distribution',
    'create_nodes',
    'build_topology',
    'visualize_and_save_topology',
    'local_train',
    'evaluate',
    'summarize_model_parameters',
    'send_update',
    'gossip_step',
    'neighboring_step',
    'setup_logging',
    'calculate_selection_probabilities',
    'visualize_data_distribution',
    'plot_metrics_with_convergence',
    'plot_loss_line',
    'plot_training_aggregation_times',
    'plot_additional_metrics'
]

# Initialize default logging if not already set up
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
    #logging.info("Default logging initialized in dfl_simulator package.")
