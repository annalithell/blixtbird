# fenics/topology/__init__.py

from fenics.topology.base import create_nodes
from fenics.topology.builder import build_topology
from fenics.topology.visualization import visualize_and_save_topology
from fenics.topology.factory import TopologyFactory
from fenics.topology.types import (
    FullyConnectedTopology,
    RingTopology,
    RandomTopology,
    CustomTopology
)

__all__ = [
    'create_nodes',
    'build_topology',
    'visualize_and_save_topology',
    'TopologyFactory',
    'FullyConnectedTopology',
    'RingTopology',
    'RandomTopology',
    'CustomTopology'
]