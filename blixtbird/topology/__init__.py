# fenics/topology/__init__.py

from blixtbird.topology.base import create_nodes
from blixtbird.topology.builder import build_topology
from blixtbird.topology.visualization import visualize_and_save_topology
from blixtbird.topology.factory import TopologyFactory
from blixtbird.topology.types import (
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