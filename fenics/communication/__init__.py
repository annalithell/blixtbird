# fenics/communication/__init__.py

from fenics.communication.sender import send_update
from fenics.communication.gossip import gossip_step, gossip_exchange
from fenics.communication.neighboring import neighboring_step, neighboring_exchange
from fenics.communication.factory import ProtocolFactory
from fenics.communication.base import CommunicationProtocol

__all__ = [
    'send_update',
    'CommunicationProtocol',
    'ProtocolFactory',
    'gossip_step',
    'gossip_exchange',
    'neighboring_step',
    'neighboring_exchange'
]