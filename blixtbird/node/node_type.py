from enum import Enum, auto

class NodeType(Enum):
    """ Enumeration for different types of nodes. """
    NORMAL = auto()
    ATTACK = auto()
    MITIGATION = auto()

