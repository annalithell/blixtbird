from enum import Enum, auto

class NodeType(Enum):
    """ Enumeration for different types of nodes. """
    NORMAL = auto()
    ATTACK = auto()
    MITIGATION = auto()
  # COORDINATION = auto() DO WE NEED ONE SO FAR ? 
  # TO FUNCTION AS THE MAIN POROCESS KEEOPING TRACK OF THE PROCESSES?
