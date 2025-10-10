from enum import Enum, auto


## OBS: THIS IS REMOVED IN THE FIXED ATTACK REGISTRY APPROACH

class AttackType(Enum):
    """ Enumeration for different types of attacks. """
    NONE      = auto()
    DELAY     = auto()
    FREERIDER = auto()
    POISON    = auto()
    # Add more attack types ?