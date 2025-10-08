from enum import Enum, auto

class AttackType(Enum):
    """ Enumeration for different types of attacks. """
    DELAY     = auto()
    FREERIDER = auto()
    POISON    = auto()
    # Add more attack types ?