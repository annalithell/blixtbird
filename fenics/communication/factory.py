# fenics/communication/factory.py

from typing import Dict, Type, Optional, Any
import logging

from fenics.communication.base import CommunicationProtocol
from fenics.communication.gossip import gossip_step, gossip_exchange
from fenics.communication.neighboring import neighboring_step, neighboring_exchange

# Create protocol classes
class GossipProtocol(CommunicationProtocol):
    def exchange(self, nodes, G, local_models, executor):
        return gossip_step(nodes, G, local_models, executor)

class NeighboringProtocol(CommunicationProtocol):
    def exchange(self, nodes, G, local_models, executor):
        return neighboring_step(nodes, G, local_models, executor)

class ProtocolFactory:
    """
    Factory class for creating communication protocol instances.
    This makes it easy for users to select different protocols.
    """
    
    # Registry of available protocols
    _protocols: Dict[str, Type[CommunicationProtocol]] = {
        'gossip': GossipProtocol,
        'neighboring': NeighboringProtocol,
    }
    
    @classmethod
    def register_protocol(cls, name: str, protocol_class: Type[CommunicationProtocol]) -> None:
        """
        Register a new communication protocol.
        
        Args:
            name: Name of the protocol
            protocol_class: Protocol class
        """
        cls._protocols[name] = protocol_class
    
    @classmethod
    def get_protocol(cls, protocol_name: str, logger: Optional[logging.Logger] = None, **kwargs) -> CommunicationProtocol:
        """
        Get a protocol instance by name.
        
        Args:
            protocol_name: Name of the protocol
            logger: Logger instance
            **kwargs: Additional arguments to pass to the protocol constructor
            
        Returns:
            Instance of the requested protocol
            
        Raises:
            ValueError: If the protocol name is not recognized
        """
        if protocol_name not in cls._protocols:
            available_protocols = list(cls._protocols.keys())
            raise ValueError(f"Unknown protocol: '{protocol_name}'. Available protocols: {available_protocols}")
        
        protocol_class = cls._protocols[protocol_name]
        return protocol_class(logger=logger, **kwargs)
    
    @classmethod
    def list_available_protocols(cls) -> Dict[str, Type[CommunicationProtocol]]:
        """
        List all available protocols.
        
        Returns:
            Dictionary mapping protocol names to protocol classes
        """
        return cls._protocols.copy()