# fenics/communication/base.py

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional


class CommunicationProtocol(ABC):
    """
    Base class for all communication protocols.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the communication protocol.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger()
    
    @abstractmethod
    def exchange(self, nodes: List[int], G: Any, local_models: Dict[int, Any], executor: Any) -> None:
        """
        Perform model exchange between nodes according to the protocol.
        
        Args:
            nodes: List of node IDs
            G: Network graph
            local_models: Dictionary mapping node IDs to models
            executor: Executor for parallel execution
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of the protocol.
        
        Returns:
            Name of the protocol
        """
        return self.__class__.__name__.replace('Protocol', '')