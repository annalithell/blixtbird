# fenics/client_selection/examples/custom_strategy_example.py
# 
# This file shows how to create and register a custom client selection strategy with Fenics.
# You can use this as a template for creating your own selection strategies.

import numpy as np
import logging
from typing import List, Optional

# Import the necessary factory from Fenics
from fenics.client_selection.factory import SelectionFactory


def select_clients_round_robin(nodes: List[int], 
                              num_participants: int, 
                              current_round: int, 
                              logger: Optional[logging.Logger] = None) -> List[int]:
    """
    Example of a custom client selection strategy for Fenics.
    This implements round-robin selection of clients, cycling through all nodes.
    
    Args:
        nodes: List of all node IDs
        num_participants: Number of nodes to select
        current_round: Current round number (used for cycling)
        logger: Logger instance
        
    Returns:
        List of selected node IDs
    """
    logger = logger or logging.getLogger()
    
    # Calculate starting index for this round
    start_idx = (current_round * num_participants) % len(nodes)
    
    # Select nodes in a round-robin fashion
    selected_nodes = []
    for i in range(num_participants):
        idx = (start_idx + i) % len(nodes)
        selected_nodes.append(nodes[idx])
    
    logger.info(f"Round {current_round}: Selected nodes using round-robin strategy: {selected_nodes}")
    return selected_nodes


# Function to extend ClientSelector with the custom strategy
def extend_client_selector_with_round_robin(client_selector):
    """
    Extend a ClientSelector instance with the round-robin selection strategy.
    
    Args:
        client_selector: ClientSelector instance to extend
    """
    def select_round_robin(self, current_round):
        return select_clients_round_robin(
            self.nodes, 
            self.num_participants, 
            current_round, 
            self.logger
        )
    
    # Add the method to the ClientSelector instance
    import types
    client_selector.select_clients_round_robin = types.MethodType(select_round_robin, client_selector)


# Register the strategy with the factory
# This makes it available to use via the factory
SelectionFactory.register_strategy('round_robin', select_clients_round_robin)


# =======================================================================
# How to use this custom selection strategy:
# =======================================================================
#
# 1. Place this file in your project directory
#
# 2. Import this module before using the strategy:
#    ```
#    # In your script or notebook
#    import custom_strategy_example
#    ```
#
# 3. Use the strategy directly:
#    ```python
#    from fenics.client_selection.factory import SelectionFactory
#    
#    # Get the round-robin strategy
#    round_robin_strategy = SelectionFactory.get_strategy('round_robin')
#    
#    # Use it to select nodes
#    selected_nodes = round_robin_strategy(
#        nodes=list(range(10)),
#        num_participants=3,
#        current_round=2
#    )
#    ```
#
# 4. Or extend an existing ClientSelector instance:
#    ```python
#    from fenics.client_selection import ClientSelector
#    import custom_strategy_example
#    
#    # Create a client selector
#    selector = ClientSelector(nodes=list(range(10)), participation_rate=0.3)
#    
#    # Extend it with the round-robin strategy
#    custom_strategy_example.extend_client_selector_with_round_robin(selector)
#    
#    # Now you can use the new method
#    selected_nodes = selector.select_clients_round_robin(current_round=2)
#    ```
#
# 5. If you want to precompute nodes with your custom strategy:
#    ```python
#    # Precompute participating nodes for all rounds
#    participating_nodes_per_round = []
#    for rnd in range(num_rounds):
#        selected_nodes = selector.select_clients_round_robin(current_round=rnd)
#        participating_nodes_per_round.append(selected_nodes)
#    ```