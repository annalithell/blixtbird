# fenics/communication/sender.py

import time
import logging
from fenics.attack.attack_factory import AttackFactory


def send_update(node_id, attacker_type):
    """
    Send an update from a node, using attack if specified.
    
    Args:
        node_id: ID of the sender node
        attacker_type: Type of attack to simulate (if any)
        
    Returns:
        Time taken to send the update
    """
    start_time = time.time()
    
    # If an attack type is specified, use the AttackFactory to create and execute it
    if attacker_type:
        # Create the attack using the factory
        attack = AttackFactory.get_attack(attacker_type, node_id=node_id)
        # Execute the attack
        attack.execute()
    
    # Simulate sending update (no actual operation)
    end_time = time.time()
    sending_time = end_time - start_time
    
    return sending_time