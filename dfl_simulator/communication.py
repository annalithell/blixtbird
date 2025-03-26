# communication.py

import random
import time
import logging
from concurrent.futures import as_completed

def send_update(node_id, attacker_type):
    start_time = time.time()
    if attacker_type == 'delay':
        delay_duration = random.uniform(500, 700)
        logging.info(f"[node_{node_id}] Delaying sending updates by {delay_duration:.2f} seconds.")
        time.sleep(delay_duration)
    # Simulate sending update (no actual operation)
    end_time = time.time()
    sending_time = end_time - start_time
    return sending_time

def gossip_exchange(node_a, node_b, local_models):
    """
    Exchange and average the models between two nodes.
    """
    model_a = local_models[node_a]
    model_b = local_models[node_b]
    
    # Average the parameters
    for param_key in model_a.state_dict().keys():
        param_a = model_a.state_dict()[param_key]
        param_b = model_b.state_dict()[param_key]
        averaged_param = (param_a + param_b) / 2.0
        model_a.state_dict()[param_key].copy_(averaged_param)
        model_b.state_dict()[param_key].copy_(averaged_param)

def gossip_step(nodes, G, local_models, executor):
    """
    Perform one gossip iteration where each node exchanges models with a randomly selected neighbor
    and averages their parameters.
    """
    gossip_futures = {}
    
    for node in nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            selected_neighbor = random.choice(neighbors)
            # Ensure each pair is processed only once
            if (selected_neighbor, node) in gossip_futures:
                continue
            # Submit gossip exchange task
            future = executor.submit(gossip_exchange, node, selected_neighbor, local_models)
            gossip_futures[future] = (node, selected_neighbor)
    
    # Wait for all gossip exchanges to complete
    for future in as_completed(gossip_futures):
        node_a, node_b = gossip_futures[future]
        try:
            future.result()
            logging.info(f"Gossip exchange completed between node_{node_a} and node_{node_b}.")
        except Exception as exc:
            logging.error(f"Gossip exchange between node_{node_a} and node_{node_b} generated an exception: {exc}")

def neighboring_exchange(node_a, node_b, local_models):
    """
    Exchange and average the models between two nodes (all neighbors).
    """
    model_a = local_models[node_a]
    model_b = local_models[node_b]
    
    # Average the parameters
    for param_key in model_a.state_dict().keys():
        param_a = model_a.state_dict()[param_key]
        param_b = model_b.state_dict()[param_key]
        averaged_param = (param_a + param_b) / 2.0
        model_a.state_dict()[param_key].copy_(averaged_param)
        model_b.state_dict()[param_key].copy_(averaged_param)

def neighboring_step(nodes, G, local_models, executor):
    """
    Perform one neighboring iteration where each node exchanges models with all its neighbors
    and averages their parameters.
    """
    neighboring_futures = {}
    
    for node in nodes:
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            # To prevent duplicate exchanges
            if node < neighbor:
                future = executor.submit(neighboring_exchange, node, neighbor, local_models)
                neighboring_futures[future] = (node, neighbor)
    
    # Wait for all neighboring exchanges to complete
    for future in as_completed(neighboring_futures):
        node_a, node_b = neighboring_futures[future]
        try:
            future.result()
            logging.info(f"Neighboring exchange completed between node_{node_a} and node_{node_b}.")
        except Exception as exc:
            logging.error(f"Neighboring exchange between node_{node_a} and node_{node_b} generated an exception: {exc}")
