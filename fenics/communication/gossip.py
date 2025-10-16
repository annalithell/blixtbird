# fenics/communication/gossip.py

import random
import logging
from concurrent.futures import as_completed


def gossip_exchange(node_a, node_b, local_models):
    """
    Exchange and average the models between two nodes.
    
    Args:
        node_a: First node ID
        node_b: Second node ID
        local_models: Dictionary mapping node IDs to models
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
    
    Args:
        nodes: List of node IDs
        G: Network graph
        local_models: Dictionary mapping node IDs to models
        executor: ThreadPoolExecutor for parallel execution
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