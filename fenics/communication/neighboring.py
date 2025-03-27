# fenics/communication/neighboring.py

import logging
from concurrent.futures import as_completed


def neighboring_exchange(node_a, node_b, local_models):
    """
    Exchange and average the models between two nodes (all neighbors).
    
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


def neighboring_step(nodes, G, local_models, executor):
    """
    Perform one neighboring iteration where each node exchanges models with all its neighbors
    and averages their parameters.
    
    Args:
        nodes: List of node IDs
        G: Network graph
        local_models: Dictionary mapping node IDs to models
        executor: ThreadPoolExecutor for parallel execution
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