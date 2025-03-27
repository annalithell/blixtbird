# fenics/communication/examples/custom_protocol_example.py
# 
# This file shows how to create and register a custom communication protocol with Fenics.
# You can use this as a template for creating your own protocols.

import random
from concurrent.futures import as_completed

# Import the necessary base classes from Fenics
from fenics.communication import CommunicationProtocol, ProtocolFactory


class RingProtocol(CommunicationProtocol):
    """
    Example of a custom protocol for Fenics.
    This creates a ring communication pattern where each node only communicates with
    its successor in a virtual ring.
    """
    
    def exchange(self, nodes, G, local_models, executor):
        """
        Perform model exchange in a ring pattern.
        
        Args:
            nodes: List of node IDs
            G: Network graph
            local_models: Dictionary mapping node IDs to models
            executor: Executor for parallel execution
        """
        self.logger.info("Starting Ring Protocol exchange")
        
        # Sort nodes to create a deterministic ring
        sorted_nodes = sorted(nodes)
        num_nodes = len(sorted_nodes)
        
        if num_nodes < 2:
            self.logger.info("Not enough nodes for ring communication")
            return
        
        # Create pairs: each node with its successor in the ring
        pairs = [(sorted_nodes[i], sorted_nodes[(i + 1) % num_nodes]) 
                 for i in range(num_nodes)]
        
        # Submit exchange tasks
        exchange_futures = {}
        for node_a, node_b in pairs:
            future = executor.submit(self._exchange_pair, node_a, node_b, local_models)
            exchange_futures[future] = (node_a, node_b)
        
        # Wait for all exchanges to complete
        for future in as_completed(exchange_futures):
            node_a, node_b = exchange_futures[future]
            try:
                future.result()
                self.logger.info(f"Ring exchange completed between node_{node_a} and node_{node_b}.")
            except Exception as exc:
                self.logger.error(f"Ring exchange between node_{node_a} and node_{node_b} generated an exception: {exc}")
    
    def _exchange_pair(self, node_a, node_b, local_models):
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


# Register the protocol with the factory
# This makes it available to use with --protocol ring_comm
ProtocolFactory.register_protocol('ring_comm', RingProtocol)


# =======================================================================
# How to use this custom protocol:
# =======================================================================
#
# 1. Place this file in your project directory
#
# 2. Import this module before running Fenics:
#    ```
#    # In your script or notebook
#    import custom_protocol_example
#    ```
#
# 3. Run Fenics with the protocol parameter:
#    ```
#    # Command line
#    python fenics.py setup --protocol ring_comm
#    python fenics.py run
#    ```
#
#    or in the configuration file (config.yaml):
#    ```yaml
#    simulations:
#      my_simulation:
#        # other parameters...
#        protocol: ring_comm
#    ```
#
# 4. For multiple custom protocols, you can create additional files
#    and import them all before running Fenics.