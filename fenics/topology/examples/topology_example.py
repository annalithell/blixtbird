# topology_example.py
# 
# This file shows how to create and register a custom topology with Fenics.
# You can use this as a template for creating your own topologies.

import networkx as nx

# Import the necessary base classes from Fenics
from fenics.topology import TopologyBase, TopologyFactory


class GridTopology(TopologyBase):
    """
    Example of a custom topology for Fenics.
    This creates a 2D grid network with customizable dimensions.
    """
    
    def __init__(self, num_nodes, grid_width=None, grid_height=None):
        """
        Initialize the grid topology.
        
        Args:
            num_nodes: Total number of nodes
            grid_width: Width of the grid (number of columns)
            grid_height: Height of the grid (number of rows)
                         If not specified, will try to create a square grid
        """
        super().__init__(num_nodes)
        
        # Calculate grid dimensions if not provided
        if grid_width is None and grid_height is None:
            # Try to make a square grid
            import math
            grid_width = int(math.sqrt(num_nodes))
            grid_height = (num_nodes + grid_width - 1) // grid_width  # Ceiling division
        elif grid_width is None:
            grid_width = (num_nodes + grid_height - 1) // grid_height
        elif grid_height is None:
            grid_height = (num_nodes + grid_width - 1) // grid_width
            
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def build(self):
        """
        Build a 2D grid network.
        
        Returns:
            NetworkX graph representing a 2D grid
        """
        # Create a grid graph with the calculated dimensions
        width = min(self.grid_width, self.num_nodes)
        height = min(self.grid_height, (self.num_nodes + width - 1) // width)
        
        # Create a 2D grid
        G = nx.grid_2d_graph(height, width)
        
        # Relabel nodes to be integers from 0 to num_nodes-1
        mapping = {(i, j): i * width + j for i in range(height) for j in range(width)}
        G = nx.relabel_nodes(G, mapping)
        
        # Ensure we have exactly num_nodes
        if G.number_of_nodes() > self.num_nodes:
            # Remove extra nodes
            nodes_to_remove = list(range(self.num_nodes, G.number_of_nodes()))
            G.remove_nodes_from(nodes_to_remove)
        
        return G


# Register the topology with the factory
# This makes it available to use with --topology grid
TopologyFactory.register_topology('grid', GridTopology)


# =======================================================================
# How to use this custom topology:
# =======================================================================
#
# 1. Place this file in your project directory
#
# 2. Import this module before running Fenics:
#    ```
#    # In your script or notebook
#    import custom_topology_example
#    ```
#
# 3. Run Fenics with the topology parameter:
#    ```
#    # Command line
#    python fenics.py setup --topology grid
#    python fenics.py run
#    ```
#
#    or in the configuration file (config.yaml):
#    ```yaml
#    simulations:
#      my_simulation:
#        # other parameters...
#        topology: grid
#    ```
#
# 4. To pass parameters to your topology:
#    ```python
#    from fenics.topology import TopologyFactory
#    
#    # This creates a grid with 5 columns
#    G = TopologyFactory.build_topology('grid', num_nodes=20, grid_width=5)
#    ```
#
# 5. For multiple custom topologies, you can create additional files
#    and import them all before running Fenics.