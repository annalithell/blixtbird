# Fenics Network Topologies

**OBSERVE: This functionality has been inherited from Fenics and not been thoroughly tested for Blixtbird**

**TODO: Integrate this into the Blixtbird framework**

This directory contains the network topology implementations for the Fenics simulator.

## Available Topologies

1. **Fully Connected (fully_connected)**: Complete graph where every node is connected to every other node.
2. **Ring (ring)**: Nodes arranged in a single cycle.
3. **Random (random)**: Random graph with probabilistic edge creation.
4. **Custom (custom)**: Custom topology loaded from an edge list file.

## How to Use Different Topologies

You can specify which topology to use by setting the `topology` parameter when running Fenics:

```bash
# Command line
python fenics.py setup --topology ring
python fenics.py run
```

Or in your YAML configuration file:

```yaml
simulations:
  my_simulation:
    # other parameters...
    topology: ring
```

## Creating Custom Topologies

### Method 1: Using an Edge List File (Recommended)

The simplest way to create a custom topology is to define your network structure in an edge list file:

1. **Create an Edge List File**:
   Create a file (e.g., `topology.edgelist`) where each line represents an edge between two nodes:

   ```
   0 1
   1 2
   2 3
   3 0
   ```

   This represents a network where node 0 connects to 1, node 1 connects to 2, etc.

2. **Use the Custom Topology Type**:
   ```bash
   python fenics.py setup --topology custom --topology_file topology.edgelist
   python fenics.py run
   ```

   Or in your configuration file:
   ```yaml
   simulations:
     my_simulation:
       # other parameters...
       topology: custom
       topology_file: topology.edgelist
   ```

This approach doesn't require any coding and is sufficient for most custom network structures.

### Method 2: Programmatically Defined Topologies (Advanced)

For more complex topologies that can't be easily defined in an edge list, you can create a custom topology class:

1. Create a new Python file with your topology class that inherits from `TopologyBase`
2. Register your topology with `TopologyFactory`
3. Import your custom topology file before running Fenics

See the `examples/custom_topology_example.py` file for a complete example of how to create and register a programmatically defined topology.

#### Example Code

```python
# my_topology.py
from fenics.topology import TopologyBase, TopologyFactory
import networkx as nx

class StarTopology(TopologyBase):
    def __init__(self, num_nodes, center_node=0):
        super().__init__(num_nodes)
        self.center_node = center_node
    
    def build(self):
        # Create a star topology with the specified center node
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        
        # Connect all nodes to the center node
        for node in range(self.num_nodes):
            if node != self.center_node:
                G.add_edge(self.center_node, node)
        
        return G

# Register the topology
TopologyFactory.register_topology('star', StarTopology)
```

Then import your file before running Fenics:

```python
import my_topology  # This registers your topology
```

And use it:

```bash
python fenics.py setup --topology star
python fenics.py run
```

## The TopologyFactory

The `TopologyFactory` class in `factory.py` is responsible for managing the available topologies. It provides three main methods:

1. `register_topology(name, topology_class)`: Register a new topology
2. `build_topology(topology_type, num_nodes, topology_file=None, **kwargs)`: Build a topology of the specified type
3. `list_available_topologies()`: Get a dictionary of all registered topologies

You can use these methods to dynamically register and build topologies at runtime.