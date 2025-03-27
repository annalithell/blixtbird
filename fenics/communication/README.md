# Fenics Communication Module

This directory contains the communication functionality for the Fenics simulator.

## Components

1. **sender.py**: Contains functions for sending updates, including handling of delay attacks.
2. **gossip.py**: Contains functions for gossip-based communication protocol.
3. **neighboring.py**: Contains functions for neighbor-based communication protocol.
4. **base.py**: Contains the base class for communication protocols.
5. **factory.py**: Contains the factory for creating protocol instances.

## Available Protocols

1. **Gossip Protocol (gossip)**: Each node exchanges information with a randomly selected neighbor.
2. **Neighboring Protocol (neighboring)**: Each node exchanges information with all of its neighbors.

## Using Different Protocols

You can specify which protocol to use by setting the `--protocol` parameter when running Fenics:

```bash
# Command line
python fenics.py setup --protocol gossip
python fenics.py run
```

Or in your configuration file:

```yaml
simulations:
  my_simulation:
    # other parameters...
    protocol: neighboring
```

## Creating Custom Protocols

To add your own custom communication protocol:

1. Create a new Python file with your protocol class that inherits from `CommunicationProtocol`
2. Implement the required methods, especially `exchange()`
3. Register your protocol with `ProtocolFactory`
4. Import your custom protocol file before running Fenics

### Example Code

```python
# my_protocol.py
from fenics.communication.base import CommunicationProtocol
from fenics.communication.factory import ProtocolFactory
import random

class BroadcastProtocol(CommunicationProtocol):
    """
    Broadcast protocol where a randomly selected node broadcasts its model to all others.
    """
    
    def exchange(self, nodes, G, local_models, executor):
        # Select a random node as broadcaster
        broadcaster = random.choice(nodes)
        self.logger.info(f"Node {broadcaster} selected as broadcaster")
        
        broadcast_model = local_models[broadcaster]
        broadcast_state = broadcast_model.state_dict()
        
        # Broadcast to all other nodes
        for node in nodes:
            if node != broadcaster:
                model = local_models[node]
                model.load_state_dict(broadcast_state)
                self.logger.info(f"Node {node} received broadcast from node {broadcaster}")

# Register the protocol
ProtocolFactory.register_protocol('broadcast', BroadcastProtocol)
```

Then import your file before running Fenics:

```python
import my_protocol  # This registers your protocol
```

And use it:

```bash
python fenics.py setup --protocol broadcast
python fenics.py run
```

See the `examples/custom_protocol_example.py` file for a more complete example.

## The ProtocolFactory

The `ProtocolFactory` class in `factory.py` is responsible for managing the available protocols. It provides methods for:

1. Registering new protocol types
2. Creating protocol instances by name
3. Listing all available protocols

This allows for easy extensibility and runtime selection of communication protocols.