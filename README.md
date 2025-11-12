# Blixtbird: A Simulation Framework for Modeling Attacks in DFL networks

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-purple.svg)
![MPI4PY](https://img.shields.io/badge/MPI4PY-Required-brightgreen.svg)
![Threading](https://img.shields.io/badge/Concurrent-Multithreading-blueviolet.svg)
![Modular](https://img.shields.io/badge/Architecture-Modular-yellow.svg)


## Prerequisites

Before installing and using Blixtbird, ensure that your system meets the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.11 or higher
- **MPI**: An installed MPI distribution. This framework was built and tested using Microsoft MPI v10.0 [download link](https://www.microsoft.com/en-us/download/details.aspx?id=57467). 
- **Virtual Environment (Recommended):** It's advisable to use a virtual environment to manage dependencies

---

## Installation

### Installing via `pip`

1. **Clone the Repository:**
    
    ```bash
    git clone https://github.com/yourusername/blixtbird.git
    cd blixtbird
    ```
    
2. **Create and Activate a Virtual Environment (Optional but Recommended):**
    
    - **Windows:**
    
      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```
    
    - **Unix/Linux:**
    
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    
3. **Install Required Dependencies:**
    
    Ensure you have `pip` updated to the latest version:
    
    ```bash
    pip install --upgrade pip
    ```
    
4. **Install the Package in Editable Mode:**
    
    This allows you to modify the code without reinstalling the package each time.
    
    ```bash
    pip install -e .
    ```

    **Note:** The `setup.py` is configured to install all required dependencies.
    
5. **Verify Installation:**
    
    After installation, the `blixtbird` command-line tool should be available.
    
    ```bash
    blixtbird
    ```

    **Expected Output:**
    
    ```
    ______ _     _______   _____________ _________________ 
    | ___ \ |   |_   _\ \ / /_   _| ___ \_   _| ___ \  _  \
    | |_/ / |     | |  \ V /  | | | |_/ / | | | |_/ / | | |
    | ___ \ |     | |  /   \  | | | ___ \ | | |    /| | | |
    | |_/ / |_____| |_/ /^\ \ | | | |_/ /_| |_| |\ \| |/ /
    \____/\_____/\___/\/   \/ \_/ \____/ \___/\_| \_|___/



    Welcome to BLIXTBIRD Shell! Type 'help' to see available commands.
    ```


## Configuration

Blixtbird uses a `config.yaml` file to manage simulation parameters. Proper configuration ensures that simulations run as intended with the desired settings.
    
### Parameters in `config.yaml`
    
Below is an explanation of the primary parameters you can configure:
    
```yaml
simulations:
  simulation1:
    rounds: 10                  # Number of simulation rounds
    epochs: 10                  # Number of training epochs per simulation round
    num_nodes: 4                # Total number of nodes in the simulation
    node_type_map:              # Define a node class for each node id
      0: 'poison'
      1: 'base'
      2: 'freerider'
      3: 'base'
    use_attackers: true         # Flag to enable or disable attacker simulation 
    topology: fully_connected   # Network topology type
    topology_file: null         # Path to a custom topology edge list file
    max_attacks: 5              # Maximum number of attacks 
    gossip_steps: 3             # Number of gossiping steps during aggregation
    protocol: neighboring       # Aggregation protocol 
    alpha: 0.5                  # Dirichlet distribution parameter
    model_type: cnn             # Model type to use ('cnn')
```


## Usage
    
Blixtbird provides a command-line interface called **Blixtbird Shell** to interact with the simulator. Below are instructions on how to set up and run simulations.
    
### Using the Blixtbird Shell
    
1. **Launch the Fenics Shell:**
    
    ```bash
    blixtbird
    ```
    
    **Sample Output:**
    ```
    ______ _     _______   _____________ _________________ 
    | ___ \ |   |_   _\ \ / /_   _| ___ \_   _| ___ \  _  \
    | |_/ / |     | |  \ V /  | | | |_/ / | | | |_/ / | | |
    | ___ \ |     | |  /   \  | | | ___ \ | | |    /| | | |
    | |_/ / |_____| |_/ /^\ \ | | | |_/ /_| |_| |\ \| |/ / 
    \____/\_____/\___/\/   \/ \_/ \____/ \___/\_| \_|___/  


    Welcome to BLIXTBIRD Shell! Type 'help' to see available commands.

    BLIXTBIRD> 
    ```
    
2. **Setup the Simulation Environment:**
    
    Use the `setup` command to initialize simulation parameters defined in the configuration file.
    
      ```bash
      setup --config config.yaml --simulation_name simulation1
      ```
    
      **Explanation:**
      - **`--config`**: Specifies the path to the `config.yaml` file.
      - **`--simulation_name`**: Identifies which simulation parameters to load from the configuration file.
    
4. **Run the Simulation:**
    
    After setup, execute the simulation using the `run` command.
    
    ```bash
    run
    ```
    
    **Sample Output:**
    
    ```
    BLIXTBIRD> run
    Final Simulation Arguments:
    Rounds: 3
    Epochs: 1
    Num nodes: 3
    Node type map: {0: 'base', 1: 'base', 2: 'freerider'}
    Use attackers: True
    Max attacks: 5
    Participation rate: 0.6
    Topology: fully_connected
    Topology file: None
    Gossip steps: 3
    Protocol: neighboring
    Alpha: 0.5
    Model type: cnn
    Starting simulation...
    data saved

    --- Standard Output (stdout): Starting MPI---
    Node: 2 created node instance:freerider
    Node: 1 created node instance:base
    Node: 0 created node instance:base
    Node: 2 with negighbors:[0, 1], type: freerider, data_path: ./results/federated_data/node_2_train_data.pt and epochs: 1
    Node: 1 with negighbors:[0, 2], type: base, data_path: ./results/federated_data/node_1_train_data.pt and epochs: 1
    Node: 0 with negighbors:[1, 2], type: base, data_path: ./results/federated_data/node_0_train_data.pt and epochs: 1
    [Free-rider node 2] fakes training...
    [Node 1] Training for 1 epochs...
    [Node 0] Training for 1 epochs...
    ```
    
    **Explanation:**
    - `results` directory: contains plots showing network topology, data distribution and average performance of all nodes' models.  
      - The partitioned data for each node is saved in `federated_data`.  
      - The `metrics` folder contains detailed results after evaluating a specific node's model for the test and training dataset. 
      - The (middle-man) MPI configuration created by an MPI instance is saved in `mpi_config`. 
    
5. **Exit the Fenics Shell:**
    
    Use the `exit` or `quit` command to exit.
    
    ```bash
    exit
    ```


## License

This project is licensed under the [MIT License](LICENSE).

---
## Authors
This framework was implemented by Gabriel Bengtsson, Zaid Haj-Ibrhaim, Piotr Krzyczkowski and Anna Lithell. 

---
## Acknowledgments

  This project was part of a course project (DAT-300: Data-diven support for cyberphysical systems) at Chalmers University of Technology under the supervision of Romaric Duvignau and Yixing Zhang. 

