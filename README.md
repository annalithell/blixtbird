# Blixtbird: A Simulation Framework for Modeling Attacks in DFL networks

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-purple.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0%2B-yellow.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-2.5%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-red.svg)
![Threading](https://img.shields.io/badge/Concurrent-Multithreading-blueviolet.svg)
![Modular](https://img.shields.io/badge/Architecture-Modular-brightgreen.svg)

**THIS README FILE IS CURRENTLY UNDER DEVELOPMENT** 

## Prerequisites

Before installing and using Blixtbird, ensure that your system meets the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.7 or higher
- **MPI**: An installed MPI distribution. This framework was built and tested using Microsoft MPI v10.0 [download link](https://www.microsoft.com/en-us/download/details.aspx?id=57467). 
- **Virtual Environment (Recommended):** It's advisable to use a virtual environment to manage dependencies

---

## Installation

### Installing via `pip`

1. **Clone the Repository:**
    
    ```bash
    git clone https://github.com/yourusername/blixtbird.git
    cd fenics
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
    
    **Note:** The `setup.py` is configured to install all necessary dependencies, including `torch`, `numpy`, `psutil`, `colorama`, `pyfiglet`, and `pydantic`.
    
5. **Verify Installation:**
    
    After installation, the `fenics` command-line tool should be available.
    
    ```bash
    fenics --help
    ```

    **Expected Output:**
    
    ```
    Usage: phoenix [OPTIONS] COMMAND [ARGS]...
    
      Distributed Federated Learning Simulator
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      setup            Initialize the simulation environment with desired...
      run              Execute the simulation with the provided options.
      list_simulations List all available simulation configurations.
      parameters       Display all available simulation parameters and...
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

**TODO UPDATE FOR BLIXTBIRD**
    
1. **Launch the Fenics Shell:**
    
    ```bash
    fenics
    ```
    
    **Sample Output:**
    
    ```
       ______              _            
      |  ____|            (_)           
      | |__ ___ _ __  ___ _  ___ ___ 
      |  __/ _ \ '_ \/ __| |/ __/ __|
      | | |  __/ | | \__ \ | (__\__ \
      |_|  \___|_| |_|___/_|\___|___/
                                        
    Welcome to Fenics Shell! Type 'help' to see available commands.
    Fenics> 
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
    Final Simulation Arguments:
    Rounds: 3
    Epochs: 1
    Topology: fully_connected
    Participation rate: 0.6
    Protocol: gossip
    Num nodes: 5
    Alpha: 0.5
    
    Starting simulation...
    Simulation Progress: 100%|██████████████████████████████████████████████████████████| 3/3 [00:01<00:00, 2.50round/s, CPU Usage: 55%]
    
    Simulation completed successfully. Check the 'results' directory for outputs.
    ```
    
    **Explanation:**
    - **Progress Bar:** A real-time progress bar displays the simulation's progress, showing the percentage completed and the latest CPU usage.
    - **Logs and Plots:** Detailed logs and visual plots are saved in the `results` directory.
    
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

