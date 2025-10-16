import yaml
import networkx as nx
import os
from typing import Dict

def create_yaml(G, node_type_map: Dict[int, str], output_dir: str):

    node_type_map

    mpi_config_folder = f'{output_dir}/mpi_config'

    if not os.path.exists(mpi_config_folder):
        os.makedirs(mpi_config_folder)

    node_type = None

    data = {'nodes':{}}
    for node_id, node_type in node_type_map.items():
        data['nodes'][node_id] = {}
        data['nodes'][node_id]['type'] = node_type
        data['nodes'][node_id]['neighbors'] = get_neighbors(node_id, G)
        data['nodes'][node_id]['dataset'] = f'./{output_dir}/federated_data/node_{node_id}_train_data.pt'

    # Writing the data to a YAML file
    with open(f'./{mpi_config_folder}/local_config.yaml', 'w') as file:
        yaml.dump(data, file)
    return

def get_node_data(node_id: int, output_dir: str):

    mpi_config_folder = f'{output_dir}/mpi_config/local_config.yaml'

    with open(mpi_config_folder, 'r') as file:
        config = yaml.safe_load(file)

        nodes = config['nodes']
        dataset_path = nodes[node_id]['dataset']
        neighbors = nodes[node_id]['neighbors']
        type = nodes[node_id]['type']

        return type, neighbors, dataset_path

def get_neighbors(node_id, G):
    return list(G.adj[node_id])

def get_output_dir():
    base_dir = "results"
    if not os.path.exists(base_dir):
        output_dir = base_dir
    else:
        i = 1
        while True:
            new_dir = f"{base_dir}_{i}"
            if not os.path.exists(new_dir):
                current_dir_num = i-1
                output_dir = f"{base_dir}_{current_dir_num}"
                break
            i += 1
    
    return output_dir