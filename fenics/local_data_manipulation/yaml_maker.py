import yaml
import networkx as nx

def create_yaml(nodes, topology_file):

    node_type = None

    data = {}

    for node_id in nodes:
        data['nodes'][node_id]['type'] = node_type
        data['nodes'][node_id]['neighbors'] = get_neighbors(node_id, topology_file)
        data['nodes'][node_id]['dataset'] = f'node_{node_id}_train_data.pt'

    # Writing the data to a YAML file
    with open('local_config.yaml', 'w') as file:
        yaml.dump(data, file)
    pass

def get_neighbors(node_id, topology_file):
    G = nx.read_edgelist(topology_file, nodetype=int)
    return list(G.adj[node_id])