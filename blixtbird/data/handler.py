# handler.py

import torch
from torchvision import datasets, transforms
import numpy as np
import os

def save_datasets(train_datasets, output_dir):
    """
    Save datasets (torch.utils.data.Subset) in separate files.
    
    Args:
        train_datasets: List of training data for each node.
        save_path: Path for folder where data will be saved.
    """

    #federated_data_folder = f'{output_dir}/federated_data'
    federated_data_folder = os.path.join(output_dir, 'federated_data')
    #os.makedirs(federated_data_folder, exist_ok=True)

    if not os.path.exists(federated_data_folder):
        os.makedirs(federated_data_folder)

    os.chmod(federated_data_folder, 0o777)
    
    for i, dataset in enumerate(train_datasets):
        file_name = os.path.join(federated_data_folder, f'node_{i}_train_data.pt')
        torch.save(dataset, file_name)
        os.chmod(file_name, 0o666) 


def distribute_data_dirichlet(labels, num_nodes, alpha):
    """
    Distribute data indices among nodes according to a Dirichlet distribution.
    
    Args:
        labels: Array of data labels
        num_nodes: Number of nodes
        alpha: Dirichlet distribution parameter
        
    Returns:
        Dictionary mapping node indices to lists of data indices
    """
    num_classes = np.unique(labels).shape[0]
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    node_indices = {i: [] for i in range(num_nodes)}
    
    for cls in range(num_classes):
        # Shuffle the indices for the current class to ensure random distribution
        np.random.shuffle(class_indices[cls])
        
        # Sample a Dirichlet distribution for the current class
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_nodes))
        
        # Scale proportions to the number of samples in the current class
        proportions = (proportions * len(class_indices[cls])).astype(int)
        
        # Adjust proportions to ensure all samples are allocated
        diff = len(class_indices[cls]) - np.sum(proportions)
        for i in range(diff):
            proportions[i % num_nodes] += 1
        
        # Assign indices to each client based on the proportions
        start = 0
        for node in range(num_nodes):
            end = start + proportions[node]
            node_indices[node].extend(class_indices[cls][start:end].tolist())
            start = end
    
    return node_indices


def load_datasets_dirichlet(num_nodes, alpha, save_to_file, output_dir):
    """
    Load and distribute the FashionMNIST dataset among nodes using Dirichlet distribution.
    
    Args:
        num_nodes: Number of nodes
        alpha: Dirichlet distribution parameter
        
    Returns:
        Tuple of (train_datasets, test_dataset, labels)
    """
    # Define the transformation for the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the training and test datasets
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    # Extract labels from the training dataset
    labels = np.array(train_dataset.targets)

    # Distribute data among clients using the Dirichlet distribution
    node_indices = distribute_data_dirichlet(labels, num_nodes, alpha)

    # Create Subsets for each client based on the distributed indices
    train_datasets = [torch.utils.data.Subset(train_dataset, node_indices[i]) for i in range(num_nodes)]

    # Verification: Ensure all samples are assigned
    total_assigned = sum(len(dataset) for dataset in train_datasets)
    total_available = len(train_dataset)
    assert total_assigned == total_available, "Data assignment mismatch!"

    # Save datasets to separate files
    if save_to_file:
        save_datasets(train_datasets, output_dir)
        print("data saved")

    return train_datasets, test_dataset, labels


def print_class_distribution(train_datasets, logger):
    """
    Print the class distribution for each node's dataset.
    
    Args:
        train_datasets: List of training datasets for each node
        logger: Logger instance
    """
    for i, dataset in enumerate(train_datasets):
        labels = np.array(dataset.dataset.targets)[dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        logger.info(f"node_{i} Class Distribution: {class_counts}")