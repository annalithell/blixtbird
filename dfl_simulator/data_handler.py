# data_handler.py

import torch
from torchvision import datasets, transforms
import numpy as np
import random
from collections import defaultdict
import logging

def distribute_data_dirichlet(labels, num_nodes, alpha):
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

def load_datasets_dirichlet(num_nodes, alpha):
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

    return train_datasets, test_dataset, labels

def print_class_distribution(train_datasets, logger):
    for i, dataset in enumerate(train_datasets):
        labels = np.array(dataset.dataset.targets)[dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        logger.info(f"node_{i} Class Distribution: {class_counts}")
