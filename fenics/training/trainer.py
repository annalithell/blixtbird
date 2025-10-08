# fenics/training/trainer.py

import torch
import torch.nn as nn
import time
import logging
import os


def local_train(node_id, local_model, train_dataset, epochs, attacker_type):
    """
    Train a local model for a specific node.
    
    Args:
        node_id: ID of the node
        local_model: Model to train
        train_dataset: Training dataset
        epochs: Number of training epochs
        attacker_type: Type of attack to simulate (if any)
        
    Returns:
        Tuple of (model state dictionary, training time)
    """
    device = torch.device("cpu")
    model = local_model
    model.to(device)
    # Added weight decay for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Weight decay added
    criterion = nn.NLLLoss()
    model.train()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    start_time = time.time()  # Start time for training

    logging.info(f"[node_{node_id}] Starting training for {epochs} epochs.")

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Simulate model poisoning
    if attacker_type == 'poison':
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.5)  # Add significant noise
        logging.info(f"[node_{node_id}] Model poisoned.")

    end_time = time.time()  # End time for training
    training_time = end_time - start_time  # Calculate training time

    # Return updated model parameters and training time
    return model.state_dict(), training_time  # Return tuple

def load_datasets(nodes_ids, save_path='./federated_data'):

    node_datasets = {}

    for node_id in nodes_ids:
        file_name = os.path.join(save_path, f'node_{node_id}_train_data.pt')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File for node {node_id} not found: {file_name}")
        node_dataset = torch.load(file_name, weights_only=False)
        node_datasets[node_id] = node_dataset
    
    return node_datasets