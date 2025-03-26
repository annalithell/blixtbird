# training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from sklearn.metrics import f1_score, precision_score, recall_score

def local_train(node_id, local_model, train_dataset, epochs, attacker_type):
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

def evaluate(model, test_loader):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    criterion = nn.NLLLoss()  # Negative log likelihood loss function
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)

    return test_loss, accuracy, f1, precision, recall

def summarize_model_parameters(node_name, model_state_dict, logger):
    logger.info(f"\n[Summary] Model parameters for node_{node_name} after local training:")
    for key, param in model_state_dict.items():
        param_np = param.cpu().numpy()
        mean = param_np.mean()
        std = param_np.std()
        logger.info(f" Layer: {key:<20} | Mean: {mean:.6f} | Std: {std:.6f}")
