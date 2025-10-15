# fenics/training/evaluator.py

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(model, test_loader):
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        
    Returns:
        Tuple of (test loss, accuracy, f1 score, precision, recall)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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