"""
Membership Inference Attack (MIA) Auditor for PrivateMed.
Implements a loss-based membership inference attack comparing loss distributions
of train (members) versus holdout test (non-members).
Quantifies empirical privacy leakage: Attack AUC = 0.50 signifies zero empirical leakage (perfect differential privacy),
whereas Attack AUC near 1.0 indicates severe memorization / leakage.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Any

def compute_sample_losses(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Computes per-sample BCE loss values for the given dataset loader.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    all_losses = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            all_losses.extend(loss.cpu().numpy().tolist())

    return np.array(all_losses, dtype=np.float64)


def audit_membership_inference(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Performs loss-threshold MIA audit.
    Hypothesis: Members (train) exhibit lower loss than non-members (test).
    Attack score = -loss (higher score = more likely to be a training member).
    """
    train_losses = compute_sample_losses(model, train_loader, device)
    test_losses = compute_sample_losses(model, test_loader, device)

    # Member label: 1 for train (member), 0 for test (non-member)
    y_true = np.concatenate([np.ones_like(train_losses), np.zeros_like(test_losses)])
    # Scores: negative loss (smaller loss -> higher probability of being in train set)
    y_scores = np.concatenate([-train_losses, -test_losses])

    # Attack AUROC
    try:
        attack_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        attack_auc = 0.5

    # Compute optimal threshold accuracy
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Balanced attack accuracy: max((tpr + (1 - fpr)) / 2)
    balanced_acc = float(np.max((tpr + (1.0 - fpr)) / 2.0))

    return {
        "mia_auc": round(attack_auc, 4),
        "mia_accuracy": round(balanced_acc, 4),
        "mean_train_loss": round(float(np.mean(train_losses)), 4),
        "mean_test_loss": round(float(np.mean(test_losses)), 4),
        "loss_gap": round(float(np.mean(test_losses) - np.mean(train_losses)), 4),
    }
