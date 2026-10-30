"""
Core Training Engine for PrivateMed.
Supports:
1. Non-Private Baseline SGD/Adam
2. Differentially Private Training (DP-SGD) with Opacus PrivacyEngine (RDP accountant),
   make_private_with_epsilon(), and per-epoch Epsilon Budget Guardrails.
"""

import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from opacus import PrivacyEngine
from typing import Dict, Any, List, Optional, Callable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluates binary classification metrics (AUROC, AUPRC, F1, Loss).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(y_batch.cpu().numpy().tolist())

    n_samples = max(1, len(all_targets))
    mean_loss = total_loss / n_samples
    targets_arr = (torch.tensor(all_targets).numpy() >= 0.5).astype(int)
    probs_arr = torch.tensor(all_probs).numpy()
    preds_arr = (probs_arr >= 0.5).astype(int)

    try:
        auroc = float(roc_auc_score(targets_arr, probs_arr))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(targets_arr, probs_arr))
    except Exception:
        auprc = float(targets_arr.mean()) if len(targets_arr) > 0 else 0.0

    f1 = float(f1_score(targets_arr, preds_arr, zero_division=0))

    return {
        "loss": round(mean_loss, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(f1, 4),
    }


def train_privatemed_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    target_epsilon: Optional[float] = 2.0,
    target_delta: float = 1e-4,
    max_grad_norm: float = 1.0,
    epochs: int = 20,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Trains the clinical model under either Non-Private baseline (target_epsilon is None or float('inf'))
    or Differentially Private Training (DP-SGD) with strict Epsilon Budget Guardrails.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    is_dp = (target_epsilon is not None) and not math.isinf(target_epsilon)
    privacy_engine = None
    calibrated_sigma = 0.0

    if is_dp:
        # Opacus 1.x PrivacyEngine with Rényi Differential Privacy (RDP) accountant
        privacy_engine = PrivacyEngine(accountant="rdp")
        # make_private_with_epsilon automatically searches for noise multiplier sigma
        # that satisfies (target_epsilon, target_delta) over the planned number of epochs
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            criterion=criterion,
        )
        calibrated_sigma = float(optimizer.noise_multiplier)

    epoch_logs: List[Dict[str, Any]] = []
    budget_halted = False

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_sz = len(y_batch)
            running_loss += loss.item() * batch_sz
            total_samples += batch_sz

        train_loss = running_loss / max(1, total_samples)

        # Privacy Accounting after epoch
        if is_dp and privacy_engine is not None:
            spent_eps = float(privacy_engine.get_epsilon(delta=target_delta))
            current_sigma = float(optimizer.noise_multiplier)
        else:
            spent_eps = float("inf")
            current_sigma = 0.0

        # Clinical evaluation on validation/test set
        eval_metrics = evaluate_model(model, test_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": eval_metrics["loss"],
            "val_auroc": eval_metrics["auroc"],
            "val_auprc": eval_metrics["auprc"],
            "val_f1": eval_metrics["f1"],
            "spent_epsilon": round(spent_eps, 4) if not math.isinf(spent_eps) else None,
            "noise_multiplier": round(current_sigma, 4),
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "budget_spent_pct": round(min(100.0, (spent_eps / target_epsilon) * 100.0), 1) if is_dp else 100.0,
        }
        epoch_logs.append(epoch_record)

        if progress_callback:
            progress_callback(epoch_record)

        # Strict Epsilon Budget Guardrail Halt
        if is_dp and (spent_eps >= target_epsilon):
            budget_halted = True
            break

    # Final test evaluation
    final_metrics = evaluate_model(model, test_loader, device)
    final_eps = epoch_logs[-1]["spent_epsilon"] if epoch_logs else None

    return {
        "is_dp": is_dp,
        "target_epsilon": target_epsilon,
        "target_delta": target_delta,
        "max_grad_norm": max_grad_norm,
        "epochs_completed": len(epoch_logs),
        "total_epochs_requested": epochs,
        "budget_halted": budget_halted,
        "noise_multiplier": round(calibrated_sigma, 4),
        "final_spent_epsilon": final_eps,
        "final_metrics": final_metrics,
        "epoch_history": epoch_logs,
        "trained_model": model,
    }
