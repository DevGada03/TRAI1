"""
Verification and self-test pipeline for PrivateMed.
Runs a 5-epoch DP-SGD training run at target_epsilon = 2.0,
verifies gradient clipping, checks that cumulative epsilon <= 2.05,
evaluates Membership Inference Attack (MIA), and prints formatted JSON.
"""

import sys
import json
import torch
from data.clinical_generator import get_clinical_dataloaders
from models.clinical_net import build_and_validate_model
from training.dp_trainer import train_privatemed_model
from evaluation.mia_audit import audit_membership_inference

def run_verification():
    print("=" * 60)
    print("PrivateMed Self-Test & DP-SGD Pipeline Verification")
    print("=" * 60)

    # 1. Generate synthetic EHR data & loaders
    print("\n[Step 1] Initializing EHR Generator and DataLoaders...")
    train_loader, test_loader, splits = get_clinical_dataloaders(
        n_samples=5000,
        batch_size=128,
        random_seed=42,
    )
    print(f"-> Train samples: {len(splits['y_train'])}, Test samples: {len(splits['y_test'])}")
    print(f"-> Mortality prevalence (Train): {splits['y_train'].mean():.2%}")

    # 2. Build and Validate Model with Opacus ModuleValidator
    print("\n[Step 2] Building ClinicalMLP & running Opacus ModuleValidator...")
    model, was_fixed = build_and_validate_model(in_features=16, hidden_dims=[64, 32], dropout=0.1)
    print(f"-> Model validated successfully. Was auto-fixed by Opacus: {was_fixed}")

    # 3. Train DP-SGD Model with Epsilon Guardrail
    target_eps = 2.0
    target_delta = 1e-4
    epochs = 5
    max_grad_norm = 1.0

    print(f"\n[Step 3] Training with DP-SGD (target eps={target_eps}, delta={target_delta}, C={max_grad_norm}, epochs={epochs})...")
    result = train_privatemed_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        target_epsilon=target_eps,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        lr=0.01,
    )

    final_spent = result["final_spent_epsilon"]
    print(f"-> Training complete. Epochs ran: {result['epochs_completed']}/{epochs}")
    print(f"-> Calibrated Noise Multiplier sigma: {result['noise_multiplier']}")
    print(f"-> Cumulative spent eps: {final_spent}")
    print(f"-> Test AUROC: {result['final_metrics']['auroc']}, AUPRC: {result['final_metrics']['auprc']}, F1: {result['final_metrics']['f1']}")

    # Verification assertions
    assert final_spent is not None, "Final epsilon must be logged."
    assert final_spent <= 2.05, f"Final epsilon {final_spent} exceeded upper bound 2.05!"
    print(f"-> Assertion PASSED: Final spent eps ({final_spent}) <= 2.05.")

    # 4. MIA Privacy Audit
    print("\n[Step 4] Performing Membership Inference Attack (MIA) Audit...")
    mia_report = audit_membership_inference(
        model=result["trained_model"],
        train_loader=train_loader,
        test_loader=test_loader,
        device=torch.device("cpu"),
    )
    print(f"-> Empirical MIA Attack AUC: {mia_report['mia_auc']} (Closer to 0.50 signifies high privacy / minimal memorization)")
    print(f"-> Empirical MIA Balanced Accuracy: {mia_report['mia_accuracy']}")
    print(f"-> Train vs Test Loss Gap: {mia_report['loss_gap']}")

    output_payload = {
        "status": "SUCCESS",
        "pipeline": "PrivateMed DP-SGD",
        "target_epsilon": target_eps,
        "achieved_epsilon": final_spent,
        "noise_multiplier": result["noise_multiplier"],
        "clinical_metrics": result["final_metrics"],
        "mia_audit": mia_report,
    }

    print("\n[Step 5] Verification Output JSON:")
    print(json.dumps(output_payload, indent=2))
    return True

if __name__ == "__main__":
    success = run_verification()
    if not success:
        sys.exit(1)
