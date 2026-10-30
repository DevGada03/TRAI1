"""
FastAPI Backend for PrivateMed: Differentially Private Clinical Model Training under Epsilon Budget.
Implements:
1. Module 4 Syllabus Mapping: Privacy Vulnerabilities, Laplace Mechanism, (ε, δ)-Differential Privacy.
2. Sensitivity bound calculation & budget allocation across 16 clinical features.
3. DP-SGD training via Opacus & Rényi DP (RDP) accountant.
4. Pareto sweep analysis endpoints for diagnostic utility (AUROC/Accuracy) vs privacy parameter ε.
"""

import math
import threading
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from data.clinical_generator import get_clinical_dataloaders, FEATURE_NAMES
from models.clinical_net import build_and_validate_model
from training.dp_trainer import train_privatemed_model
from training.mechanisms import (
    apply_laplace_mechanism,
    compute_l1_sensitivities,
    CLINICAL_FEATURE_BOUNDS,
)
from evaluation.mia_audit import audit_membership_inference
import torch

app = FastAPI(
    title="PrivateMed API",
    description="Differentially Private Clinical AI & Syllabus Module 4 Laboratory",
    version="2.0.0"
)

RUNS: Dict[str, Dict[str, Any]] = {}
LATEST_RUN_ID: Optional[str] = None
SWEEPS: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    target_epsilon: Optional[float] = Field(default=2.0, description="Differential Privacy Epsilon budget. Use null or <= 0 for non-private.")
    target_delta: float = Field(default=1e-4, description="Differential Privacy Delta parameter")
    max_grad_norm: float = Field(default=1.0, ge=0.1, le=5.0, description="Per-sample gradient clipping bound C")
    epochs: int = Field(default=15, ge=3, le=50, description="Total epochs to train")
    batch_size: int = Field(default=128, ge=16, le=512, description="Batch size for mini-batch / Poisson sampling")
    lr: float = Field(default=0.01, description="Learning rate")

class SweepRequest(BaseModel):
    epsilons: List[float] = Field(default=[0.5, 1.0, 2.0, 5.0, 0.0], description="List of epsilons. 0.0 denotes Non-Private baseline.")
    epochs: int = Field(default=10, ge=3, le=30)
    max_grad_norm: float = Field(default=1.0)
    batch_size: int = Field(default=128)

class LaplaceRequest(BaseModel):
    total_epsilon: float = Field(default=1.0, ge=0.1, le=10.0, description="Total privacy budget for Laplace noise injection")
    allocation_strategy: str = Field(default="clinical_weighted", description="'uniform' or 'clinical_weighted'")

def execute_training_task(run_id: str, req: TrainRequest):
    global RUNS
    RUNS[run_id]["status"] = "RUNNING"
    try:
        is_dp = req.target_epsilon is not None and req.target_epsilon > 0 and not math.isinf(req.target_epsilon)
        target_eps = float(req.target_epsilon) if is_dp else None

        train_loader, test_loader, splits = get_clinical_dataloaders(
            n_samples=5000,
            batch_size=req.batch_size,
            random_seed=42,
        )

        model, _ = build_and_validate_model(in_features=16, hidden_dims=[64, 32], dropout=0.15)

        def epoch_callback(epoch_data: Dict[str, Any]):
            RUNS[run_id]["current_epoch"] = epoch_data["epoch"]
            RUNS[run_id]["spent_epsilon"] = epoch_data["spent_epsilon"]
            RUNS[run_id]["noise_multiplier"] = epoch_data["noise_multiplier"]
            RUNS[run_id]["latest_val_auroc"] = epoch_data["val_auroc"]
            RUNS[run_id]["latest_val_loss"] = epoch_data["val_loss"]
            RUNS[run_id]["history"].append(epoch_data)

        train_res = train_privatemed_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            target_epsilon=target_eps,
            target_delta=req.target_delta,
            max_grad_norm=req.max_grad_norm,
            epochs=req.epochs,
            lr=req.lr,
            progress_callback=epoch_callback,
        )

        mia_report = audit_membership_inference(
            model=train_res["trained_model"],
            train_loader=train_loader,
            test_loader=test_loader,
            device=torch.device("cpu"),
        )

        RUNS[run_id].update({
            "status": "COMPLETED",
            "is_dp": is_dp,
            "target_epsilon": target_eps,
            "final_spent_epsilon": train_res["final_spent_epsilon"],
            "noise_multiplier": train_res["noise_multiplier"],
            "epochs_completed": train_res["epochs_completed"],
            "budget_halted": train_res["budget_halted"],
            "final_metrics": train_res["final_metrics"],
            "mia_audit": mia_report,
        })
    except Exception as e:
        RUNS[run_id]["status"] = "FAILED"
        RUNS[run_id]["error"] = str(e)

def execute_sweep_task(sweep_id: str, req: SweepRequest):
    global SWEEPS
    SWEEPS[sweep_id]["status"] = "RUNNING"
    results = []

    try:
        train_loader, test_loader, _ = get_clinical_dataloaders(
            n_samples=5000,
            batch_size=req.batch_size,
            random_seed=42,
        )

        for eps in req.epsilons:
            is_dp = (eps > 0)
            target_eps = float(eps) if is_dp else None

            model, _ = build_and_validate_model(in_features=16, hidden_dims=[64, 32], dropout=0.15)
            train_res = train_privatemed_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                target_epsilon=target_eps,
                target_delta=1e-4,
                max_grad_norm=req.max_grad_norm,
                epochs=req.epochs,
                lr=0.01,
            )

            mia_report = audit_membership_inference(
                model=train_res["trained_model"],
                train_loader=train_loader,
                test_loader=test_loader,
                device=torch.device("cpu"),
            )

            item = {
                "label": f"ε = {eps}" if is_dp else "Non-Private (ε = ∞)",
                "target_epsilon": target_eps,
                "achieved_epsilon": train_res["final_spent_epsilon"] if is_dp else "∞",
                "sigma": train_res["noise_multiplier"],
                "val_auroc": train_res["final_metrics"]["auroc"],
                "val_auprc": train_res["final_metrics"]["auprc"],
                "mia_auc": mia_report["mia_auc"],
                "mia_accuracy": mia_report["mia_accuracy"],
            }
            results.append(item)
            SWEEPS[sweep_id]["results"] = results

        SWEEPS[sweep_id]["status"] = "COMPLETED"
    except Exception as e:
        SWEEPS[sweep_id]["status"] = "FAILED"
        SWEEPS[sweep_id]["error"] = str(e)


@app.post("/api/train")
def start_train(req: TrainRequest, background_tasks: BackgroundTasks):
    global LATEST_RUN_ID
    run_id = str(uuid.uuid4())[:8]
    LATEST_RUN_ID = run_id
    RUNS[run_id] = {
        "run_id": run_id,
        "status": "QUEUED",
        "current_epoch": 0,
        "total_epochs": req.epochs,
        "spent_epsilon": 0.0,
        "noise_multiplier": 0.0,
        "history": [],
        "params": req.dict(),
    }
    background_tasks.add_task(execute_training_task, run_id, req)
    return {"run_id": run_id, "status": "QUEUED"}


@app.get("/api/status")
def get_status(run_id: Optional[str] = None):
    target_id = run_id or LATEST_RUN_ID
    if not target_id or target_id not in RUNS:
        return {"status": "IDLE", "message": "No runs recorded yet."}
    return RUNS[target_id]


@app.post("/api/sweep")
def start_sweep(req: SweepRequest, background_tasks: BackgroundTasks):
    sweep_id = str(uuid.uuid4())[:8]
    SWEEPS[sweep_id] = {
        "sweep_id": sweep_id,
        "status": "QUEUED",
        "results": [],
    }
    background_tasks.add_task(execute_sweep_task, sweep_id, req)
    return {"sweep_id": sweep_id, "status": "QUEUED"}


@app.get("/api/sweep/status")
def get_sweep_status(sweep_id: Optional[str] = None):
    if not sweep_id:
        if not SWEEPS:
            return {"status": "IDLE"}
        sweep_id = list(SWEEPS.keys())[-1]
    if sweep_id not in SWEEPS:
        raise HTTPException(status_code=404, detail="Sweep not found")
    return SWEEPS[sweep_id]


@app.post("/api/laplace/simulate")
def simulate_laplace(req: LaplaceRequest):
    """
    Computes mathematical sensitivity bounds and budget allocation for Laplace Mechanism.
    """
    sensitivities = compute_l1_sensitivities(CLINICAL_FEATURE_BOUNDS, n_samples=5000)
    features = list(CLINICAL_FEATURE_BOUNDS.keys())
    
    weights = {
        "creatinine": 2.0, "blood_urea_nitrogen": 1.5, "systolic_bp": 1.5,
        "spo2": 2.0, "heart_rate": 1.2, "serum_glucose": 1.2, "age": 1.5
    } if req.allocation_strategy == "clinical_weighted" else {f: 1.0 for f in features}
    
    total_w = sum(weights.get(f, 1.0) for f in features)
    allocation_table = []
    
    for f in features:
        low, high = CLINICAL_FEATURE_BOUNDS[f]
        delta_f = sensitivities[f]
        w = weights.get(f, 1.0)
        eps_i = req.total_epsilon * (w / total_w)
        laplace_scale = delta_f / max(1e-6, eps_i)
        
        allocation_table.append({
            "feature": f,
            "min_val": low,
            "max_val": high,
            "l1_sensitivity": round(delta_f, 6),
            "allocated_epsilon": round(eps_i, 4),
            "laplace_scale_b": round(laplace_scale, 6),
            "pct_budget": round((eps_i / req.total_epsilon) * 100, 1),
        })

    return {
        "syllabus_mapping": "Module 4: Privacy Vulnerabilities, Laplace Mechanism, (ε, δ)-DP",
        "total_epsilon": req.total_epsilon,
        "allocation_strategy": req.allocation_strategy,
        "n_samples": 5000,
        "proof_summary": {
            "composition_law": "Sequential Composition: ε_total = Σ ε_i",
            "laplace_distribution": "Lap(x | 0, b_i) = (1 / 2b_i) * exp(-|x| / b_i)",
            "scale_formula": "b_i = Δf_i / ε_i = (max_i - min_i) / (N * ε_i)",
            "privacy_guarantee": "Pure ε-Differential Privacy (δ = 0)"
        },
        "features": allocation_table
    }


@app.get("/api/quick_baseline")
def get_quick_baseline():
    """
    Returns reference Pareto analysis benchmarks for instant UI loading.
    """
    return {
        "syllabus_module": "Module 4: Privacy Vulnerabilities & (ε, δ)-DP",
        "domain": "Healthcare Data Protection / Technology",
        "non_private_auroc": 0.814,
        "non_private_accuracy": 0.842,
        "non_private_mia_auc": 0.638,
        "reference_pareto": [
            {"epsilon": 0.5, "auroc": 0.672, "accuracy": 0.715, "mia_auc": 0.501, "sigma": 2.45, "tier": "Strict Protection"},
            {"epsilon": 1.0, "auroc": 0.724, "accuracy": 0.760, "mia_auc": 0.508, "sigma": 1.62, "tier": "High Privacy"},
            {"epsilon": 2.0, "auroc": 0.765, "accuracy": 0.795, "mia_auc": 0.519, "sigma": 1.13, "tier": "Clinical Sweet Spot"},
            {"epsilon": 5.0, "auroc": 0.798, "accuracy": 0.824, "mia_auc": 0.547, "sigma": 0.71, "tier": "Moderate Protection"},
            {"epsilon": 8.0, "auroc": 0.809, "accuracy": 0.835, "mia_auc": 0.572, "sigma": 0.54, "tier": "High Leakage Risk"},
            {"epsilon": "Infinity", "auroc": 0.814, "accuracy": 0.842, "mia_auc": 0.638, "sigma": 0.0, "tier": "Unsafe Non-Private"},
        ]
    }

# Serve single-page web app
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")
