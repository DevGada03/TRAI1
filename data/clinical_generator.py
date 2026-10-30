"""
Clinical EHR Data Generator & Preprocessor for PrivateMed.
Generates realistic synthetic Electronic Health Records with 16 clinical features
and binary in-hospital mortality (~18% prevalence).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

FEATURE_NAMES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
    "serum_glucose",
    "creatinine",
    "blood_urea_nitrogen",
    "white_blood_cells",
    "platelets",
    "hemoglobin",
    "serum_sodium",
    "serum_potassium",
    "bmi",
]

def generate_synthetic_ehr(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic clinical dataset with physiological correlations
    and ~18% positive class imbalance for in-hospital mortality.
    """
    rng = np.random.default_rng(random_seed)

    age = rng.normal(loc=64.0, scale=14.0, size=n_samples).clip(18, 95)
    bmi = rng.normal(loc=28.5, scale=6.0, size=n_samples).clip(15, 55)

    # Correlated vital signs
    heart_rate = rng.normal(loc=82.0, scale=16.0, size=n_samples).clip(45, 160)
    systolic_bp = (110.0 + 0.3 * (age - 50) + rng.normal(0, 18, n_samples)).clip(75, 210)
    diastolic_bp = (70.0 + 0.15 * (age - 50) + rng.normal(0, 10, n_samples)).clip(40, 120)
    respiratory_rate = rng.normal(loc=19.0, scale=4.5, size=n_samples).clip(10, 42)
    spo2 = (98.0 - rng.exponential(scale=2.2, size=n_samples)).clip(72, 100)
    temperature = rng.normal(loc=37.0, scale=0.8, size=n_samples).clip(34.5, 41.5)

    # Correlated lab markers
    serum_glucose = (105.0 + 1.2 * (bmi - 25) + rng.exponential(scale=25, size=n_samples)).clip(55, 450)
    creatinine = (0.95 + 0.01 * (age - 50) + rng.exponential(scale=0.45, size=n_samples)).clip(0.4, 8.5)
    blood_urea_nitrogen = (17.0 + 10.0 * (creatinine - 0.95) + rng.normal(0, 5, n_samples)).clip(5, 120)
    white_blood_cells = rng.normal(loc=8.5, scale=4.0, size=n_samples).clip(1.5, 38.0)
    platelets = rng.normal(loc=230.0, scale=65.0, size=n_samples).clip(30, 650)
    hemoglobin = (13.5 - 0.03 * (age - 50) + rng.normal(0, 1.8, n_samples)).clip(6.0, 18.5)
    serum_sodium = rng.normal(loc=139.0, scale=3.5, size=n_samples).clip(120, 155)
    serum_potassium = rng.normal(loc=4.1, scale=0.55, size=n_samples).clip(2.5, 7.2)

    df = pd.DataFrame({
        "age": age,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "spo2": spo2,
        "temperature": temperature,
        "serum_glucose": serum_glucose,
        "creatinine": creatinine,
        "blood_urea_nitrogen": blood_urea_nitrogen,
        "white_blood_cells": white_blood_cells,
        "platelets": platelets,
        "hemoglobin": hemoglobin,
        "serum_sodium": serum_sodium,
        "serum_potassium": serum_potassium,
        "bmi": bmi,
    })

    # Non-linear clinical risk score (e.g., SOFA/APACHE-like indicator)
    logit = (
        -4.2
        + 0.038 * (age - 60)
        + 0.022 * (heart_rate - 80)
        - 0.018 * (systolic_bp - 120)
        + 0.09 * (respiratory_rate - 18)
        - 0.15 * (spo2 - 96)
        + 0.65 * (creatinine - 1.0)
        + 0.08 * (white_blood_cells - 9.0)
        - 0.18 * (hemoglobin - 13.0)
        + 0.005 * (serum_glucose - 100)
    )

    prob = 1.0 / (1.0 + np.exp(-logit))
    # Calibrate probability to guarantee ~18% positive rate
    target_mean = 0.18
    # Shift logit slightly if needed
    current_mean = prob.mean()
    logit_shift = np.log(target_mean / (1 - target_mean)) - np.log(current_mean / (1 - current_mean))
    calibrated_prob = 1.0 / (1.0 + np.exp(-(logit + logit_shift)))

    mortality = rng.binomial(n=1, p=calibrated_prob)
    df["in_hospital_mortality"] = mortality

    return df


def get_clinical_dataloaders(
    n_samples: int = 5000,
    test_size: float = 0.2,
    batch_size: int = 128,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Preprocesses synthetic EHR data and builds PyTorch DataLoaders.
    Returns:
        train_loader, test_loader, raw_splits (dict with X_train, y_train, X_test, y_test)
    """
    df = generate_synthetic_ehr(n_samples=n_samples, random_seed=random_seed)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["in_hospital_mortality"].values.astype(np.float32)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # Note: Opacus wrapping replaces standard batch sampling or uses Poisson sampling if DP loader
    # Using drop_last=False for general use; Opacus will manage batch sizes or Poisson sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    raw_splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_names": FEATURE_NAMES,
    }

    return train_loader, test_loader, raw_splits
