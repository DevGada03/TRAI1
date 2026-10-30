# PrivateMed: Differentially Private Clinical Model Training under Epsilon Budget

PrivateMed provides an end-to-end framework for training deep neural networks on clinical electronic health records (EHR) under formal Differential Privacy guarantees using **Opacus** and **PyTorch**.

## 🚀 Key Features
- **Realistic Synthetic Clinical EHR Data**: 5,000 patient records across 16 vital signs and laboratory biomarkers with correlated clinical pathophysiology and ~18% positive class imbalance for in-hospital mortality.
- **Opacus-Compatible Architecture (`ClinicalMLP`)**: Built with `nn.Linear`, `nn.LayerNorm`, `nn.ReLU`, and `nn.Dropout`. Avoids leaky `BatchNorm1d` and passes strict validation with `opacus.validators.ModuleValidator`.
- **Dual Training Modes**:
  - **Non-Private Baseline**: Standard deep learning optimization.
  - **DP-SGD**: Rigorous Rényi Differential Privacy (RDP) accountant with automated noise calibration ($\sigma$) via `make_private_with_epsilon()` and per-sample clipping ($C$).
- **Strict Epsilon Budget Guardrail**: Continuously monitors cumulative spent privacy $\varepsilon$ after each epoch and halts training immediately if the privacy budget is exhausted.
- **Membership Inference Attack (MIA) Audit**: Evaluates empirical privacy leakage via loss distribution comparison (Attack AUC ~ 0.50 denotes zero privacy leakage).
- **Interactive Dark-Mode Dashboard**: FastAPI backend with a responsive single-page web interface, live telemetry metrics, and interactive Chart.js visualizations.

---

## 📁 Directory Structure
```
TRAI1/
├── data/
│   └── clinical_generator.py   # Synthetic EHR data generation, normalization, & DataLoader creation
├── models/
│   └── clinical_net.py         # ClinicalMLP & Opacus ModuleValidator checking
├── training/
│   └── dp_trainer.py           # DP-SGD & Non-private training engine with epsilon guardrail
├── evaluation/
│   └── mia_audit.py            # Membership Inference Attack (loss-based) privacy auditor
├── api/
│   └── app.py                  # FastAPI server with training & benchmark sweep endpoints
├── static/
│   └── index.html              # Dark-mode dashboard with Chart.js visualizations
├── test_pipeline.py            # Self-test script verifying DP bounds, clipping, and MIA
└── README.md
```

---

## ⚙️ Quick Start

### 1. Run Verification Self-Test
Verify that data generation, DP-SGD model training, clipping, and MIA audits run cleanly:
```bash
uv run --python .venv python test_pipeline.py
```

### 2. Launch Interactive Web Dashboard
Start the FastAPI server:
```bash
uv run --python .venv python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://127.0.0.1:8000`**
