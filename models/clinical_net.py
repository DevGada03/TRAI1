"""
Clinical Neural Network architecture for PrivateMed.
Strictly compatible with Opacus DP-SGD:
- Uses nn.Linear, nn.LayerNorm (or GroupNorm), nn.ReLU, and nn.Dropout
- Avoids BatchNorm1d (which leaks cross-sample statistics)
- Automatic validation with opacus.validators.ModuleValidator
"""

import torch
import torch.nn as nn
from opacus.validators import ModuleValidator
from typing import Tuple, List

class ClinicalMLP(nn.Module):
    """
    Multi-layer Perceptron designed for clinical risk prediction under DP constraints.
    """
    def __init__(self, in_features: int = 16, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # LayerNorm is per-sample normalized, making it DP-compliant unlike BatchNorm
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Final projection layer to scalar logit
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze output to shape (N,) to match binary BCE targets
        return self.network(x).squeeze(-1)


def build_and_validate_model(
    in_features: int = 16,
    hidden_dims: List[int] = [64, 32],
    dropout: float = 0.2,
) -> Tuple[nn.Module, bool]:
    """
    Instantiates ClinicalMLP and runs Opacus ModuleValidator to ensure compatibility.
    """
    model = ClinicalMLP(in_features=in_features, hidden_dims=hidden_dims, dropout=dropout)
    errors = ModuleValidator.validate(model, strict=False)

    if errors:
        model = ModuleValidator.fix(model)
        is_fixed = True
    else:
        is_fixed = False

    # Double check validation
    post_check_errors = ModuleValidator.validate(model, strict=False)
    if post_check_errors:
        raise ValueError(f"Model remains invalid for Opacus DP-SGD: {post_check_errors}")

    return model, is_fixed
