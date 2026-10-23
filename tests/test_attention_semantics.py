"""Stage-0 source-semantic tests for DEM and TRACE-GS.

The first module deliberately reproduces the Transformer call in DEM's
Extract1Omics: a 2-D [batch, feature] tensor is passed to a TransformerEncoder
whose encoder layer has batch_first=False.  PyTorch interprets the first axis as
sequence length, so one material can affect another material's representation.
"""
from __future__ import annotations

import json

import torch
from torch import nn


class OriginalDEMCall(nn.Module):
    """Exact attention calling convention used by DEM Extract1Omics."""

    def __init__(self, feature_dim: int = 12) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=2,
                dim_feedforward=24,
                dropout=0.0,
            ),
            num_layers=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MinimalDEMFix(nn.Module):
    """Closest sample-independent correction to DEM's 2-D representation."""

    def __init__(self, feature_dim: int = 12) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=2,
                dim_feedforward=24,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.unsqueeze(1)).squeeze(1)


class ProperTokenAttention(nn.Module):
    """TRACE-GS calling convention: attention over tokens within each sample."""

    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=2,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x, x, x, need_weights=False)[0]


def changed_first_sample(model: nn.Module, x: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        y1 = model(x.clone())
        changed = x.clone()
        changed[-1] = changed[-1] * -7.0 + 11.0
        y2 = model(changed)
    return float((y1[0] - y2[0]).abs().max())


def main() -> None:
    torch.manual_seed(20260902)
    batch_features = torch.randn(4, 12)
    token_features = torch.randn(4, 6, 8)

    original_delta = changed_first_sample(OriginalDEMCall(), batch_features)
    fixed_delta = changed_first_sample(MinimalDEMFix(), batch_features)
    token_delta = changed_first_sample(ProperTokenAttention(), token_features)

    # Within-sample token interaction must remain: changing token 5 in sample 0
    # should alter token 0 in sample 0.
    token_model = ProperTokenAttention().eval()
    with torch.no_grad():
        y1 = token_model(token_features.clone())
        changed = token_features.clone()
        changed[0, -1] += 9.0
        y2 = token_model(changed)
    within_sample_delta = float((y1[0, 0] - y2[0, 0]).abs().max())

    result = {
        "torch_version": torch.__version__,
        "original_dem_cross_sample_delta": original_delta,
        "minimal_dem_fix_cross_sample_delta": fixed_delta,
        "proper_token_cross_sample_delta": token_delta,
        "proper_token_within_sample_delta": within_sample_delta,
    }
    print(json.dumps(result, indent=2))

    assert original_delta > 1e-6, "Original DEM calling convention did not reproduce sample mixing"
    assert fixed_delta < 1e-6, "Minimal DEM-Fix is not batch invariant"
    assert token_delta < 1e-6, "Proper token attention is not batch invariant"
    assert within_sample_delta > 1e-6, "Proper attention lost within-sample token interaction"
    print("ATTENTION_SEMANTICS_TESTS_PASSED")


if __name__ == "__main__":
    main()


