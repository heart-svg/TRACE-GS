"""Minimal sample-independent repair of the published DEM architecture.

Only the attention tensor semantics are changed.  Every tabular sample is
treated as a sequence of length one: [B, P] -> [B, 1, P].  All linear heads,
fusion weights, losses, optimizer logic, and hyperparameters remain inherited
from the original implementation.
"""
from __future__ import annotations

import torch
from torch import nn

import biodem.constants as const
from biodem.dem.model import DEMLTN as OriginalDEMLTN


class _Extract1OmicsFix(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            n_encoders,
        )
        dims = const.hparam_candidates.linear_dims_single_omics[0]
        self.linears = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.Linear(dims[0], dims[1]),
            nn.Mish(),
            nn.Linear(dims[1], output_dim),
        )

    def forward(self, x):
        x = self.encoders(x.unsqueeze(1)).squeeze(1)
        x = torch.flatten(x, start_dim=1)
        h_out = self.linears[0](x)
        return self.linears(x), h_out


class _ExtractConcOmicsFix(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            n_encoders,
        )
        dims = const.hparam_candidates.linear_dims_conc_omics[0]
        self.linears = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.Mish(),
            nn.LayerNorm(dims[0]),
            nn.Linear(dims[0], dims[1]),
            nn.Linear(dims[1], output_dim),
        )

    def forward(self, x):
        x = self.encoders(x.unsqueeze(1)).squeeze(1)
        x = torch.flatten(x, start_dim=1)
        h_out = self.linears[0](x)
        return self.linears(x), h_out


class _IntegrateExtractionsFix(nn.Module):
    def __init__(self, n_heads, n_encoders, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            n_encoders,
        )
        dims = const.hparam_candidates.linear_dims_integrated[0]
        self.linears = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.Linear(dims[0], dims[1]),
            nn.Mish(),
            nn.Linear(dims[1], dims[2]),
            nn.Linear(dims[2], output_dim),
        )

    def forward(self, x):
        x = self.encoders(x.unsqueeze(1)).squeeze(1)
        x = torch.flatten(x, start_dim=1)
        return self.linears(x)


class DEMFix(nn.Module):
    def __init__(self, omics_dim, n_heads, n_encoders, hidden_dim, output_dim, dropout):
        super().__init__()
        self.omics_dim = omics_dim
        self.extract_conc = _ExtractConcOmicsFix(
            n_heads, n_encoders, sum(omics_dim), hidden_dim, output_dim, dropout
        )
        self.extract_each_omics = nn.ModuleList(
            [
                _Extract1OmicsFix(
                    n_heads, n_encoders, omics_dim[index], hidden_dim, output_dim, dropout
                )
                for index in range(len(omics_dim))
            ]
        )
        integrated_input_dim = (
            const.hparam_candidates.linear_dims_conc_omics[0][0]
            + const.hparam_candidates.linear_dims_single_omics[0][0] * len(omics_dim)
        )
        self.integrate_extractions = _IntegrateExtractionsFix(
            n_heads, n_encoders, integrated_input_dim, hidden_dim, output_dim, dropout
        )
        self.weights_each_omics = nn.ParameterList(
            [nn.Parameter(torch.ones(1) / len(omics_dim)) for _ in omics_dim]
        )
        self.weight_conc = nn.Parameter(torch.ones(1))
        self.weight_integrated = nn.Parameter(torch.ones(1))

    def forward(self, x: list[torch.Tensor]):
        y_pred_conc, h_conc = self.extract_conc(torch.cat(x, dim=1))
        predictions, hidden = [], []
        for index in range(len(self.omics_dim)):
            prediction, representation = self.extract_each_omics[index](x[index])
            predictions.append(self.weights_each_omics[index] * prediction)
            hidden.append(representation)
        y_pred_each = torch.sum(torch.stack(predictions), dim=0)
        h_integrated = torch.cat([h_conc, torch.cat(hidden, dim=1)], dim=1)
        y_pred_integrated = self.integrate_extractions(h_integrated)
        return self.weight_conc * y_pred_conc + self.weight_integrated * y_pred_integrated + y_pred_each


class DEMLTNFix(OriginalDEMLTN):
    def __init__(
        self,
        omics_dim: list[int],
        n_heads: int,
        n_encoders: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        learning_rate: float,
        is_regression: bool,
    ):
        super().__init__(
            omics_dim=omics_dim,
            n_heads=n_heads,
            n_encoders=n_encoders,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            is_regression=is_regression,
        )
        self.DEM_model = DEMFix(
            omics_dim=omics_dim,
            n_heads=n_heads,
            n_encoders=n_encoders,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

