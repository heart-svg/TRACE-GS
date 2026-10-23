"""TRACE-GS stage-2 RNA program encoder and trait-conditioned query."""
from __future__ import annotations

import torch
from torch import nn


class RNAProgramEncoder(nn.Module):
    """Convert fold-fitted low-rank RNA scores into compact program tokens."""

    def __init__(
        self,
        input_dim: int,
        n_programs: int = 32,
        d_model: int = 64,
        bottleneck: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.d_model = d_model
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, n_programs * d_model),
        )
        self.program_identity = nn.Parameter(torch.randn(1, n_programs, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        programs = self.encoder(scores).view(-1, self.n_programs, self.d_model)
        return self.norm(programs + self.program_identity)


class TraitRNAQuery(nn.Module):
    """Let each phenotype query a shared set of RNA expression programs."""

    def __init__(
        self,
        n_traits: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.trait_embedding = nn.Embedding(n_traits, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.program_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, programs: torch.Tensor, trait_ids: torch.Tensor):
        query = self.query_norm(self.trait_embedding(trait_ids)).unsqueeze(1)
        normalized = self.program_norm(programs)
        context, weights = self.attention(
            query,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        context = context + self.ffn(self.ffn_norm(context))
        return context.squeeze(1), weights.squeeze(2)


class TRACERNAOnly(nn.Module):
    def __init__(self, input_dim: int, n_traits: int = 4, d_model: int = 64) -> None:
        super().__init__()
        self.program_encoder = RNAProgramEncoder(input_dim, 32, d_model)
        self.query = TraitRNAQuery(n_traits, d_model, 4)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, scores: torch.Tensor, trait_ids: torch.Tensor):
        programs = self.program_encoder(scores)
        context, attention = self.query(programs, trait_ids)
        trait = self.query.trait_embedding(trait_ids)
        prediction = self.head(torch.cat([context, trait], dim=-1)).squeeze(1)
        return {
            "prediction": prediction,
            "programs": programs,
            "context": context,
            "program_attention": attention,
        }

