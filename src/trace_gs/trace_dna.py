"""TRACE-GS stage-1 chromosome-aware DNA backbone."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from token_allocate import allocate_tokens


class MultiScaleChromosomeCNN(nn.Module):
    """Shared local encoder applied independently to every chromosome."""

    def __init__(self, input_channels: int = 3, d_model: int = 64, local_channels: int = 32) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_channels, local_channels)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    local_channels,
                    local_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=local_channels,
                )
                for kernel in (3, 5, 9)
            ]
        )
        self.mix = nn.Conv1d(local_channels * 3, d_model, kernel_size=1)
        self.residual = nn.Linear(input_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.input_projection(x).transpose(1, 2)
        h = torch.cat([branch(h) for branch in self.branches], dim=1)
        h = self.mix(F.gelu(h)).transpose(1, 2)
        return F.gelu(self.norm(h + residual))


class GenomeTokenizer(nn.Module):
    """Encode chromosomes separately and pool to a fixed genome-wide budget."""

    def __init__(self, d_model: int = 64, total_tokens: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.total_tokens = total_tokens
        self.chromosome_cnn = MultiScaleChromosomeCNN(3, d_model)

    def forward(
        self,
        chromosome_tensors: dict[int, torch.Tensor],
        token_allocation: dict[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not chromosome_tensors:
            raise ValueError("No chromosome tensors were supplied")
        batch_size = next(iter(chromosome_tensors.values())).shape[0]
        tokens = []
        for chromosome in sorted(chromosome_tensors):
            x = chromosome_tensors[chromosome]
            encoded = self.chromosome_cnn(x)
            n_tokens = token_allocation[chromosome]
            pooled = F.adaptive_avg_pool1d(encoded.transpose(1, 2), n_tokens).transpose(1, 2)
            tokens.append(pooled)
        genome = torch.cat(tokens, dim=1)
        effective = genome.shape[1]
        if effective > self.total_tokens:
            raise RuntimeError("Allocated more tokens than the configured budget")
        mask = torch.zeros(batch_size, self.total_tokens, dtype=torch.bool, device=genome.device)
        if effective < self.total_tokens:
            pad = genome.new_zeros(batch_size, self.total_tokens - effective, self.d_model)
            genome = torch.cat([genome, pad], dim=1)
            mask[:, effective:] = True
        return genome, mask


class SampleIndependentAttentionBlock(nn.Module):
    """Explicit within-sample MHA followed by a compact feed-forward block."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attention, _ = self.attention(
            h,
            h,
            h,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attention
        x = x + self.ffn(self.norm2(x))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class TraitAttentionPool(nn.Module):
    def __init__(self, n_traits: int = 4, d_model: int = 64) -> None:
        super().__init__()
        self.trait_embedding = nn.Embedding(n_traits, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5

    def forward(
        self,
        tokens: torch.Tensor,
        trait_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trait = self.trait_embedding(trait_ids)
        query = self.query(trait).unsqueeze(1)
        scores = torch.matmul(query, self.key(tokens).transpose(1, 2)).squeeze(1) * self.scale
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * self.value(tokens), dim=1)
        return pooled, weights


class TRACEGenomicBackbone(nn.Module):
    """Stage-1 multi-trait DNA-only predictor."""

    def __init__(self, n_genes: int, n_traits: int = 4, d_model: int = 64, total_tokens: int = 64) -> None:
        super().__init__()
        self.tokenizer = GenomeTokenizer(d_model, total_tokens)
        self.attention = SampleIndependentAttentionBlock(d_model, 4)
        self.pool = TraitAttentionPool(n_traits, d_model)
        # The global additive path preserves information that local pooling may lose.
        self.additive_head = nn.Linear(n_genes, n_traits)
        self.local_residual_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        full_genome: torch.Tensor,
        chromosome_tensors: dict[int, torch.Tensor],
        token_allocation: dict[int, int],
        trait_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local, mask = self.tokenizer(chromosome_tensors, token_allocation)
        genome_tokens = self.attention(local, mask)
        pooled, attention_weights = self.pool(genome_tokens, trait_ids, mask)
        trait = self.pool.trait_embedding(trait_ids)
        additive_all = self.additive_head(full_genome)
        additive = additive_all.gather(1, trait_ids.unsqueeze(1)).squeeze(1)
        local_residual = self.local_residual_head(torch.cat([pooled, trait], dim=-1)).squeeze(1)
        return {
            "prediction": additive + local_residual,
            "additive": additive,
            "local_residual": local_residual,
            "tokens": genome_tokens,
            "padding_mask": mask,
            "trait_attention": attention_weights,
        }


class GenomeLayout:
    """Convert a flat S2G matrix into ordered chromosome tensors."""

    def __init__(self, path: str | Path, total_tokens: int = 64) -> None:
        layout = json.loads(Path(path).read_text())
        self.n_genes = len(layout["feature_names"])
        self.chromosomes = {}
        counts = {}
        for key, value in layout["chromosomes"].items():
            chromosome = int(key)
            start, end = int(value["start_index"]), int(value["end_index"])
            self.chromosomes[chromosome] = {
                "start": start,
                "end": end,
                "position": torch.tensor(value["position_norm"], dtype=torch.float32),
                "distance": torch.tensor(value["log_distance_z"], dtype=torch.float32),
            }
            counts[chromosome] = end - start
        self.token_allocation = allocate_tokens(counts, total_tokens)

    def make(self, genome: torch.Tensor) -> dict[int, torch.Tensor]:
        result = {}
        for chromosome, value in self.chromosomes.items():
            start, end = value["start"], value["end"]
            position = value["position"].to(genome.device).expand(len(genome), -1)
            distance = value["distance"].to(genome.device).expand(len(genome), -1)
            result[chromosome] = torch.stack([genome[:, start:end], position, distance], dim=-1)
        return result


