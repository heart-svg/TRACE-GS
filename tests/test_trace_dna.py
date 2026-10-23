"""CPU tests for the TRACE-GS stage-1 DNA backbone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from token_allocate import allocate_tokens
from trace_dna import GenomeLayout, TRACEGenomicBackbone


def test_synthetic() -> dict:
    torch.manual_seed(20260903)
    counts = {1: 7, 2: 5, 3: 3, 4: 2, 5: 1}
    allocation = allocate_tokens(counts, 64)
    assert sum(allocation.values()) == sum(counts.values())
    batch, genes = 4, sum(counts.values())
    genome = torch.randn(batch, genes, requires_grad=True)
    chromosomes = {}
    offset = 0
    for chromosome, count in counts.items():
        value = genome[:, offset : offset + count]
        position = torch.linspace(0, 1, count).expand(batch, -1)
        distance = torch.zeros(batch, count)
        chromosomes[chromosome] = torch.stack([value, position, distance], dim=-1)
        offset += count

    model = TRACEGenomicBackbone(genes, total_tokens=64).eval()
    trait_ids = torch.tensor([0, 1, 2, 3])
    output = model(genome, chromosomes, allocation, trait_ids)
    assert output["prediction"].shape == (batch,)
    assert output["tokens"].shape == (batch, 64, 64)
    assert output["padding_mask"].shape == (batch, 64)
    assert output["padding_mask"][:, sum(counts.values()) :].all()
    assert torch.all(output["trait_attention"][output["padding_mask"]] == 0)

    # Replacing another material must not change sample 0.
    with torch.no_grad():
        first = model(genome, chromosomes, allocation, trait_ids)["prediction"][0]
        changed_genome = genome.detach().clone()
        changed_genome[-1] += 100
        changed_chromosomes = {}
        offset = 0
        for chromosome, count in counts.items():
            value = changed_genome[:, offset : offset + count]
            position = torch.linspace(0, 1, count).expand(batch, -1)
            distance = torch.zeros(batch, count)
            changed_chromosomes[chromosome] = torch.stack([value, position, distance], dim=-1)
            offset += count
        second = model(changed_genome, changed_chromosomes, allocation, trait_ids)["prediction"][0]
    batch_delta = float((first - second).abs())
    assert batch_delta < 1e-6

    model.train()
    loss = model(genome, chromosomes, allocation, trait_ids)["prediction"].square().mean()
    loss.backward()
    gradient_parameters = sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters())
    assert gradient_parameters > 0
    return {
        "synthetic_allocation": allocation,
        "synthetic_effective_tokens": sum(allocation.values()),
        "batch_invariance_delta": batch_delta,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_with_gradient": gradient_parameters,
    }


def test_real_layout(path: Path) -> dict:
    layout = GenomeLayout(path, 64)
    torch.manual_seed(20260904)
    genome = torch.randn(2, layout.n_genes)
    model = TRACEGenomicBackbone(layout.n_genes, total_tokens=64).eval()
    with torch.no_grad():
        output = model(genome, layout.make(genome), layout.token_allocation, torch.tensor([0, 1]))
    assert output["tokens"].shape == (2, 64, 64)
    assert not output["padding_mask"].any()
    assert torch.isfinite(output["prediction"]).all()
    return {
        "real_n_genes": layout.n_genes,
        "real_allocation": layout.token_allocation,
        "real_prediction_shape": list(output["prediction"].shape),
        "real_padding_tokens": int(output["padding_mask"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=Path, required=True)
    args = parser.parse_args()
    result = {"synthetic": test_synthetic(), "real": test_real_layout(args.layout)}
    print(json.dumps(result, indent=2))
    print("TRACE_DNA_TESTS_PASSED")


if __name__ == "__main__":
    main()


