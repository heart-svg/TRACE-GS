"""Allocate a fixed genome-wide token budget across non-empty chromosomes."""
from __future__ import annotations


def allocate_tokens(gene_counts: dict[int, int], total_tokens: int = 64) -> dict[int, int]:
    active = {int(c): int(n) for c, n in gene_counts.items() if int(n) > 0}
    if not active:
        raise ValueError("At least one non-empty chromosome is required")
    if len(active) > total_tokens:
        raise ValueError("Token budget must be at least the number of chromosomes")

    effective = min(total_tokens, sum(active.values()))
    allocation = {c: 1 for c in active}
    remaining = effective - len(active)
    capacity = {c: active[c] - 1 for c in active}

    if remaining:
        total_genes = sum(active.values())
        raw = {c: remaining * active[c] / total_genes for c in active}
        for c in active:
            add = min(capacity[c], int(raw[c]))
            allocation[c] += add
            remaining -= add
        order = sorted(active, key=lambda c: (raw[c] - int(raw[c]), active[c]), reverse=True)
        while remaining:
            progressed = False
            for c in order:
                if allocation[c] < active[c] and remaining:
                    allocation[c] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("Unable to allocate the requested token budget")

    assert sum(allocation.values()) == effective
    assert all(1 <= allocation[c] <= active[c] for c in active)
    return dict(sorted(allocation.items()))


