#!/usr/bin/env python3
"""Train the TRACE DNA backbone for one structure-CV inner fold."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--phenotype", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--trace-code-dir", required=True)
    p.add_argument("--seed", type=int, default=20261016)
    a = p.parse_args()
    sys.path.insert(0, a.trace_code_dir)
    from trace_dna import GenomeLayout, TRACEGenomicBackbone

    run = Path(a.run)
    trans = json.loads((run / "TRANS_COMPLETE.json").read_text())
    assert trans["status"] == "PASS"
    done = run / "TRACE_COMPLETE.json"
    if done.exists():
        assert json.loads(done.read_text())["status"] == "PASS"
        print(done.read_text())
        return
    out = run / "trace"
    if out.exists() and (out / "best.pt").exists():
        raise RuntimeError(f"Partial TRACE output exists: {out}")
    out.mkdir(exist_ok=True)
    outer, inner = int(trans["outer"]), int(trans["inner"])
    seed = a.seed + 100 * outer + inner
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    spec = json.loads(Path(a.layout).read_text())
    ordered = [str(x).removeprefix("gene:") for x in spec["feature_names"]]
    phenotype = pd.read_csv(a.phenotype, dtype={"ID": str, "vcf_sample_id": str})
    if "ID" in phenotype:
        id_col = "ID"
    elif "vcf_sample_id" in phenotype:
        id_col = "vcf_sample_id"
    else:
        id_col = phenotype.columns[0]
    phenotype[id_col] = phenotype[id_col].astype(str)
    phenotype = phenotype.set_index(id_col)
    raw = {}
    for split in ("trn", "val", "tst"):
        frame = pd.read_parquet(run / f"transf/transformed_genotypes_0_0_{split}.parquet")
        column = "ID" if "ID" in frame else frame.columns[0]
        frame[column] = frame[column].astype(str)
        mapping = {str(x).removeprefix("gene:"): x for x in frame.columns if x != column}
        assert set(mapping) == set(ordered)
        ids = frame[column].tolist()
        x = frame[[mapping[g] for g in ordered]].to_numpy(np.float32)
        y = phenotype.loc[ids, a.trait].to_numpy(np.float32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        raw[split] = (ids, x, y)
    sets = {key: set(value[0]) for key, value in raw.items()}
    assert not (sets["trn"] & sets["val"] or sets["trn"] & sets["tst"] or sets["val"] & sets["tst"])
    mean = float(raw["trn"][2].mean()); std = float(raw["trn"][2].std()); assert std > 0
    loaders = {
        split: DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(((y - mean) / std).astype(np.float32))),
            batch_size=32, shuffle=(split == "trn"), num_workers=0
        ) for split, (_, x, y) in raw.items()
    }
    device = torch.device("cuda")
    layout = GenomeLayout(Path(a.layout), total_tokens=64)
    model = TRACEGenomicBackbone(layout.n_genes, n_traits=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()
    best, patience = float("inf"), 20
    history, start = [], time.time()

    @torch.no_grad()
    def predict(split):
        model.eval(); values = []
        for x, _ in loaders[split]:
            x = x.to(device)
            trait = torch.zeros(len(x), dtype=torch.long, device=device)
            values.append(model(x, layout.make(x), layout.token_allocation, trait)["prediction"].cpu().numpy())
        return np.concatenate(values)

    for epoch in range(1, 151):
        model.train(); total = 0.0; count = 0
        for x, y in loaders["trn"]:
            x, y = x.to(device), y.to(device)
            trait = torch.zeros(len(x), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, layout.make(x), layout.token_allocation, trait)["prediction"]
            loss = loss_fn(prediction, y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(y); count += len(y)
        validation = predict("val")
        y_validation = (raw["val"][2] - mean) / std
        mse = float(np.mean((y_validation - validation) ** 2))
        history.append({"epoch": epoch, "train_smooth_l1": total / count, "val_mse": mse})
        if mse < best - 1e-6:
            best, patience = mse, 20
            torch.save(model.state_dict(), out / "best.pt")
        else:
            patience -= 1
        if patience == 0:
            break
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    for split in ("trn", "val", "tst"):
        ids, _, y = raw[split]
        prediction = predict(split) * std + mean
        pd.DataFrame({"ID": ids, "y_raw": y, "pred_raw": prediction}).to_csv(
            out / f"predictions_{split}.csv", index=False
        )
    record = {
        "status": "PASS", "trait": a.trait, "outer": outer, "inner": inner,
        "counts": {key: len(value[0]) for key, value in raw.items()},
        "best_val_mse": best, "epochs": len(history), "seconds": time.time() - start,
        "n_holdout_predictions": len(raw["tst"][0]),
        "parameters": sum(x.numel() for x in model.parameters()),
        "note": "Inner-holdout truth was not used for fitting or early stopping."
    }
    done.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()

