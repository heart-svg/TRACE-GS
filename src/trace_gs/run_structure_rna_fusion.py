#!/usr/bin/env python3
"""Strict structure-grouped RNA residual fusion for one maize trait."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
for k in ("trait", "inner-root", "outer-root", "rna", "phenotype", "output"):
    p.add_argument("--" + k, required=True)
p.add_argument("--seed", type=int, default=20261017)
p.add_argument("--n-resamples", type=int, default=1000)
a = p.parse_args()
inner_root, outer_root, out = Path(a.inner_root), Path(a.outer_root), Path(a.output)
out.mkdir(parents=True, exist_ok=False)
PCA_DIMS = (8, 16, 32, 64)
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(pearsonr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else float("nan")

def project(xf, xh, q, seed):
    sc = StandardScaler().fit(xf)
    af, ah = sc.transform(xf), sc.transform(xh)
    n = min(q, len(xf) - 1, af.shape[1])
    pc = PCA(n_components=n, svd_solver="randomized", random_state=seed).fit(af)
    return pc.transform(af), pc.transform(ah)

def grouped_crossfit(x, residual, fold, q, alpha, seed):
    pred = np.empty(len(residual), float)
    for k in sorted(np.unique(fold)):
        hold = np.flatnonzero(fold == k); fit = np.flatnonzero(fold != k)
        zf, zh = project(x[fit], x[hold], q, seed + int(k))
        pred[hold] = Ridge(alpha=alpha).fit(zf, residual[fit]).predict(zh)
    return pred

rna = pd.read_parquet(a.rna)
if "ID" not in rna.columns and "vcf_sample_id" not in rna.columns:
    rna = rna.reset_index()
if "ID" not in rna.columns:
    rna = rna.rename(columns={"vcf_sample_id": "ID"})
rna["ID"] = rna["ID"].astype(str)
assert not rna.ID.duplicated().any()
rna = rna.set_index("ID")
assert np.isfinite(rna.to_numpy(float)).all()
ph = pd.read_csv(a.phenotype, dtype={"ID": str, "vcf_sample_id": str})
if "ID" not in ph.columns:
    ph = ph.rename(columns={"vcf_sample_id": "ID"})
ph["ID"] = ph["ID"].astype(str)
ph = ph.set_index("ID")
assert a.trait in ph.columns

reports, predictions = [], []
for outer in range(5):
    parts = []
    for inner in range(3):
        q = inner_root / f"outer_{outer}/inner_{inner}/trace/predictions_tst.csv"
        d = pd.read_csv(q, dtype={"ID": str})
        assert list(d.columns) == ["ID", "y_raw", "pred_raw"]
        d["inner_holdout"] = inner
        parts.append(d)
    oof = pd.concat(parts, ignore_index=True)
    assert not oof.ID.duplicated().any()
    oof["residual_raw"] = oof.y_raw - oof.pred_raw
    tst = pd.read_csv(outer_root / f"outer_{outer}/trace/predictions_tst.csv", dtype={"ID": str})
    assert list(tst.columns) == ["ID", "y_raw", "pred_raw"] and not tst.ID.duplicated().any()
    assert not set(oof.ID) & set(tst.ID)
    expected = set(ph.index)
    assert set(oof.ID) | set(tst.ID) == expected
    assert expected <= set(rna.index)

    x = rna.loc[oof.ID].to_numpy(np.float32)
    xt = rna.loc[tst.ID].to_numpy(np.float32)
    res = oof.residual_raw.to_numpy(float)
    fold = oof.inner_holdout.to_numpy(int)
    scores = {}
    for q in PCA_DIMS:
        for alpha in ALPHAS:
            cp = grouped_crossfit(x, res, fold, q, alpha, a.seed + outer * 100)
            scores[(q, alpha)] = mean_squared_error(res, cp)
    q, alpha = min(scores, key=scores.get)
    residual_oof_pred = grouped_crossfit(x, res, fold, q, alpha, a.seed + outer * 1000)

    scaler = StandardScaler().fit(x)
    xs, xts = scaler.transform(x), scaler.transform(xt)
    pca = PCA(n_components=min(q, len(x) - 1, xs.shape[1]), svd_solver="randomized",
              random_state=a.seed + outer).fit(xs)
    model = Ridge(alpha=alpha).fit(pca.transform(xs), res)
    residual_test_pred = model.predict(pca.transform(xts))
    gammas = np.linspace(0.0, 1.5, 151)
    gamma = float(min(gammas, key=lambda g: mean_squared_error(
        oof.y_raw, oof.pred_raw + g * residual_oof_pred)))

    y, dna = tst.y_raw.to_numpy(float), tst.pred_raw.to_numpy(float)
    fused = dna + gamma * residual_test_pred
    dpc, fpc = corr(y, dna), corr(y, fused)
    rng = np.random.default_rng(a.seed + outer)
    null = np.asarray([corr(y, dna + gamma * rng.permutation(residual_test_pred))
                       for _ in range(a.n_resamples)])
    rec = {
        "outer": outer, "n_oof": len(oof), "n_test": len(tst),
        "pca_components": int(q), "ridge_alpha": float(alpha), "gamma": gamma,
        "rna_oof_residual_pcc": corr(res, residual_oof_pred),
        "test_residual_pcc": corr(y - dna, residual_test_pred),
        "dna_PCC": dpc, "fusion_PCC": fpc, "PCC_gain": fpc - dpc,
        "dna_RMSE": float(np.sqrt(mean_squared_error(y, dna))),
        "fusion_RMSE": float(np.sqrt(mean_squared_error(y, fused))),
        "dna_MAE": float(mean_absolute_error(y, dna)),
        "fusion_MAE": float(mean_absolute_error(y, fused)),
        "test_increment_permutation_p_one_sided": float((1 + np.sum(null >= fpc)) / (a.n_resamples + 1)),
    }
    reports.append(rec)
    pred = pd.DataFrame({"ID": tst.ID, "true_raw": y, "dna_pred_raw": dna,
                         "rna_residual_pred_raw": residual_test_pred,
                         "fusion_pred_raw": fused, "outer": outer})
    pred.to_csv(out / f"outer_{outer}_predictions.csv", index=False)
    (out / f"outer_{outer}_metrics.json").write_text(json.dumps(rec, indent=2))
    predictions.append(pred)
    print("OUTER_PASS", json.dumps(rec), flush=True)

tab = pd.DataFrame(reports)
pooled = pd.concat(predictions, ignore_index=True)
tab.to_csv(out / "fivefold_metrics.csv", index=False)
pooled.to_csv(out / "predictions_oof.csv", index=False)
pooled_dna = corr(pooled.true_raw, pooled.dna_pred_raw)
pooled_fusion = corr(pooled.true_raw, pooled.fusion_pred_raw)

rng = np.random.default_rng(a.seed + 9999)
n = len(pooled); boot = []
for _ in range(a.n_resamples):
    ix = rng.integers(0, n, n)
    boot.append(corr(pooled.true_raw.iloc[ix], pooled.fusion_pred_raw.iloc[ix]) -
                corr(pooled.true_raw.iloc[ix], pooled.dna_pred_raw.iloc[ix]))
lo, hi = np.nanquantile(boot, [0.025, 0.975])
summary = {
    "status": "PASS", "trait": a.trait, "protocol": "structure-grouped 5x3 nested CV",
    "folds": 5, "per_fold": reports,
    "mean_fold_dna_PCC": float(tab.dna_PCC.mean()),
    "mean_fold_fusion_PCC": float(tab.fusion_PCC.mean()),
    "mean_fold_PCC_gain": float(tab.PCC_gain.mean()),
    "positive_gain_folds": int((tab.PCC_gain > 0).sum()),
    "pooled_dna_PCC": pooled_dna, "pooled_fusion_PCC": pooled_fusion,
    "pooled_PCC_gain": pooled_fusion - pooled_dna,
    "pooled_gain_bootstrap_95CI": [float(lo), float(hi)],
    "n_bootstrap": a.n_resamples,
    "outer_test_used_for_selection": False,
}
(out / "COMPLETE.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

