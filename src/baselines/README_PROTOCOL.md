# TRACE-GS unified benchmark protocol (v2)

This directory is an adapter layer. It never edits the downloaded upstream
repositories. Formal comparisons must satisfy all of the following rules.

1. Reuse the exact five outer folds already used by TRACE-GS. Never generate a
   new `KFold(seed=42)` split.
2. Join every table by material ID. Positional joins are forbidden.
3. For the representation-controlled benchmark, each outer fold consumes only
   the S2G representation trained for that outer fold. This prevents an S2G
   encoder trained with outer-test phenotypes from entering a baseline.
4. Hyperparameters are selected with that outer fold's validation materials.
   The outer test labels are evaluated once and never select features, epochs,
   or hyperparameters.
5. Every run writes predictions, metrics, the selected hyperparameters, and an
   auditable completion marker under `results/<species>/<trait>/<model>/`.
6. DNA-only baselines are not described as DNA+RNA methods. The multi-omics
   table is restricted to TRACE-GS, DEM, and DEM-Fix.

## Main-table models

- Ridge/S2G-Ridge (linear reference)
- LASSO or Elastic Net (sparse linear reference)
- SVR (kernel reference)
- XGBoost or LightGBM (one tree ensemble; avoid redundant headline models)
- rrBLUP/GBLUP and BayesB/C (after the R environment is installed)
- LCNN and leakage-corrected Cropformer (deep references)
- DEM, DEM-Fix, TRACE-GS

DeepGS and the custom low-rank gMLP are supplementary until their dependency
and implementation equivalence audits pass.

## Required artifacts

`build_manifest.py` converts existing TRACE prediction splits into a locked
ID-level manifest. `run_s2g_baselines.py` consumes fold-specific S2G parquet
files and the locked manifest. Both abort on duplicate IDs, overlap, missing
samples, inconsistent phenotypes, or a test fold that is not unique.
# Locked benchmark execution layout

The comparison set is frozen before inspecting formal benchmark results:
rrBLUP, BayesA, BayesB, BayesC, LASSO, SVR, XGBoost, DeepGS, LCNN,
Cropformer, DEM, DEM-Fix, and TRACE-GS. A method must not be removed because
it outperforms TRACE-GS.

All methods use the same outer-five-fold manifest, fold-specific S2G features,
phenotype scale, and test IDs. Hyperparameter selection and early stopping use
training/validation data only. Test labels are used exactly once for reporting.

Execution is separated by architecture, not by data protocol:

- Spark/aarch64: DeepGS, LCNN, Cropformer, DEM, DEM-Fix, TRACE-GS.
- Seed/x86_64: XGBoost and the R models (rrBLUP and BGLR BayesA/B/C).
- LASSO and SVR can run on either host and already share the common runner.

For Arabidopsis R/XGBoost runs, `export_r_inputs.py` exports the locked
fold-specific matrices by ID. Those immutable exports may be copied to the
x86_64 host. This changes only the execution host, not samples or features.

DeepGS, LCNN, and Cropformer are audited S2G-input adaptations, not claims of
bit-for-bit execution of the original raw-SNP pipelines. The manuscript must
label them accordingly and cite the original methods.


