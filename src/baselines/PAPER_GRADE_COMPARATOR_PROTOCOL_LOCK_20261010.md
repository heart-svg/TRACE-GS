# TRACE-GS paper-grade comparator protocol lock

Locked on: 2026-10-10

## Scientific objective

The purpose of this benchmark is to compare TRACE-GS with representative
statistical, machine-learning and deep-learning genomic-prediction methods.
It is not to force every method to consume the same derived feature matrix.

## Common starting point (mandatory)

For every trait and species, all models must start from:

1. the same eligible individuals;
2. the same phenotype values after the same sample-level quality control;
3. the same post-QC SNP universe;
4. the same locked five outer folds, with exactly the same outer-test IDs;
5. the same random-seed and HPO-budget policy at the comparison level.

No model may silently exclude difficult samples or use a different test split.

## Model-native processing (mandatory)

After the common starting point, each comparator must follow its original
paper, author repository, or official package documentation for:

- genotype encoding;
- marker filtering or feature selection;
- scaling and imputation;
- architecture and loss;
- optimization, validation and early stopping.

Examples include relationship-matrix/marker encodings for rrBLUP and BGLR,
native matrix input for LASSO/SVR/XGBoost, and paper-defined fixed-length or
sequence encodings for DeepGS, LCNN and Cropformer.

## Leakage barrier (mandatory)

Any phenotype-dependent or distribution-fitted operation must be fitted using
training data only. This includes MIC/association-based SNP selection,
imputation statistics, scaling, PCA, hyperparameter selection and checkpoint
selection. The fitted transformation is then applied unchanged to validation
and outer-test samples. Outer-test phenotypes may be accessed only once after
the full model-selection procedure has been frozen.

Unsupervised population-level filtering that is explicitly part of the common
genotype QC must be applied once and identically to all methods. If an original
method additionally requires LD pruning or a fixed marker count, that operation
must be recorded as model-native preprocessing and must not alter sample IDs or
outer folds.

## Evidence required before a result enters the manuscript

Each model must have:

1. a verified primary paper and official source/package when available;
2. a documented mapping from the original method to the executable runner;
3. an input audit (samples, SNP universe, encoding, fold hashes);
4. per-fold predictions containing material ID, observed value and prediction;
5. PCC, RMSE and MAE for every fold and pooled OOF predictions;
6. HPO/early-stopping logs and proof that the outer test was not used for tuning;
7. parameter count, runtime and peak-resource records under the stated hardware;
8. an explicit label if a compatibility-only patch was required.

Simplified substitutes, S2G adapters that are not part of the original method,
and development-only results are excluded from the formal comparison table.

## Final comparison

All accepted comparator predictions are evaluated on the identical locked
outer-test individuals and compared with TRACE-GS using the same metrics and
paired OOF statistical analyses. Differences in model-native feature processing
are reported transparently as part of the methods, not hidden as a common input.

