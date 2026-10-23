# GS comparator source and input audit (2026-10-10)

## Decision

All previous outputs produced by `run_deep_s2g.py` for DeepGS, LCNN and
Cropformer are development-only.  That runner uses custom PyTorch models and
S2G features, whereas all three source methods were defined for marker-level
genotype inputs.  They must not be reported under the original model names in
the formal manuscript table.

## Locked implementation and input policy

| Model | Authoritative implementation | Native input used in the formal rerun | Status of old result |
|---|---|---|---|
| rrBLUP | `rrBLUP` R package, `mixed.solve`/documented rrBLUP interface | sample × SNP dosage matrix (0/1/2), same QC marker panel and outer folds | rerun/audit on native SNP input |
| BayesA/B/C | `BGLR` R package (`model="BayesA"`, `"BayesB"`, `"BayesC"`) | sample × SNP dosage matrix (0/1/2), same QC marker panel and outer folds | rerun/audit on native SNP input |
| LASSO | official scikit-learn `Lasso`/`LassoCV` | sample × SNP dosage matrix; imputation, scaling and alpha selection fitted only on training data | S2G result excluded; native-SNP rerun required |
| SVR | official scikit-learn `SVR` (LIBSVM) | sample × SNP dosage matrix; imputation/scaling and C/gamma/epsilon selection fitted only on training data | S2G result excluded; native-SNP rerun required |
| XGBoost | official `xgboost.XGBRegressor` | sample × SNP dosage matrix; regression objective; all tuning confined to inner validation data | S2G result excluded; native-SNP rerun required |
| DeepGS | author R package `cma2015/DeepGS`, version 1.2 | ordered genome-wide marker matrix, one row per sample; official CNN/pooling/dense architecture | custom PyTorch/S2G result excluded |
| LCNN | Pook et al. supplementary Keras scripts (Files S2/S3) | ordered 0/1/2 SNP matrix; phenotype centered using training data | custom pooling/S2G result excluded |
| Cropformer | author repository `jiekesen/Cropformer` | PLINK PED-derived marker representation converted by official `ped_to_matrix.py`; official 0–9/one-hot processing | custom S2G adapter excluded |

## Source records

- rrBLUP: Endelman JB (2011), *The Plant Genome*, DOI
  `10.3835/plantgenome2011.08.0024`; official CRAN package `rrBLUP`.
- BayesA/B/C: Pérez P, de los Campos G (2014), *Genetics*, DOI
  `10.1534/genetics.114.164442`; official package/repository `BGLR`.
- LASSO: Tibshirani (1996); official implementation used here is
  `sklearn.linear_model.Lasso`/`LassoCV`.
- SVR: LIBSVM implementation exposed by official `sklearn.svm.SVR`.
- XGBoost: Chen and Guestrin (2016), DOI `10.1145/2939672.2939785`;
  official `dmlc/xgboost` package.
- DeepGS: Ma et al. (2018), *Planta* 248:1307–1318, DOI
  `10.1007/s00425-018-2976-9`; author repository `cma2015/DeepGS`.
- LCNN: Pook et al. (2020), *Frontiers in Genetics* 11:561497, DOI
  `10.3389/fgene.2020.561497`; scripts in article Supplementary Files 2/3.
- Cropformer: *Plant Communications* (2025) 6:101223, DOI
  `10.1016/j.xplc.2024.101223`; author repository `jiekesen/Cropformer`.

## Fairness constraints

1. The sample IDs and five outer test folds are identical across all models
   for a given trait.
2. The same post-QC SNP universe is the starting information for every
   SNP-based comparator.  A model-specific encoding is allowed only when the
   source method requires it.
3. Missing-value imputation, allele-frequency statistics, scaling, LD pruning,
   feature selection and hyperparameter selection are fitted without access to
   the outer test fold.
4. S2G is not substituted for SNP input in DeepGS, LCNN or Cropformer.
5. Epoch counts need not be identical.  Deep models use source-compatible
   maximum epochs plus validation-only early stopping; tuning/search budgets
   and seeds are reported explicitly.
6. Each output must include per-fold predictions, PCC/RMSE/MAE, learning curves
   for neural models, runtime, peak memory/GPU memory, package/repository
   version and source commit/checksum.
7. Predictive performance is never used as a reason to remove a preregistered
   comparator.

## Immediate execution order

1. Freeze checksums/commits for the official implementations.
2. Build one audited sample-by-SNP matrix and fold manifest per trait.
3. Run rrBLUP/BayesA/B/C, LASSO, SVR and XGBoost.
4. Run the unmodified author DeepGS package with only an external fold/data
   adapter.
5. Run Pook LCNN from supplementary code; if legacy Keras is incompatible,
   perform a line-by-line faithful port and label it as a faithful port rather
   than official binary execution.
6. Run the author Cropformer repository, retaining its preprocessing and model
   code while replacing hard-coded paths/splits only.
7. Audit all OOF files before merging them with DEM, DEM-Fix and TRACE-GS.
# Protocol precedence

The paper-grade comparison is governed by
`PAPER_GRADE_COMPARATOR_PROTOCOL_LOCK_20261010.md`. All models start from the
same samples, post-QC SNP universe and locked five folds, then use the input
encoding and preprocessing required by their original paper or official
software. Any supervised processing is fit within the training fold only.


