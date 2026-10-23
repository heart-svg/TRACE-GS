# TRACE-GS reproducibility archive

This repository is the frozen code-and-results companion for the TRACE-GS manuscript. TRACE-GS treats DNA as the primary predictor and admits RNA only as a bounded correction learned from strictly cross-fitted DNA residuals.

## What is included

- `src/trace_gs/`: DNA backbone, fold-specific S2G reference implementation, strict residual RNA expert, and structure-aware variants.
- `src/baselines/`: formal entry points and provenance records for the comparator models. Third-party repositories are referenced rather than redistributed.
- `splits/`: exact outer-test assignments for random, structure-grouped, and stickleback family-grouped evaluations. Inner training roles are included for the structure-grouped analyses.
- `results/oof/`: per-sample observed values and OOF predictions for all formal models.
- `results/statistics/`: paired bootstrap/randomization results and their locked configuration.
- `src/figures/` and `results/figure_source_data/`: scripts and source tables for manuscript figures.

## Reproducibility boundary

Raw genotype, transcriptome, and phenotype files are not redistributed because they originate from third-party studies and remain subject to the original providers' terms. `docs/THIRD_PARTY_DATA.md` must be used to obtain them. All phenotype-dependent preprocessing, feature selection, residual construction, and calibration must be fitted within the corresponding training folds.

Several scripts under `src/trace_gs/` are preserved from the server-side publication snapshot. Private absolute roots have been replaced by public path tokens; configure local dataset roots through `config/datasets.example.yml` before execution. The frozen OOF tables and manifests can be audited without downloading the raw datasets.

## Quick integrity checks

```bash
python src/data_audit/derive_random_split_manifests.py
python src/data_audit/validate_release.py
python tests/run_smoke_tests.py
```

## Environment

The primary environment is documented in `environment.yml` and `requirements.txt`. The manuscript environment used Python 3.10.21, PyTorch 2.12.0.dev20260408+cu128, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.2, scikit-learn 1.7.2, PyArrow 25.0.1, R 4.4.3, BGLR 1.1.4, rrBLUP 4.6.3, and XGBoost 3.2.0.

## Citation and version

Version: 1.0.1. The public source repository is available at https://github.com/heart-svg/TRACE-GS. The frozen archival release is available at https://doi.org/10.5281/zenodo.23584355. Cite the archive using `CITATION.cff` and this version-specific DOI.

## License

TRACE-GS software is released under the MIT License. Third-party biological datasets are not covered by this software license and remain subject to their source repositories' terms.

