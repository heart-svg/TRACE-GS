# Reproduction entry points

## Immediate, data-independent audit

```bash
python src/data_audit/validate_release.py
python src/data_audit/derive_random_split_manifests.py
```

## Figure regeneration

Run the R scripts under `src/figures/` after setting their input root to `results/figure_source_data/`. Each manuscript figure has a matching source-data CSV. The Python GO/GSEA preparation script must be run before the updated Figure 4 scripts when enrichment outputs change.

## Model refitting

Full refitting requires the third-party raw data listed in `docs/THIRD_PARTY_DATA.md`. Historical server paths in the publication snapshot must be replaced with local paths. The exact fold identities in `splits/` must not be regenerated with different seeds when reproducing the published values.

## Baselines

Use `src/baselines/FORMAL_MODEL_LOCK.json` and `docs/Supplementary_Table_S3_baseline_implementations.xlsx` to select the corresponding entry point and native input representation. All models start from the same materials, QC-filtered SNP information, and frozen outer folds; model-specific encoding is fitted on training data only.

