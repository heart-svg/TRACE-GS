# Reproducibility file map

| Manuscript component | Code | Frozen input/output |
|---|---|---|
| TRACE-GS DNA backbone | `src/trace_gs/trace_dna.py` | fold manifests under `splits/` |
| Strict RNA residual expert and bounded fusion | `src/trace_gs/run_strict_rna_residual_fivefold_reference.py` | `results/oof/seven_trait_formal_oof_predictions_long.csv` |
| Structure-grouped sensitivity analysis | `src/trace_gs/train_structure_trace_inner.py`, `run_structure_rna_fusion.py` | `splits/structure_grouped/` |
| Stickleback family-grouped analysis | `src/data_audit/prepare_formal_stickleback.py`, `src/trace_gs/run_stickleback_strict_rna_fusion.py` | `splits/stickleback_family_grouped/` |
| Statistical and machine-learning baselines | `src/baselines/run_r_gs_formal.R`, `run_s2g_baselines.py` | `docs/Supplementary_Table_S3_baseline_implementations.xlsx` |
| Deep-learning baselines | `src/baselines/run_deep_s2g.py` | `docs/Supplementary_Table_S3_baseline_implementations.xlsx` |
| Paired bootstrap and randomization | `src/statistics/paired_oof_statistics.py` | `results/statistics/` |
| RNA stability and GO/GSEA | `src/statistics/`, `src/figures/GO_GSEA语义去冗余与绘图数据准备.py` | `results/figure_source_data/` |
| Figures 2–6 and GSEA supplement | `src/figures/` | `results/figure_source_data/` |

Figure 1 is a conceptual architecture diagram and is not generated from numerical data.

