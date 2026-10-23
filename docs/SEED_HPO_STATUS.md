# Seed and HPO export status

The frozen archive records the inferential-analysis seed (`results/statistics/analysis_config.json`), the locked comparator list, and the HPO spaces encoded in the frozen comparator entry scripts (`config/model_hpo_spaces.yml`). The HPO-space file is an implementation snapshot and is not yet claimed to be identity-matched to every reported OOF result. The archive also contains all 35 outer-fold TRACE DNA records for the seven reported traits (`config/trait_fold_seeds.csv`). Each row distinguishes values written directly in the metric record from seeds reconstructed from the archived submission/training script.

Where available, selected fold-specific comparator configurations may additionally be exported to:

- `config/final_selected_hyperparameters.csv`

The manuscript may state that the repository contains the seven-trait TRACE seed table and comparator HPO spaces. It must not claim that a complete fold-specific selected-configuration table exists unless `config/final_selected_hyperparameters.csv` is added and identity-audited against the frozen OOF predictions.

