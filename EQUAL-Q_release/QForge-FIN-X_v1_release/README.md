# QForge-FIN-X v1.0 / EQUAL-Q release

**Author:** Hossein Tabasi
**Citation:** QForge-FIN-X v1.0 (2026), equal-budget quantum vs classical finance panel (superset of QForge-FIN v1.0).
**Licenses:** Code MIT; Data CC-BY 4.0
**Master seed:** 20260907

## H0 headline

Not rejected. No quantum outperformance vs equal-n RBF/LIN on AUC or Spearman IC at Holm alpha=0.05.

## Key files

- Panel: qforge_fin_x_v1_panel.parquet
- Oracle: qforge_fin_x_v1_oracle_factors.csv (PIT probs hidden here)
- quality_gates.txt (before models)
- preregistration.md (before models)
- experiment_equalq.py / generate_qforge_fin_x.py
- results_tclass.csv, results_treg.csv, results_tbook.csv, results_geometry.csv
- results_equalq.md, paper_draft_equalq.md, hypothesis_eval.txt
- figures/fig1_tclass_auc.png ... fig4_geometry.png
- ENVIRONMENT.txt, zz_kernel_verification.txt, frozen_hps.json

## Referee-quotable sentences

1. After equalizing n, Pack-4 feature budget, and walk-forward folds, neither ZZ-quantum kernels nor AngleEmbedding+SEL VQCs beat matched RBF/linear models on AUC or Spearman IC at Holm alpha=0.05.
2. Representation geometry (CKA about 0.74 for QK-RBF; lower KTA for QK) changes without delivering portfolio Sharpe gains after 5 bps.
3. On this NISQ budget, the honest equal-budget conclusion is a null quantum edge in cross-sectional return prediction.
