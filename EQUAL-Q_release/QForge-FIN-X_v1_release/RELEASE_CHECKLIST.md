# RELEASE CHECKLIST

Author: Hossein Tabasi

## Files
- [x] Dataset panel/oracle/splits/metadata/codebook/preview/market
- [x] quality_gates.txt PASS before models
- [x] preregistration.md before models
- [x] experiment_equalq.py + generate_qforge_fin_x.py
- [x] results_*.csv + results_equalq.md + paper_draft_equalq.md
- [x] figures fig1-fig4
- [x] ENVIRONMENT.txt, LICENSE, README, frozen_hps.json
- [x] tests leakage + seed repro
- [x] root README, LICENSE, .gitignore, requirements.txt, reports/RESULTS.md

## H0
- [x] H0 rejected for quantum outperformance? NO (not rejected)

## Referee-quotable sentences
1. After equalizing n, Pack-4 feature budget, and walk-forward folds, neither ZZ-quantum kernels nor AngleEmbedding+SEL VQCs beat matched RBF/linear models on AUC or Spearman IC at Holm alpha=0.05.
2. Representation geometry (CKA about 0.74 for QK-RBF; lower KTA for QK) changes without delivering portfolio Sharpe gains after 5 bps.
3. On this NISQ budget, the honest equal-budget conclusion is a null quantum edge in cross-sectional return prediction.
