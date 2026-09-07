# EQUAL-Q

**Equal-Budget Anatomy of Quantum Representations in Cross-Sectional Finance**

**Author:** Hossein Tabasi
**Benchmark:** QForge-FIN-X v1.0 (superset of QForge-FIN v1.0)
**Licenses:** Code MIT; Data CC-BY 4.0
**Master seed:** 20260907

## Claim

After equalizing n, Pack-4 / 4-qubit feature budget, and walk-forward protocol, any NISQ quantum kernel/VQC edge over classical RBF/linear models in cross-sectional return prediction is absent, confined to stress/small-n, or a geometry effect that fails portfolio checks.

## Release bundle

See EQUAL-Q_release/QForge-FIN-X_v1_release/.

## Quick start

    pip install -r requirements.txt
    python scripts/generate_qforge_fin_x.py
    python scripts/experiment_equalq.py
    pytest -q

## Finding (honest)

H0 not rejected. Equal-n LIN/RBF match or beat QK/VQC on AUC and Spearman IC (Holm alpha=0.05). Geometry (CKA/KTA) can move without accuracy or net Sharpe gains after 5 bps.

## Not investment advice

100% synthetic data. No market APIs. Not for trading.
