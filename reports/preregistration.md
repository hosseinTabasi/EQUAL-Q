# EQUAL-Q Preregistration

**Study:** Equal-Budget Anatomy of Quantum Representations in Cross-Sectional Finance  
**Benchmark:** QForge-FIN-X v1.0 (superset of QForge-FIN v1.0)  
**Author:** Hossein Tabasi  
**Master seed (data):** 20260907  
**Date locked (UTC):** 2026-09-07  
**Status:** Locked BEFORE model fitting. Hypotheses below are confirmatory; deviations must be labeled exploratory.

## Claim under test

After equalizing sample size \(n\), feature budget (4-qubit / Pack-4 geometry), and walk-forward protocol, any NISQ quantum kernel / VQC edge over classical RBF / linear models in cross-sectional return prediction is absent, confined to stress / small-\(n\), or a geometry effect that fails portfolio and calibration checks.

## Design locked

| Item | Choice |
|------|--------|
| Backend | PennyLane `default.qubit`, 4 qubits; numpy analytic fallback documented if needed |
| Feature budget | Pack-4 primary; PCA(4) on Pack-8 allowed only as robustness (exploratory) |
| Models | LIN (logistic / ridge), RBF (SVC / KernelRidge), QK (ZZFeatureMap reps=2 + SVM/KRR), VQC (AngleEmbedding + 2 StronglyEntanglingLayers + classical head) |
| Train sizes | \(n \in \{150, 400\}\) equal-budget; LIN/RBF **full-\(n\)** as REFERENCE only (not in H0 family) |
| Sampling | Same stratified indices across representations within (window, seed, \(n\)) |
| Windows | 8 walk-forward expanding windows (63bd test, step 63); no time shuffle |
| Seeds | 0, 1, 2 |
| HP | Freeze on window-0 only; reuse frozen HP for windows 1–7 |
| Scaler / PCA | Fit on train fold only |
| Costs (book) | 5 bps per rebalance (primary); 0 bps reported as diagnostic |

## Endpoints

- **T-class:** `y_dir_5d` — ROC-AUC (primary classification), also report Brier
- **T-reg:** `y_exret_5d` — Spearman IC (primary regression)
- **T-book:** long-short top-3 / bottom-3, 5-day hold, \(n=400\) — net Sharpe after 5 bps
- **Geometry (H3):** CKA and KTA between kernels; sector nearest-neighbor purity

## Hypotheses (confirmatory)

### H0 (primary)

QK and VQC do **not** outperform equal-\(n\) RBF and LIN on Spearman IC (T-reg) or AUC (T-class) at family-wise error rate \(\alpha = 0.05\).

**Test:** Paired tests across (window × seed) cells. Compare QK vs RBF and VQC vs LIN (and cross pairs as sensitivity) at matched \(n \in \{150,400\}\). Multiplicity: Holm–Bonferroni over the pre-specified primary contrasts (4 contrasts: QK–RBF and VQC–LIN × {AUC, Spearman IC} at \(n=400\); \(n=150\) reported but secondary to H1).

**Decision rule:** Reject H0 only if at least one primary contrast shows quantum **better** than classical with Holm-adjusted \(p < 0.05\). Higher classical performance that is significant does **not** reject H0 in favor of quantum advantage (H0 is “no quantum outperformance”).

### H1 (secondary)

The IC gap (QK − RBF) differs at \(n \in \{150, 400\}\) versus full-\(n\) RBF reference (interaction / gap comparison).

### H2 (secondary)

On Stress/Crash days (causal regime label from oracle, evaluation-only), quantum models are relatively better vs Calm than RBF/ridge (regime × model interaction on AUC or IC).

### H3 (secondary)

Geometry metrics (CKA/KTA; sector NN purity) can differ between quantum and classical kernels even if H0 is not rejected.

### H4 (secondary)

Long-short book: quantum does **not** beat RBF/ridge on net Sharpe after 5 bps at \(n=400\).

## Framing findings F1–F5 (interpretive; not hypotheses)

- **F1:** Fragile advantage — any apparent edge is sensitive to \(n\), window, or seed.
- **F2:** Classification is kinder than regression for shallow quantum models.
- **F3:** Geometry can move (CKA/KTA/purity) without accuracy gains.
- **F4:** Simulator ≠ hardware — `default.qubit` results do not imply device advantage.
- **F5:** Reuse QForge-FIN v1 — EQUAL-Q is an equal-budget anatomy on the extended panel, not a new data claim.

## Exclusions / integrity

- Oracle columns (`regime`, macros, PIT probs) never enter training features.
- No hyperparameter search after seeing test windows 1–7.
- Prefer NULL / no-advantage wording when earned; do not invent metrics.

## Analysis code

`experiment_equalq.py` in this release. Results tables: `results_tclass.csv`, `results_treg.csv`, `results_tbook.csv`, `results_geometry.csv`.
