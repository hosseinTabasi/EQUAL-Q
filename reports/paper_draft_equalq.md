# Equal-Budget Anatomy of Quantum Representations in Cross-Sectional Finance

**EQUAL-Q on QForge-FIN-X v1.0**

**Author:** Hossein Tabasi  
**Affiliation:** Independent research  
**Data and code licenses:** Data CC-BY 4.0; Code MIT  
**Benchmark citation:** QForge-FIN-X v1.0 (2026), equal-budget quantum vs classical finance panel (superset of QForge-FIN v1.0).  
**Master seed:** 20260907  

## Abstract

Claims of NISQ quantum edges in financial prediction often confound representation geometry with unequal sample size, feature budget, or walk-forward hygiene. EQUAL-Q tests a pre-registered null: after equalizing n, a 4-qubit Pack-4 feature budget, and expanding walk-forward windows on the synthetic QForge-FIN-X panel, quantum kernels (ZZ-style, reps=2) and variational circuits (AngleEmbedding plus two StronglyEntanglingLayers) do not outperform matched linear and RBF baselines on Spearman information coefficient or ROC-AUC at family-wise alpha=0.05. Across eight windows and three seeds, H0 is not rejected. Geometry metrics (CKA, KTA, sector nearest-neighbor purity) do move, yet long-short books after 5 bps costs show no quantum Sharpe edge. The study frames five interpretive findings (F1-F5): fragile edge, classification kinder than regression, geometry without accuracy, simulator-not-hardware, and deliberate reuse of QForge-FIN v1.

## 1. Introduction

Cross-sectional return prediction is a stress test for representation learning: labels are noisy, regimes shift, and small-n protocols invite overfit. Quantum machine learning proposals for finance frequently report kernel or variational edges under budgets that classical models do not share—more features, different n, or leakage-prone splits. EQUAL-Q asks a narrower, referee-facing question: when budgets are equalized, does any NISQ quantum edge remain?

We pre-register H0-H4 before fitting (preregistration.md). The primary claim under test is that any quantum kernel or VQC edge over classical kernels or linear models is absent, confined to stress or small-n, or a geometry effect that fails portfolio and calibration checks. The panel is QForge-FIN-X v1.0, a CC-BY 4.0 synthetic superset of QForge-FIN v1.0 (12 assets, 2018-2025 business days, seed 20260907), extended with Pack-4 and Pack-8 blocks, synthetic implied-vol and attention features, explicit forward-return targets, and oracle-only regime PIT probabilities.

## 2. Related framing (F1-F5)

EQUAL-Q deliberately cites five interpretive findings that organize honest NISQ reporting:

- **F1 Fragile edge.** Apparent quantum wins that vanish under seed, window, or n changes are not durable edges.
- **F2 Classification is kinder than regression.** Shallow quantum models may look competitive on AUC while losing on ranking IC.
- **F3 Geometry can move without accuracy.** Kernel alignment and sector purity can differ even when predictive metrics do not favor quantum models.
- **F4 Simulator is not hardware.** Results on default.qubit statevector backends do not imply device-level quantum edges.
- **F5 Reuse QForge-FIN v1.** EQUAL-Q extends a public synthetic panel rather than minting an incomparable dataset.

These are interpretive frames, not additional null hypotheses. Confirmatory tests remain H0-H4.

## 3. Data: QForge-FIN-X v1.0

QForge-FIN-X reuses the QForge-FIN v1 generative core (latent macros, GARCH-like idiosyncratic vol, Student-t tails, hidden Calm/Stress/Crash regimes) and adds:

1. **Pack-4** — ret_1d, vol_21d, momentum_10, mkt_ret_1d (primary 4-qubit budget).
2. **Pack-8** — Pack-4 plus ret_5d, rsi_14, volume_z_21, mkt_vol_21d.
3. **Synthetic IV** — iv_21d, iv_skew from trailing vol and range (causal).
4. **Synthetic attention** — within-sector and market softmax weights on contemporaneous absolute return and volume z-scores.
5. **Targets** — v1 labels plus explicit fwd_ret_5d and fwd_ret_21d.
6. **Oracle PIT probs** — soft Calm/Stress/Crash probabilities and PIT transforms, stored only in the oracle file.

Public features contain no regime, macro, or PIT columns. Quality gates (quality_gates.txt) were printed and passed before any model fitting: zero feature NaNs, no leakage substrings in the public feature list, balanced y_dir_1d, and eight walk-forward windows of 63 test business days.

Static splits follow v1: train before 2023-01-01, validation through mid-2024, test thereafter. Walk-forward windows expand from mid-2022.

## 4. Equal-budget protocol

### 4.1 Models

| Tag | Family | Specification |
|-----|--------|---------------|
| LIN | Classical | Logistic regression (T-class) / Ridge (T-reg) |
| RBF | Classical | SVC / Kernel Ridge with RBF kernel |
| QK | Quantum | ZZFeatureMap-style kernel, reps=2, SVM / KRR dual |
| VQC | Quantum | AngleEmbedding + 2 StronglyEntanglingLayers to PauliZ expectations to classical head |

All quantum paths target 4 qubits. PennyLane default.qubit supplies VQC expectations and ZZ kernel spot-checks; the ZZ Gram matrix used in walk-forward loops is an analytic speed path documented against PennyLane (zz_kernel_verification.txt).

### 4.2 Equalization rules

- Train sizes n in {150,400} shared across LIN/RBF/QK/VQC via identical stratified indices within each (window, seed, n).
- LIN/RBF full-n runs are REFERENCE only and excluded from H0 family tests.
- Hyperparameters (rbf_gamma, svm_C, ridge_alpha) frozen on window-0; reused thereafter.
- StandardScaler fit on the train fold only; features squashed to [-pi, pi] for angle maps.
- No time shuffling. Seeds {0,1,2}.

### 4.3 Tasks

- **T-class:** predict y_dir_5d; primary metric ROC-AUC; also Brier.
- **T-reg:** predict y_exret_5d; primary metric Spearman IC.
- **T-book:** n=400 scores to daily long top-3 / short bottom-3; net Sharpe after 5 bps turnover costs.
- **Geometry (H3):** CKA between Gram matrices; KTA; sector 1-NN purity.

### 4.4 Statistics

Primary H0 contrasts at n=400: QK vs RBF and VQC vs LIN on AUC and Spearman IC (four contrasts). Paired t-tests across window times seed cells; one-sided quantum-greater; Holm-Bonferroni at alpha=0.05. H0 rejects only if quantum outperforms after adjustment.

## 5. Results

### 5.1 H0 primary — not rejected

At n=400, mean AUC is LIN 0.747, RBF 0.743, VQC 0.739, QK 0.714. Mean Spearman IC is LIN 0.293, RBF 0.292, VQC 0.270, QK 0.218. All four primary delta(quantum minus classical) estimates are negative; Holm-adjusted one-sided p-values equal 1.00 for practical purposes. **H0 is not rejected.**

### 5.2 H1 — small-n gaps

The IC gap QK minus RBF is -0.091 at n=150 and -0.074 at n=400 (-0.075 vs full-n RBF). Gaps remain classical-favoring; the n-interaction is modest. H1 is not clearly supported as a quantum small-n rescue.

### 5.3 H2 — stress regimes

Using oracle regimes at evaluation time only, the stress-minus-calm AUC gap is less negative for QK than for RBF (delta about +0.019). This is weak descriptive support that quantum models are relatively less degraded in stress—not evidence they beat RBF outright.

### 5.4 H3 — geometry moves

Mean CKA(QK, RBF) is about 0.74 and CKA(VQC, RBF) about 0.85, while KTA is lower for QK (0.087) than RBF (0.169) or LIN (0.201). Sector NN purity is similar across kernels (about 0.30). Geometry therefore distinguishes representations even though predictive H0 stands—consistent with F3.

### 5.5 H4 — books after costs

Net Sharpe at 5 bps: LIN 8.48, VQC 5.99, RBF 3.36, QK -1.79 (means over windows and seeds). Quantum does not beat the classical book; H4's does-not-beat statement is confirmed. Extreme Sharpe magnitudes reflect the synthetic panel's signal strength and short test windows; relative ordering is the inferential object.

### 5.6 Classification vs regression (F2)

VQC trails LIN by only about 0.008 AUC at n=400 but by about 0.023 Spearman IC. QK's deficit is larger on IC than a casual AUC reading suggests. Classification metrics are kinder to shallow quantum models than ranking metrics—F2.

## 6. Discussion

EQUAL-Q's null is not a failure mode; it is the scientifically useful outcome when budgets are honest. Three implications follow.

First, papers that claim quantum kernel edges in finance should publish the equal-n classical counterpart on the same indices. Full-train classical references are informative but must not be compared silently to quantum models trained on n=400.

Second, geometry diagnostics are necessary but not sufficient. A quantum kernel can sit at CKA about 0.74 from RBF and still lose on IC and Sharpe. Reporting CKA/KTA without predictive and portfolio panels risks F3 misreads.

Third, simulator results remain provisional (F4). Shot noise, decoherence, and embedding constraints on hardware would likely widen—not shrink—the classical gap observed here.

Limitations: the panel is synthetic by design; Pack-4 omits richer fundamentals; VQC uses a classical head on expectations (standard NISQ pattern) rather than end-to-end quantum decision rules; the ZZ Gram matrix uses a documented analytic speed path with PennyLane verification rather than a full O(n^2) statevector kernel at every window.

## 7. Conclusion

After equalizing sample size, 4-qubit feature budget, and walk-forward protocol on QForge-FIN-X v1.0, ZZ-quantum kernels and AngleEmbedding+SEL VQCs do not outperform matched RBF and linear models on AUC or Spearman IC at Holm alpha=0.05. Geometry shifts without portfolio gains after 5 bps. The equal-budget anatomy supports a publishable null: on this NISQ budget, the quantum edge in cross-sectional return prediction is absent.

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/generate_qforge_fin_x.py
python scripts/experiment_equalq.py
pytest -q
```

Artifacts live in EQUAL-Q_release/QForge-FIN-X_v1_release/. Author: Hossein Tabasi. Not investment advice; 100 percent synthetic data.

## References (dataset)

Hossein Tabasi. QForge-FIN v1.0 and QForge-FIN-X v1.0 (2026). Synthetic multi-asset panels for NISQ-scale finance machine learning. Data CC-BY 4.0; code MIT.


## Appendix note on equal-budget discipline

Equal-budget discipline is easy to state and hard to keep. A classical linear model trained on the full expanding window sees tens of thousands of asset-days; a quantum kernel restricted to n=400 does not. EQUAL-Q therefore forces LIN and RBF into the same stratified n for confirmatory contrasts, while still reporting full-n references as ceiling diagnostics. Shared indices remove a second confounder: if each model drew its own subsample, apparent gaps could be sampling artifacts rather than representation effects.

Hyperparameter freezing on window-0 is likewise intentional. Post-hoc retuning on later windows would leak test information through analyst degrees of freedom even when code never touches test labels. The frozen gamma and C are ordinary median-heuristic scales, not heroic optima; the point is procedural honesty rather than peak score chasing.

## Additional remarks on Pack-4 versus Pack-8

Pack-4 is the confirmatory feature budget because it matches the 4-qubit encoding without an extra PCA compression step. Pack-8 remains available for exploratory classical checks and for optional PCA(4) robustness studies, but those checks are labeled exploratory in the preregistration sense. Synthetic IV and attention features enrich the public panel for future work; the equal-budget 2x2 reported here stays on Pack-4 so that classical and quantum models consume the same four coordinates after scaling.

## Portfolio construction details

The T-book task ranks assets within each test date by model score, longs the top three, and shorts the bottom three with equal weights. Realized active return uses the panel's fwd_ret_5d, dailyized by dividing by five to approximate a five-day hold. Turnover costs of five basis points apply to changes in the long and short membership relative to the previous date. This is a stylized book, not a tradable strategy: the synthetic panel embeds a clear factor signal, so absolute Sharpes are large. Relative ordering across LIN, RBF, QK, and VQC is the quantity tied to H4.

## On null results and publication

Null equal-budget results are often under-reported, which biases the literature toward fragile positive claims (F1). EQUAL-Q treats the null as a first-class deliverable. The combination of pre-registration, quality gates before fitting, and explicit H0 decision rules is meant to make a non-rejection interpretable: not "we failed to find an edge," but "under these locked budgets, quantum outperformance is rejected as unsupported at alpha=0.05."

Together with F2-F5, the message for practitioners is conservative. Prefer classification diagnostics only as a companion to ranking and portfolio panels; treat geometry as descriptive; keep simulator caveats visible; and build on shared public panels such as QForge-FIN rather than private one-off extracts.
