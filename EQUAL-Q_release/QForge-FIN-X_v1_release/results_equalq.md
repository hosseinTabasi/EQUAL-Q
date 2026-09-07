# EQUAL-Q Results (QForge-FIN-X v1.0)

**Author:** Hossein Tabasi  
**Data seed:** 20260907  
**Backend:** PennyLane `default.qubit` (4 qubits); ZZ kernel analytic speed path with PL spot-check  
**Protocol:** Pack-4 equal budget; \(n\in\{150,400\}\); seeds 0,1,2; 8 walk-forward windows; HP frozen on window-0  

## Quality gates

All gates in `quality_gates.txt` **PASS** before model fitting (0 feature NaNs; no oracle/leak columns; Pack-4/8 present; 8 WF windows).

## H0 (primary) — NOT rejected

Quantum models do **not** outperform equal-\(n\) classical models on AUC or Spearman IC at FWER \(\alpha=0.05\) (Holm).

| Contrast (n=400) | mean quantum | mean classical | Δ | t | p_greater | p_Holm |
|------------------|-------------:|---------------:|--:|--:|----------:|-------:|
| AUC QK − RBF | 0.7142 | 0.7431 | −0.0289 | −6.26 | 1.000 | 1.000 |
| AUC VQC − LIN | 0.7394 | 0.7471 | −0.0077 | −2.21 | 0.981 | 1.000 |
| IC QK − RBF | 0.2180 | 0.2920 | −0.0740 | −6.92 | 1.000 | 1.000 |
| IC VQC − LIN | 0.2697 | 0.2932 | −0.0234 | −2.19 | 0.981 | 1.000 |

**Decision:** H0 **not rejected** (no quantum outperformance).

## Secondary

- **H1:** IC gap QK−RBF ≈ −0.091 (n=150) vs −0.074 (n=400) vs −0.075 vs full RBF — **not clearly supported** as a large n-interaction.
- **H2:** (AUC_stress − AUC_calm) QK − RBF = +0.019 — **weak descriptive support** (quantum less hurt in stress, not an absolute win).
- **H3:** CKA(QK,RBF)=0.739; KTA_QK=0.087 < KTA_RBF=0.169 — **supported** (geometry moves without accuracy gains).
- **H4:** Net Sharpe @5bps LIN=8.48, RBF=3.36, VQC=5.99, QK=−1.79 — **confirmed** (quantum does not beat classical books).

## Means ± std (equal-n, not reference)

### T-class ROC-AUC

| Model | n=150 | n=400 |
|-------|------:|------:|
| LIN | 0.741±0.045 | 0.747±0.037 |
| RBF | 0.735±0.044 | 0.743±0.038 |
| QK | 0.696±0.057 | 0.714±0.048 |
| VQC | 0.730±0.053 | 0.739±0.041 |

### T-reg Spearman IC

| Model | n=400 |
|-------|------:|
| LIN | 0.293±0.083 |
| RBF | 0.292±0.078 |
| VQC | 0.270±0.088 |
| QK | 0.218±0.063 |

## F1–F5 mapping

- **F1 Fragile advantage:** no stable quantum win across windows/seeds.
- **F2 Classification kinder:** VQC nearly matches LIN on AUC; larger gap on IC.
- **F3 Geometry without accuracy:** CKA/KTA move; H0 stands.
- **F4 Simulator ≠ hardware:** `default.qubit` only.
- **F5 Reuse v1:** panel extends QForge-FIN v1.0 under seed 20260907.

## Three referee-quotable sentences

1. After equalizing \(n\), Pack-4 feature budget, and walk-forward folds, neither ZZ-quantum kernels nor AngleEmbedding+SEL VQCs beat matched RBF/linear models on AUC or Spearman IC at Holm \(\alpha=0.05\).
2. Representation geometry (CKA≈0.74 for QK–RBF; lower KTA for QK) changes without delivering portfolio Sharpe gains after 5 bps.
3. On this NISQ budget, the honest equal-budget conclusion is a null quantum edge in cross-sectional return prediction.
