# EQUAL-Q Results Summary

**Author:** Hossein Tabasi
**Benchmark:** QForge-FIN-X v1.0 / EQUAL-Q
**Master seed:** 20260907
**Backend:** pennylane.default.qubit (4 qubits)

## Quality gates

ALL PASS (printed before model fitting).

## H0 primary - NOT rejected

| Contrast n=400 | mean Q | mean C | delta | p_Holm |
|----------------|-------:|-------:|------:|-------:|
| AUC QK-RBF | 0.714 | 0.743 | -0.029 | 1.000 |
| AUC VQC-LIN | 0.739 | 0.747 | -0.008 | 1.000 |
| IC QK-RBF | 0.218 | 0.292 | -0.074 | 1.000 |
| IC VQC-LIN | 0.270 | 0.293 | -0.023 | 1.000 |

## Secondary one-liners

- H1: IC gap QK-RBF similar at n=150 (-0.091) vs n=400 (-0.074); not clearly supported.
- H2: Stress-calm AUC gap less negative for QK than RBF (+0.019 relative); weak descriptive support.
- H3: CKA(QK,RBF)=0.739; KTA_QK=0.087 < KTA_RBF=0.169; geometry moves supported.
- H4: Net Sharpe@5bps LIN=8.48 > VQC=5.99 > RBF=3.36 > QK=-1.79; quantum does not beat classical.

## Conclusion

On this equal-budget NISQ protocol, the quantum edge in cross-sectional return prediction is absent.
