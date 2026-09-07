# QForge-FIN-X v1.0 Codebook

**Brand:** Quantum-Forge Finance Extended Benchmark  
**Version:** 1.0  
**Author:** Hossein Tabasi  
**Citation:** QForge-FIN-X v1.0 (2026), equal-budget quantum vs classical finance panel (superset of QForge-FIN v1.0).  
**Data license:** CC-BY 4.0  
**Master seed:** 20260907  
**Relation:** Strict superset of QForge-FIN v1.0 public keys (same assets, dates, core features/targets).

## Assets
QF01–QF12 across Tech, Energy, Finance, Healthcare, Staples, Crypto-beta (same as v1).

## Public features
### Core (v1)
ret_1d, ret_5d, ret_21d, vol_21d, vol_63d, rsi_14, momentum_10, momentum_21, volume_z_21, range_hl, sector_id, mkt_ret_1d, mkt_vol_21d

### Pack-4 (4-qubit equal-budget block)
ret_1d, vol_21d, momentum_10, mkt_ret_1d

### Pack-8 (extended classical / optional 8-dim map)
ret_1d, vol_21d, momentum_10, mkt_ret_1d, ret_5d, rsi_14, volume_z_21, mkt_vol_21d

### X extensions (causal; no look-ahead)
- `iv_21d` — synthetic implied-vol proxy from trailing vol × range
- `iv_skew` — short-minus-long vol proxy
- `attn_sector` — within-sector softmax attention on |ret| + volume_z
- `attn_mkt` — cross-sectional market softmax attention (sums to 1 per date)
- `attn_dispersion` — dispersion of market attention that day

## Targets
- `y_dir_1d`, `y_dir_5d`, `y_exret_5d`, `y_stress_5d`, `rank_21d` (v1)
- `fwd_ret_5d`, `fwd_ret_21d` — explicit forward simple-return sums

## Oracle (NOT for public ML features)
`qforge_fin_x_v1_oracle_factors.csv` includes v1 latent macros/regime/crash plus:
- `p_calm`, `p_stress`, `p_crash` — soft regime probabilities
- `p_calm_pit`, `p_stress_pit`, `p_crash_pit` — PIT transforms of those probs

## Splits
Static train/valid/test and 8 walk-forward windows — identical scheme to QForge-FIN v1.0.
See `qforge_fin_x_v1_splits.json`.

## Regeneration
```bash
python scripts/generate_qforge_fin_x.py
```
Requires QForge-FIN v1.0 release at `/workspace/QForge-FIN/QForge-FIN_v1_release/` (or regenerate v1 first).
