#!/usr/bin/env python3
"""
QForge-FIN-X v1.0 — Superset panel generator for EQUAL-Q
Author: Hossein Tabasi
License: Code MIT; Data CC-BY 4.0
Master seed: 20260907

Extends QForge-FIN v1.0 with Pack-4 / Pack-8 feature blocks, synthetic IV,
synthetic attention features, forward-return targets, and regime PIT
probabilities (oracle only). Core panel keys remain identical to v1.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_SEED = 20260907
VERSION = "1.0"
BRAND = "QForge-FIN-X v1.0"
CITATION = (
    "QForge-FIN-X v1.0 (2026), equal-budget quantum vs classical finance panel "
    "(superset of QForge-FIN v1.0)."
)
AUTHOR = "Hossein Tabasi"

ROOT = Path("/workspace/EQUAL-Q")
RELEASE = ROOT / "EQUAL-Q_release" / "QForge-FIN-X_v1_release"
V1_RELEASE = Path("/workspace/QForge-FIN/QForge-FIN_v1_release")

PACK4 = ["ret_1d", "vol_21d", "momentum_10", "mkt_ret_1d"]
PACK8 = PACK4 + ["ret_5d", "rsi_14", "volume_z_21", "mkt_vol_21d"]

V1_PUBLIC_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "vol_21d", "vol_63d", "rsi_14",
    "momentum_10", "momentum_21", "volume_z_21", "range_hl", "sector_id",
    "mkt_ret_1d", "mkt_vol_21d",
]
X_PUBLIC_EXTRA = [
    "iv_21d", "iv_skew", "attn_sector", "attn_mkt", "attn_dispersion",
]
PUBLIC_FEATURES = V1_PUBLIC_FEATURES + X_PUBLIC_EXTRA
PUBLIC_TARGETS = [
    "y_dir_1d", "y_dir_5d", "y_exret_5d", "y_stress_5d", "rank_21d",
    "fwd_ret_5d", "fwd_ret_21d",
]


def sub_seed(name: str) -> int:
    h = hashlib.sha256(f"{MASTER_SEED}:{name}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def soft_pit(z: np.ndarray) -> np.ndarray:
    """Approximate PIT via logistic CDF of standardized z."""
    z = np.asarray(z, dtype=float)
    mu = np.nanmean(z)
    sd = np.nanstd(z) + 1e-12
    return 1.0 / (1.0 + np.exp(-(z - mu) / sd))


def add_x_features(df: pd.DataFrame, oracle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add synthetic IV, attention, fwd returns; extend oracle with regime PIT."""
    rng_iv = np.random.default_rng(sub_seed("synth_iv"))
    rng_att = np.random.default_rng(sub_seed("synth_attention"))
    rng_pit = np.random.default_rng(sub_seed("regime_pit"))

    d = df.sort_values(["asset_id", "date"]).reset_index(drop=True).copy()

    # --- Synthetic IV (causal: uses trailing vol + range only) ---
    # iv_21d ~ annualized trailing vol with mild microstructure noise
    base_iv = d["vol_21d"].to_numpy(float) * np.sqrt(252)
    noise = rng_iv.normal(0, 0.01, size=len(d))
    d["iv_21d"] = np.clip(base_iv * (1.0 + 0.15 * d["range_hl"].to_numpy(float)) + noise, 0.05, 1.5)
    # iv_skew: short-minus-long vol proxy
    d["iv_skew"] = (
        d["vol_21d"].to_numpy(float) - d["vol_63d"].to_numpy(float)
    ) * np.sqrt(252) + rng_iv.normal(0, 0.005, size=len(d))

    # --- Synthetic attention (cross-sectional, same-date, no future) ---
    # Softmax-like weights vs sector peers and market on |ret_1d| + volume_z
    attn_sector = np.zeros(len(d))
    attn_mkt = np.zeros(len(d))
    attn_disp = np.zeros(len(d))
    for date, g in d.groupby("date", sort=False):
        idx = g.index.to_numpy()
        score = np.abs(g["ret_1d"].to_numpy(float)) + 0.25 * np.abs(g["volume_z_21"].to_numpy(float))
        score = score + rng_att.normal(0, 0.01, size=len(score))
        # market attention: softmax over all assets that day
        e = np.exp(score - score.max())
        p_mkt = e / e.sum()
        attn_mkt[idx] = p_mkt
        attn_disp[idx] = float(np.std(p_mkt))
        # sector attention: within-sector softmax
        for sid, sg in g.groupby("sector_id"):
            sidx = sg.index.to_numpy()
            sc = score[np.isin(idx, sidx)]
            # map back
            sc_full = np.abs(sg["ret_1d"].to_numpy(float)) + 0.25 * np.abs(sg["volume_z_21"].to_numpy(float))
            sc_full = sc_full + rng_att.normal(0, 0.01, size=len(sc_full))
            ee = np.exp(sc_full - sc_full.max())
            attn_sector[sidx] = ee / ee.sum()
    d["attn_sector"] = attn_sector
    d["attn_mkt"] = attn_mkt
    d["attn_dispersion"] = attn_disp

    # --- Forward returns as explicit targets (aligned; trailing NaN at end) ---
    fwd5 = []
    fwd21 = []
    for aid, g in d.groupby("asset_id", sort=False):
        ret = g["ret_1d"].to_numpy(float)
        n = len(ret)
        f5 = np.array([np.sum(ret[t + 1 : t + 6]) if t + 5 < n else np.nan for t in range(n)])
        f21 = np.array([np.sum(ret[t + 1 : t + 22]) if t + 21 < n else np.nan for t in range(n)])
        fwd5.append(pd.Series(f5, index=g.index))
        fwd21.append(pd.Series(f21, index=g.index))
    d["fwd_ret_5d"] = pd.concat(fwd5).sort_index()
    d["fwd_ret_21d"] = pd.concat(fwd21).sort_index()

    # Pack indicator columns (documentation; values are the features themselves)
    # No leakage: packs are subsets of public features already present.

    # --- Oracle regime PIT probabilities ---
    # Merge regime from oracle if present
    o = oracle.copy()
    if "regime" not in o.columns and "_regime" in o.columns:
        o = o.rename(columns={"_regime": "regime"})
    # Build soft probs from regime + macros
    # Use growth/risk_appetite style columns if available
    merge_cols = ["date", "asset_id"]
    o_use = o.copy()
    for c in list(o_use.columns):
        if c.startswith("_"):
            o_use = o_use.rename(columns={c: c[1:]})

    # Align oracle to panel dates/assets
    key = ["date", "asset_id"]
    # Ensure date string
    d["date"] = d["date"].astype(str)
    o_use["date"] = o_use["date"].astype(str)

    # Regime one-hot soft with noise → PIT
    if "regime" in o_use.columns:
        reg = o_use["regime"].to_numpy(float)
    else:
        reg = np.zeros(len(o_use))

    # latent macros for soft probs
    ra = o_use["risk_appetite"].to_numpy(float) if "risk_appetite" in o_use.columns else np.zeros(len(o_use))
    liq = o_use["liquidity"].to_numpy(float) if "liquidity" in o_use.columns else np.zeros(len(o_use))
    crash = o_use["crash"].to_numpy(float) if "crash" in o_use.columns else np.zeros(len(o_use))

    # Unnormalized logits for Calm/Stress/Crash (causal at t; still ORACLE — not for ML features)
    logit_crash = 2.5 * crash + 1.2 * (reg == 2).astype(float) - 1.5 * ra - 0.8 * liq
    logit_stress = 1.0 * (reg == 1).astype(float) - 0.6 * ra + 0.4 * (reg == 2).astype(float)
    logit_calm = 1.5 * (reg == 0).astype(float) + 0.8 * ra + 0.5 * liq
    logit_crash = logit_crash + rng_pit.normal(0, 0.15, size=len(logit_crash))
    logit_stress = logit_stress + rng_pit.normal(0, 0.15, size=len(logit_stress))
    logit_calm = logit_calm + rng_pit.normal(0, 0.15, size=len(logit_calm))
    logits = np.column_stack([logit_calm, logit_stress, logit_crash])
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    probs = ex / ex.sum(axis=1, keepdims=True)

    o_use["p_calm_pit"] = soft_pit(probs[:, 0])
    o_use["p_stress_pit"] = soft_pit(probs[:, 1])
    o_use["p_crash_pit"] = soft_pit(probs[:, 2])
    o_use["p_calm"] = probs[:, 0]
    o_use["p_stress"] = probs[:, 1]
    o_use["p_crash"] = probs[:, 2]

    # Ensure fwd in oracle too
    if "fwd_ret_21d" not in o_use.columns and "fwd_ret_21d" in d.columns:
        o_use = o_use.merge(d[key + ["fwd_ret_5d", "fwd_ret_21d"]], on=key, how="left")
    else:
        # still attach fwd_ret_5d
        if "fwd_ret_5d" not in o_use.columns:
            o_use = o_use.merge(d[key + ["fwd_ret_5d", "fwd_ret_21d"]], on=key, how="left", suffixes=("", "_x"))
            if "fwd_ret_21d_x" in o_use.columns:
                o_use["fwd_ret_21d"] = o_use["fwd_ret_21d"].fillna(o_use["fwd_ret_21d_x"])
                o_use = o_use.drop(columns=["fwd_ret_21d_x"])

    # Strip any accidental oracle cols from public
    for bad in ["regime", "crash", "growth", "inflation", "risk_appetite", "liquidity",
                "p_calm", "p_stress", "p_crash", "p_calm_pit", "p_stress_pit", "p_crash_pit"]:
        if bad in d.columns:
            d = d.drop(columns=[bad])

    return d, o_use


def quality_gates(df: pd.DataFrame, splits: dict, out_path: Path) -> str:
    lines = []
    lines.append("QForge-FIN-X v1.0 Quality Gates")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Master seed: {MASTER_SEED}")
    lines.append(f"Author: {AUTHOR}")
    lines.append("=" * 60)

    # 1. Feature NaNs
    feat_nans = df[PUBLIC_FEATURES].isna().sum().sum()
    lines.append(f"\n[1] NaNs in public features: {feat_nans}")
    lines.append(df[PUBLIC_FEATURES].isna().sum().to_string())
    assert feat_nans == 0, "Feature NaNs present"

    # 2. No leakage in public features
    forbidden = ["_fwd", "future", "lead", "oracle", "regime", "crash", "growth",
                 "inflation", "liquidity", "risk_appetite", "p_calm", "p_stress", "p_crash"]
    bad = []
    for c in PUBLIC_FEATURES:
        if c.startswith("y_"):
            bad.append(c)
        if any(f in c.lower() for f in ["_fwd", "future", "lead", "oracle", "regime", "crash"]):
            bad.append(c)
    # fwd_ret_* are TARGETS not features — ensure not in PUBLIC_FEATURES
    for t in ["fwd_ret_5d", "fwd_ret_21d"]:
        assert t not in PUBLIC_FEATURES
    lines.append(f"\n[2] Leak columns in public features: {bad}")
    lines.append(f"    Public features ({len(PUBLIC_FEATURES)}): {PUBLIC_FEATURES}")
    lines.append(f"    Pack-4: {PACK4}")
    lines.append(f"    Pack-8: {PACK8}")
    lines.append("    PASS: no look-ahead / oracle in public features" if not bad else "    FAIL")
    assert not bad

    # 3. Oracle hidden
    for c in df.columns:
        assert not c.startswith("p_"), f"PIT prob leaked into public: {c}"
        assert c not in ("regime", "crash", "growth", "inflation", "risk_appetite", "liquidity")

    # 4. Moments
    lines.append("\n[3] Return moments (ret_1d):")
    lines.append(f"    overall mean={df['ret_1d'].mean():.6f}, ann_vol={df['ret_1d'].std()*np.sqrt(252)*100:.1f}%")
    lines.append(f"    n_assets={df['asset_id'].nunique()}, n_dates={df['date'].nunique()}, n_rows={len(df)}")

    # 5. Target balance
    ybal = df["y_dir_1d"].dropna()
    pos = (ybal == 1).mean() * 100
    lines.append(f"\n[4] y_dir_1d balance: positive={pos:.2f}% (target 45–55%)")
    assert 45 <= pos <= 55

    y5 = df["y_dir_5d"].dropna()
    lines.append(f"    y_dir_5d positive={(y5==1).mean()*100:.2f}%")

    # 6. Pack integrity
    for p in PACK4 + PACK8:
        assert p in df.columns
    lines.append("\n[5] Pack-4 / Pack-8 columns present: PASS")

    # 7. IV / attention ranges
    lines.append(f"\n[6] Synthetic IV: iv_21d mean={df['iv_21d'].mean():.4f}, "
                 f"iv_skew mean={df['iv_skew'].mean():.4f}")
    lines.append(f"    Attention: attn_mkt sum/day≈1 check: "
                 f"{df.groupby('date')['attn_mkt'].sum().mean():.4f}")
    lines.append(f"    attn_sector mean={df['attn_sector'].mean():.4f}")

    # 8. Splits
    dates = sorted(df["date"].unique())
    n_train = sum(1 for x in dates if x < "2023-01-01")
    n_valid = sum(1 for x in dates if "2023-01-01" <= x < "2024-07-01")
    n_test = sum(1 for x in dates if x >= "2024-07-01")
    lines.append(f"\n[7] Static splits: train_dates={n_train}, valid={n_valid}, test={n_test}")
    lines.append(f"    Walk-forward windows: {len(splits['walk_forward'])}")
    for w in splits["walk_forward"]:
        lines.append(
            f"      W{w['window_id']}: train_end={w['train_end']}, "
            f"test={w['test_start']}→{w['test_end']} (n_test_dates={w['n_test_dates']})"
        )
    assert len(splits["walk_forward"]) >= 6

    # 9. Targets present
    for t in PUBLIC_TARGETS:
        assert t in df.columns, t
    lines.append(f"\n[8] Targets present: {PUBLIC_TARGETS}")

    lines.append("\n" + "=" * 60)
    lines.append("Quality gates: ALL PASS")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    return text


def write_codebook(path: Path):
    text = f"""# QForge-FIN-X v1.0 Codebook

**Brand:** Quantum-Forge Finance Extended Benchmark  
**Version:** 1.0  
**Author:** {AUTHOR}  
**Citation:** {CITATION}  
**Data license:** CC-BY 4.0  
**Master seed:** {MASTER_SEED}  
**Relation:** Strict superset of QForge-FIN v1.0 public keys (same assets, dates, core features/targets).

## Assets
QF01–QF12 across Tech, Energy, Finance, Healthcare, Staples, Crypto-beta (same as v1).

## Public features
### Core (v1)
{', '.join(V1_PUBLIC_FEATURES)}

### Pack-4 (4-qubit equal-budget block)
{', '.join(PACK4)}

### Pack-8 (extended classical / optional 8-dim map)
{', '.join(PACK8)}

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
"""
    path.write_text(text)


def main():
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "figures").mkdir(exist_ok=True)

    assert V1_RELEASE.exists(), f"Missing v1 release at {V1_RELEASE}"
    panel_path = V1_RELEASE / "qforge_fin_v1_panel.parquet"
    if not panel_path.exists():
        panel_path = V1_RELEASE / "qforge_fin_v1_panel.csv"
    if str(panel_path).endswith(".parquet"):
        df_v1 = pd.read_parquet(panel_path)
    else:
        df_v1 = pd.read_csv(panel_path)

    oracle_v1 = pd.read_csv(V1_RELEASE / "qforge_fin_v1_oracle_factors.csv")
    with open(V1_RELEASE / "qforge_fin_v1_splits.json") as f:
        splits = json.load(f)
    meta_v1 = json.loads((V1_RELEASE / "qforge_fin_v1_metadata.json").read_text())

    df_x, oracle_x = add_x_features(df_v1, oracle_v1)

    # Public panel columns
    id_cols = ["date", "asset_id", "sector", "sector_id", "open", "high", "low", "close",
               "volume", "bid", "ask", "spread_bps"]
    pub_cols = [c for c in id_cols if c in df_x.columns] + PUBLIC_FEATURES + PUBLIC_TARGETS
    # dedupe preserve order
    seen = set()
    pub_cols = [c for c in pub_cols if not (c in seen or seen.add(c))]
    df_pub = df_x[pub_cols].copy()

    # Quality gates BEFORE writing success marker / before models
    qpath = RELEASE / "quality_gates.txt"
    quality_gates(df_pub, splits, qpath)
    # also root reports
    (ROOT / "reports").mkdir(exist_ok=True)
    shutil.copy(qpath, ROOT / "reports" / "quality_gates.txt")

    # Write panel
    df_pub.to_csv(RELEASE / "qforge_fin_x_v1_panel.csv", index=False)
    df_pub.to_parquet(RELEASE / "qforge_fin_x_v1_panel.parquet", index=False)
    df_pub.head(20).to_csv(RELEASE / "qforge_fin_x_v1_preview_20rows.csv", index=False)

    # Market file: equal-weight
    mkt = (
        df_pub.groupby("date", as_index=False)
        .agg(mkt_ret_1d=("ret_1d", "mean"), mkt_close=("close", "mean"),
             mkt_vol_21d=("mkt_vol_21d", "first"))
    )
    mkt.to_csv(RELEASE / "qforge_fin_x_v1_market.csv", index=False)

    # Oracle
    oracle_cols_keep = [
        c for c in oracle_x.columns
        if c in ("date", "asset_id", "regime", "crash", "growth", "inflation",
                 "risk_appetite", "liquidity", "fwd_ret_5d", "fwd_ret_21d",
                 "p_calm", "p_stress", "p_crash",
                 "p_calm_pit", "p_stress_pit", "p_crash_pit")
        or c.startswith("p_")
    ]
    # ensure essentials
    for c in ["date", "asset_id"]:
        assert c in oracle_x.columns
    oracle_out = oracle_x[oracle_cols_keep].copy()
    oracle_out.to_csv(RELEASE / "qforge_fin_x_v1_oracle_factors.csv", index=False)

    # Splits (same as v1)
    with open(RELEASE / "qforge_fin_x_v1_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Metadata
    meta = {
        "brand": BRAND,
        "full_name": "Quantum-Forge Finance Extended Benchmark",
        "version": VERSION,
        "citation": CITATION,
        "license_data": "CC-BY 4.0",
        "license_code": "MIT",
        "author": AUTHOR,
        "master_seed": MASTER_SEED,
        "parent_benchmark": "QForge-FIN v1.0",
        "parent_master_seed": meta_v1.get("master_seed", MASTER_SEED),
        "sub_seeds": {
            "synth_iv": sub_seed("synth_iv"),
            "synth_attention": sub_seed("synth_attention"),
            "regime_pit": sub_seed("regime_pit"),
            **meta_v1.get("sub_seeds", {}),
        },
        "n_assets": int(df_pub["asset_id"].nunique()),
        "n_rows": int(len(df_pub)),
        "n_dates": int(df_pub["date"].nunique()),
        "date_range_panel": [str(df_pub["date"].min()), str(df_pub["date"].max())],
        "public_features": PUBLIC_FEATURES,
        "pack4": PACK4,
        "pack8": PACK8,
        "targets": PUBLIC_TARGETS,
        "oracle_file": "qforge_fin_x_v1_oracle_factors.csv",
        "qubit_recommendations": {
            "primary": 4,
            "encoding": "Pack-4 or PCA(4) on Pack-8/public; StandardScaler train-only",
            "depth": "ZZFeatureMap reps=2; VQC AngleEmbedding + 2 StronglyEntanglingLayers",
            "backend": "pennylane.default.qubit (statevector); numpy fallback documented",
        },
        "limitations": [
            "100% synthetic; not real market data; no investment advice.",
            "Superset of QForge-FIN v1.0; core returns identical to parent panel.",
            "Regime PIT probabilities are ORACLE-only; must not enter training features.",
            "NISQ experiments use small-n and 4 qubits; simulator ≠ hardware.",
            "Equal-budget null results are expected and publishable.",
        ],
    }
    with open(RELEASE / "qforge_fin_x_v1_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    write_codebook(RELEASE / "qforge_fin_x_v1_codebook.md")

    # Copy LICENSE notice
    lic = f"""MIT License

Copyright (c) 2026 {AUTHOR}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------
DATA LICENSE

The QForge-FIN-X v1.0 dataset files (panel, market, splits, metadata, codebook,
preview, oracle factors, and regenerated outputs of generate_qforge_fin_x.py)
are licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0).
See https://creativecommons.org/licenses/by/4.0/
"""
    (RELEASE / "LICENSE").write_text(lic)

    print(f"Wrote QForge-FIN-X v1.0 to {RELEASE}")
    print(f"Rows={len(df_pub)} features={len(PUBLIC_FEATURES)} targets={len(PUBLIC_TARGETS)}")


if __name__ == "__main__":
    main()
