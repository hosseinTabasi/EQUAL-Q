#!/usr/bin/env python3
"""
QForge-FIN v1.0 — Synthetic multi-asset panel generator
Author: Hossein Tabasi
License: Code MIT; Data CC-BY 4.0
Master seed: 20260907
Deterministic bit-identical regeneration.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MASTER_SEED = 20260907
VERSION = "1.0"
BRAND = "QForge-FIN v1.0"
CITATION = "QForge-FIN v1.0 (2026), synthetic multi-asset panel for NISQ quantum machine learning."

# Sub-seeds derived from master for documented reproducibility
def sub_seed(name: str) -> int:
    h = hashlib.sha256(f"{MASTER_SEED}:{name}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)

ASSETS = [
    ("QF01", "Tech", 0),
    ("QF02", "Tech", 0),
    ("QF03", "Tech", 0),
    ("QF04", "Energy", 1),
    ("QF05", "Energy", 1),
    ("QF06", "Finance", 2),
    ("QF07", "Finance", 2),
    ("QF08", "Healthcare", 3),
    ("QF09", "Healthcare", 3),
    ("QF10", "Staples", 4),
    ("QF11", "Staples", 4),
    ("QF12", "Crypto-beta", 5),
]

SECTOR_NAMES = ["Tech", "Energy", "Finance", "Healthcare", "Staples", "Crypto-beta"]


def business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="C")


def ar1_with_jumps(n: int, phi: float, sigma: float, jump_prob: float, jump_scale: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    jumps = rng.random(n) < jump_prob
    jump_vals = rng.normal(0, jump_scale, n) * jumps
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t] + jump_vals[t]
    return x


def generate_latent_macros(n: int, rng: np.random.Generator):
    """Latent macro: growth, inflation, risk_appetite, liquidity — AR(1)+jumps; crash windows."""
    growth = ar1_with_jumps(n, 0.92, 0.015, 0.008, 0.08, rng)
    inflation = ar1_with_jumps(n, 0.95, 0.008, 0.005, 0.04, rng)
    risk_appetite = ar1_with_jumps(n, 0.88, 0.025, 0.012, 0.12, rng)
    liquidity = ar1_with_jumps(n, 0.90, 0.020, 0.010, 0.10, rng)

    # 3–6 crash windows of 5–15 business days
    n_crashes = int(rng.integers(3, 5))
    crash_mask = np.zeros(n, dtype=bool)
    crash_windows = []
    used = set()
    for _ in range(n_crashes):
        length = int(rng.integers(5, 12))
        # Prefer mid-range to avoid edges
        start = int(rng.integers(80, max(81, n - length - 80)))
        # Avoid heavy overlap
        attempts = 0
        while any(abs(start - s) < 40 for s in used) and attempts < 50:
            start = int(rng.integers(80, max(81, n - length - 80)))
            attempts += 1
        used.add(start)
        crash_mask[start : start + length] = True
        crash_windows.append((start, start + length - 1))
        # Crash shocks on macros
        risk_appetite[start : start + length] -= rng.uniform(0.08, 0.22)
        liquidity[start : start + length] -= rng.uniform(0.06, 0.18)
        growth[start : start + length] -= rng.uniform(0.03, 0.10)

    macros = {
        "growth": growth,
        "inflation": inflation,
        "risk_appetite": risk_appetite,
        "liquidity": liquidity,
    }
    return macros, crash_mask, crash_windows


def hidden_markov_regimes(n: int, crash_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """3-state Markov: Calm=0, Stress=1, Crash=2. Crash forced on crash windows."""
    # Transition matrix (Calm, Stress, Crash)
    P = np.array([
        [0.96, 0.035, 0.005],
        [0.08, 0.88, 0.04],
        [0.15, 0.25, 0.60],
    ])
    states = np.zeros(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        if crash_mask[t]:
            states[t] = 2
        else:
            states[t] = rng.choice(3, p=P[states[t - 1]])
    return states


def garch_like_vol(n: int, omega: float, alpha: float, beta: float, base: float, rng: np.random.Generator, innovations: np.ndarray) -> np.ndarray:
    """Simple GARCH(1,1)-like conditional volatility path."""
    sigma2 = np.zeros(n)
    sigma2[0] = base ** 2
    for t in range(1, n):
        sigma2[t] = omega + alpha * (innovations[t - 1] ** 2) + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)
    return np.sqrt(sigma2)


def generate_returns(
    n: int,
    macros: dict,
    regimes: np.ndarray,
    crash_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate returns for 12 assets. Shape (n, 12)."""
    n_assets = len(ASSETS)
    returns = np.zeros((n, n_assets))
    vols = np.zeros((n, n_assets))

    # Factor loadings (betas) per asset — growth, inflation, risk_appetite, liquidity
    # Crypto has higher crash beta / vol
    beta_specs = {
        "Tech": [0.9, -0.2, 0.7, 0.3],
        "Energy": [0.4, 0.6, 0.3, 0.2],
        "Finance": [0.7, -0.3, 0.9, 0.5],
        "Healthcare": [0.5, -0.1, 0.4, 0.2],
        "Staples": [0.3, 0.2, 0.2, 0.1],
        "Crypto-beta": [0.6, -0.1, 1.4, 0.8],
    }
    base_vols = {
        "Tech": 0.018,
        "Energy": 0.022,
        "Finance": 0.017,
        "Healthcare": 0.015,
        "Staples": 0.0185,
        "Crypto-beta": 0.028,
    }

    factor_mat = np.column_stack([
        macros["growth"],
        macros["inflation"],
        macros["risk_appetite"],
        macros["liquidity"],
    ])

    # Mild non-stationarity: slowly drifting betas
    t_frac = np.linspace(0, 1, n)

    asset_meta = {}
    for i, (aid, sector, sid) in enumerate(ASSETS):
        rng_a = np.random.default_rng(sub_seed(f"asset_{aid}"))
        base_beta = np.array(beta_specs[sector], dtype=float)
        # Per-asset idiosyncratic beta jitter
        beta0 = base_beta + rng_a.normal(0, 0.08, 4)
        # Drift
        drift = rng_a.normal(0, 0.15, 4)
        betas_t = beta0[None, :] + drift[None, :] * t_frac[:, None] * 0.3

        # Student-t df
        df = float(rng_a.uniform(5.0, 8.0))
        # Idiosyncratic innovations (Student-t scaled)
        raw_innov = rng_a.standard_t(df, n) / np.sqrt(df / (df - 2))
        # GARCH params
        omega = (base_vols[sector] * 0.3) ** 2
        alpha_g = 0.08 + rng_a.uniform(0, 0.06)
        beta_g = 0.85 + rng_a.uniform(-0.05, 0.05)
        beta_g = min(beta_g, 0.94)
        # Preliminary for GARCH recursion
        idio = raw_innov * base_vols[sector]
        sigma = garch_like_vol(n, omega, alpha_g, beta_g, base_vols[sector], rng_a, idio)
        # Regime scaling
        regime_scale = np.ones(n)
        regime_scale[regimes == 1] = 1.25
        regime_scale[regimes == 2] = 1.9
        if sector == "Crypto-beta":
            regime_scale[regimes == 2] = 2.3
            regime_scale[crash_mask] *= 1.15

        # Crash beta boost for crypto
        crash_beta_extra = 0.0
        if sector == "Crypto-beta":
            crash_beta_extra = 0.35

        factor_ret = np.sum(betas_t * factor_mat * 0.22, axis=1)
        # Extra crash contribution via risk_appetite when crashing
        factor_ret = factor_ret + crash_beta_extra * crash_mask.astype(float) * macros["risk_appetite"] * 0.5

        idio_ret = raw_innov * sigma * regime_scale * 0.42
        # Mild autocorrelation of returns (microstructure / habit)
        r = factor_ret + idio_ret + 0.00015
        r[1:] = r[1:] + 0.03 * r[:-1]

        returns[:, i] = r
        vols[:, i] = sigma * regime_scale
        asset_meta[aid] = {
            "sector": sector,
            "sector_id": sid,
            "df": df,
            "base_vol": base_vols[sector],
            "beta0": beta0.tolist(),
            "garch_omega": omega,
            "garch_alpha": alpha_g,
            "garch_beta": beta_g,
        }

    return returns, vols, asset_meta


def prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    """Compound simple returns from start price."""
    n, k = returns.shape
    prices = np.zeros((n, k))
    prices[0] = start
    for t in range(1, n):
        prices[t] = prices[t - 1] * (1.0 + returns[t])
        prices[t] = np.maximum(prices[t], 0.01)  # floor
    return prices


def microstructure(prices: np.ndarray, returns: np.ndarray, rng: np.random.Generator):
    """Bid-ask proxy and lognormal volume."""
    n, k = prices.shape
    # Spread ~ vol-linked
    daily_vol = np.abs(returns)
    spread_bps = 5.0 + 80.0 * daily_vol * 100  # rough
    spread_bps = np.clip(spread_bps, 2.0, 200.0)
    half = prices * (spread_bps / 10000.0) / 2.0
    bid = prices - half
    ask = prices + half
    # Volume lognormal, higher on big moves / stress
    vol_scale = 1e6 * (1.0 + 5.0 * daily_vol)
    volume = rng.lognormal(mean=np.log(np.maximum(vol_scale, 1.0)), sigma=0.45)
    return bid, ask, volume, spread_bps


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder-style EMA approximation via rolling mean for simplicity/determinism
    avg_gain = pd.Series(gain).rolling(period, min_periods=period).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(period, min_periods=period).mean().to_numpy()
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def build_panel(dates: pd.DatetimeIndex, returns: np.ndarray, prices: np.ndarray,
                bid, ask, volume, spread_bps, regimes, macros, asset_meta, crash_mask):
    n = len(dates)
    rows = []
    # Market equal-weight return
    mkt_ret = returns.mean(axis=1)

    for i, (aid, sector, sid) in enumerate(ASSETS):
        px = prices[:, i]
        ret = returns[:, i]
        high = np.maximum(ask[:, i], px * (1 + np.abs(ret) * 0.3))
        low = np.minimum(bid[:, i], px * (1 - np.abs(ret) * 0.3))
        # Features (no look-ahead: use past only)
        ret_1d = ret.copy()
        ret_5d = pd.Series(ret).rolling(5, min_periods=5).sum().to_numpy()
        ret_21d = pd.Series(ret).rolling(21, min_periods=21).sum().to_numpy()
        vol_21d = pd.Series(ret).rolling(21, min_periods=21).std().to_numpy()
        vol_63d = pd.Series(ret).rolling(63, min_periods=63).std().to_numpy()
        rsi_14 = rsi(px, 14)
        mom_10 = pd.Series(px).pct_change(10).to_numpy()
        mom_21 = pd.Series(px).pct_change(21).to_numpy()
        vol_z_21 = (
            (pd.Series(volume[:, i]) - pd.Series(volume[:, i]).rolling(21, min_periods=21).mean())
            / pd.Series(volume[:, i]).rolling(21, min_periods=21).std().replace(0, np.nan)
        ).to_numpy()
        range_hl = (high - low) / np.maximum(px, 1e-8)
        mkt_ret_1d = mkt_ret
        mkt_vol_21d = pd.Series(mkt_ret).rolling(21, min_periods=21).std().to_numpy()

        # Targets — aligned no leakage: label at t uses future outcomes known at t+h close
        # y_dir_1d: sign of next-day return (predictable from features at t)
        # Convention: features at date t use info available at close t; target is future
        y_dir_1d = np.full(n, np.nan)
        y_dir_5d = np.full(n, np.nan)
        y_exret_5d = np.full(n, np.nan)
        y_stress_5d = np.full(n, np.nan)
        rank_21d = np.full(n, np.nan)

        # Will fill after all assets for rank; dir/exret now
        fut_1 = np.roll(ret, -1)
        fut_1[-1] = np.nan
        y_dir_1d = (fut_1 > 0).astype(float)
        y_dir_1d[np.isnan(fut_1)] = np.nan

        fut_5 = pd.Series(ret).shift(-5).rolling(5).sum().shift(4).to_numpy()  # wrong — fix below
        # Correct: forward 5-day sum of returns from t+1..t+5
        fut_5 = np.array([np.sum(ret[t + 1 : t + 6]) if t + 5 < n else np.nan for t in range(n)])
        y_dir_5d = (fut_5 > 0).astype(float)
        y_dir_5d[np.isnan(fut_5)] = np.nan
        # Excess return vs market over next 5 days
        fut_mkt_5 = np.array([np.sum(mkt_ret[t + 1 : t + 6]) if t + 5 < n else np.nan for t in range(n)])
        y_exret_5d = fut_5 - fut_mkt_5
        # Stress: whether max drawdown or large negative in next 5 days, or regime stress
        y_stress_5d = np.array([
            1.0 if (t + 5 < n and (np.min(np.cumsum(ret[t + 1 : t + 6])) < -0.03 or np.any(regimes[t + 1 : t + 6] >= 1))) else (np.nan if t + 5 >= n else 0.0)
            for t in range(n)
        ])

        for t in range(n):
            rows.append({
                "date": dates[t].strftime("%Y-%m-%d"),
                "asset_id": aid,
                "sector": sector,
                "sector_id": sid,
                "open": px[t] / (1.0 + ret[t]) if t > 0 else px[t],
                "high": high[t],
                "low": low[t],
                "close": px[t],
                "volume": volume[t, i],
                "bid": bid[t, i],
                "ask": ask[t, i],
                "spread_bps": spread_bps[t, i],
                "ret_1d": ret_1d[t],
                "ret_5d": ret_5d[t],
                "ret_21d": ret_21d[t],
                "vol_21d": vol_21d[t],
                "vol_63d": vol_63d[t],
                "rsi_14": rsi_14[t],
                "momentum_10": mom_10[t],
                "momentum_21": mom_21[t],
                "volume_z_21": vol_z_21[t],
                "range_hl": range_hl[t],
                "mkt_ret_1d": mkt_ret_1d[t],
                "mkt_vol_21d": mkt_vol_21d[t],
                "y_dir_1d": y_dir_1d[t],
                "y_dir_5d": y_dir_5d[t],
                "y_exret_5d": y_exret_5d[t],
                "y_stress_5d": y_stress_5d[t],
                # oracle / hidden — will strip from public later for a separate file
                "_regime": int(regimes[t]),
                "_crash": int(crash_mask[t]),
                "_growth": macros["growth"][t],
                "_inflation": macros["inflation"][t],
                "_risk_appetite": macros["risk_appetite"][t],
                "_liquidity": macros["liquidity"][t],
                "_fwd_ret_21d": np.sum(ret[t + 1 : t + 22]) if t + 21 < n else np.nan,
            })

    df = pd.DataFrame(rows)
    # Cross-sectional rank of forward 21d return within date (target)
    df["rank_21d"] = df.groupby("date")["_fwd_ret_21d"].rank(pct=True)
    df.loc[df["_fwd_ret_21d"].isna(), "rank_21d"] = np.nan
    return df


def drop_warmup(df: pd.DataFrame, warmup: int = 63) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    keep = set(dates[warmup:])
    return df[df["date"].isin(keep)].reset_index(drop=True)


PUBLIC_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "vol_21d", "vol_63d", "rsi_14",
    "momentum_10", "momentum_21", "volume_z_21", "range_hl", "sector_id",
    "mkt_ret_1d", "mkt_vol_21d",
]
PUBLIC_TARGETS = ["y_dir_1d", "y_dir_5d", "y_exret_5d", "y_stress_5d", "rank_21d"]
ORACLE_COLS = ["_regime", "_crash", "_growth", "_inflation", "_risk_appetite", "_liquidity", "_fwd_ret_21d"]


def make_splits(dates: list[str]) -> dict:
    train = [d for d in dates if d < "2023-01-01"]
    valid = [d for d in dates if "2023-01-01" <= d < "2024-07-01"]
    test = [d for d in dates if d >= "2024-07-01"]
    # Walk-forward: 8 expanding windows, test next 63 bd, step 63, inside 2022–2025
    all_bd = [d for d in dates if "2022-01-01" <= d <= "2025-12-31"]
    windows = []
    # Need enough history before first test
    step = 63
    test_len = 63
    # Start first test window around mid-2022 so train has data
    start_candidates = [d for d in all_bd if d >= "2022-06-01"]
    if not start_candidates:
        start_candidates = all_bd[max(0, len(all_bd) // 4):]
    # Index-based
    date_to_idx = {d: i for i, d in enumerate(dates)}
    idx_all = [date_to_idx[d] for d in all_bd]
    # Find first index in dates for first test start
    first_test_date = start_candidates[0]
    first_test_idx = date_to_idx[first_test_date]
    wi = 0
    t0 = first_test_idx
    while wi < 8:
        test_end = t0 + test_len
        if test_end > len(dates):
            break
        train_dates = dates[:t0]
        test_dates = dates[t0:test_end]
        # Only keep windows whose test is within 2022-2025
        if test_dates[0] < "2022-01-01" or test_dates[-1] > "2025-12-31":
            t0 += step
            continue
        windows.append({
            "window_id": wi,
            "train_end": train_dates[-1],
            "train_start": train_dates[0],
            "n_train_dates": len(train_dates),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "n_test_dates": len(test_dates),
        })
        wi += 1
        t0 += step
    # Ensure min 6
    if len(windows) < 6:
        # relax: start earlier
        windows = []
        t0 = date_to_idx[[d for d in dates if d >= "2022-01-03"][0]]
        wi = 0
        while wi < 8 and t0 + test_len <= len(dates):
            train_dates = dates[:t0]
            test_dates = dates[t0:t0 + test_len]
            windows.append({
                "window_id": wi,
                "train_end": train_dates[-1],
                "train_start": train_dates[0],
                "n_train_dates": len(train_dates),
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "n_test_dates": len(test_dates),
            })
            wi += 1
            t0 += step

    return {
        "train": {"start": train[0] if train else None, "end": train[-1] if train else None, "n_dates": len(train)},
        "valid": {"start": valid[0] if valid else None, "end": valid[-1] if valid else None, "n_dates": len(valid)},
        "test": {"start": test[0] if test else None, "end": test[-1] if test else None, "n_dates": len(test)},
        "walk_forward": windows,
    }


def quality_checks(df_pub: pd.DataFrame, df_full: pd.DataFrame, splits: dict, out_path: Path) -> str:
    lines = []
    lines.append("QForge-FIN v1.0 Quality Checks")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Master seed: {MASTER_SEED}")
    lines.append("=" * 60)

    # 1. NaNs after warmup
    feat_cols = PUBLIC_FEATURES + PUBLIC_TARGETS
    nan_counts = df_pub[feat_cols].isna().sum()
    # Targets have trailing NaNs by design (last 21 days) — check features primarily
    feat_nans = df_pub[PUBLIC_FEATURES].isna().sum().sum()
    lines.append(f"\n[1] NaNs in public features after warmup: {feat_nans}")
    lines.append(f"    Feature NaN counts:\n{nan_counts[PUBLIC_FEATURES].to_string()}")
    # Drop rows with any feature NaN for moment checks
    df_c = df_pub.dropna(subset=PUBLIC_FEATURES).copy()
    lines.append(f"    Rows after dropna(features): {len(df_c)} (from {len(df_pub)})")

    # 2. No future columns in public features
    forbidden = ["_fwd", "future", "lead", "y_", "oracle", "regime", "crash", "growth", "inflation"]
    bad = [c for c in PUBLIC_FEATURES if any(f in c.lower() for f in ["_fwd", "future", "lead", "oracle"])]
    # y_ should not be in features
    y_in_feat = [c for c in PUBLIC_FEATURES if c.startswith("y_")]
    lines.append(f"\n[2] Future/leak columns in public features: {bad + y_in_feat}")
    lines.append(f"    Public feature list: {PUBLIC_FEATURES}")
    lines.append(f"    PASS: no look-ahead feature columns" if not (bad or y_in_feat) else "    FAIL")

    # 3. Moments
    lines.append("\n[3] Return moments by asset:")
    crash_day_fracs = []
    for aid, sector, _ in ASSETS:
        sub = df_c[df_c["asset_id"] == aid]["ret_1d"]
        mu = sub.mean()
        ann_vol = sub.std() * np.sqrt(252) * 100
        crash_frac = (sub < -0.05).mean() * 100  # < -5% days
        crash_day_fracs.append(crash_frac)
        lines.append(f"    {aid} ({sector}): mean_daily={mu:.6f}, ann_vol={ann_vol:.1f}%, crash_days(<-5%)={crash_frac:.2f}%")
    lines.append(f"    Mean daily ret near 0: overall mean={df_c['ret_1d'].mean():.6f}")
    lines.append(f"    Ann vol range: check 15–80% by asset (see above)")
    lines.append(f"    Crash days <4%: max={max(crash_day_fracs):.2f}%, mean={np.mean(crash_day_fracs):.2f}%")

    # 4. Corr matrix block by sector
    lines.append("\n[4] Correlation matrix (ret_1d, pivot by asset):")
    pivot = df_c.pivot_table(index="date", columns="asset_id", values="ret_1d")
    corr = pivot.corr()
    lines.append(corr.round(3).to_string())
    # Sector block means
    lines.append("\n    Mean within-sector vs cross-sector corr:")
    asset_sector = {a: s for a, s, _ in ASSETS}
    within, cross = [], []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            if asset_sector[a] == asset_sector[b]:
                within.append(corr.loc[a, b])
            else:
                cross.append(corr.loc[a, b])
    lines.append(f"    within-sector mean corr={np.mean(within):.3f}, cross-sector mean corr={np.mean(cross):.3f}")

    # 5. y_dir_1d balance
    ybal = df_c["y_dir_1d"].dropna()
    pos = (ybal == 1).mean() * 100
    lines.append(f"\n[5] y_dir_1d balance: positive={pos:.2f}% (target 45–55%)")

    # 6. Split counts
    dates = sorted(df_c["date"].unique())
    n_train = sum(1 for d in dates if d < "2023-01-01")
    n_valid = sum(1 for d in dates if "2023-01-01" <= d < "2024-07-01")
    n_test = sum(1 for d in dates if d >= "2024-07-01")
    rows_train = len(df_c[df_c["date"] < "2023-01-01"])
    rows_valid = len(df_c[(df_c["date"] >= "2023-01-01") & (df_c["date"] < "2024-07-01")])
    rows_test = len(df_c[df_c["date"] >= "2024-07-01"])
    lines.append(f"\n[6] Train/valid/test counts:")
    lines.append(f"    Dates: train={n_train}, valid={n_valid}, test={n_test}")
    lines.append(f"    Rows:  train={rows_train}, valid={rows_valid}, test={rows_test}")
    lines.append(f"    Walk-forward windows: {len(splits['walk_forward'])}")
    for w in splits["walk_forward"]:
        lines.append(f"      W{w['window_id']}: train_end={w['train_end']}, test={w['test_start']}→{w['test_end']} (n_test_dates={w['n_test_dates']})")

    lines.append("\n" + "=" * 60)
    lines.append("Quality checks complete.")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    return text


def main():
    root = Path("/workspace/QForge-FIN")
    release = root / "QForge-FIN_v1_release"
    release.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(MASTER_SEED)
    dates = business_days("2018-01-02", "2025-12-31")
    n = len(dates)
    print(f"Business days: {n} ({dates[0].date()} → {dates[-1].date()})")

    macros, crash_mask, crash_windows = generate_latent_macros(n, np.random.default_rng(sub_seed("macros")))
    regimes = hidden_markov_regimes(n, crash_mask, np.random.default_rng(sub_seed("regimes")))
    returns, vols, asset_meta = generate_returns(
        n, macros, regimes, crash_mask, np.random.default_rng(sub_seed("returns"))
    )
    prices = prices_from_returns(returns, 100.0)
    bid, ask, volume, spread_bps = microstructure(
        prices, returns, np.random.default_rng(sub_seed("microstructure"))
    )

    df_full = build_panel(dates, returns, prices, bid, ask, volume, spread_bps, regimes, macros, asset_meta, crash_mask)
    df_full = drop_warmup(df_full, 63)

    # Public ML file
    pub_cols = (
        ["date", "asset_id", "sector", "sector_id", "open", "high", "low", "close", "volume", "bid", "ask", "spread_bps"]
        + PUBLIC_FEATURES
        + PUBLIC_TARGETS
    )
    # sector_id already in features and identity — keep once
    pub_cols = list(dict.fromkeys(pub_cols))
    df_pub = df_full[pub_cols].copy()

    # Oracle factors file
    oracle_cols = ["date", "asset_id"] + ORACLE_COLS
    df_oracle = df_full[oracle_cols].copy()
    df_oracle = df_oracle.rename(columns={c: c.lstrip("_") for c in ORACLE_COLS})

    # Market file (cross-sectional daily)
    mkt = (
        df_pub.groupby("date")
        .agg(
            mkt_ret_1d=("ret_1d", "mean"),
            mkt_vol_proxy=("ret_1d", "std"),
            n_assets=("asset_id", "count"),
            mean_volume=("volume", "mean"),
            mean_spread_bps=("spread_bps", "mean"),
        )
        .reset_index()
    )

    dates_list = sorted(df_pub["date"].unique())
    splits = make_splits(dates_list)

    # Save panel
    panel_parquet = release / "qforge_fin_v1_panel.parquet"
    panel_csv = release / "qforge_fin_v1_panel.csv"
    df_pub.to_parquet(panel_parquet, index=False)
    df_pub.to_csv(panel_csv, index=False)
    mkt.to_csv(release / "qforge_fin_v1_market.csv", index=False)
    df_oracle.to_csv(release / "qforge_fin_v1_oracle_factors.csv", index=False)
    df_pub.head(20).to_csv(release / "qforge_fin_v1_preview_20rows.csv", index=False)

    with open(release / "qforge_fin_v1_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    metadata = {
        "brand": BRAND,
        "full_name": "Quantum-Forge Finance Benchmark",
        "version": VERSION,
        "citation": CITATION,
        "license_data": "CC-BY 4.0",
        "license_code": "MIT",
        "author": "Hossein Tabasi",
        "master_seed": MASTER_SEED,
        "sub_seeds": {
            "macros": sub_seed("macros"),
            "regimes": sub_seed("regimes"),
            "returns": sub_seed("returns"),
            "microstructure": sub_seed("microstructure"),
            **{f"asset_{a}": sub_seed(f"asset_{a}") for a, _, _ in ASSETS},
        },
        "n_assets": 12,
        "assets": [{"id": a, "sector": s, "sector_id": sid} for a, s, sid in ASSETS],
        "date_range_raw": ["2018-01-02", "2025-12-31"],
        "warmup_days_dropped": 63,
        "n_rows": len(df_pub),
        "n_dates": len(dates_list),
        "date_range_panel": [dates_list[0], dates_list[-1]],
        "public_features": PUBLIC_FEATURES,
        "targets": PUBLIC_TARGETS,
        "oracle_file": "qforge_fin_v1_oracle_factors.csv",
        "limitations": [
            "100% synthetic; not real market data; no investment advice.",
            "Generative process is stylized (AR(1)+GARCH-like+Student-t); not calibrated to a specific exchange.",
            "Hidden regimes and oracle factors are NOT available in the public ML panel (no look-ahead).",
            "NISQ experiments use small-n (n=400) and 4–8 qubits; results do not imply quantum advantage on real hardware.",
            "Crash windows and fat tails are simulated; frequency and severity are design choices.",
        ],
        "qubit_recommendations": {
            "primary": 4,
            "optional": 6,
            "encoding": "PCA(n_qubits) on public features, StandardScaler fit on train only",
            "depth": "2–3 variational layers",
            "backend": "statevector preferred; optional finite shots",
        },
        "crash_windows_indices_raw": crash_windows,
        "generative_notes": (
            "Latent macros AR(1)+jumps; asset returns = beta·factors + idiosyncratic GARCH(1,1)-like "
            "with Student-t (df 5–8); crypto higher vol/crash beta; prices from 100 via compound simple returns; "
            "microstructure bid-ask + lognormal volume; hidden 3-state Markov Calm/Stress/Crash."
        ),
    }
    with open(release / "qforge_fin_v1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Codebook
    codebook = f"""# QForge-FIN v1.0 Codebook

**Brand:** Quantum-Forge Finance Benchmark  
**Version:** {VERSION}  
**Author:** Hossein Tabasi  
**Citation:** {CITATION}  
**Data license:** CC-BY 4.0  
**Master seed:** {MASTER_SEED}

## Assets

| ID | Sector | sector_id |
|----|--------|-----------|
""" + "\n".join(f"| {a} | {s} | {sid} |" for a, s, sid in ASSETS) + """

## Public panel columns

### Identifiers & prices
- `date` — ISO business date
- `asset_id` — QF01–QF12
- `sector`, `sector_id` — sector label and integer id
- `open`, `high`, `low`, `close` — synthetic OHLC
- `volume` — lognormal microstructure volume
- `bid`, `ask`, `spread_bps` — bid-ask proxy

### Features (no look-ahead; available at close of `date`)
- `ret_1d`, `ret_5d`, `ret_21d` — trailing simple-return sums
- `vol_21d`, `vol_63d` — trailing return std
- `rsi_14` — 14-day RSI on close
- `momentum_10`, `momentum_21` — price percent change
- `volume_z_21` — volume z-score vs 21-day window
- `range_hl` — (high-low)/close
- `sector_id` — categorical sector encoding
- `mkt_ret_1d`, `mkt_vol_21d` — equal-weight market return and 21d vol

### Targets (aligned; use only as labels)
- `y_dir_1d` — 1 if next-day return > 0 else 0
- `y_dir_5d` — 1 if sum of returns over t+1..t+5 > 0
- `y_exret_5d` — asset forward 5d return minus market forward 5d return
- `y_stress_5d` — 1 if stress/crash-like conditions in next 5 days
- `rank_21d` — cross-sectional percentile rank of forward 21d return

## Oracle file (separate; NOT for public ML features)
`qforge_fin_v1_oracle_factors.csv`: regime, crash flag, latent macros, forward 21d return.  
Do not merge into training features for leakage-free experiments.

## Splits
- Train: date < 2023-01-01
- Valid: 2023-01-01 ≤ date < 2024-07-01
- Test: date ≥ 2024-07-01
- Walk-forward: see `qforge_fin_v1_splits.json`

## Regeneration
```bash
python scripts/generate_qforge_fin.py
```
Master seed **20260907** plus documented sub-seeds in metadata yield bit-identical outputs (same library versions).
"""
    (release / "qforge_fin_v1_codebook.md").write_text(codebook)

    # Copy generator into release
    import shutil
    src_gen = Path(__file__).resolve()
    shutil.copy(src_gen, release / "generate_qforge_fin.py")
    # Also mirror at repo root scripts already is source

    quality_checks(df_pub, df_full, splits, release / "quality_checks.txt")
    # Also save at repo root reports
    (root / "reports").mkdir(exist_ok=True)
    shutil.copy(release / "quality_checks.txt", root / "reports" / "quality_checks.txt")

    print(f"\nWrote panel rows={len(df_pub)} dates={len(dates_list)} → {release}")


if __name__ == "__main__":
    main()
