"""Leakage and schema tests for QForge-FIN-X v1.0. Author: Hossein Tabasi."""
from pathlib import Path
import json
import pandas as pd

RELEASE = Path(__file__).resolve().parents[1] / "EQUAL-Q_release" / "QForge-FIN-X_v1_release"
FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "vol_21d", "vol_63d", "rsi_14",
    "momentum_10", "momentum_21", "volume_z_21", "range_hl", "sector_id",
    "mkt_ret_1d", "mkt_vol_21d", "iv_21d", "iv_skew", "attn_sector", "attn_mkt", "attn_dispersion",
]
FORBIDDEN_SUBSTR = ["_fwd", "future", "lead", "oracle", "regime", "crash", "growth", "inflation", "liquidity", "risk_appetite"]


def test_no_leakage_in_public_features():
    df = pd.read_parquet(RELEASE / "qforge_fin_x_v1_panel.parquet")
    for c in FEATURES:
        assert c in df.columns
        assert not c.startswith("y_")
        assert not any(f in c.lower() for f in FORBIDDEN_SUBSTR)
    # oracle columns must not be in public panel
    for bad in ["regime", "crash", "growth", "inflation", "risk_appetite", "liquidity", "p_calm", "p_stress", "p_crash"]:
        assert bad not in df.columns


def test_feature_nans_absent():
    df = pd.read_parquet(RELEASE / "qforge_fin_x_v1_panel.parquet")
    assert df[FEATURES].isna().sum().sum() == 0


def test_splits_ordered():
    with open(RELEASE / "qforge_fin_x_v1_splits.json") as f:
        splits = json.load(f)
    assert splits["train"]["end"] < splits["valid"]["start"]
    assert splits["valid"]["end"] < splits["test"]["start"]
    assert len(splits["walk_forward"]) >= 6
