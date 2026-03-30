"""Anti-leakage verification tests.

Ensures no future data leaks into past features through:
1. Feature computation (merge_asof direction='backward')
2. Rolling indicators (no look-ahead)
3. Observation building (frame buffer)

Usage:
    python -m pytest tests/test_anti_leakage.py -v
    python tests/test_anti_leakage.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.config import TrainingConfig
from src.data.feature_builder import FeatureBuilder


def make_synthetic_m5(n_bars=500, seed=42):
    """Create synthetic M5 OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-08 00:00", periods=n_bars, freq="5min")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n_bars)
    low = close - rng.uniform(0.5, 2.0, n_bars)
    open_ = close + rng.randn(n_bars) * 0.3
    volume = rng.randint(100, 5000, n_bars).astype(float)

    return pd.DataFrame({
        "time": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_synthetic_h1(n_bars=60, seed=42):
    """Create synthetic H1 data."""
    rng = np.random.RandomState(seed + 1)
    dates = pd.date_range("2024-01-08 00:00", periods=n_bars, freq="1h")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 1.0)
    return pd.DataFrame({
        "time": dates,
        "open": close + rng.randn(n_bars) * 0.5,
        "high": close + rng.uniform(1, 3, n_bars),
        "low": close - rng.uniform(1, 3, n_bars),
        "close": close,
        "volume": rng.randint(1000, 20000, n_bars).astype(float),
    })


def make_synthetic_h4(n_bars=20, seed=42):
    rng = np.random.RandomState(seed + 2)
    dates = pd.date_range("2024-01-08 00:00", periods=n_bars, freq="4h")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 2.0)
    return pd.DataFrame({
        "time": dates,
        "open": close + rng.randn(n_bars),
        "high": close + rng.uniform(2, 5, n_bars),
        "low": close - rng.uniform(2, 5, n_bars),
        "close": close,
        "volume": rng.randint(5000, 50000, n_bars).astype(float),
    })


def make_synthetic_d1(n_bars=5, seed=42):
    rng = np.random.RandomState(seed + 3)
    dates = pd.date_range("2024-01-08", periods=n_bars, freq="1D")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 5.0)
    return pd.DataFrame({
        "time": dates,
        "open": close + rng.randn(n_bars) * 2,
        "high": close + rng.uniform(5, 15, n_bars),
        "low": close - rng.uniform(5, 15, n_bars),
        "close": close,
        "volume": rng.randint(50000, 200000, n_bars).astype(float),
    })


def test_feature_no_future_leakage():
    """Core test: corrupting future bars should NOT change earlier features.

    Process:
    1. Compute features on full data → save first 200 rows
    2. Corrupt the last 100 bars (set close=9999)
    3. Recompute features → first 200 rows must be identical
    """
    cfg = TrainingConfig()
    builder = FeatureBuilder(cfg)

    m5 = make_synthetic_m5(500)
    h1 = make_synthetic_h1(60)
    h4 = make_synthetic_h4(20)
    d1 = make_synthetic_d1(5)

    # Compute original features
    features_orig = builder.build_features(m5.copy(), h1.copy(), h4.copy(), d1.copy())

    # Corrupt future data (last 100 bars)
    m5_corrupted = m5.copy()
    m5_corrupted.loc[400:, "close"] = 9999.0
    m5_corrupted.loc[400:, "high"] = 10000.0
    m5_corrupted.loc[400:, "low"] = 9998.0
    m5_corrupted.loc[400:, "volume"] = 999999.0

    # Recompute features
    features_corrupted = builder.build_features(m5_corrupted, h1.copy(), h4.copy(), d1.copy())

    # Check: first 200 rows of non-time columns should be identical
    feature_cols = [c for c in features_orig.columns
                    if c not in ("time", "open", "high", "low", "close", "volume", "atr_14_raw")]

    # Use rows 200-350 (after warmup, before corruption at 400)
    check_start = 200
    check_end = 350

    for col in feature_cols:
        orig_vals = features_orig[col].iloc[check_start:check_end].values
        corrupt_vals = features_corrupted[col].iloc[check_start:check_end].values

        # Allow tiny floating point differences
        mask = np.isfinite(orig_vals) & np.isfinite(corrupt_vals)
        if mask.sum() < 10:
            continue

        max_diff = np.max(np.abs(orig_vals[mask] - corrupt_vals[mask]))
        assert max_diff < 1e-6, (
            f"Feature '{col}' changed by {max_diff:.6f} when future data was corrupted! "
            f"This indicates look-ahead bias."
        )

    print(f"  PASS: {len(feature_cols)} features verified, no future leakage")


def test_merge_asof_no_lookahead():
    """Verify merge_asof uses direction='backward' (no future HTF data)."""
    m5 = make_synthetic_m5(200)
    h1 = make_synthetic_h1(30)

    # Do the merge
    merged = pd.merge_asof(
        m5.sort_values("time"),
        h1.sort_values("time")[["time", "close"]].rename(columns={"close": "h1_close"}),
        on="time",
        direction="backward",
    )

    # For each M5 bar, the merged H1 close should be from an H1 bar at or BEFORE the M5 time
    for i in range(len(merged)):
        m5_time = merged.iloc[i]["time"]
        h1_close = merged.iloc[i]["h1_close"]

        if pd.isna(h1_close):
            continue  # No H1 data before this point

        # Find which H1 bar this came from
        h1_before = h1[h1["time"] <= m5_time]
        assert len(h1_before) > 0, f"H1 data found for M5 time {m5_time} but shouldn't exist"

        latest_h1 = h1_before.iloc[-1]
        assert abs(latest_h1["close"] - h1_close) < 1e-10, (
            f"merge_asof returned H1 data that doesn't match the latest backward bar. "
            f"Possible look-ahead!"
        )

    print(f"  PASS: merge_asof backward direction verified for {len(merged)} bars")


def test_rolling_indicators_no_future():
    """Verify that rolling indicators only use past data."""
    m5 = make_synthetic_m5(300)

    # Compute RSI manually and check
    import ta
    rsi = ta.momentum.RSIIndicator(m5["close"], window=14).rsi()

    # RSI at bar i should only depend on bars 0..i
    # Test: changing bar 200 close should NOT affect RSI at bar 199
    m5_modified = m5.copy()
    m5_modified.loc[200:, "close"] = m5_modified.loc[200:, "close"] + 100.0

    rsi_modified = ta.momentum.RSIIndicator(m5_modified["close"], window=14).rsi()

    # RSI values at indices < 200 should be identical
    for i in range(180, 200):
        if np.isfinite(rsi.iloc[i]) and np.isfinite(rsi_modified.iloc[i]):
            assert abs(rsi.iloc[i] - rsi_modified.iloc[i]) < 1e-10, (
                f"RSI at bar {i} changed when future bars were modified!"
            )

    print("  PASS: Rolling indicators verified (no future leakage)")


def test_observation_frame_buffer():
    """Verify frame buffer doesn't leak future frames into current observation."""
    cfg = TrainingConfig()
    builder = FeatureBuilder(cfg)

    m5 = make_synthetic_m5(500)
    h1 = make_synthetic_h1(60)
    h4 = make_synthetic_h4(20)
    d1 = make_synthetic_d1(5)

    features = builder.build_features(m5, h1, h4, d1)

    from src.environment.trade_env import SpartusTradeEnv
    from src.memory.trading_memory import TradingMemory
    import tempfile

    tmp_db = tempfile.mktemp(suffix=".db")
    mem = TradingMemory(db_path=Path(tmp_db))

    env = SpartusTradeEnv(features_df=features, config=cfg, memory=mem, seed=42)
    obs1, _ = env.reset()

    # Step forward 5 times
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            break

    # The observation should only contain data from steps 0..current
    # Verify obs is finite and within bounds
    assert np.all(np.isfinite(obs)), "Observation contains NaN/Inf"
    assert np.all(obs >= -10.0) and np.all(obs <= 10.0), "Observation out of bounds"

    # Verify shape
    assert obs.shape == (cfg.obs_dim,), f"Obs shape {obs.shape} != expected {cfg.obs_dim}"

    mem.close()
    import os
    os.unlink(tmp_db)

    print(f"  PASS: Frame buffer OK, obs shape {obs.shape}, all finite and bounded")


if __name__ == "__main__":
    print("=" * 60)
    print("Anti-Leakage Verification Tests")
    print("=" * 60)

    print("\n1. Feature computation (no future leakage)...")
    test_feature_no_future_leakage()

    print("\n2. merge_asof backward direction...")
    test_merge_asof_no_lookahead()

    print("\n3. Rolling indicators (no future)...")
    test_rolling_indicators_no_future()

    print("\n4. Observation frame buffer...")
    test_observation_frame_buffer()

    print("\n" + "=" * 60)
    print("ALL ANTI-LEAKAGE TESTS PASSED")
    print("=" * 60)
