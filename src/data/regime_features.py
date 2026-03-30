"""Intermarket regime detection features (Upgrade 4).

2 features capturing gold's shifting correlations with other markets:
- corr_gold_usd_100: 100-bar rolling correlation of gold vs EURUSD 5-bar returns
- corr_gold_spx_100: 100-bar rolling correlation of gold vs US500 5-bar returns

Both features get z-score normalization (added to market_feature_names).
Prerequisite: Upgrade 1 data (EURUSD, US500 M5 data must be available).
"""

import numpy as np
import pandas as pd

from src.config import TrainingConfig


def calc_regime_features(
    xau_m5: pd.DataFrame,
    eurusd_m5: pd.DataFrame,
    us500_m5: pd.DataFrame,
    config: TrainingConfig = None,
) -> pd.DataFrame:
    """Compute 2 regime detection features.

    Uses 5-bar returns (not 1-bar) to reduce noise in correlation estimates.
    100-bar rolling window (~8.3 hours on M5).

    Args:
        xau_m5: XAUUSD M5 OHLCV with 'time', 'close' columns.
        eurusd_m5: EURUSD M5 OHLCV (may be None).
        us500_m5: US500 M5 OHLCV (may be None).
        config: TrainingConfig.

    Returns:
        DataFrame with 2 columns: corr_gold_usd_100, corr_gold_spx_100.
        Indexed like xau_m5.
    """
    cfg = config or TrainingConfig()
    window = cfg.regime_corr_window  # 100
    n = len(xau_m5)

    # Gold 5-bar returns
    gold_ret5 = np.log(xau_m5["close"] / xau_m5["close"].shift(5))

    # --- Gold-USD correlation ---
    corr_usd = _rolling_corr_aligned(
        xau_m5, gold_ret5, eurusd_m5, window
    )

    # --- Gold-SPX correlation ---
    corr_spx = _rolling_corr_aligned(
        xau_m5, gold_ret5, us500_m5, window
    )

    return pd.DataFrame({
        "corr_gold_usd_100": corr_usd,
        "corr_gold_spx_100": corr_spx,
    })


def _rolling_corr_aligned(
    xau_m5: pd.DataFrame,
    gold_ret5: pd.Series,
    other_m5: pd.DataFrame,
    window: int,
) -> np.ndarray:
    """Compute rolling Pearson correlation between gold and another instrument.

    Aligns other instrument to gold timeline via merge_asof(direction='backward').
    Returns array of correlation values, NaN-filled where insufficient data.
    """
    n = len(xau_m5)

    if other_m5 is None or other_m5.empty:
        return np.zeros(n, dtype=np.float64)

    other = other_m5[["time", "close"]].copy().sort_values("time")
    other["other_ret5"] = np.log(other["close"] / other["close"].shift(5))

    # Align to gold timeline
    merged = pd.merge_asof(
        xau_m5[["time"]].copy().sort_values("time"),
        other[["time", "other_ret5"]],
        on="time",
        direction="backward",
    )

    # Forward-fill up to 60 bars (5 hours) — beyond that the data is too stale
    # for meaningful correlation. During instrument closures, returns stay NaN
    # and rolling correlation gracefully degrades via min_periods.
    other_ret5_aligned = merged["other_ret5"].ffill(limit=60)

    # Rolling Pearson correlation
    corr = gold_ret5.rolling(window, min_periods=50).corr(other_ret5_aligned)

    # Clip to valid range and fill NaN (0.0 = no correlation = neutral)
    return corr.clip(-1.0, 1.0).fillna(0.0).values
