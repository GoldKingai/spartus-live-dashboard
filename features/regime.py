"""Intermarket regime detection features (Upgrade 4): 2 features.

IDENTICAL to training regime_features.py logic.
Both features get z-score normalization (market_feature_names).

Features:
    - corr_gold_usd_100: 100-bar rolling Pearson correlation between
      XAUUSD 5-bar returns and EURUSD 5-bar returns.
    - corr_gold_spx_100: 100-bar rolling Pearson correlation between
      XAUUSD 5-bar returns and US500 5-bar returns.

Uses 5-bar returns (not 1-bar) to reduce noise in correlation estimates.
100-bar rolling window (~8.3 hours on M5).
ffill(limit=60) for instrument closures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_regime_features(
    xau_m5: pd.DataFrame,
    eurusd_m5: Optional[pd.DataFrame] = None,
    us500_m5: Optional[pd.DataFrame] = None,
    regime_corr_window: int = 100,
) -> Dict[str, float]:
    """Compute 2 regime detection features for the latest XAUUSD M5 bar.

    IDENTICAL to training regime_features.calc_regime_features.

    Args:
        xau_m5: XAUUSD M5 OHLCV with 'time', 'close' columns.
            At least 105 bars (100 window + 5 for returns).
        eurusd_m5: EURUSD M5 OHLCV (may be None).
        us500_m5: US500 M5 OHLCV (may be None).
        regime_corr_window: Rolling correlation window (default 100).

    Returns:
        Dict mapping feature_name -> float value (2 features).
        0.0 (neutral) when correlated data is unavailable.
    """
    # Gold 5-bar returns
    gold_close = xau_m5["close"]
    gold_ret5 = np.log(gold_close / gold_close.shift(5))

    # Gold-USD correlation
    corr_usd = _rolling_corr_aligned(
        xau_m5, gold_ret5, eurusd_m5, regime_corr_window
    )

    # Gold-SPX correlation
    corr_spx = _rolling_corr_aligned(
        xau_m5, gold_ret5, us500_m5, regime_corr_window
    )

    return {
        "corr_gold_usd_100": corr_usd,
        "corr_gold_spx_100": corr_spx,
    }


def _rolling_corr_aligned(
    xau_m5: pd.DataFrame,
    gold_ret5: pd.Series,
    other_m5: Optional[pd.DataFrame],
    window: int,
) -> float:
    """Compute rolling Pearson correlation between gold and another instrument.

    Aligns other instrument to gold timeline via merge_asof(direction='backward').
    Returns the latest correlation value.

    Matches training regime_features._rolling_corr_aligned exactly.
    """
    if other_m5 is None or other_m5.empty:
        return 0.0

    other = other_m5[["time", "close"]].copy().sort_values("time")
    other["other_ret5"] = np.log(other["close"] / other["close"].shift(5))

    # Align to gold timeline
    merged = pd.merge_asof(
        xau_m5[["time"]].copy().sort_values("time"),
        other[["time", "other_ret5"]],
        on="time",
        direction="backward",
    )

    # Forward-fill up to 60 bars (5 hours) -- beyond that too stale
    other_ret5_aligned = merged["other_ret5"].ffill(limit=60)

    # Rolling Pearson correlation
    corr = gold_ret5.rolling(window, min_periods=50).corr(other_ret5_aligned)

    # Get latest value, clip to valid range
    val = corr.iloc[-1]
    if np.isnan(val):
        return 0.0

    return float(np.clip(val, -1.0, 1.0))
