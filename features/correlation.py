"""Correlated instrument features (Upgrade 1): 11 features.

IDENTICAL to training correlation_features.py logic.
All 11 features get z-score normalization (market_feature_names).

For each correlated instrument, computes from their M5 bars:
    - returns_20: log return 20 bars
    - rsi_14: RSI(14) / 100 (for instruments that have it)
    - trend: EMA(20) slope over 3 bars (for instruments that have it)

Plus gold_silver_ratio_z = close_xau / close_xag (RAW ratio, normalizer handles z-scoring).

EURUSD: returns_20, rsi_14, trend (3 features)
XAGUSD: returns_20, rsi_14 (2 features)
USDJPY: returns_20, trend (2 features)
US500:  returns_20, rsi_14 (2 features)
USOIL:  returns_20 (1 feature)
gold_silver_ratio_z (1 feature)

NEUTRAL_FILLS for missing instruments:
    - RSI features: 0.5
    - Returns/trend/ratio: 0.0

Uses merge_asof backward for time alignment.
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, Optional


# Feature definitions per instrument (matches training exactly)
_INSTRUMENT_FEATURES = {
    "EURUSD": ("returns", "rsi", "trend"),
    "XAGUSD": ("returns", "rsi"),
    "USDJPY": ("returns", "trend"),
    "US500":  ("returns", "rsi"),
    "USOIL":  ("returns",),
}

# Neutral fill values per feature type
_NEUTRAL_RSI = 0.5
_NEUTRAL_OTHER = 0.0


def compute_correlation_features(
    xau_m5: pd.DataFrame,
    correlated_m5: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, float]:
    """Compute all 11 correlated instrument features for the latest XAUUSD M5 bar.

    IDENTICAL to training correlation_features.calc_correlation_features.

    Args:
        xau_m5: XAUUSD M5 OHLCV with 'time', 'close' columns. At least 21 bars.
        correlated_m5: Dict mapping symbol name to M5 OHLCV DataFrames.
            Expected keys: "EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL".
            None or empty dict if correlated data not available.

    Returns:
        Dict mapping feature_name -> float value for the latest bar (11 features).
        Missing instruments produce neutral fill values.
    """
    if correlated_m5 is None:
        correlated_m5 = {}

    current_time = xau_m5["time"].iloc[-1]
    features = {}

    # Per-instrument features
    for symbol, feat_types in _INSTRUMENT_FEATURES.items():
        prefix = symbol.lower()
        df = correlated_m5.get(symbol)

        if df is not None and not df.empty:
            inst_feats = _calc_instrument_features(df, prefix, feat_types, current_time)
            features.update(inst_feats)
        else:
            # Fill with neutral values
            for ft in feat_types:
                if ft == "returns":
                    features[f"{prefix}_returns_20"] = _NEUTRAL_OTHER
                elif ft == "rsi":
                    features[f"{prefix}_rsi_14"] = _NEUTRAL_RSI
                elif ft == "trend":
                    features[f"{prefix}_trend"] = _NEUTRAL_OTHER

    # Gold/silver ratio
    features["gold_silver_ratio_z"] = _calc_gold_silver_ratio(
        xau_m5, correlated_m5.get("XAGUSD"), current_time
    )

    return features


def _calc_instrument_features(
    df: pd.DataFrame,
    prefix: str,
    feat_types: tuple,
    current_time,
) -> Dict[str, float]:
    """Compute selected features for one instrument at the latest time.

    Matches training _calc_instrument_features exactly.

    Args:
        df: M5 OHLCV for the instrument.
        prefix: Feature name prefix (e.g., "eurusd").
        feat_types: Tuple of feature types to compute.
        current_time: Current XAUUSD bar timestamp for alignment.

    Returns:
        Dict of feature_name -> float for the latest aligned bar.
    """
    result = {}

    df = df.copy().sort_values("time").reset_index(drop=True)

    # Only use bars at or before current time (no look-ahead)
    mask = df["time"] <= current_time
    if mask.sum() < 21:
        # Not enough data -- return neutrals
        for ft in feat_types:
            if ft == "returns":
                result[f"{prefix}_returns_20"] = _NEUTRAL_OTHER
            elif ft == "rsi":
                result[f"{prefix}_rsi_14"] = _NEUTRAL_RSI
            elif ft == "trend":
                result[f"{prefix}_trend"] = _NEUTRAL_OTHER
        return result

    df_valid = df.loc[mask]
    close = df_valid["close"]

    if "returns" in feat_types:
        if len(close) >= 21:
            val = float(np.log(close.iloc[-1] / close.iloc[-21]))
            result[f"{prefix}_returns_20"] = val if not np.isnan(val) else _NEUTRAL_OTHER
        else:
            result[f"{prefix}_returns_20"] = _NEUTRAL_OTHER

    if "rsi" in feat_types:
        if len(close) >= 15:
            rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            val = rsi.iloc[-1] / 100.0
            result[f"{prefix}_rsi_14"] = float(val) if not np.isnan(val) else _NEUTRAL_RSI
        else:
            result[f"{prefix}_rsi_14"] = _NEUTRAL_RSI

    if "trend" in feat_types:
        if len(close) >= 21:
            ema_20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            if len(ema_20.dropna()) >= 4:
                slope = (ema_20.iloc[-1] - ema_20.iloc[-4]) / (close.iloc[-1] * 0.01 + 1e-8)
                val = float(np.clip(slope, -3.0, 3.0))
                result[f"{prefix}_trend"] = val if not np.isnan(val) else _NEUTRAL_OTHER
            else:
                result[f"{prefix}_trend"] = _NEUTRAL_OTHER
        else:
            result[f"{prefix}_trend"] = _NEUTRAL_OTHER

    return result


def _calc_gold_silver_ratio(
    xau_m5: pd.DataFrame,
    xag_m5: Optional[pd.DataFrame],
    current_time,
) -> float:
    """Compute gold/silver price ratio for the latest bar.

    Returns RAW ratio (z-score normalization is handled by the pipeline normalizer).
    Matches training _calc_gold_silver_ratio_z logic.

    Args:
        xau_m5: XAUUSD M5 OHLCV.
        xag_m5: XAGUSD M5 OHLCV (may be None).
        current_time: Current timestamp.

    Returns:
        Gold/silver ratio, or 0.0 if silver data unavailable.
    """
    if xag_m5 is None or xag_m5.empty:
        return _NEUTRAL_OTHER

    xag = xag_m5.copy().sort_values("time")
    mask = xag["time"] <= current_time
    if mask.sum() == 0:
        return _NEUTRAL_OTHER

    xag_valid = xag.loc[mask]
    xag_close = xag_valid["close"].iloc[-1]
    xau_close = xau_m5["close"].iloc[-1]

    ratio = xau_close / (xag_close + 1e-8)

    return float(ratio) if not np.isnan(ratio) else _NEUTRAL_OTHER
