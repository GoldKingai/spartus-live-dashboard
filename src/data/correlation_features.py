"""Correlated instrument features (Upgrade 1).

11 features from 5 instruments (EURUSD, XAGUSD, USDJPY, US500, USOIL):
- Per-instrument: 20-bar log returns, RSI(14), EMA(20) trend slope
- Cross-ratio: gold/silver ratio z-score

All features get z-score normalization (added to market_feature_names).
Uses merge_asof(direction='backward') for timestamp alignment — no look-ahead bias.
"""

import numpy as np
import pandas as pd
import ta

from src.config import TrainingConfig


def calc_correlation_features(
    xau_m5: pd.DataFrame,
    correlated_m5: dict,
    config: TrainingConfig = None,
) -> pd.DataFrame:
    """Compute 11 correlated instrument features aligned to XAUUSD M5 timeline.

    Args:
        xau_m5: XAUUSD M5 OHLCV with 'time', 'close' columns.
        correlated_m5: Dict mapping symbol name to M5 OHLCV DataFrames.
            Expected keys: "EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL".
        config: TrainingConfig.

    Returns:
        DataFrame with 11 feature columns, indexed like xau_m5.
    """
    cfg = config or TrainingConfig()
    n = len(xau_m5)

    # Build base timeline for merge_asof
    base = xau_m5[["time"]].copy().sort_values("time").reset_index(drop=True)

    # --- Per-instrument features ---
    # EURUSD: returns_20, rsi_14, trend (3 features)
    eurusd_feats = _calc_instrument_features(
        correlated_m5.get("EURUSD"), "eurusd", features=("returns", "rsi", "trend")
    )
    base = _merge_features(base, eurusd_feats)

    # XAGUSD: returns_20, rsi_14 (2 features)
    xagusd_feats = _calc_instrument_features(
        correlated_m5.get("XAGUSD"), "xagusd", features=("returns", "rsi")
    )
    base = _merge_features(base, xagusd_feats)

    # USDJPY: returns_20, trend (2 features)
    usdjpy_feats = _calc_instrument_features(
        correlated_m5.get("USDJPY"), "usdjpy", features=("returns", "trend")
    )
    base = _merge_features(base, usdjpy_feats)

    # US500: returns_20, rsi_14 (2 features)
    us500_feats = _calc_instrument_features(
        correlated_m5.get("US500"), "us500", features=("returns", "rsi")
    )
    base = _merge_features(base, us500_feats)

    # USOIL: returns_20 (1 feature)
    usoil_feats = _calc_instrument_features(
        correlated_m5.get("USOIL"), "usoil", features=("returns",)
    )
    base = _merge_features(base, usoil_feats)

    # --- Cross-ratio: gold/silver ratio z-score ---
    gold_silver_z = _calc_gold_silver_ratio_z(xau_m5, correlated_m5.get("XAGUSD"))
    base = _merge_features(base, gold_silver_z)

    # Drop time column, return feature columns only
    feature_cols = [c for c in base.columns if c != "time"]
    result = base[feature_cols].reset_index(drop=True)

    # Fill remaining NaN with semantically neutral values per feature type
    # RSI features: 0.5 is neutral (not overbought or oversold)
    # Returns/trend/ratio features: 0.0 is neutral (no change)
    rsi_cols = [c for c in result.columns if "rsi" in c]
    other_cols = [c for c in result.columns if "rsi" not in c]
    for col in rsi_cols:
        result[col] = result[col].fillna(0.5)
    for col in other_cols:
        result[col] = result[col].fillna(0.0)

    return result


def _calc_instrument_features(
    df: pd.DataFrame,
    prefix: str,
    features: tuple = ("returns", "rsi", "trend"),
) -> pd.DataFrame:
    """Compute selected features for one instrument.

    Args:
        df: M5 OHLCV for the instrument. May be None if data unavailable.
        prefix: Feature name prefix (e.g., "eurusd").
        features: Tuple of feature types to compute.

    Returns:
        DataFrame with 'time' + computed feature columns.
    """
    if df is None or df.empty:
        return None

    df = df.copy().sort_values("time").reset_index(drop=True)
    close = df["close"]
    result = pd.DataFrame({"time": df["time"]})

    if "returns" in features:
        result[f"{prefix}_returns_20"] = np.log(close / close.shift(20))

    if "rsi" in features:
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi() / 100.0
        result[f"{prefix}_rsi_14"] = rsi

    if "trend" in features:
        ema_20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        # Slope over 3 bars, normalized by price
        slope = ema_20.diff(3) / (close * 0.01 + 1e-8)
        result[f"{prefix}_trend"] = slope.clip(-3, 3)

    return result


def _calc_gold_silver_ratio_z(
    xau_m5: pd.DataFrame,
    xag_m5: pd.DataFrame,
) -> pd.DataFrame:
    """Compute z-score of gold/silver price ratio vs 200-bar rolling mean.

    Returns DataFrame with 'time' and 'gold_silver_ratio_z' columns.
    """
    if xag_m5 is None or xag_m5.empty:
        return None

    # Align silver to gold timeline via merge_asof
    merged = pd.merge_asof(
        xau_m5[["time", "close"]].sort_values("time"),
        xag_m5[["time", "close"]].sort_values("time").rename(columns={"close": "xag_close"}),
        on="time",
        direction="backward",
    )

    merged["xag_close"] = merged["xag_close"].ffill()

    # Raw ratio — z-score normalization is handled by the pipeline normalizer
    # (was previously double z-scored: here AND in normalizer)
    ratio = merged["close"] / (merged["xag_close"] + 1e-8)

    return pd.DataFrame({
        "time": merged["time"],
        "gold_silver_ratio_z": ratio,
    })


def _merge_features(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Merge instrument features into base timeline using merge_asof.

    direction='backward' ensures no look-ahead bias.
    Forward-fills up to 5 bars for minor timestamp misalignments.
    """
    if features is None:
        return base

    features = features.sort_values("time")
    result = pd.merge_asof(
        base.sort_values("time"),
        features,
        on="time",
        direction="backward",
    )

    # Forward-fill new feature columns (up to 5 bars for minor gaps)
    new_cols = [c for c in features.columns if c != "time"]
    for col in new_cols:
        result[col] = result[col].ffill(limit=5)

    return result
