"""Technical indicator features (Groups A-E): 25 features from M5/H1/H4/D1 bars.

IDENTICAL to training feature_builder.py Groups A-E logic.
Uses the `ta` library (NOT pandas-ta).
Implements fractional differentiation manually (fracdiff package needs Python <3.10).

Group A: Price & Returns (7 features)
Group B: Volatility (4 features)
Group C: Momentum & Trend (6 features)
Group D: Volume (2 features)
Group E: Multi-Timeframe Context (6 features)
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Fractional differentiation (manual implementation)
# ---------------------------------------------------------------------------

def _get_frac_diff_weights(d: float, threshold: float = 1e-5,
                           max_width: int = 200) -> np.ndarray:
    """Compute fractional differentiation weights using the expanding window method.

    The weights follow: w_k = -w_{k-1} * (d - k + 1) / k
    We truncate when |w_k| < threshold or width exceeds max_width.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k >= max_width:
            break
    return np.array(weights[::-1])  # Oldest weight first


def frac_diff(series: pd.Series, d: float = 0.35, threshold: float = 1e-5,
              max_width: int = 200) -> pd.Series:
    """Apply fractional differentiation to a price series.

    Args:
        series: Price series (e.g., close prices).
        d: Differentiation order (0 < d < 1). 0.35 is default for XAUUSD.
        threshold: Weight cutoff for truncation.
        max_width: Maximum window width for weights.

    Returns:
        Fractionally differentiated series (NaN where insufficient history).
    """
    weights = _get_frac_diff_weights(d, threshold, max_width)
    width = len(weights)
    result = pd.Series(index=series.index, dtype=np.float64)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        result.iloc[i] = np.dot(weights, window)

    return result


def frac_diff_latest(series: pd.Series, d: float = 0.35, threshold: float = 1e-5,
                     max_width: int = 200) -> float:
    """Compute fractional differentiation for the latest bar only.

    More efficient than frac_diff() when only the last value is needed.
    Requires at least `max_width` bars of history.

    Args:
        series: Price series (at least max_width bars).
        d: Differentiation order.
        threshold: Weight cutoff.
        max_width: Maximum window width.

    Returns:
        Fractionally differentiated value for the latest bar, or NaN.
    """
    weights = _get_frac_diff_weights(d, threshold, max_width)
    width = len(weights)

    if len(series) < width:
        return np.nan

    window = series.iloc[-width:].values
    return float(np.dot(weights, window))


# ---------------------------------------------------------------------------
# Group A: Price & Returns (7 features)
# ---------------------------------------------------------------------------

def calc_price_returns(close: pd.Series, high: pd.Series,
                       low: pd.Series, opn: pd.Series,
                       d: float = 0.35) -> Dict[str, float]:
    """Compute Group A features from M5 OHLCV for the latest bar.

    Args:
        close: M5 close prices (at least 200 bars for frac_diff).
        high: M5 high prices.
        low: M5 low prices.
        opn: M5 open prices.
        d: Fractional differentiation order.

    Returns:
        Dict of feature_name -> float for the latest bar.
    """
    c = close.iloc[-1]
    h = high.iloc[-1]
    lo = low.iloc[-1]
    o = opn.iloc[-1]

    # #1: Fractionally differentiated close
    close_frac = frac_diff_latest(close, d=d)

    # #2-4: Log returns at different horizons
    ret_1 = np.log(c / close.iloc[-2]) if len(close) >= 2 else 0.0
    ret_5 = np.log(c / close.iloc[-6]) if len(close) >= 6 else 0.0
    ret_20 = np.log(c / close.iloc[-21]) if len(close) >= 21 else 0.0

    # #5: Bar range as % of close
    bar_range = (h - lo) / c if c != 0 else 0.0

    # #6: Where close sits within bar [0, 1]
    close_pos = (c - lo) / (h - lo + 1e-8)

    # #7: Candle body strength [0, 1]
    body = abs(c - o) / (h - lo + 1e-8)

    return {
        "close_frac_diff": close_frac if not np.isnan(close_frac) else 0.0,
        "returns_1bar": ret_1 if not np.isnan(ret_1) else 0.0,
        "returns_5bar": ret_5 if not np.isnan(ret_5) else 0.0,
        "returns_20bar": ret_20 if not np.isnan(ret_20) else 0.0,
        "bar_range": bar_range if not np.isnan(bar_range) else 0.0,
        "close_position": close_pos if not np.isnan(close_pos) else 0.0,
        "body_ratio": body if not np.isnan(body) else 0.0,
    }


# ---------------------------------------------------------------------------
# Group B: Volatility (4 features)
# ---------------------------------------------------------------------------

def calc_volatility(close: pd.Series, high: pd.Series, low: pd.Series,
                    atr_14: pd.Series) -> Dict[str, float]:
    """Compute Group B features from M5 OHLCV for the latest bar.

    Args:
        close: M5 close prices.
        high: M5 high prices.
        low: M5 low prices.
        atr_14: Pre-computed ATR(14) series.

    Returns:
        Dict of feature_name -> float for the latest bar.
    """
    # ATR(7) and ATR(21) for the ratio
    atr_7 = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=7
    ).average_true_range()
    atr_21 = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=21
    ).average_true_range()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

    c = close.iloc[-1]
    a14 = atr_14.iloc[-1]
    a7 = atr_7.iloc[-1]
    a21 = atr_21.iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_mavg = bb.bollinger_mavg().iloc[-1]

    # #8: ATR normalized by price
    atr_14_norm = a14 / c * 100 if c != 0 else 0.0

    # #9: Volatility expansion/contraction
    atr_ratio = np.clip(a7 / (a21 + 1e-8), 0.0, 10.0)

    # #10: BB width
    bb_width = (bb_upper - bb_lower) / (bb_mavg + 1e-8)

    # #11: Position within bands [0, 1]
    bb_pos = (c - bb_lower) / (bb_upper - bb_lower + 1e-8)

    return {
        "atr_14_norm": _safe(atr_14_norm),
        "atr_ratio": _safe(atr_ratio),
        "bb_width": _safe(bb_width),
        "bb_position": _safe(bb_pos),
    }


# ---------------------------------------------------------------------------
# Group C: Momentum & Trend (6 features)
# ---------------------------------------------------------------------------

def calc_momentum_trend(close: pd.Series, high: pd.Series, low: pd.Series,
                        atr_14: pd.Series) -> Dict[str, float]:
    """Compute Group C features from M5 OHLCV for the latest bar.

    Args:
        close: M5 close prices (at least 200 bars for EMA200).
        high: M5 high prices.
        low: M5 low prices.
        atr_14: Pre-computed ATR(14) series.

    Returns:
        Dict of feature_name -> float for the latest bar.
    """
    # RSI(14)
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # MACD (12, 26, 9)
    macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd_ind.macd_diff()

    # ADX(14)
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # EMAs
    ema_20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    ema_50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    ema_200 = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    # Stochastic %K
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close,
        window=14, smooth_window=3,
    )

    a14 = atr_14.iloc[-1]
    c = close.iloc[-1]

    return {
        # #12: RSI scaled to [0, 1]
        "rsi_14": _safe(rsi.iloc[-1] / 100.0),
        # #13: MACD histogram scaled by volatility
        "macd_signal": _safe(macd_hist.iloc[-1] / (a14 + 1e-8)),
        # #14: ADX scaled to [0, 1]
        "adx_14": _safe(adx.iloc[-1] / 100.0),
        # #15: EMA crossover signal
        "ema_cross": _safe((ema_20.iloc[-1] - ema_50.iloc[-1]) / (a14 + 1e-8)),
        # #16: Long-term trend position
        "price_vs_ema200": _safe((c - ema_200.iloc[-1]) / (a14 + 1e-8)),
        # #17: Stochastic %K scaled to [0, 1]
        "stoch_k": _safe(stoch.stoch().iloc[-1] / 100.0),
    }


# ---------------------------------------------------------------------------
# Group D: Volume (2 features)
# ---------------------------------------------------------------------------

def calc_volume(close: pd.Series, volume: pd.Series) -> Dict[str, float]:
    """Compute Group D features from M5 OHLCV for the latest bar.

    Args:
        close: M5 close prices.
        volume: M5 tick volume.

    Returns:
        Dict of feature_name -> float for the latest bar.
    """
    # Volume relative to 20-bar SMA
    vol_sma_20 = volume.rolling(20).mean()

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()

    # OBV slope: linear regression over 10 bars, normalized by 50-bar std
    obv_slope = obv.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0,
        raw=False,
    )
    obv_slope_std = obv_slope.rolling(50).std()
    obv_slope_norm = obv_slope / (obv_slope_std + 1e-8)

    vol_ratio = volume.iloc[-1] / (vol_sma_20.iloc[-1] + 1e-8)
    obv_s = obv_slope_norm.iloc[-1]

    return {
        # #18: Volume ratio
        "volume_ratio": _safe(vol_ratio),
        # #19: OBV slope (normalized)
        "obv_slope": _safe(obv_s),
    }


# ---------------------------------------------------------------------------
# Group E: Multi-Timeframe Context (6 features)
# ---------------------------------------------------------------------------

def calc_mtf_context(m5: pd.DataFrame, h1: pd.DataFrame,
                     h4: pd.DataFrame, d1: pd.DataFrame) -> Dict[str, float]:
    """Compute Group E multi-timeframe features for the latest M5 bar.

    Uses merge_asof(direction='backward') to prevent look-ahead bias.
    Replicates training feature_builder._calc_mtf_context exactly.

    Args:
        m5: M5 OHLCV with 'time' column.
        h1: H1 OHLCV with 'time' column.
        h4: H4 OHLCV with 'time' column.
        d1: D1 OHLCV with 'time' column.

    Returns:
        Dict of feature_name -> float for the latest M5 bar.
    """
    current_time = m5["time"].iloc[-1] if "time" in m5.columns else None

    # H1 trend direction: EMA(20) slope over 3 bars, normalized by price
    h1_trend = _htf_trend_dir(h1, current_time)
    h4_trend = _htf_trend_dir(h4, current_time)
    d1_trend = _htf_trend_dir(d1, current_time)

    # H1 RSI
    h1_rsi = _htf_rsi(h1, current_time)

    # H4 MACD momentum
    htf_mom = _htf_momentum(h4, current_time)

    # MTF alignment: average of trend directions clipped to [-1, 1]
    mtf_align = np.clip((h1_trend + h4_trend + d1_trend) / 3.0, -1.0, 1.0)

    return {
        "h1_trend_dir": h1_trend,
        "h4_trend_dir": h4_trend,
        "d1_trend_dir": d1_trend,
        "h1_rsi": h1_rsi,
        "mtf_alignment": mtf_align,
        "htf_momentum": htf_mom,
    }


def _htf_trend_dir(htf: pd.DataFrame, current_time) -> float:
    """Compute trend direction for a higher-timeframe DataFrame at current time.

    Returns trend direction value clipped to [-3, 3].
    Uses only bars at or before current_time (no look-ahead).
    """
    if htf is None or htf.empty:
        return 0.0

    if current_time is not None and "time" in htf.columns:
        htf = htf.sort_values("time")
        mask = htf["time"] <= current_time
        if mask.sum() < 4:
            return 0.0
        htf_valid = htf.loc[mask]
    else:
        htf_valid = htf
        if len(htf_valid) < 4:
            return 0.0
    close = htf_valid["close"]
    ema_20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()

    # Slope over 3 bars, normalized by price
    if len(ema_20) < 4:
        return 0.0

    slope = (ema_20.iloc[-1] - ema_20.iloc[-4]) / (close.iloc[-1] * 0.01 + 1e-8)
    return float(np.clip(slope, -3.0, 3.0))


def _htf_rsi(h1: pd.DataFrame, current_time) -> float:
    """Compute H1 RSI(14) / 100 at current time."""
    if h1 is None or h1.empty:
        return 0.5

    if current_time is not None and "time" in h1.columns:
        h1 = h1.sort_values("time")
        mask = h1["time"] <= current_time
        if mask.sum() < 15:
            return 0.5
        h1_valid = h1.loc[mask]
    else:
        h1_valid = h1
        if len(h1_valid) < 15:
            return 0.5
    rsi = ta.momentum.RSIIndicator(close=h1_valid["close"], window=14).rsi()
    val = rsi.iloc[-1]
    return _safe(val / 100.0, default=0.5)


def _htf_momentum(h4: pd.DataFrame, current_time) -> float:
    """Compute H4 MACD histogram / H4 ATR(14), clipped [-3, 3]."""
    if h4 is None or h4.empty:
        return 0.0

    if current_time is not None and "time" in h4.columns:
        h4 = h4.sort_values("time")
        mask = h4["time"] <= current_time
        if mask.sum() < 27:
            return 0.0
        h4_valid = h4.loc[mask]
    else:
        h4_valid = h4
        if len(h4_valid) < 27:
            return 0.0
    close = h4_valid["close"]
    high = h4_valid["high"]
    low = h4_valid["low"]

    h4_atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    h4_macd = ta.trend.MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9
    ).macd_diff()

    atr_val = h4_atr.iloc[-1]
    macd_val = h4_macd.iloc[-1]

    if np.isnan(atr_val) or np.isnan(macd_val):
        return 0.0

    return float(np.clip(macd_val / (atr_val + 1e-8), -3.0, 3.0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_technical_features(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    frac_diff_d: float = 0.35,
) -> Dict[str, float]:
    """Compute all 25 technical features (Groups A-E) for the latest M5 bar.

    IDENTICAL computation to training feature_builder.py.

    Args:
        m5: M5 OHLCV DataFrame with columns [time, open, high, low, close, volume].
            Must have at least 200 bars of history for proper warmup.
        h1: H1 OHLCV DataFrame with same columns.
        h4: H4 OHLCV DataFrame with same columns.
        d1: D1 OHLCV DataFrame with same columns.
        frac_diff_d: Fractional differentiation order (default 0.35).

    Returns:
        Dict mapping feature_name -> float value for the latest bar (25 features).
        Any NaN values are replaced with 0.0.
    """
    close = m5["close"]
    high = m5["high"]
    low = m5["low"]
    opn = m5["open"]
    vol = m5["volume"]

    # Pre-compute ATR(14) -- used by multiple groups
    atr_14 = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    features = {}

    # Group A: Price & Returns (7)
    features.update(calc_price_returns(close, high, low, opn, d=frac_diff_d))

    # Group B: Volatility (4)
    features.update(calc_volatility(close, high, low, atr_14))

    # Group C: Momentum & Trend (6)
    features.update(calc_momentum_trend(close, high, low, atr_14))

    # Group D: Volume (2)
    features.update(calc_volume(close, vol))

    # Group E: Multi-Timeframe Context (6)
    features.update(calc_mtf_context(m5, h1, h4, d1))

    return features


def get_atr_14(m5: pd.DataFrame) -> float:
    """Get latest ATR(14) value from M5 data. Utility for other modules.

    Args:
        m5: M5 OHLCV DataFrame.

    Returns:
        Latest ATR(14) value, or 1.0 if insufficient data.
    """
    if len(m5) < 15:
        return 1.0

    atr_14 = ta.volatility.AverageTrueRange(
        high=m5["high"], low=m5["low"], close=m5["close"], window=14
    ).average_true_range()

    val = atr_14.iloc[-1]
    return float(val) if not np.isnan(val) and val > 0 else 1.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe(val, default: float = 0.0) -> float:
    """Return val as float, replacing NaN/Inf with default."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default
