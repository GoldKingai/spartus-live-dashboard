"""Session microstructure features (Upgrade 5).

4 features capturing intraday gold trading patterns:
- Asian range (overnight range reference for London breakouts)
- Asian range position (price vs Asian range — breakout detection)
- Session momentum (how far price has moved this session)
- London-NY overlap (highest liquidity window binary)

All features are EXEMPT from z-score normalization (already bounded/clipped).
No look-ahead bias: Asian range only available after 07:00 UTC.
"""

import numpy as np
import pandas as pd

from src.config import TrainingConfig


def calc_session_features(m5: pd.DataFrame, config: TrainingConfig = None) -> pd.DataFrame:
    """Compute all 4 session microstructure features from M5 OHLCV.

    Args:
        m5: M5 OHLCV with 'time', 'open', 'high', 'low', 'close' columns.
            'time' must be timezone-aware UTC.
        config: TrainingConfig for session boundaries and ATR reference.

    Returns:
        DataFrame with columns: asian_range_norm, asian_range_position,
        session_momentum, london_ny_overlap. Indexed like m5.
    """
    cfg = config or TrainingConfig()
    times = m5["time"]
    close = m5["close"].values
    high = m5["high"].values
    low = m5["low"].values
    opn = m5["open"].values

    # Need ATR for normalization — compute inline (14-bar)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]  # First bar: no prior close
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values

    hour = times.dt.hour.values
    date = times.dt.date

    # --- Asian range (00:00-07:00 UTC each day) ---
    asian_high, asian_low = _calc_asian_range(m5, hour, date, cfg.asian_session_end_utc)

    asian_range = asian_high - asian_low
    asian_mid = (asian_high + asian_low) / 2.0

    # asian_range_norm: range / ATR(14), clipped to [0, 5]
    asian_range_norm = (asian_range / (atr_14 + 1e-8)).clip(0.0, 5.0)

    # asian_range_position: (close - midpoint) / (half_range), clipped [-2, 2]
    half_range = asian_range / 2.0 + 1e-8
    asian_range_position = np.clip((close - asian_mid) / half_range, -2.0, 2.0)

    # --- Session momentum ---
    session_momentum = _calc_session_momentum(
        close, opn, hour, atr_14, cfg.session_boundaries_utc
    )

    # --- London-NY overlap (12:00-16:00 UTC) ---
    london_ny_overlap = ((hour >= 12) & (hour < 16)).astype(np.float32)

    return pd.DataFrame({
        "asian_range_norm": asian_range_norm,
        "asian_range_position": asian_range_position,
        "session_momentum": session_momentum,
        "london_ny_overlap": london_ny_overlap,
    })


def _calc_asian_range(
    m5: pd.DataFrame, hour: np.ndarray, date: pd.Series,
    asian_end: int,
) -> tuple:
    """Compute Asian session high/low per day, forward-filled.

    Asian session = 00:00 to asian_end (exclusive) UTC.
    Before 07:00 on each day, uses PREVIOUS day's range (no look-ahead).
    """
    n = len(m5)
    asian_high = np.full(n, np.nan)
    asian_low = np.full(n, np.nan)

    dates = date.values
    high = m5["high"].values
    low = m5["low"].values

    # Group bars by date, compute Asian range per day
    unique_dates = np.unique(dates)

    prev_day_high = np.nan
    prev_day_low = np.nan

    for d in unique_dates:
        day_mask = dates == d
        day_hours = hour[day_mask]
        day_high = high[day_mask]
        day_low = low[day_mask]
        day_indices = np.where(day_mask)[0]

        # Asian bars for this day
        asian_mask = day_hours < asian_end
        if asian_mask.any():
            today_asian_high = day_high[asian_mask].max()
            today_asian_low = day_low[asian_mask].min()
        else:
            today_asian_high = prev_day_high
            today_asian_low = prev_day_low

        # Before asian_end: use previous day's range (no look-ahead)
        for idx in day_indices:
            if hour[idx] < asian_end:
                asian_high[idx] = prev_day_high
                asian_low[idx] = prev_day_low
            else:
                asian_high[idx] = today_asian_high
                asian_low[idx] = today_asian_low

        prev_day_high = today_asian_high
        prev_day_low = today_asian_low

    # Forward-fill any remaining NaN (first day has no previous range)
    asian_high = pd.Series(asian_high).ffill().fillna(0.0).values
    asian_low = pd.Series(asian_low).ffill().fillna(0.0).values

    return asian_high, asian_low


def _calc_session_momentum(
    close: np.ndarray, opn: np.ndarray, hour: np.ndarray,
    atr_14: np.ndarray, boundaries: tuple,
) -> np.ndarray:
    """Compute price movement since current session open, normalized by ATR.

    Session boundaries are UTC hours where a new session starts.
    Returns values clipped to [-3, 3].
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    boundaries_set = set(boundaries)

    session_open_price = opn[0]  # Default to first bar's open

    for i in range(n):
        if hour[i] in boundaries_set and (i == 0 or hour[i] != hour[i - 1]):
            # New session starts — use this bar's open as reference
            session_open_price = opn[i]
        momentum[i] = (close[i] - session_open_price) / (atr_14[i] + 1e-8)

    return np.clip(momentum, -3.0, 3.0)
