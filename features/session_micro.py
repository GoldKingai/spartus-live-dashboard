"""Session microstructure features (Upgrade 5): 4 features.

IDENTICAL to training session_features.py logic.
All features are EXEMPT from z-score normalization (already bounded/clipped/binary).

Features:
    - asian_range_norm: Asian session (0-7 UTC) high-low range / ATR, clipped [0, 5]
    - asian_range_position: (current_price - asian_midpoint) / half_range, clipped [-2, 2]
    - session_momentum: cumulative return within current session / ATR, clipped [-3, 3]
    - london_ny_overlap: 1.0 if 12-16 UTC, else 0.0

No look-ahead bias: Asian range only available after 07:00 UTC.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


# Default session boundaries (matches training config)
_DEFAULT_ASIAN_END_UTC = 7
_DEFAULT_SESSION_BOUNDARIES_UTC = (0, 7, 12)


def compute_session_micro_features(
    m5: pd.DataFrame,
    atr_14: float,
    asian_session_end_utc: int = _DEFAULT_ASIAN_END_UTC,
    session_boundaries_utc: tuple = _DEFAULT_SESSION_BOUNDARIES_UTC,
) -> Dict[str, float]:
    """Compute all 4 session microstructure features for the latest M5 bar.

    IDENTICAL to training session_features.calc_session_features.

    Args:
        m5: M5 OHLCV DataFrame with columns [time, open, high, low, close].
            Must have at least a day's worth of bars (288 M5 bars) for
            proper Asian range computation. The 'time' column must be UTC.
        atr_14: Current ATR(14) value in price units (for normalization).
        asian_session_end_utc: UTC hour when Asian session ends (default 7).
        session_boundaries_utc: Tuple of UTC hours where new sessions start.

    Returns:
        Dict mapping feature_name -> float value (4 features).
    """
    if len(m5) < 2:
        return _neutral_features()

    current_time = m5["time"].iloc[-1]
    current_close = m5["close"].iloc[-1]
    current_hour = current_time.hour if hasattr(current_time, "hour") else 12

    # --- Asian range ---
    asian_high, asian_low = _get_asian_range(m5, current_time, asian_session_end_utc)

    if asian_high is not None and asian_low is not None and asian_high > asian_low:
        asian_range = asian_high - asian_low
        asian_mid = (asian_high + asian_low) / 2.0

        # asian_range_norm: range / ATR(14), clipped [0, 5]
        asian_range_norm = float(np.clip(asian_range / (atr_14 + 1e-8), 0.0, 5.0))

        # asian_range_position: (close - midpoint) / half_range, clipped [-2, 2]
        half_range = asian_range / 2.0 + 1e-8
        asian_range_pos = float(np.clip(
            (current_close - asian_mid) / half_range, -2.0, 2.0
        ))
    else:
        asian_range_norm = 0.0
        asian_range_pos = 0.0

    # --- Session momentum ---
    session_momentum = _calc_session_momentum_latest(
        m5, current_hour, atr_14, session_boundaries_utc
    )

    # --- London-NY overlap (12:00-16:00 UTC) ---
    london_ny_overlap = 1.0 if (12 <= current_hour < 16) else 0.0

    return {
        "asian_range_norm": asian_range_norm,
        "asian_range_position": asian_range_pos,
        "session_momentum": session_momentum,
        "london_ny_overlap": london_ny_overlap,
    }


def _get_asian_range(
    m5: pd.DataFrame,
    current_time: datetime,
    asian_end: int,
) -> tuple:
    """Get Asian session high/low for use at current_time.

    Asian session = 00:00 to asian_end (exclusive) UTC.
    Before asian_end on current day: uses PREVIOUS day's range (no look-ahead).
    After asian_end: uses current day's range.

    Matches training session_features._calc_asian_range logic.

    Returns:
        (asian_high, asian_low) or (None, None) if insufficient data.
    """
    current_hour = current_time.hour if hasattr(current_time, "hour") else 12
    current_date = current_time.date() if hasattr(current_time, "date") else None

    if current_date is None:
        return None, None

    times = m5["time"]
    highs = m5["high"].values
    lows = m5["low"].values

    # Get dates and hours
    try:
        dates = times.dt.date
        hours = times.dt.hour.values
    except AttributeError:
        return None, None

    if current_hour >= asian_end:
        # After Asian session end: use today's Asian range
        today_asian_mask = (dates == current_date) & (hours < asian_end)
        if today_asian_mask.any():
            return highs[today_asian_mask].max(), lows[today_asian_mask].min()

    # Before Asian end OR no Asian bars today: use previous day's range
    # Find the most recent complete Asian session
    unique_dates = sorted(dates.unique(), reverse=True)

    for d in unique_dates:
        if d >= current_date and current_hour < asian_end:
            continue  # Skip current day if Asian session not yet complete
        if d == current_date and current_hour >= asian_end:
            # Use today's completed Asian session
            day_asian = (dates == d) & (hours < asian_end)
            if day_asian.any():
                return highs[day_asian].max(), lows[day_asian].min()
            continue

        # Previous days
        day_asian = (dates == d) & (hours < asian_end)
        if day_asian.any():
            return highs[day_asian].max(), lows[day_asian].min()

    return None, None


def _calc_session_momentum_latest(
    m5: pd.DataFrame,
    current_hour: int,
    atr_14: float,
    boundaries: tuple,
) -> float:
    """Compute session momentum for the latest bar.

    Session momentum = (current_close - session_open_price) / ATR.
    Session open price is the open of the first bar in the current session.

    Matches training session_features._calc_session_momentum logic.

    Returns:
        Session momentum clipped to [-3, 3].
    """
    if len(m5) < 2:
        return 0.0

    boundaries_set = set(boundaries)
    times = m5["time"]
    closes = m5["close"].values
    opens = m5["open"].values

    try:
        hours = times.dt.hour.values
    except AttributeError:
        return 0.0

    # Walk backwards to find the session open
    session_open_price = opens[-1]  # Default: current bar's open

    for i in range(len(m5) - 1, -1, -1):
        h = hours[i]
        if h in boundaries_set:
            # Check if this is actually a session boundary (not same bar repeated)
            if i == 0 or hours[i] != hours[i - 1]:
                session_open_price = opens[i]
                break

    current_close = closes[-1]
    momentum = (current_close - session_open_price) / (atr_14 + 1e-8)

    return float(np.clip(momentum, -3.0, 3.0))


def _neutral_features() -> Dict[str, float]:
    """Return neutral default session features."""
    return {
        "asian_range_norm": 0.0,
        "asian_range_position": 0.0,
        "session_momentum": 0.0,
        "london_ny_overlap": 0.0,
    }
