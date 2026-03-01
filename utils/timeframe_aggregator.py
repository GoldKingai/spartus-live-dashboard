"""Timeframe aggregation utilities.

Provides functions to aggregate M5 bars into higher timeframes (H1, H4, D1)
as a backup in case the MT5 bridge cannot fetch HTF bars directly.

This is a safety net -- the primary path is to fetch H1/H4/D1 bars
directly from MT5 via ``MT5Bridge.get_latest_bars()``.  Use these
functions only when direct HTF bar retrieval fails.

Usage:
    from utils.timeframe_aggregator import aggregate_m5_to_h1, aggregate_m5_to_h4, aggregate_m5_to_d1

    h1 = aggregate_m5_to_h1(m5_df)
    h4 = aggregate_m5_to_h4(m5_df)
    d1 = aggregate_m5_to_d1(m5_df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def aggregate_m5(
    m5: pd.DataFrame,
    freq: str,
    label: str = "left",
    closed: str = "left",
) -> pd.DataFrame:
    """Aggregate M5 OHLCV bars into a higher timeframe.

    Args:
        m5: M5 OHLCV DataFrame with columns [time, open, high, low, close, volume].
            ``time`` must be a datetime column (tz-aware UTC preferred).
        freq: Pandas frequency string for the target timeframe.
            Common values: ``"1h"`` (H1), ``"4h"`` (H4), ``"1D"`` (D1).
        label: Which edge of the resampled interval to label (default ``"left"``).
        closed: Which side of the interval is closed (default ``"left"``).

    Returns:
        DataFrame with the same columns, aggregated to the target timeframe.
        Incomplete (partial) bars at the end are included.
        Returns an empty DataFrame if input is empty.
    """
    if m5.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = m5.copy()

    # Ensure time is the index for resampling
    if "time" in df.columns:
        df = df.set_index("time")

    # Resample
    agg = df.resample(freq, label=label, closed=closed).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Drop rows where all OHLCV are NaN (no M5 bars in that interval)
    agg = agg.dropna(subset=["open", "close"], how="all")

    # Reset index to get 'time' back as a column
    agg = agg.reset_index()
    agg.rename(columns={"index": "time"}, inplace=True)
    if "time" not in agg.columns and agg.index.name == "time":
        agg = agg.reset_index()

    # Fill any remaining NaN in volume with 0
    agg["volume"] = agg["volume"].fillna(0)

    return agg


def aggregate_m5_to_h1(m5: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M5 bars into H1 (1-hour) bars.

    Args:
        m5: M5 OHLCV DataFrame.

    Returns:
        H1 OHLCV DataFrame.
    """
    result = aggregate_m5(m5, freq="1h")
    log.debug("Aggregated %d M5 bars -> %d H1 bars", len(m5), len(result))
    return result


def aggregate_m5_to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M5 bars into H4 (4-hour) bars.

    Args:
        m5: M5 OHLCV DataFrame.

    Returns:
        H4 OHLCV DataFrame.
    """
    result = aggregate_m5(m5, freq="4h")
    log.debug("Aggregated %d M5 bars -> %d H4 bars", len(m5), len(result))
    return result


def aggregate_m5_to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M5 bars into D1 (daily) bars.

    Args:
        m5: M5 OHLCV DataFrame.

    Returns:
        D1 OHLCV DataFrame.
    """
    result = aggregate_m5(m5, freq="1D")
    log.debug("Aggregated %d M5 bars -> %d D1 bars", len(m5), len(result))
    return result
