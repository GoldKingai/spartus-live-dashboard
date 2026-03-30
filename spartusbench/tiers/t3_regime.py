"""Tier 3: Regime Segmentation -- slice trades across 5 dimensions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from ..metrics import avg_hold_bars, avg_pnl, net_pnl, profit_factor, win_rate
from ..types import RegimeSlice, TradeRecord

log = logging.getLogger("spartusbench.t3")


def run_t3(trades: List[TradeRecord]) -> List[RegimeSlice]:
    """Run Tier 3: Regime Segmentation.

    Slices trades by: ATR quartile, session, day of week, direction, close reason.
    """
    slices = []

    # A. ATR Quartiles
    slices.extend(_slice_by_atr(trades))

    # B. Trading Session
    slices.extend(_slice_by_field(trades, "session", "session"))

    # C. Day of Week (derived from week + step, simplified to session bucket)
    # Since we don't have exact calendar day, we use session as proxy
    # In a full implementation, the trade entry datetime would provide this

    # D. Trade Direction
    slices.extend(_slice_by_field(trades, "side", "direction"))

    # E. Close Reason
    slices.extend(_slice_by_field(trades, "close_reason", "close_reason"))

    # F. Lesson Type
    slices.extend(_slice_by_field(trades, "lesson_type", "lesson_type"))

    log.info("T3: Generated %d regime slices from %d trades", len(slices), len(trades))
    return slices


def _slice_by_atr(trades: List[TradeRecord]) -> List[RegimeSlice]:
    """Slice trades by ATR quartile at entry."""
    if not trades:
        return []

    atrs = np.array([t.atr_at_entry for t in trades])
    if np.all(atrs == 0):
        return []

    p25, p50, p75 = np.percentile(atrs, [25, 50, 75])

    buckets = {"Q1_low_vol": [], "Q2": [], "Q3": [], "Q4_high_vol": []}
    for t in trades:
        if t.atr_at_entry < p25:
            buckets["Q1_low_vol"].append(t)
        elif t.atr_at_entry < p50:
            buckets["Q2"].append(t)
        elif t.atr_at_entry < p75:
            buckets["Q3"].append(t)
        else:
            buckets["Q4_high_vol"].append(t)

    slices = []
    for bucket_name, bucket_trades in buckets.items():
        if bucket_trades:
            slices.append(_make_slice("atr_quartile", bucket_name, bucket_trades))
    return slices


def _slice_by_field(
    trades: List[TradeRecord],
    field_name: str,
    slice_type: str,
) -> List[RegimeSlice]:
    """Slice trades by a string field value."""
    buckets: Dict[str, List[TradeRecord]] = {}
    for t in trades:
        val = getattr(t, field_name, "UNKNOWN")
        if val not in buckets:
            buckets[val] = []
        buckets[val].append(t)

    return [
        _make_slice(slice_type, val, bucket_trades)
        for val, bucket_trades in sorted(buckets.items())
        if bucket_trades
    ]


def _make_slice(slice_type: str, slice_value: str, trades: List[TradeRecord]) -> RegimeSlice:
    return RegimeSlice(
        slice_type=slice_type,
        slice_value=slice_value,
        trades=len(trades),
        win_pct=win_rate(trades),
        net_pnl=net_pnl(trades),
        pf=profit_factor(trades),
        avg_pnl=avg_pnl(trades),
        avg_hold=avg_hold_bars(trades),
    )
