"""Tier 4: Churn Diagnostic -- frequency, edge, and cost analysis."""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np

from ..types import ChurnResult, TradeRecord

log = logging.getLogger("spartusbench.t4")


def run_t4(
    trades: List[TradeRecord],
    val_weeks_count: int,
    config: Any,
) -> ChurnResult:
    """Run Tier 4: Churn Diagnostic.

    Analyzes whether the model has a real edge or is just churning.
    """
    if not trades:
        return ChurnResult()

    trading_days = val_weeks_count * 5
    tpd = len(trades) / trading_days if trading_days > 0 else 0

    # Average spread estimate (across sessions)
    avg_spread = np.mean([
        config.spread_london_pips,
        config.spread_ny_pips,
        config.spread_asia_pips,
        config.spread_off_hours_pips,
    ])

    # Average slippage (entry + exit)
    slip_mean = getattr(config, 'slippage_mean_pips', 0.5)
    avg_slippage = slip_mean * 2.0

    avg_cost_pips = avg_spread + avg_slippage
    pip_price = getattr(config, 'pip_price', 0.1)
    avg_cost_points = avg_cost_pips * pip_price

    avg_lot = float(np.mean([t.lots for t in trades])) if trades else 0.0
    vpp = config.trade_tick_value / config.trade_tick_size
    est_cost_per_trade = avg_cost_points * avg_lot * vpp
    total_cost = est_cost_per_trade * len(trades)

    total_pnl = sum(t.pnl for t in trades)
    est_gross = total_pnl + total_cost

    net_edge = total_pnl / len(trades) if trades else 0.0
    gross_edge = est_gross / len(trades) if trades else 0.0
    cost_edge_ratio = total_cost / abs(total_pnl) if abs(total_pnl) > 1e-12 else 999.0

    result = ChurnResult(
        trading_days=trading_days,
        trades_per_day=tpd,
        avg_spread_pips=avg_spread,
        avg_slippage_pips=avg_slippage,
        avg_cost_pips=avg_cost_pips,
        avg_cost_points=avg_cost_points,
        avg_lot=avg_lot,
        est_cost_per_trade=est_cost_per_trade,
        total_est_cost=total_cost,
        net_pnl=total_pnl,
        est_gross_pnl=est_gross,
        net_edge_per_trade=net_edge,
        gross_edge_per_trade=gross_edge,
        cost_to_edge_ratio=cost_edge_ratio,
    )

    log.info(
        "T4: trades/day=%.1f | net_edge=%.3f | cost_ratio=%.2f | gross_edge=%.3f",
        tpd, net_edge, cost_edge_ratio, gross_edge,
    )

    return result
