"""Pure metric computation functions for SpartusBench.

All functions are stateless and operate on lists/arrays.
No side effects, no file I/O, no model loading.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import TradeRecord


# ---------------------------------------------------------------------------
# Core performance metrics
# ---------------------------------------------------------------------------

def sharpe_annualized(weekly_returns: List[float]) -> float:
    """Annualized Sharpe ratio from weekly returns."""
    if len(weekly_returns) < 2:
        return 0.0
    arr = np.array(weekly_returns, dtype=np.float64)
    std = np.std(arr, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((np.mean(arr) / std) * math.sqrt(52))


def sortino_annualized(weekly_returns: List[float]) -> float:
    """Annualized Sortino ratio from weekly returns."""
    if len(weekly_returns) < 2:
        return 0.0
    arr = np.array(weekly_returns, dtype=np.float64)
    neg = arr[arr < 0]
    if len(neg) < 2:
        return 100.0  # too few negative returns for meaningful std
    down_std = np.std(neg, ddof=1)
    if down_std < 1e-12:
        return 100.0
    return float((np.mean(arr) / down_std) * math.sqrt(52))


def profit_factor(trades: List[TradeRecord]) -> float:
    """Profit factor = gross_profit / gross_loss."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    if gross_loss < 1e-12:
        return 100.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def profit_factor_from_pnls(pnls: List[float]) -> float:
    """Profit factor from raw P/L list."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    if gross_loss < 1e-12:
        return 100.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(trades: List[TradeRecord]) -> float:
    """Win rate percentage."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades) * 100


def max_drawdown_pct(trades: List[TradeRecord], initial_balance: float = 100.0) -> float:
    """Max drawdown as percentage of peak equity from trade sequence."""
    if not trades:
        return 0.0
    equity = initial_balance
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def max_drawdown_dollars(trades: List[TradeRecord], initial_balance: float = 100.0) -> float:
    """Max drawdown in dollar terms."""
    if not trades:
        return 0.0
    equity = initial_balance
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def net_pnl(trades: List[TradeRecord]) -> float:
    """Total net P/L."""
    return sum(t.pnl for t in trades)


def avg_pnl(trades: List[TradeRecord]) -> float:
    """Average P/L per trade."""
    if not trades:
        return 0.0
    return sum(t.pnl for t in trades) / len(trades)


def gross_profit(trades: List[TradeRecord]) -> float:
    return sum(t.pnl for t in trades if t.pnl > 0)


def gross_loss(trades: List[TradeRecord]) -> float:
    return abs(sum(t.pnl for t in trades if t.pnl <= 0))


def avg_win(trades: List[TradeRecord]) -> float:
    winners = [t.pnl for t in trades if t.pnl > 0]
    return float(np.mean(winners)) if winners else 0.0


def avg_loss(trades: List[TradeRecord]) -> float:
    losers = [abs(t.pnl) for t in trades if t.pnl <= 0]
    return float(np.mean(losers)) if losers else 0.0


def win_loss_ratio(trades: List[TradeRecord]) -> float:
    aw = avg_win(trades)
    al = avg_loss(trades)
    if al < 1e-12:
        return 100.0 if aw > 0 else 0.0
    return aw / al


def expectancy(trades: List[TradeRecord]) -> float:
    """Expected $ per trade: win_rate * avg_win - loss_rate * avg_loss."""
    if not trades:
        return 0.0
    wr = win_rate(trades) / 100.0
    aw = avg_win(trades)
    al = avg_loss(trades)
    return wr * aw - (1 - wr) * al


def calmar_ratio(weekly_returns: List[float], max_dd_pct_val: float) -> float:
    """Calmar ratio = annualized return / max drawdown %."""
    if max_dd_pct_val < 0.01:
        return 0.0
    ann_return_pct = float(np.mean(weekly_returns) * 52 * 100) if weekly_returns else 0.0
    return ann_return_pct / max_dd_pct_val


def recovery_factor(trades: List[TradeRecord], initial_balance: float = 100.0) -> float:
    """Recovery factor = net_pnl / max_drawdown_dollars."""
    net = net_pnl(trades)
    max_dd = max_drawdown_dollars(trades, initial_balance)
    if max_dd < 1e-12:
        return 100.0 if net > 0 else 0.0
    return net / max_dd


def tail_ratio(trades: List[TradeRecord]) -> float:
    """Tail ratio = abs(95th percentile) / abs(5th percentile)."""
    if len(trades) < 10:
        return 0.0
    pnls = np.array([t.pnl for t in trades])
    p95 = abs(np.percentile(pnls, 95))
    p5 = abs(np.percentile(pnls, 5))
    if p5 < 1e-12:
        return 100.0 if p95 > 0 else 0.0
    return float(p95 / p5)


def max_consecutive(trades: List[TradeRecord], winning: bool = False) -> int:
    """Max consecutive wins or losses."""
    if not trades:
        return 0
    max_run = 0
    current_run = 0
    for t in trades:
        if (winning and t.pnl > 0) or (not winning and t.pnl <= 0):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def avg_hold_bars(trades: List[TradeRecord]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.hold_bars for t in trades]))


def median_hold_bars(trades: List[TradeRecord]) -> float:
    if not trades:
        return 0.0
    return float(np.median([t.hold_bars for t in trades]))


def time_in_market_pct(in_position_steps: int, total_steps: int) -> float:
    if total_steps == 0:
        return 0.0
    return in_position_steps / total_steps * 100


def flat_bar_pct(in_position_steps: int, total_steps: int) -> float:
    if total_steps == 0:
        return 100.0
    return (total_steps - in_position_steps) / total_steps * 100


def trades_per_day(total_trades: int, val_weeks: int) -> float:
    trading_days = val_weeks * 5
    if trading_days == 0:
        return 0.0
    return total_trades / trading_days


def entry_timing_score(trades: List[TradeRecord]) -> float:
    """% of trades where price moved favorably after entry."""
    if not trades:
        return 0.0
    favorable = sum(1 for t in trades if t.max_favorable > 0)
    return favorable / len(trades) * 100


def sl_quality_score(trades: List[TradeRecord]) -> float:
    """For SL-hit trades: % where price moved favorably before hitting SL."""
    sl_trades = [t for t in trades if t.close_reason == "SL_HIT"]
    if not sl_trades:
        return 0.0
    favorable = sum(1 for t in sl_trades if t.max_favorable > 0)
    return favorable / len(sl_trades) * 100


# ---------------------------------------------------------------------------
# Side split metrics
# ---------------------------------------------------------------------------

def side_split(trades: List[TradeRecord]) -> Dict[str, Dict]:
    """Split metrics by LONG/SHORT."""
    result = {}
    for side in ("LONG", "SHORT"):
        side_trades = [t for t in trades if t.side == side]
        result[side] = {
            "count": len(side_trades),
            "pnl": sum(t.pnl for t in side_trades),
            "pf": profit_factor(side_trades),
            "win_pct": win_rate(side_trades),
        }
    return result


# ---------------------------------------------------------------------------
# Conviction statistics
# ---------------------------------------------------------------------------

def conviction_distribution(trades: List[TradeRecord]) -> Dict[str, float]:
    """Statistics of conviction values across trades."""
    if not trades:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    convictions = np.array([t.conviction for t in trades])
    return {
        "mean": float(np.mean(convictions)),
        "std": float(np.std(convictions)),
        "p10": float(np.percentile(convictions, 10)),
        "p50": float(np.percentile(convictions, 50)),
        "p90": float(np.percentile(convictions, 90)),
    }


# ---------------------------------------------------------------------------
# Weekly returns helper
# ---------------------------------------------------------------------------

def compute_weekly_returns(
    trades: List[TradeRecord],
    val_weeks: List[int],
    initial_balance: float = 100.0,
) -> List[float]:
    """Compute weekly P/L returns as fraction of starting weekly equity."""
    if not val_weeks:
        return []
    week_pnl: Dict[int, float] = {w: 0.0 for w in val_weeks}
    for t in trades:
        if t.week in week_pnl:
            week_pnl[t.week] += t.pnl
    equity = initial_balance
    returns = []
    for w in sorted(val_weeks):
        pnl = week_pnl.get(w, 0.0)
        r = pnl / equity if equity > 0 else 0.0
        returns.append(r)
        equity += pnl
    return returns


# ---------------------------------------------------------------------------
# Compute all T1 metrics at once
# ---------------------------------------------------------------------------

def compute_all_t1_metrics(
    trades: List[TradeRecord],
    weekly_rets: List[float],
    total_steps: int,
    in_position_steps: int,
    val_weeks_count: int,
    initial_balance: float = 100.0,
) -> Dict[str, float]:
    """Compute all Tier 1 validation metrics in one pass."""
    dd = max_drawdown_pct(trades, initial_balance)
    sides = side_split(trades)

    return {
        "val_trades": len(trades),
        "val_win_pct": win_rate(trades),
        "val_pf": profit_factor(trades),
        "val_sharpe": sharpe_annualized(weekly_rets),
        "val_sortino": sortino_annualized(weekly_rets),
        "val_max_dd_pct": dd,
        "val_net_pnl": net_pnl(trades),
        "val_tim_pct": time_in_market_pct(in_position_steps, total_steps),
        "val_trades_day": trades_per_day(len(trades), val_weeks_count),
        "val_avg_hold": avg_hold_bars(trades),
        "val_median_hold": median_hold_bars(trades),
        "val_calmar": calmar_ratio(weekly_rets, dd),
        "val_recovery_factor": recovery_factor(trades, initial_balance),
        "val_tail_ratio": tail_ratio(trades),
        "val_expectancy": expectancy(trades),
        "val_max_consec_loss": max_consecutive(trades, winning=False),
        "val_max_consec_win": max_consecutive(trades, winning=True),
        "val_gross_profit": gross_profit(trades),
        "val_gross_loss": gross_loss(trades),
        "val_avg_win": avg_win(trades),
        "val_avg_loss": avg_loss(trades),
        "val_win_loss_ratio": win_loss_ratio(trades),
        "val_flat_bar_pct": flat_bar_pct(in_position_steps, total_steps),
        "val_entry_timing": entry_timing_score(trades),
        "val_sl_quality": sl_quality_score(trades),
        "val_long_count": sides["LONG"]["count"],
        "val_short_count": sides["SHORT"]["count"],
        "val_long_pnl": sides["LONG"]["pnl"],
        "val_short_pnl": sides["SHORT"]["pnl"],
        "val_long_pf": sides["LONG"]["pf"],
        "val_short_pf": sides["SHORT"]["pf"],
    }
