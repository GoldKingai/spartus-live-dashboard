"""Tier 7: Locked Test -- single-pass evaluation on held-out test weeks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..metrics import (
    compute_weekly_returns, max_drawdown_pct, net_pnl, profit_factor,
    sharpe_annualized, sortino_annualized, time_in_market_pct, win_rate,
)
from ..rollout import make_eval_config, rollout_weeks, seed_all
from ..types import BenchmarkResult

log = logging.getLogger("spartusbench.t7")


def run_t7(
    model: Any,
    config: Any,
    test_weeks: List[int],
    seed: int = 42,
    initial_balance: float = 100.0,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run Tier 7: Locked Test.

    Same as T1 but on test weeks. Every run is permanently recorded.

    Returns dict with test_* metrics.
    """
    if not test_weeks:
        log.warning("T7: No test weeks available")
        return {}

    seed_all(seed)
    eval_cfg = make_eval_config(config)

    log.info("T7: Running locked test on %d weeks (seed=%d)", len(test_weeks), seed)

    rollout = rollout_weeks(
        model=model,
        week_indices=test_weeks,
        config=eval_cfg,
        master_seed=seed,
        initial_balance=initial_balance,
        collect_reward_components=False,
        collect_actions=False,
        scenario="locked_test",
        progress_callback=progress_callback,
    )

    weekly_rets = compute_weekly_returns(rollout.trades, test_weeks, initial_balance)

    test_metrics = {
        "test_trades": len(rollout.trades),
        "test_win_pct": win_rate(rollout.trades),
        "test_pf": profit_factor(rollout.trades),
        "test_sharpe": sharpe_annualized(weekly_rets),
        "test_sortino": sortino_annualized(weekly_rets),
        "test_max_dd_pct": max_drawdown_pct(rollout.trades, initial_balance),
        "test_net_pnl": net_pnl(rollout.trades),
        "test_tim_pct": time_in_market_pct(
            rollout.in_position_steps, rollout.total_steps
        ),
        "test_weeks_used": test_weeks,
    }

    log.info(
        "T7 complete: %d trades | Sharpe=%.2f | PF=%.2f | MaxDD=%.1f%%",
        test_metrics["test_trades"], test_metrics["test_sharpe"],
        test_metrics["test_pf"], test_metrics["test_max_dd_pct"],
    )

    return test_metrics
