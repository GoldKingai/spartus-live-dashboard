"""Tier 1: Validation Eval -- deterministic rollout across all validation weeks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..metrics import (
    compute_all_t1_metrics, compute_weekly_returns,
    conviction_distribution,
)
from ..rollout import (
    RolloutResult, make_eval_config, rollout_weeks, seed_all,
)
from ..types import BenchmarkResult, ConvictionStats

log = logging.getLogger("spartusbench.t1")


def run_t1(
    model: Any,
    config: Any,
    val_weeks: List[int],
    seed: int = 42,
    initial_balance: float = 100.0,
    progress_callback: Optional[Callable] = None,
) -> BenchmarkResult:
    """Run Tier 1: Validation Eval.

    Returns BenchmarkResult with T1 metrics populated.
    """
    seed_all(seed)

    eval_cfg = make_eval_config(config)

    log.info("T1: Running validation eval on %d weeks (seed=%d)", len(val_weeks), seed)

    rollout = rollout_weeks(
        model=model,
        week_indices=val_weeks,
        config=eval_cfg,
        master_seed=seed,
        initial_balance=initial_balance,
        collect_reward_components=True,
        collect_actions=True,
        scenario="base",
        progress_callback=progress_callback,
    )

    # Compute weekly returns
    weekly_rets = compute_weekly_returns(rollout.trades, val_weeks, initial_balance)

    # Compute all T1 metrics
    metrics = compute_all_t1_metrics(
        trades=rollout.trades,
        weekly_rets=weekly_rets,
        total_steps=rollout.total_steps,
        in_position_steps=rollout.in_position_steps,
        val_weeks_count=len(val_weeks),
        initial_balance=initial_balance,
    )

    # Conviction stats
    conv = conviction_distribution(rollout.trades)

    log.info(
        "T1 complete: %d trades | Sharpe=%.2f | PF=%.2f | Win=%.1f%% | MaxDD=%.1f%%",
        metrics["val_trades"], metrics["val_sharpe"],
        metrics["val_pf"], metrics["val_win_pct"], metrics["val_max_dd_pct"],
    )

    result = BenchmarkResult()

    # Populate T1 metrics
    for k, v in metrics.items():
        setattr(result, k, v)

    result.base_trades = rollout.trades
    result.weekly_returns = weekly_rets
    result.total_steps = rollout.total_steps
    result.in_position_steps = rollout.in_position_steps

    result.conviction_stats = ConvictionStats(
        mean=conv["mean"], std=conv["std"],
        p10=conv["p10"], p50=conv["p50"], p90=conv["p90"],
    )

    # Store raw rollout data for T5 and T6
    result._rollout = rollout  # transient, not persisted

    return result
