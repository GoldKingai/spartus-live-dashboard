"""Tier 2: Stress Matrix -- 7 cost scenarios across validation weeks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..metrics import (
    compute_weekly_returns, max_drawdown_pct, net_pnl, profit_factor,
    sharpe_annualized, sortino_annualized, time_in_market_pct,
    trades_per_day, win_rate, avg_hold_bars, side_split,
)
from ..rollout import (
    STRESS_SCENARIOS, apply_stress_scenario, make_eval_config,
    rollout_weeks, seed_all,
)
from ..types import BenchmarkResult, StressResult

log = logging.getLogger("spartusbench.t2")


def run_t2(
    model: Any,
    config: Any,
    val_weeks: List[int],
    seed: int = 42,
    initial_balance: float = 100.0,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, StressResult]:
    """Run Tier 2: Stress Matrix.

    Returns dict mapping scenario name -> StressResult.
    """
    seed_all(seed)

    results: Dict[str, StressResult] = {}
    base_pf = 0.0
    base_sharpe = 0.0

    for idx, scenario in enumerate(STRESS_SCENARIOS):
        log.info("T2: Running scenario '%s' (%d/%d)",
                 scenario, idx + 1, len(STRESS_SCENARIOS))

        # Apply stress to base eval config
        eval_cfg = make_eval_config(config)
        stressed_cfg = apply_stress_scenario(eval_cfg, scenario)

        rollout = rollout_weeks(
            model=model,
            week_indices=val_weeks,
            config=stressed_cfg,
            master_seed=seed,
            initial_balance=initial_balance,
            collect_reward_components=False,
            collect_actions=False,
            scenario=scenario,
            progress_callback=progress_callback,
        )

        weekly_rets = compute_weekly_returns(rollout.trades, val_weeks, initial_balance)
        sides = side_split(rollout.trades)

        sr = StressResult(
            scenario=scenario,
            trades=len(rollout.trades),
            win_pct=win_rate(rollout.trades),
            net_pnl=net_pnl(rollout.trades),
            pf=profit_factor(rollout.trades),
            sharpe=sharpe_annualized(weekly_rets),
            sortino=sortino_annualized(weekly_rets),
            max_dd_pct=max_drawdown_pct(rollout.trades, initial_balance),
            tim_pct=time_in_market_pct(rollout.in_position_steps, rollout.total_steps),
            avg_hold=avg_hold_bars(rollout.trades),
            trades_per_day=trades_per_day(len(rollout.trades), len(val_weeks)),
            long_count=sides["LONG"]["count"],
            short_count=sides["SHORT"]["count"],
            long_pnl=sides["LONG"]["pnl"],
            short_pnl=sides["SHORT"]["pnl"],
            trade_list=rollout.trades,
        )

        if scenario == "base":
            base_pf = sr.pf
            base_sharpe = sr.sharpe
            sr.pf_retention = 1.0
            sr.sharpe_retention = 1.0
        else:
            sr.pf_retention = sr.pf / base_pf if base_pf > 0 else 0.0
            sr.sharpe_retention = sr.sharpe / base_sharpe if base_sharpe > 0 else 0.0

        results[scenario] = sr

        log.info(
            "  %s: PF=%.2f | retention=%.2f | trades=%d",
            scenario, sr.pf, sr.pf_retention, sr.trades,
        )

    return results
