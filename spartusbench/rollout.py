"""Deterministic rollout engine for SpartusBench.

Runs a model through the SpartusTradeEnv collecting trades, reward
components, action statistics, and gating data. Single environment
(n_envs=1) for determinism.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import TradeRecord

log = logging.getLogger("spartusbench.rollout")


@dataclass
class RolloutResult:
    """Raw data collected from a single or multi-week rollout."""
    trades: List[TradeRecord] = field(default_factory=list)
    weekly_pnls: Dict[int, float] = field(default_factory=dict)
    total_steps: int = 0
    in_position_steps: int = 0

    # Reward components (per step, for ablation)
    r1_values: List[float] = field(default_factory=list)
    r2_values: List[float] = field(default_factory=list)
    r3_values: List[float] = field(default_factory=list)
    r4_values: List[float] = field(default_factory=list)
    r5_values: List[float] = field(default_factory=list)

    # Raw actions (per step, for gating + action stats)
    raw_actions: List[np.ndarray] = field(default_factory=list)

    # Final balance
    final_balance: float = 0.0


def seed_all(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_eval_config(base_config: Any, **overrides) -> Any:
    """Create a deterministic eval config from base TrainingConfig.

    Disables all randomization for benchmark reproducibility.
    """
    cfg = deepcopy(base_config)

    # Disable randomization
    if hasattr(cfg, 'start_offset_max'):
        cfg.start_offset_max = 0
    if hasattr(cfg, 'observation_noise_std'):
        cfg.observation_noise_std = 0.0
    if hasattr(cfg, 'spread_jitter'):
        cfg.spread_jitter = 0.0
    if hasattr(cfg, 'slippage_jitter'):
        cfg.slippage_jitter = 0.0
    if hasattr(cfg, 'commission_jitter'):
        cfg.commission_jitter = 0.0

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def apply_stress_scenario(
    base_config: Any,
    scenario: str,
) -> Any:
    """Apply a stress scenario to a config by multiplying cost parameters."""
    SCENARIOS = {
        "base": {"spread_mult": 1.0, "slip_mean_mult": 1.0, "slip_std_mult": 1.0},
        "2x_spread": {"spread_mult": 2.0, "slip_mean_mult": 1.0, "slip_std_mult": 1.0},
        "combined_2x2x": {"spread_mult": 2.0, "slip_mean_mult": 2.0, "slip_std_mult": 2.0},
        "3x_spread": {"spread_mult": 3.0, "slip_mean_mult": 1.0, "slip_std_mult": 1.0},
        "2x_slip_mean": {"spread_mult": 1.0, "slip_mean_mult": 2.0, "slip_std_mult": 1.0},
        "2x_slip_std": {"spread_mult": 1.0, "slip_mean_mult": 1.0, "slip_std_mult": 2.0},
        "5x_spread": {"spread_mult": 5.0, "slip_mean_mult": 1.0, "slip_std_mult": 1.0},
    }

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown stress scenario: {scenario}")

    s = SCENARIOS[scenario]
    cfg = deepcopy(base_config)

    # Multiply spread across all sessions
    spread_mult = s["spread_mult"]
    cfg.spread_london_pips = cfg.spread_london_pips * spread_mult
    cfg.spread_ny_pips = cfg.spread_ny_pips * spread_mult
    cfg.spread_asia_pips = cfg.spread_asia_pips * spread_mult
    cfg.spread_off_hours_pips = cfg.spread_off_hours_pips * spread_mult

    # Multiply slippage
    if hasattr(cfg, 'slippage_mean_pips'):
        cfg.slippage_mean_pips = cfg.slippage_mean_pips * s["slip_mean_mult"]
    if hasattr(cfg, 'slippage_std_pips'):
        cfg.slippage_std_pips = cfg.slippage_std_pips * s["slip_std_mult"]

    return cfg


STRESS_SCENARIOS = [
    "base", "2x_spread", "combined_2x2x", "3x_spread",
    "2x_slip_mean", "2x_slip_std", "5x_spread",
]


def rollout_week(
    model: Any,
    features_df: Any,
    config: Any,
    week_idx: int,
    seed: int = 42,
    initial_balance: float = 100.0,
    collect_reward_components: bool = False,
    collect_actions: bool = False,
    scenario: str = "base",
) -> RolloutResult:
    """Run a deterministic rollout for a single week.

    Args:
        model: SB3 SAC model (loaded on CPU)
        features_df: Pre-computed features DataFrame for this week
        config: TrainingConfig (with eval overrides applied)
        week_idx: Week index (for tracking)
        seed: RNG seed
        initial_balance: Starting balance
        collect_reward_components: If True, collect R1-R5 per step
        collect_actions: If True, collect raw actions per step
        scenario: Stress scenario name for trade tagging

    Returns:
        RolloutResult with trades, steps, and optionally reward/action data
    """
    from src.environment.trade_env import SpartusTradeEnv
    from src.data.normalizer import ExpandingWindowNormalizer
    from src.memory.trading_memory import TradingMemory

    result = RolloutResult()

    # Create normalizer and normalize features
    normalizer = ExpandingWindowNormalizer(config)
    norm_df = normalizer.normalize(features_df)

    # Use an in-memory SQLite database so benchmarks don't conflict
    # with the training process that holds a lock on the main DB
    bench_memory = TradingMemory(db_path=":memory:", config=config)

    # Create environment (single env, no VecEnv wrapper)
    env = SpartusTradeEnv(
        features_df=norm_df,
        config=config,
        memory=bench_memory,
        initial_balance=initial_balance,
        week=week_idx,
        seed=seed,
        is_validation=True,
    )

    obs, info = env.reset(seed=seed)
    done = False
    trade_counter = 0
    week_pnl = 0.0

    while not done:
        # Stochastic prediction — SAC policy mean is near 0 for direction,
        # so deterministic=True produces 0 trades. Stochastic with fixed seed
        # is both reproducible (seed_all called before rollout) and correct.
        action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        result.total_steps += 1
        if info.get("has_position", False):
            result.in_position_steps += 1

        # Collect reward components
        if collect_reward_components:
            result.r1_values.append(info.get("r1_position_pnl", 0.0))
            result.r2_values.append(info.get("r2_trade_quality", 0.0))
            result.r3_values.append(info.get("r3_drawdown", 0.0))
            result.r4_values.append(info.get("r4_sharpe", 0.0))
            result.r5_values.append(info.get("r5_risk_bonus", 0.0))

        # Collect raw actions
        if collect_actions:
            result.raw_actions.append(action.copy())

        # Check for closed trade
        if "last_trade" in info:
            trade = info["last_trade"]
            trade_counter += 1
            pnl = trade.get("pnl", 0.0)
            week_pnl += pnl

            # Determine session from entry step
            session = _classify_session(trade.get("entry_conditions", {}))

            record = TradeRecord(
                trade_num=trade_counter,
                week=week_idx,
                step=info.get("step", 0),
                side=trade.get("side", "UNKNOWN"),
                entry_price=trade.get("entry_price", 0.0),
                exit_price=trade.get("exit_price", 0.0),
                lots=trade.get("lots", 0.0),
                pnl=pnl,
                pnl_pct=pnl / max(initial_balance, 1.0) * 100,
                hold_bars=trade.get("hold_bars", 0),
                conviction=trade.get("conviction", 0.0),
                close_reason=trade.get("reason", "UNKNOWN"),
                lesson_type=_classify_lesson(trade),
                session=session,
                atr_at_entry=trade.get("entry_conditions", {}).get("atr", 0.0),
                max_favorable=trade.get("max_favorable",
                                       trade.get("entry_conditions", {}).get("max_favorable", 0.0)),
                initial_sl=trade.get("entry_conditions", {}).get("stop_loss", 0.0),
                initial_tp=trade.get("entry_conditions", {}).get("take_profit", 0.0),
                final_sl=trade.get("final_sl", 0.0),
                scenario=scenario,
            )
            result.trades.append(record)

    result.weekly_pnls[week_idx] = week_pnl
    result.final_balance = info.get("balance", initial_balance)

    env.close()
    return result


def rollout_weeks(
    model: Any,
    week_indices: List[int],
    config: Any,
    master_seed: int = 42,
    initial_balance: float = 100.0,
    collect_reward_components: bool = False,
    collect_actions: bool = False,
    scenario: str = "base",
    progress_callback: Any = None,
) -> RolloutResult:
    """Run deterministic rollout across multiple weeks with balance persistence."""
    from .discovery import load_week_features

    combined = RolloutResult()
    balance = initial_balance
    trade_counter = 0

    for i, week_idx in enumerate(sorted(week_indices)):
        week_seed = master_seed + week_idx * 100

        try:
            features_df = load_week_features(week_idx)
        except FileNotFoundError:
            log.warning("Skipping week %d: no feature cache", week_idx)
            continue

        week_result = rollout_week(
            model=model,
            features_df=features_df,
            config=config,
            week_idx=week_idx,
            seed=week_seed,
            initial_balance=balance,
            collect_reward_components=collect_reward_components,
            collect_actions=collect_actions,
            scenario=scenario,
        )

        # Renumber trades for continuity
        for t in week_result.trades:
            trade_counter += 1
            t.trade_num = trade_counter

        combined.trades.extend(week_result.trades)
        combined.weekly_pnls.update(week_result.weekly_pnls)
        combined.total_steps += week_result.total_steps
        combined.in_position_steps += week_result.in_position_steps

        if collect_reward_components:
            combined.r1_values.extend(week_result.r1_values)
            combined.r2_values.extend(week_result.r2_values)
            combined.r3_values.extend(week_result.r3_values)
            combined.r4_values.extend(week_result.r4_values)
            combined.r5_values.extend(week_result.r5_values)

        if collect_actions:
            combined.raw_actions.extend(week_result.raw_actions)

        # Balance persists across weeks
        balance = week_result.final_balance

        if progress_callback:
            progress_callback(i + 1, len(week_indices), week_idx)

    combined.final_balance = balance
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_MAP = {
    "london": "London",
    "ny": "NY",
    "ny_overlap": "NY_Overlap",
    "ny_pm": "NY_PM",
    "asia": "Asia",
    "off_hours": "Off_hours",
}


def _classify_session(entry_conditions: Dict) -> str:
    """Classify trading session from entry conditions."""
    session = entry_conditions.get("session", "")
    if session:
        return SESSION_MAP.get(session.lower(), session)

    # Fallback: use hour
    hour = entry_conditions.get("hour", 12)
    if 8 <= hour < 12:
        return "London"
    elif 12 <= hour < 20:
        return "NY"
    elif 0 <= hour < 8:
        return "Asia"
    return "Off_hours"


def _classify_lesson(trade: Dict) -> str:
    """Classify trade lesson type (simplified version of TradeAnalyzer)."""
    pnl = trade.get("pnl", 0.0)
    reason = trade.get("reason", "UNKNOWN")
    hold_bars = trade.get("hold_bars", 0)

    if pnl > 0:
        if reason == "TP_HIT":
            return "GOOD_TRADE"
        elif reason == "AGENT_CLOSE":
            return "SMALL_WIN"
        else:
            return "SMALL_WIN"
    else:
        if reason == "SL_HIT":
            if hold_bars <= 2:
                return "BAD_ENTRY"
            return "STOPPED_OUT"
        elif reason == "EMERGENCY_STOP":
            return "EMERGENCY"
        elif reason == "TIMEOUT":
            return "TIMEOUT_LOSS"
        return "LOSS"
