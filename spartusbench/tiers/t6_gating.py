"""Tier 6: Gating Diagnostics -- simulate the gate pipeline per bar."""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np

from ..types import ActionStats, GatingResult

log = logging.getLogger("spartusbench.t6")


def run_t6(
    raw_actions: List[np.ndarray],
    config: Any,
) -> GatingResult:
    """Run Tier 6: Gating Diagnostics.

    Simulates the gating chain on every step's raw action output.
    """
    if not raw_actions:
        return GatingResult()

    total_bars = len(raw_actions)
    dir_pass = 0
    conv_pass_live = 0
    conv_pass_train = 0
    lot_pass = 0
    spread_pass = 0
    overall_pass = 0
    promote_count = 0
    lot_attempts = 0

    for action in raw_actions:
        direction = action[0]
        # Map conviction from [-1,1] -> [0,1]
        conviction = (action[1] + 1.0) / 2.0

        # Direction gate
        passes_direction = abs(direction) >= getattr(config, 'direction_threshold', 0.3)
        if passes_direction:
            dir_pass += 1

            # Conviction gates (only for candidate bars)
            passes_conv_live = conviction >= 0.15
            passes_conv_train = conviction >= 0.30

            if passes_conv_live:
                conv_pass_live += 1
            if passes_conv_train:
                conv_pass_train += 1

            # Lot sizing gate (simplified: conviction > 0 implies lots > 0)
            lot_attempts += 1
            lots_ok = conviction > 0.05  # Proxy: very low conviction = 0 lots
            if lots_ok:
                lot_pass += 1

                # Spread gate (always passes for benchmarking with configured spreads)
                spread_pass += 1
                overall_pass += 1

    result = GatingResult(
        total_bars=total_bars,
        direction_pass_count=dir_pass,
        direction_pass_pct=dir_pass / total_bars * 100 if total_bars > 0 else 0,
        conviction_pass_live_count=conv_pass_live,
        conviction_pass_live_pct=conv_pass_live / dir_pass * 100 if dir_pass > 0 else 0,
        conviction_pass_train_count=conv_pass_train,
        conviction_pass_train_pct=conv_pass_train / dir_pass * 100 if dir_pass > 0 else 0,
        lot_pass_count=lot_pass,
        lot_pass_pct=lot_pass / max(lot_attempts, 1) * 100,
        spread_pass_count=spread_pass,
        spread_pass_pct=spread_pass / max(lot_pass, 1) * 100,
        overall_pass_count=overall_pass,
        overall_pass_pct=overall_pass / total_bars * 100 if total_bars > 0 else 0,
        promote_rate=0.0,  # Would need actual lot_sizing call
    )

    log.info(
        "T6: Direction=%.1f%% | Conv(live)=%.1f%% | Conv(train)=%.1f%% | Overall=%.1f%%",
        result.direction_pass_pct, result.conviction_pass_live_pct,
        result.conviction_pass_train_pct, result.overall_pass_pct,
    )

    return result


def compute_action_stats(raw_actions: List[np.ndarray]) -> ActionStats:
    """Compute summary statistics of raw action outputs."""
    if not raw_actions:
        return ActionStats()

    actions = np.array(raw_actions)

    return ActionStats(
        direction_mean=float(np.mean(actions[:, 0])),
        direction_std=float(np.std(actions[:, 0])),
        conviction_mean=float(np.mean(actions[:, 1])),
        conviction_std=float(np.std(actions[:, 1])),
        exit_mean=float(np.mean(actions[:, 2])),
        exit_std=float(np.std(actions[:, 2])),
        sl_mean=float(np.mean(actions[:, 3])),
        sl_std=float(np.std(actions[:, 3])),
    )
