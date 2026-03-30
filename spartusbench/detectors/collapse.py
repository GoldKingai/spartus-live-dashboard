"""Conviction Collapse / Gating Deadlock Detector.

Detects: model is alive (producing varied actions) but nothing gets through
the gating pipeline. The model won't trade in production.
"""

from __future__ import annotations

from typing import Any, Dict

from ..types import DetectorResult


def detect_conviction_collapse(
    result: Dict[str, Any],
) -> DetectorResult:
    """Detect conviction collapse / gating deadlock.

    Args:
        result: Benchmark metrics dict with gating results

    Returns:
        DetectorResult with detection flag and gate analysis
    """
    trades = int(result.get("val_trades", 0))
    flat_pct = float(result.get("val_flat_bar_pct", 0))
    dir_pass = float(result.get("gate_direction_pass", 0))
    conv_pass = float(result.get("gate_conviction_pass", 0))
    spread_pass = float(result.get("gate_spread_pass", 0))
    lot_pass = float(result.get("gate_lot_pass", 0))

    # Estimate action std from action stats
    action_dir_std = float(result.get("action_direction_std", 0))
    action_conv_std = float(result.get("action_conviction_std", 0))
    mean_action_std = (action_dir_std + action_conv_std) / 2.0

    # Primary trigger: policy is active but almost no trades
    is_deadlocked = (
        mean_action_std > 0.10 and
        trades < 5 and
        flat_pct > 95.0
    )

    # Identify bottleneck gate
    gates = {
        "direction": dir_pass,
        "conviction": conv_pass,
        "spread": spread_pass,
        "lot_sizing": lot_pass,
    }
    bottleneck = min(gates, key=gates.get) if any(v > 0 for v in gates.values()) else "all"

    return DetectorResult(
        name="conviction_collapse",
        detected=is_deadlocked,
        severity=1 if is_deadlocked else 0,
        details={
            "mean_action_std": round(mean_action_std, 4),
            "flat_pct": round(flat_pct, 1),
            "total_trades": trades,
            "bottleneck": bottleneck,
            "gate_direction_pass": round(dir_pass, 1),
            "gate_conviction_pass": round(conv_pass, 1),
            "gate_spread_pass": round(spread_pass, 1),
            "gate_lot_pass": round(lot_pass, 1),
        },
    )
