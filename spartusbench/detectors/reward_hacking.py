"""Reward Hacking Detector.

Detects: model exploiting R5 risk bonus (being-in-position reward)
rather than learning real trading edge. R5 dominates total reward
while R1 (actual P/L) is weak.
"""

from __future__ import annotations

from typing import Any, Dict

from ..types import DetectorResult, RewardAblation


def detect_reward_hacking(
    ablation: RewardAblation,
) -> DetectorResult:
    """Detect reward hacking from reward ablation results.

    Args:
        ablation: RewardAblation from T5

    Returns:
        DetectorResult with detection flag
    """
    r5_pct = ablation.r5_pct
    r1_pct = ablation.r1_pct

    # Standard detection threshold
    is_hacking = (
        r5_pct > 40.0 and
        r1_pct < 20.0
    )

    # Hard-fail threshold (more severe)
    is_hard_fail = (
        r5_pct > 50.0 and
        r1_pct < 15.0
    )

    severity = 0
    if is_hacking:
        severity = 2
    if is_hard_fail:
        severity = 4

    return DetectorResult(
        name="reward_hacking",
        detected=is_hacking,
        severity=severity,
        details={
            "r5_pct": round(r5_pct, 1),
            "r1_pct": round(r1_pct, 1),
            "r5_positive_steps": ablation.r5_positive_steps,
            "r5_negative_steps": ablation.r5_negative_steps,
            "is_hard_fail": is_hard_fail,
        },
    )
