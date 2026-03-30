"""SpartusScore computation, hard-fail rules, and champion protocol."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    BenchmarkResult, DetectorResult, RewardAblation,
    ScoreBreakdown, StressResult,
)

log = logging.getLogger("spartusbench.scoring")

# SpartusScore component weights
SCORE_WEIGHTS = {
    "val_sharpe": 0.25,
    "val_pf": 0.20,
    "stress_robustness": 0.25,
    "max_dd_penalty": 0.15,
    "trade_quality": 0.15,
}

# Stress scenario priority weights
STRESS_WEIGHTS = {
    "2x_spread": 0.35,
    "combined_2x2x": 0.30,
    "3x_spread": 0.20,
    "2x_slip_mean": 0.05,
    "2x_slip_std": 0.05,
    "5x_spread": 0.05,
}


def compute_stress_score(retentions: Dict[str, float]) -> float:
    """Compute stress robustness score (0-100).

    Weighted by scenario importance with worst-case penalty.
    """
    weighted_sum = 0.0
    for scenario, weight in STRESS_WEIGHTS.items():
        r = retentions.get(scenario, 0.0)
        weighted_sum += min(r, 1.0) * weight

    worst = min(retentions.values()) if retentions else 0.0
    worst_penalty = max(0.0, 1.0 - worst) * 0.3

    score = max(0.0, weighted_sum - worst_penalty) * 100
    return min(score, 100.0)


def compute_spartus_score(result: BenchmarkResult) -> ScoreBreakdown:
    """Compute the composite SpartusScore (0-100)."""

    # val_sharpe component: min(sharpe / 5.0, 1.0) * 100
    val_sharpe_raw = min(max(result.val_sharpe, 0) / 5.0, 1.0) * 100

    # val_pf component: min((pf - 1.0) / 2.0, 1.0) * 100 (0 if pf < 1.0)
    pf = result.val_pf
    val_pf_raw = max(0, min((pf - 1.0) / 2.0, 1.0) * 100) if pf >= 1.0 else 0.0

    # stress_robustness component
    retentions = {}
    for scenario, sr in result.stress_results.items():
        if scenario == "base":
            continue
        retentions[scenario] = sr.pf_retention
    stress_raw = compute_stress_score(retentions) if retentions else 0.0

    # max_dd_penalty component: max(0, 100 - max_dd_pct * 5)
    max_dd_raw = max(0.0, 100 - result.val_max_dd_pct * 5)

    # trade_quality component: (win_rate_score * 0.5) + (lesson_score * 0.5)
    win_rate_score = min(result.val_win_pct / 60.0, 1.0) * 100
    # lesson_score: proportion of GOOD_TRADE + SMALL_WIN trades
    if result.base_trades:
        good_trades = sum(
            1 for t in result.base_trades
            if t.lesson_type in ("GOOD_TRADE", "SMALL_WIN")
        )
        lesson_score = (good_trades / len(result.base_trades)) * 100
    else:
        lesson_score = 0.0
    quality_raw = win_rate_score * 0.5 + lesson_score * 0.5

    # Weighted composite
    spartus_score = (
        val_sharpe_raw * SCORE_WEIGHTS["val_sharpe"] +
        val_pf_raw * SCORE_WEIGHTS["val_pf"] +
        stress_raw * SCORE_WEIGHTS["stress_robustness"] +
        max_dd_raw * SCORE_WEIGHTS["max_dd_penalty"] +
        quality_raw * SCORE_WEIGHTS["trade_quality"]
    )

    return ScoreBreakdown(
        val_sharpe_component=round(val_sharpe_raw, 2),
        val_pf_component=round(val_pf_raw, 2),
        stress_component=round(stress_raw, 2),
        max_dd_component=round(max_dd_raw, 2),
        quality_component=round(quality_raw, 2),
        spartus_score=round(spartus_score, 2),
    )


# ---------------------------------------------------------------------------
# Hard-fail rules
# ---------------------------------------------------------------------------

def check_hard_fails(result: BenchmarkResult) -> List[str]:
    """Check all hard-fail rules. Returns list of triggered rule names."""
    fails = []

    # stress_2x_spread_collapse
    sr_2x = result.stress_results.get("2x_spread")
    sr_base = result.stress_results.get("base")
    if sr_2x and sr_base and sr_base.pf > 0:
        if sr_2x.pf / sr_base.pf < 0.65:
            fails.append("stress_2x_spread_collapse")

    # stress_combined_collapse
    sr_comb = result.stress_results.get("combined_2x2x")
    if sr_comb and sr_base and sr_base.pf > 0:
        if sr_comb.pf / sr_base.pf < 0.50:
            fails.append("stress_combined_collapse")

    # max_dd_blowup
    if result.val_max_dd_pct > 25.0:
        fails.append("max_dd_blowup")

    # negative_pf
    if result.val_pf < 1.0:
        fails.append("negative_pf")

    # zero_trades
    if result.val_trades < 10:
        fails.append("zero_trades")

    # reward_hacking (hard-fail version)
    if result.reward_ablation:
        if result.reward_ablation.r5_pct > 50.0 and result.reward_ablation.r1_pct < 15.0:
            fails.append("reward_hacking")

    return fails


# ---------------------------------------------------------------------------
# Side warnings (not hard-fail)
# ---------------------------------------------------------------------------

def check_side_warnings(result: BenchmarkResult) -> List[str]:
    """Check side balance warnings."""
    warnings = []

    # Determine dominant side
    if result.val_long_count > result.val_short_count:
        opp_count = result.val_short_count
        opp_pf = result.val_short_pf
        opp_side = "SHORT"
    else:
        opp_count = result.val_long_count
        opp_pf = result.val_long_pf
        opp_side = "LONG"

    if opp_count < 3:
        warnings.append(f"SIDE_AVOIDANCE_WARNING: Only {opp_count} {opp_side} trades")
    elif opp_count >= 10 and opp_pf < 1.0:
        warnings.append(
            f"SIDE_WEAKNESS_WARNING: {opp_side} PF={opp_pf:.2f} across {opp_count} trades"
        )

    return warnings


# ---------------------------------------------------------------------------
# Champion protocol
# ---------------------------------------------------------------------------

def evaluate_champion_candidacy(
    result: BenchmarkResult,
    champion: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float]:
    """Evaluate whether this result should become the new champion.

    Returns:
        (verdict, delta) where verdict is "PROMOTE", "DRAW", or "REGRESSION"
    """
    if champion is None:
        # No current champion -> auto-promote if not disqualified
        if result.is_disqualified:
            return "DISQUALIFIED", 0.0
        return "PROMOTE", result.score.spartus_score

    champion_score = float(champion.get("spartus_score", 0) or 0)
    candidate_score = result.score.spartus_score
    delta = candidate_score - champion_score

    if result.is_disqualified:
        return "DISQUALIFIED", delta

    if delta > 2.0:
        return "PROMOTE", delta
    elif delta >= -2.0:
        return "DRAW", delta
    else:
        return "REGRESSION", delta
