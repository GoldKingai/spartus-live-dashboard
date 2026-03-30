"""Stress Fragility Detector.

Detects: model performs well under base conditions but collapses when costs
increase, revealing an edge too thin to survive realistic execution costs.
"""

from __future__ import annotations

from typing import Any, Dict

from ..types import DetectorResult, StressResult


def detect_stress_fragility(
    stress_results: Dict[str, StressResult],
) -> DetectorResult:
    """Detect stress fragility from stress matrix results.

    Args:
        stress_results: Dict mapping scenario name -> StressResult

    Returns:
        DetectorResult with detection flag and per-scenario retention
    """
    base = stress_results.get("base")
    if not base or base.pf <= 0:
        return DetectorResult(
            name="stress_fragility",
            detected=True,
            severity=4,
            details={"reason": "base_pf_zero_or_negative", "base_pf": base.pf if base else 0},
        )

    retentions = {}
    for scenario, sr in stress_results.items():
        if scenario == "base":
            continue
        retentions[scenario] = sr.pf / base.pf if base.pf > 0 else 0.0

    r_2x = retentions.get("2x_spread", 0)
    r_combined = retentions.get("combined_2x2x", 0)
    r_3x = retentions.get("3x_spread", 0)

    worst_retention = min(retentions.values()) if retentions else 0
    worst_scenario = min(retentions, key=retentions.get) if retentions else "none"

    is_fragile = (
        r_2x < 0.65 or
        r_combined < 0.50 or
        r_3x < 0.40
    )

    severity = sum([
        r_2x < 0.50,
        r_combined < 0.30,
        r_3x < 0.25,
        worst_retention < 0.20,
    ])

    return DetectorResult(
        name="stress_fragility",
        detected=is_fragile,
        severity=severity,
        details={
            "retentions": {k: round(v, 3) for k, v in retentions.items()},
            "r_2x_spread": round(r_2x, 3),
            "r_combined": round(r_combined, 3),
            "r_3x_spread": round(r_3x, 3),
            "worst_retention": round(worst_retention, 3),
            "worst_scenario": worst_scenario,
        },
    )
