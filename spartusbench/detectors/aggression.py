"""Aggression Drift Detector.

Detects: model becoming reckless -- trading more frequently with lower quality.
Pattern: TIM increasing + PF decreasing + Max DD increasing + churn increasing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import DetectorResult


def detect_aggression_drift(
    current: Dict[str, Any],
    reference: Optional[Dict[str, Any]] = None,
) -> DetectorResult:
    """Detect aggression drift vs a reference (champion or prior run).

    Args:
        current: Current benchmark metrics dict
        reference: Reference benchmark metrics dict (champion or prior)

    Returns:
        DetectorResult with detection flag and severity (0-4)
    """
    if reference is None:
        return DetectorResult(
            name="aggression_drift",
            detected=False,
            details={"reason": "no_reference"},
        )

    tim_delta = _get(current, "val_tim_pct") - _get(reference, "val_tim_pct")
    pf_delta = _get(current, "val_pf") - _get(reference, "val_pf")
    dd_delta = _get(current, "val_max_dd_pct") - _get(reference, "val_max_dd_pct")
    tpd_delta = _get(current, "val_trades_day") - _get(reference, "val_trades_day")

    # Primary trigger: 3 of 4 moving in the bad direction
    bad_signals = [
        tim_delta > 5.0,
        pf_delta < -0.2,
        dd_delta > 2.0,
        tpd_delta > 1.5,
    ]
    is_drifting = sum(bad_signals) >= 3

    # Severity (0-4): count of severe sub-thresholds
    severity = sum([
        tim_delta > 10.0,
        pf_delta < -0.5,
        dd_delta > 5.0,
        tpd_delta > 3.0,
    ])

    return DetectorResult(
        name="aggression_drift",
        detected=is_drifting,
        severity=severity,
        details={
            "tim_delta": round(tim_delta, 2),
            "pf_delta": round(pf_delta, 2),
            "dd_delta": round(dd_delta, 2),
            "tpd_delta": round(tpd_delta, 2),
            "bad_signals": sum(bad_signals),
        },
    )


def _get(d: Dict, key: str, default: float = 0.0) -> float:
    return float(d.get(key, default) or default)
