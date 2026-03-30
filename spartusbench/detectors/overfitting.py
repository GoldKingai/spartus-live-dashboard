"""Overfitting Detector.

Detects: performance looks great on validation but degrades on unseen data,
or training metrics improving while validation metrics stagnate/decline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import DetectorResult


def detect_overfitting(
    current: Dict[str, Any],
    prior_results: Optional[List[Dict[str, Any]]] = None,
) -> DetectorResult:
    """Detect overfitting from cross-checkpoint comparison or val/test gap.

    Args:
        current: Current benchmark metrics dict
        prior_results: List of prior benchmark result dicts (most recent last)

    Returns:
        DetectorResult with detection signals
    """
    signals = []

    # Signal 1: Val Sharpe declining across recent checkpoints
    if prior_results and len(prior_results) >= 3:
        recent_sharpes = [
            float(r.get("val_sharpe", 0) or 0) for r in prior_results[-3:]
        ] + [float(current.get("val_sharpe", 0) or 0)]

        if all(recent_sharpes[i] < recent_sharpes[i - 1]
               for i in range(1, len(recent_sharpes))):
            signals.append("val_sharpe_monotonic_decline")

    # Signal 2: Locked test vs validation gap (if test data available)
    test_pf = current.get("test_pf")
    val_pf = current.get("val_pf")
    if test_pf is not None and val_pf is not None:
        pf_gap = float(val_pf or 0) - float(test_pf or 0)
        if pf_gap > 0.5:
            signals.append(f"val_test_pf_gap={pf_gap:.2f}")

    # Signal 3: MaxDD on test much worse than validation
    test_dd = current.get("test_max_dd_pct")
    val_dd = current.get("val_max_dd_pct")
    if test_dd is not None and val_dd is not None:
        dd_gap = float(test_dd or 0) - float(val_dd or 0)
        if dd_gap > 5.0:
            signals.append(f"val_test_dd_gap={dd_gap:.1f}%")

    # Signal 4: PF declining across recent checkpoints
    if prior_results and len(prior_results) >= 3:
        recent_pfs = [
            float(r.get("val_pf", 0) or 0) for r in prior_results[-3:]
        ] + [float(current.get("val_pf", 0) or 0)]

        if all(recent_pfs[i] < recent_pfs[i - 1]
               for i in range(1, len(recent_pfs))):
            signals.append("val_pf_monotonic_decline")

    is_detected = len(signals) >= 2

    return DetectorResult(
        name="overfitting",
        detected=is_detected,
        severity=len(signals),
        details={
            "signals": signals,
            "signal_count": len(signals),
        },
    )
