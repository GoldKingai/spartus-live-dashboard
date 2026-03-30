"""Tier 5: Reward Ablation -- decompose R1-R5 contribution."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..types import RewardAblation

log = logging.getLogger("spartusbench.t5")

# Reward component weights (from config)
REWARD_WEIGHTS = {
    "r1": 0.40,
    "r2": 0.20,
    "r3": 0.15,
    "r4": 0.15,
    "r5": 0.10,
}


def run_t5(
    r1_values: List[float],
    r2_values: List[float],
    r3_values: List[float],
    r4_values: List[float],
    r5_values: List[float],
) -> RewardAblation:
    """Run Tier 5: Reward Ablation.

    Decomposes the reward signal into its 5 components.
    """
    if not r1_values:
        return RewardAblation()

    r1_sum = float(np.sum(r1_values))
    r2_sum = float(np.sum(r2_values))
    r3_sum = float(np.sum(r3_values))
    r4_sum = float(np.sum(r4_values))
    r5_sum = float(np.sum(r5_values))

    w1 = REWARD_WEIGHTS["r1"] * r1_sum
    w2 = REWARD_WEIGHTS["r2"] * r2_sum
    w3 = REWARD_WEIGHTS["r3"] * r3_sum
    w4 = REWARD_WEIGHTS["r4"] * r4_sum
    w5 = REWARD_WEIGHTS["r5"] * r5_sum

    total = abs(w1) + abs(w2) + abs(w3) + abs(w4) + abs(w5)
    if total < 1e-12:
        total = 1.0

    r5_arr = np.array(r5_values)

    result = RewardAblation(
        r1_sum=r1_sum,
        r2_sum=r2_sum,
        r3_sum=r3_sum,
        r4_sum=r4_sum,
        r5_sum=r5_sum,
        r1_weighted=w1,
        r2_weighted=w2,
        r3_weighted=w3,
        r4_weighted=w4,
        r5_weighted=w5,
        total_weighted=w1 + w2 + w3 + w4 + w5,
        r1_pct=abs(w1) / total * 100,
        r2_pct=abs(w2) / total * 100,
        r3_pct=abs(w3) / total * 100,
        r4_pct=abs(w4) / total * 100,
        r5_pct=abs(w5) / total * 100,
        r5_positive_steps=int(np.sum(r5_arr > 0)),
        r5_negative_steps=int(np.sum(r5_arr < 0)),
        r5_zero_steps=int(np.sum(r5_arr == 0)),
    )

    log.info(
        "T5: R1=%.1f%% R2=%.1f%% R3=%.1f%% R4=%.1f%% R5=%.1f%%",
        result.r1_pct, result.r2_pct, result.r3_pct,
        result.r4_pct, result.r5_pct,
    )

    if result.r5_pct > 40.0:
        log.warning("T5: R5 contribution %.1f%% > 40%% -- REWARD HACKING RISK", result.r5_pct)

    return result
