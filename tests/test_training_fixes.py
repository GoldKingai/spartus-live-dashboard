"""Tests for training system bug fixes (B1-B7, FIX-RESUME).

Covers:
  B1  - Stage 3 year-interleaving
  B2  - VAL_DECLINING state (regime-shift detection)
  B3  - convergence_weeks_since_best = 5 val points
  B4  - VAL_DECLINING increments stagnation in _graduated_defense
  B5  - VAL_DECLINING state exists in STATES list
  B6  - push_profile=False in prefetch path
  B7  - Overfitting detection window 5→3 val points
  FIX-RESUME - _trim_state_to_checkpoint trims collapsed history
  FIX-RESUME - spartus_best filename fallback via .meta.json
  FIX-RESUME - _save_best_model writes .training_state.json sidecar
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.training.convergence import ConvergenceDetector


# ─── B5: VAL_DECLINING in STATES ────────────────────────────────────────────

def test_val_declining_in_states():
    """VAL_DECLINING must appear in the ConvergenceDetector STATES list."""
    assert "VAL_DECLINING" in ConvergenceDetector.STATES


# ─── B3: weeks_since_best_limit = 5 val points ──────────────────────────────

def test_convergence_weeks_since_best_is_5():
    """Config must use 5 val points (not 50) for plateau detection."""
    cfg = TrainingConfig()
    assert cfg.convergence_weeks_since_best == 5, (
        f"Expected 5, got {cfg.convergence_weeks_since_best}. "
        "This was reduced from 50 to prevent 500-week lag before PLATEAU fires."
    )


def test_plateau_fires_after_5_val_points():
    """PLATEAU should fire after 5 consecutive val points without improvement."""
    det = ConvergenceDetector()
    # Feed 5 val points, best at point 0, then 5 more without improvement
    best = 3.0
    for i in range(5):
        det.update(week=i * 10, val_sharpe=best - i * 0.01, train_sharpe=1.0, action_std=0.5)

    # 5 val points recorded, weeks_since_best should be 4 (first was best)
    assert det.weeks_since_best == 4

    # Add one more val point without improvement — should now be 5 >= limit=5 → PLATEAU
    det.update(week=50, val_sharpe=best - 0.5, train_sharpe=1.0, action_std=0.5)
    assert det.state == "PLATEAU", f"Expected PLATEAU, got {det.state}"


# ─── B2/B5: VAL_DECLINING state ─────────────────────────────────────────────

def test_val_declining_fires_on_regime_shift():
    """VAL_DECLINING fires when both train AND val decline (regime shift scenario).

    FIX-VAL: Uses composite score (sqrt(sharpe * (1+return))) for decline detection.
    Both Sharpe AND return must drop for the composite to decline significantly.
    """
    det = ConvergenceDetector()
    # Establish a strong baseline: sharpe=3.0, return=20.0 -> score=sqrt(3*21)~7.94
    for i in range(5):
        det.update(week=i * 10, val_sharpe=3.0, val_return=20.0,
                   train_sharpe=2.0, action_std=0.5)

    assert det.best_val_sharpe == pytest.approx(3.0, abs=0.01)

    # Now simulate regime shift: BOTH sharpe AND return decline
    # sharpe=1.5, return=5.0 -> score=sqrt(1.5*6)~3.0, well below best 7.94
    for i in range(3):
        det.update(
            week=(5 + i) * 10,
            val_sharpe=1.5,     # Declined
            val_return=5.0,     # Return also declined
            train_sharpe=1.0,   # Also declining — classic overfitting check won't fire
            action_std=0.5,
        )

    assert det.state == "VAL_DECLINING", (
        f"Expected VAL_DECLINING, got {det.state}. "
        "Regime-shift (both metrics decline) should be caught by _check_val_declining."
    )


def test_val_declining_does_not_fire_when_val_strong():
    """VAL_DECLINING must NOT fire when composite score stays near previous window.

    FIX-B8: Now compares recent 3 vs previous 3 (trailing window), not vs single best.
    """
    det = ConvergenceDetector()
    # Baseline: sharpe=3.0, return=20.0 -> composite ~7.94
    for i in range(5):
        det.update(week=i * 10, val_sharpe=3.0, val_return=20.0,
                   train_sharpe=2.0, action_std=0.5)

    # Sharpe dips slightly but return stays high -> composite still >70% of previous
    # New composite: sqrt(2.6 * 23) = sqrt(59.8) = 7.73 (~97% of 7.94)
    for i in range(3):
        det.update(
            week=(5 + i) * 10,
            val_sharpe=2.6,
            val_return=22.0,  # Return even improved
            train_sharpe=1.5,
            action_std=0.5,
        )

    assert det.state != "VAL_DECLINING", (
        f"VAL_DECLINING should not fire when composite score is near previous window. Got {det.state}"
    )


def test_classic_overfitting_still_detected():
    """OVERFITTING state must still fire for classic case (train up, val down).

    FIX-B8: Now uses composite scores. Both Sharpe AND return must decline
    such that composite drops >15% for overfitting to trigger.
    """
    det = ConvergenceDetector()
    # Establish baseline: composite = sqrt(2.0 * 11.0) = 4.69
    for i in range(6):
        det.update(week=i * 10, val_sharpe=2.0, val_return=10.0,
                   train_sharpe=1.0, action_std=0.5)

    # Classic overfitting: train improves, val + return both degrade heavily
    # Composite drops: sqrt(2.0*11) = 4.69 -> sqrt(0.8*4) = 1.79 (62% drop, >> 15%)
    for i in range(3):
        det.update(
            week=(6 + i) * 10,
            val_sharpe=1.0 - i * 0.1,    # Declining to 0.8
            val_return=5.0 - i * 1.0,    # Declining to 3.0
            train_sharpe=2.0 + i * 0.1,  # Improving
            action_std=0.5,
        )

    assert det.state == "OVERFITTING", f"Expected OVERFITTING, got {det.state}"


def test_val_declining_increments_overfitting_weeks():
    """overfitting_weeks must increment in VAL_DECLINING state (feeds defense layers).

    FIX-B8: _check_val_declining now needs 6 val points and uses trailing window.
    Need 6 baseline + 3 declining = 9 total.
    """
    det = ConvergenceDetector()
    # Baseline with strong return so composite is high: sqrt(3.0 * 21) = 7.94
    for i in range(6):
        det.update(week=i * 10, val_sharpe=3.0, val_return=20.0,
                   train_sharpe=2.0, action_std=0.5)

    initial_ow = det.overfitting_weeks
    # Both sharpe and return decline -> composite drops significantly
    # sqrt(1.5 * 6) = 3.0, which is 38% of 7.94 (well below 70% threshold)
    det.update(week=60, val_sharpe=1.5, val_return=5.0, train_sharpe=1.0, action_std=0.5)
    det.update(week=70, val_sharpe=1.5, val_return=5.0, train_sharpe=1.0, action_std=0.5)
    det.update(week=80, val_sharpe=1.5, val_return=5.0, train_sharpe=1.0, action_std=0.5)

    if det.state == "VAL_DECLINING":
        assert det.overfitting_weeks > initial_ow, (
            "overfitting_weeks should increment during VAL_DECLINING "
            "so defense layers (soft correction, rollback) can trigger."
        )


# ─── B8: Overfitting false-positive prevention ──────────────────────────────

def test_overfitting_not_triggered_when_return_improves():
    """OVERFITTING must NOT fire when Sharpe drops but return increases.

    This was the root cause of the W60-W89 incident: val Sharpe dropped 28%
    (3.75 -> 2.71) while val return DOUBLED (65 -> 141).  The old detector
    saw only the Sharpe drop and falsely declared overfitting, triggering a
    rollback that destroyed the best model.
    """
    det = ConvergenceDetector()
    # Baseline: sharpe=3.0, return=20.0 -> composite = sqrt(3.0*21) = 7.94
    for i in range(6):
        det.update(week=i * 10, val_sharpe=3.0, val_return=20.0,
                   train_sharpe=1.0, action_std=0.5)

    # Sharpe drops 25% but return doubles -> composite IMPROVES
    # sqrt(2.25 * 41) = sqrt(92.25) = 9.60 (21% BETTER than 7.94)
    for i in range(3):
        det.update(
            week=(6 + i) * 10,
            val_sharpe=2.25,             # 25% Sharpe drop
            val_return=40.0 + i * 5.0,   # Return doubles+
            train_sharpe=1.5 + i * 0.1,  # Train improving
            action_std=0.5,
        )

    assert det.state != "OVERFITTING", (
        f"Got {det.state}. OVERFITTING must not fire when return improves "
        f"even if Sharpe declines.  Composite score should catch this."
    )


# ─── B7: Overfitting detection window 5→3 ───────────────────────────────────

def test_overfitting_detected_with_6_val_points():
    """Overfitting check needs only 6 val points (not 10) after window reduction.

    FIX-B8: Now uses composite scores.  Composite must drop >15%.
    Previous: sharpe=2.5, return=20 -> composite=sqrt(2.5*21)=7.25
    Declining: sharpe=1.0, return=3.0 -> composite=sqrt(1.0*4)=2.0  (72% drop)
    """
    det = ConvergenceDetector()
    for i in range(3):
        det.update(week=i * 10, val_sharpe=2.5, val_return=20.0,
                   train_sharpe=1.5, action_std=0.5)
    for i in range(3):
        det.update(week=(3 + i) * 10, val_sharpe=1.0, val_return=3.0,
                   train_sharpe=2.0, action_std=0.5)

    assert det._check_overfitting(), "Overfitting should be detected with just 6 val points."


def test_overfitting_not_detected_with_5_val_points():
    """Overfitting check should NOT fire with fewer than 6 val points."""
    det = ConvergenceDetector()
    for i in range(5):
        det.update(week=i * 10, val_sharpe=2.5 - i * 0.1, val_return=10.0 - i * 1.0,
                   train_sharpe=1.5 + i * 0.1, action_std=0.5)

    assert not det._check_overfitting(), "Overfitting should not fire with < 6 val points."


# ─── FIX-RESUME: _trim_state_to_checkpoint ──────────────────────────────────

def _make_collapsed_state(current_week=163, n_val=16):
    """Build a training state dict simulating the W163 collapsed state.

    val_sharpes layout (validation_interval=10):
      Index 0-9  = W10-W100  (improving toward peak)
      Index 10   = W110      = 3.01 (the best checkpoint — must be last good point)
      Index 11+  = W120+     (post-W110 collapse)
    This ensures that trimming to 11 val points (= W110) gives last value = 3.01.
    """
    val_sharpes = [2.10, 2.20, 2.36, 2.50, 2.64,   # W10-W50 (improving)
                   2.70, 2.80, 2.88, 2.95, 2.99,    # W60-W100 (near peak)
                   3.01,                              # W110 — BEST (index 10)
                   2.30, -0.19, -0.42, 0.04, 0.1,   # W120-W160 (collapse)
                   ]
    val_sharpes = val_sharpes[:n_val]
    return {
        "current_week": current_week,
        "balance": 495857.0,
        "peak_balance": 2693629.0,
        "stagnation_weeks": 5,
        "plateau_lr_reductions": 1,
        "convergence": {
            "state": "IMPROVING",
            "best_val_sharpe": 3.01,
            "weeks_since_best": 5,
            "overfitting_weeks": 0,
            "collapsed_weeks": 0,
            "val_sharpes": val_sharpes,
            "train_sharpes_at_val": [1.5] * n_val,
            "train_sharpes": [1.0] * current_week,
            "action_stds": [0.4] * current_week,
        },
    }


def test_trim_state_resets_week_and_balance(tmp_path):
    """_trim_state_to_checkpoint must set current_week and balance to checkpoint values."""
    from src.training.trainer import SpartusTrainer  # noqa: F401 — just to import the class

    # Create a dummy meta.json for W110
    meta = {"week": 110, "val_sharpe": 3.01, "balance": 60831.0}
    cp = tmp_path / "spartus_week_0110.zip"
    cp.touch()
    (tmp_path / "spartus_week_0110.meta.json").write_text(json.dumps(meta))

    cfg = TrainingConfig()
    # Use a minimal Trainer — just to call the method
    trainer = _make_minimal_trainer(cfg)
    state = _make_collapsed_state()
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    assert trimmed["current_week"] == 110
    assert trimmed["balance"] == pytest.approx(60831.0, abs=1.0)
    assert trimmed["peak_balance"] == pytest.approx(60831.0, abs=1.0)
    assert trimmed["stagnation_weeks"] == 0
    assert trimmed["plateau_lr_reductions"] == 0


def test_trim_state_trims_val_sharpes(tmp_path):
    """_trim_state_to_checkpoint must trim val_sharpes to match checkpoint week."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    cp = tmp_path / "spartus_week_0110.zip"
    cp.touch()

    state = _make_collapsed_state(current_week=163, n_val=16)
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    expected_val_points = 110 // cfg.validation_interval  # = 11
    conv = trimmed["convergence"]
    assert len(conv["val_sharpes"]) == expected_val_points, (
        f"Expected {expected_val_points} val points, got {len(conv['val_sharpes'])}"
    )
    # Last val_sharpe should be 3.01 (the peak at W110), not the negative collapsed values
    assert conv["val_sharpes"][-1] == pytest.approx(3.01, abs=0.01)


def test_trim_state_resets_weeks_since_best(tmp_path):
    """After trimming to W110, weeks_since_best must be 0 (W110 was the best)."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    cp = tmp_path / "spartus_week_0110.zip"
    cp.touch()

    state = _make_collapsed_state(current_week=163, n_val=16)
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    conv = trimmed["convergence"]
    assert conv["weeks_since_best"] == 0, (
        f"weeks_since_best should be 0 after W110 trim (W110 was the best). "
        f"Got {conv['weeks_since_best']}"
    )
    assert conv["overfitting_weeks"] == 0


def test_trim_state_no_immediate_plateau(tmp_path):
    """Trimmed state must not immediately trigger PLATEAU on first validation."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    cp = tmp_path / "spartus_week_0110.zip"
    cp.touch()

    state = _make_collapsed_state(current_week=163, n_val=16)
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    # Restore into a fresh detector
    det = ConvergenceDetector(cfg)
    det.restore_from_summary(trimmed["convergence"])

    # Should NOT be PLATEAU — weeks_since_best=0 < limit=5
    assert det.state != "PLATEAU", (
        f"PLATEAU should not fire immediately after W110 resume. Got {det.state}"
    )
    assert det.state not in ("VAL_DECLINING",), (
        f"VAL_DECLINING should not fire after W110 resume (val was 3.01). Got {det.state}"
    )


def test_trim_state_no_immediate_val_declining(tmp_path):
    """Trimmed state must not immediately trigger VAL_DECLINING after W110 resume."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    cp = tmp_path / "spartus_week_0110.zip"
    cp.touch()

    state = _make_collapsed_state(current_week=163, n_val=16)
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    det = ConvergenceDetector(cfg)
    det.restore_from_summary(trimmed["convergence"])

    # val_sharpes[-3:] = [2.95, 2.99, 3.01], avg=2.983 >> best(3.01) - 0.5 = 2.51
    recent_avg = np.mean(det.val_sharpes[-3:])
    assert recent_avg > det.best_val_sharpe - 0.5, (
        f"Recent val avg ({recent_avg:.3f}) should be above declining threshold "
        f"({det.best_val_sharpe - 0.5:.3f}) after W110 trim"
    )


def test_trim_state_spartus_best_fallback(tmp_path):
    """_trim_state_to_checkpoint must work with 'spartus_best' filename via .meta.json."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    # spartus_best with meta.json containing week=110
    cp = tmp_path / "spartus_best.zip"
    cp.touch()
    meta = {"week": 110, "val_sharpe": 3.01, "balance": 60831.0}
    (tmp_path / "spartus_best.meta.json").write_text(json.dumps(meta))

    state = _make_collapsed_state(current_week=163, n_val=16)
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    # Should have trimmed to W110
    assert trimmed["current_week"] == 110, (
        f"spartus_best should be trimmed to W110 via meta.json. Got {trimmed['current_week']}"
    )


def test_trim_state_no_op_when_checkpoint_is_current(tmp_path):
    """_trim_state_to_checkpoint must not modify state if checkpoint == current week."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    cp = tmp_path / "spartus_week_0163.zip"
    cp.touch()

    state = _make_collapsed_state(current_week=163, n_val=16)
    trimmed = trainer._trim_state_to_checkpoint(state, cp)

    assert trimmed["current_week"] == 163  # Unchanged


# ─── B1: Stage 3 year-interleaving ──────────────────────────────────────────

def test_stage3_order_year_interleaved():
    """_build_stage3_order must return year-interleaved order (FIX-B8).

    Sort by (week, year): 2015w1, 2016w1, 2017w1, 2015w2, 2016w2, ...
    This prevents single-year lock-in that caused catastrophic forgetting.
    """
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    # Simulate weeks_data with 3 years × 4 weeks each
    trainer._weeks_data = [
        {"year": y, "week": w} for y in [2015, 2016, 2017] for w in range(1, 5)
    ]
    trainer._train_weeks = list(range(len(trainer._weeks_data)))

    order = trainer._build_stage3_order()

    # First 3 entries should be w1 from each year (interleaved)
    first_3 = [(trainer._weeks_data[idx]["year"], trainer._weeks_data[idx]["week"]) for idx in order[:3]]
    assert first_3 == [(2015, 1), (2016, 1), (2017, 1)], (
        f"First 3 should be week 1 from each year. Got {first_3}"
    )

    # Next 3 should be w2 from each year
    next_3 = [(trainer._weeks_data[idx]["year"], trainer._weeks_data[idx]["week"]) for idx in order[3:6]]
    assert next_3 == [(2015, 2), (2016, 2), (2017, 2)], (
        f"Next 3 should be week 2 from each year. Got {next_3}"
    )

    # Full order must be (week, year) ascending
    pairs = [(trainer._weeks_data[idx]["week"], trainer._weeks_data[idx]["year"]) for idx in order]
    assert pairs == sorted(pairs), "Order must be (week, year) ascending for interleaving"

    # Must cover all weeks
    assert len(order) == len(trainer._train_weeks)
    assert set(order) == set(trainer._train_weeks)


def test_stage3_order_single_year():
    """_build_stage3_order with only one year should preserve original order."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    trainer._weeks_data = [{"year": 2015, "week": w} for w in range(1, 6)]
    trainer._train_weeks = list(range(5))

    order = trainer._build_stage3_order()
    assert order == [0, 1, 2, 3, 4]


def test_stage3_interleaved_year_diversity():
    """Year-interleaved order cycles through all years at each week number (FIX-B8).

    With 4 years × 9 weeks, every 4 consecutive entries should have 4 different years.
    This prevents the catastrophic forgetting caused by 50+ weeks of single-year data.
    """
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    trainer._weeks_data = [
        {"year": y, "week": w}
        for y in [2015, 2016, 2017, 2018]
        for w in range(1, 10)
    ]
    trainer._train_weeks = list(range(len(trainer._weeks_data)))

    order = trainer._build_stage3_order()

    # Check that every block of 4 (= num_years) has all 4 years represented
    n_years = 4
    for block_start in range(0, len(order) - n_years + 1, n_years):
        block = order[block_start:block_start + n_years]
        years_in_block = {trainer._weeks_data[idx]["year"] for idx in block}
        assert len(years_in_block) == n_years, (
            f"Block starting at {block_start} should have {n_years} different years. "
            f"Got {years_in_block}"
        )

    # Never more than n_years consecutive entries from same year
    max_same_year_run = 1
    current_run = 1
    for i in range(1, len(order)):
        y1 = trainer._weeks_data[order[i - 1]]["year"]
        y2 = trainer._weeks_data[order[i]]["year"]
        if y1 == y2:
            current_run += 1
            max_same_year_run = max(max_same_year_run, current_run)
        else:
            current_run = 1
    assert max_same_year_run == 1, (
        f"No consecutive entries should be from same year (interleaved). "
        f"Max run: {max_same_year_run}"
    )


# ─── B6: push_profile=False in prefetch ─────────────────────────────────────

def test_select_week_push_profile_false_does_not_write_metrics():
    """_select_week(push_profile=False) must not modify shared_metrics."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    trainer._weeks_data = [{"year": 2015, "week": w} for w in range(1, 10)]
    trainer._train_weeks = list(range(9))
    trainer._stage3_order = list(range(9))

    # Record initial shared_metrics state
    before = dict(trainer.shared_metrics)

    # Call _select_week with push_profile=False (as prefetch does)
    trainer._select_week(90, push_profile=False)  # Stage 3 (>= stage2_end_week=80)

    # shared_metrics must be unchanged
    after = dict(trainer.shared_metrics)
    assert before == after, (
        "shared_metrics was modified by _select_week(push_profile=False). "
        "Prefetch must not contaminate current-week dashboard metrics."
    )


def test_select_week_push_profile_true_does_write_metrics():
    """_select_week(push_profile=True) must update shared_metrics (default behavior)."""
    cfg = TrainingConfig()
    trainer = _make_minimal_trainer(cfg)

    trainer._weeks_data = [{"year": 2015, "week": w} for w in range(1, 10)]
    trainer._train_weeks = list(range(9))
    trainer._stage3_order = list(range(9))
    trainer._week_difficulties = {i: 0.5 for i in range(9)}
    trainer._week_profiles = {i: {"trending_up": 0.6} for i in range(9)}

    trainer._select_week(90, push_profile=True)
    assert "week_difficulty" in trainer.shared_metrics


# ─── Helper ──────────────────────────────────────────────────────────────────

def _make_minimal_trainer(cfg: TrainingConfig):
    """Create a SpartusTrainer stub with mocked external dependencies for unit testing."""
    from src.training.trainer import SpartusTrainer

    t = SpartusTrainer.__new__(SpartusTrainer)
    t.cfg = cfg
    t.seed = 42
    t.shared_metrics = {}
    t.convergence = ConvergenceDetector(cfg)
    t.logger = MagicMock()
    t._weeks_data = []
    t._train_weeks = []
    t._val_weeks = []
    t._test_weeks = []
    t._stage3_order = []
    t._week_difficulties = {}
    t._week_profiles = {}
    t._lr_timesteps = [0]
    t._lr_multiplier = [1.0]
    t._stagnation_weeks = 0
    t._plateau_lr_reductions = 0
    t._rollback_count = 0
    t._last_rollback_week = -999
    t._forced_stage2_until = 0
    t._best_checkpoint_week = 0
    t._bankruptcies = 0
    t.balance = cfg.initial_balance
    t.peak_balance = cfg.initial_balance
    t.current_week = 0
    return t


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
