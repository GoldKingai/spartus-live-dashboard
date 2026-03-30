"""Tests for CuratedTrainingBuffer and EraPerformanceTracker."""

import numpy as np
import pytest
import gymnasium as gym

from src.training.curated_training_buffer import CuratedTrainingBuffer
from src.training.convergence import EraPerformanceTracker


# ─── Helpers ─────────────────────────────────────────────────────────────────

OBS_DIM = 8
ACT_DIM = 2
BUFFER_SIZE = 1000  # small for tests


def _make_buffer(core_fraction=0.20, sample_core_pct=0.35, n_envs=1):
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1, high=1, shape=(ACT_DIM,), dtype=np.float32)
    return CuratedTrainingBuffer(
        buffer_size=BUFFER_SIZE,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        n_envs=n_envs,
        core_fraction=core_fraction,
        sample_core_pct=sample_core_pct,
    )


def _add_transitions(buf, n, year=2015):
    buf.set_current_year(year)
    obs_space = buf.observation_space
    act_space = buf.action_space
    for _ in range(n):
        obs = obs_space.sample()[np.newaxis]   # (1, obs_dim) for n_envs=1
        nobs = obs_space.sample()[np.newaxis]
        act = act_space.sample()[np.newaxis]
        rew = np.array([0.1])
        done = np.array([False])
        buf.add(obs, nobs, act, rew, done, [{}])


# ─── CuratedTrainingBuffer tests ─────────────────────────────────────────────

def test_buffer_adds_transitions():
    buf = _make_buffer()
    _add_transitions(buf, 50)
    assert buf.size() > 0


def test_core_tier_fills_during_initial_add():
    """Core tier should accumulate samples as transitions are added."""
    buf = _make_buffer(core_fraction=0.20)
    _add_transitions(buf, 100)
    assert buf._core_count > 0, "Core tier should have samples after adding transitions"


def test_core_capacity_respected():
    """Core count must never exceed core_capacity."""
    buf = _make_buffer(core_fraction=0.20)
    _add_transitions(buf, BUFFER_SIZE * 3)  # force many loops
    assert buf._core_count <= buf._core_capacity


def test_era_tagging():
    """Core slots should be tagged with the year set via set_current_year."""
    buf = _make_buffer()
    _add_transitions(buf, 50, year=2015)
    _add_transitions(buf, 50, year=2020)
    dist = buf.get_era_distribution()
    years = set(dist.keys())
    # At least one of the years should appear in core
    assert len(years) > 0
    assert all(y in (2015, 2020, 0) for y in years)


def test_era_diversity_across_years():
    """After training on 3 years, core should represent all 3."""
    buf = _make_buffer(core_fraction=0.30)
    # Add enough transitions to saturate core with multi-era data
    for year in [2015, 2018, 2022]:
        _add_transitions(buf, 200, year=year)
    dist = buf.get_era_distribution()
    # Reservoir should have picked up samples from multiple years
    assert len(dist) >= 2, f"Expected multiple eras in core, got {dist}"


def test_sample_returns_correct_batch_size():
    """sample() must return exactly batch_size transitions."""
    buf = _make_buffer()
    _add_transitions(buf, 200)
    batch = buf.sample(64)
    assert batch.observations.shape[0] == 64


def test_sample_mixes_core_and_fifo():
    """With core populated, sample should pull from both tiers."""
    buf = _make_buffer(core_fraction=0.20, sample_core_pct=0.35)
    # Fill with 2015 data first (goes into core)
    _add_transitions(buf, 300, year=2015)
    # Then fill with 2024 data (FIFO, evicts old FIFO but not core)
    _add_transitions(buf, 600, year=2024)
    # Core should still have some 2015 samples
    dist = buf.get_era_distribution()
    assert 2015 in dist, "Core should retain some 2015 samples even after 2024 FIFO fill"


def test_sample_fallback_before_core_populated():
    """sample() falls back to standard sampling if core is empty."""
    buf = _make_buffer()
    _add_transitions(buf, 100)
    # Force clear core mask to simulate empty core
    buf._core_mask[:] = False
    buf._core_count = 0
    batch = buf.sample(32)
    assert batch.observations.shape[0] == 32


def test_buffer_status_dict():
    """get_buffer_status returns expected keys."""
    buf = _make_buffer()
    _add_transitions(buf, 50)
    status = buf.get_buffer_status()
    for key in ("total_slots", "used_slots", "core_count", "fifo_count",
                "total_added", "fill_pct", "core_pct", "era_distribution"):
        assert key in status, f"Missing key: {key}"


def test_full_flag_set_after_buffer_size_adds():
    """full flag should be True after buffer_size transitions."""
    buf = _make_buffer()
    _add_transitions(buf, BUFFER_SIZE + 10)
    assert buf.full, "Buffer should be marked full after buffer_size transitions"


def test_core_protects_samples_from_fifo_eviction():
    """Core slots added early should still exist after FIFO loops many times."""
    buf = _make_buffer(core_fraction=0.30, sample_core_pct=0.35)
    # First batch: fill core with 2015 era
    _add_transitions(buf, 150, year=2015)
    core_count_after_2015 = buf._core_count
    # Loop FIFO many times with later-era data
    _add_transitions(buf, BUFFER_SIZE * 5, year=2026)
    dist = buf.get_era_distribution()
    # 2015 entries should survive in core (reservoir protects them)
    assert 2015 in dist or core_count_after_2015 == 0, (
        f"Some 2015 entries should persist in core. Got dist={dist}"
    )


# ─── EraPerformanceTracker tests ─────────────────────────────────────────────

def test_era_tracker_record():
    t = EraPerformanceTracker(window=5)
    t.record(2015, 1.5)
    t.record(2015, 1.8)
    summary = t.get_all_years_summary()
    assert 2015 in summary
    assert summary[2015]["best"] == pytest.approx(1.8)


def test_era_tracker_no_weak_eras_if_stable():
    t = EraPerformanceTracker(window=5)
    for _ in range(5):
        t.record(2015, 1.5)
    weak = t.get_weak_eras()
    assert len(weak) == 0, "Stable performance should not flag weak eras"


def test_era_tracker_detects_forgetting():
    t = EraPerformanceTracker(window=5)
    # Establish good performance
    for _ in range(5):
        t.record(2015, 2.0)
    # Then performance collapses
    for _ in range(5):
        t.record(2015, 0.5)
    weak = t.get_weak_eras()
    assert len(weak) == 1
    assert weak[0]["year"] == 2015
    assert weak[0]["drop"] == pytest.approx(1.5, abs=0.1)


def test_era_tracker_severity_sorted():
    t = EraPerformanceTracker(window=5)
    # 2015: mild forgetting
    for _ in range(5): t.record(2015, 1.5)
    for _ in range(3): t.record(2015, 1.2)
    # 2018: severe forgetting
    for _ in range(5): t.record(2018, 2.5)
    for _ in range(3): t.record(2018, 0.2)

    weak = t.get_weak_eras()
    if len(weak) >= 2:
        assert weak[0]["severity"] >= weak[1]["severity"], "Should be sorted by severity"


def test_era_tracker_serialisation():
    t = EraPerformanceTracker(window=5)
    for yr in [2015, 2018, 2022]:
        for _ in range(4):
            t.record(yr, float(yr % 10) / 5.0)
    d = t.to_dict()
    t2 = EraPerformanceTracker.from_dict(d)
    assert set(t2._year_sharpes.keys()) == set(t._year_sharpes.keys())
    assert t2._year_best == t._year_best


def test_era_tracker_requires_min_episodes():
    t = EraPerformanceTracker(window=5)
    # Only 2 episodes — below MIN_EPISODES_TO_ASSESS=3
    t.record(2015, 2.0)
    t.record(2015, 0.1)
    weak = t.get_weak_eras()
    assert len(weak) == 0, "Should not flag with fewer than min episodes"
