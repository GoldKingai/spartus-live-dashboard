"""Convergence detection for training loop.

8 states:
    WARMING_UP   → not enough data yet
    IMPROVING    → validation Sharpe trending up
    CONVERGED    → Sharpe stable and above threshold
    OVERFITTING  → train improving but val degrading (classic overfitting)
    VAL_DECLINING → val significantly below best (regime-shift / distribution shift)
    COLLAPSED    → action std below threshold for too long
    PLATEAU      → no improvement for extended period
    STABLE       → converged and maintaining performance
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from src.config import TrainingConfig


class ConvergenceDetector:
    """Tracks training progress and detects convergence states."""

    STATES = [
        "WARMING_UP", "IMPROVING", "CONVERGED",
        "OVERFITTING", "VAL_DECLINING", "COLLAPSED", "PLATEAU", "STABLE",
    ]

    def __init__(self, config: TrainingConfig = None):
        cfg = config or TrainingConfig()
        self.window = cfg.convergence_window           # 50
        self.sharpe_threshold = cfg.convergence_sharpe_threshold  # 0.001
        self.weeks_since_best_limit = cfg.convergence_weeks_since_best  # 5 val points = 50 training weeks (FIX-B3)
        self.collapsed_std = cfg.collapsed_action_std   # 0.05
        self.collapsed_duration = cfg.collapsed_duration  # 20

        # History
        self.val_sharpes: List[float] = []
        self.val_returns: List[float] = []  # FIX-VAL: track total return alongside Sharpe
        self.train_sharpes: List[float] = []
        self.train_sharpes_at_val: List[float] = []  # FIX-15: train sharpe sampled at val points
        self.action_stds: List[float] = []
        self.best_val_sharpe = -np.inf
        self.best_val_score = -np.inf  # FIX-VAL: composite score for checkpoint selection
        self.weeks_since_best = 0

        # Consecutive duration trackers for early stopping
        self.overfitting_weeks = 0
        self.collapsed_weeks = 0

        # State
        self.state = "WARMING_UP"

    @staticmethod
    def compute_val_score(val_sharpe: float, val_return: float) -> float:
        """Compute composite validation score for checkpoint selection.

        FIX-VAL: Pure Sharpe penalizes profitable variability — a conservative
        model with low returns can beat a profitable model that has variable
        per-week returns.  The composite score uses the geometric mean of
        Sharpe and (1+return), balancing consistency with profitability equally.

        Formula: sqrt(sharpe * (1 + max(val_return, 0)))
          - Geometric mean: neither factor dominates.
          - A model with 64% more profit but 20% less Sharpe correctly wins.
          - Negative Sharpe → score 0 (never best).
        """
        s = max(val_sharpe, 0.0)
        r = max(val_return, 0.0)
        return np.sqrt(s * (1.0 + r))

    def update(
        self,
        week: int,
        val_sharpe: float = None,
        val_return: float = None,
        train_sharpe: float = None,
        action_std: float = None,
    ) -> str:
        """Update with new metrics and return current state.

        Args:
            week: Current training week.
            val_sharpe: Validation Sharpe ratio (may be None if no validation).
            val_return: Total validation return (may be None if no validation).
            train_sharpe: Training Sharpe ratio.
            action_std: Mean action standard deviation.

        Returns:
            Current convergence state string.
        """
        if train_sharpe is not None:
            # Clamp extreme values to prevent Inf/NaN contamination
            train_sharpe = max(-10.0, min(10.0, train_sharpe))
            self.train_sharpes.append(train_sharpe)
        if action_std is not None:
            self.action_stds.append(action_std)

        if val_sharpe is not None:
            self.val_sharpes.append(val_sharpe)
            self.val_returns.append(val_return if val_return is not None else 0.0)
            # FIX-15: Record current train sharpe at val time for matched comparison
            if self.train_sharpes:
                self.train_sharpes_at_val.append(self.train_sharpes[-1])

            # FIX-VAL: Use composite score (Sharpe * sqrt(1+return)) for best model
            val_score = self.compute_val_score(val_sharpe, val_return or 0.0)
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.weeks_since_best = 0
            else:
                self.weeks_since_best += 1

            # Still track best raw Sharpe for logging/display
            if val_sharpe > self.best_val_sharpe:
                self.best_val_sharpe = val_sharpe
        else:
            # Only increment on validation weeks — non-val weeks don't update
            # this counter because we can't assess improvement without validation.
            # Previously incremented every week, triggering PLATEAU 10x too fast.
            pass

        self.state = self._determine_state(week)

        # Track consecutive weeks in critical states
        # VAL_DECLINING treated same as OVERFITTING for defense-layer triggers (FIX-B2/B5)
        if self.state in ("OVERFITTING", "VAL_DECLINING"):
            self.overfitting_weeks += 1
        else:
            self.overfitting_weeks = 0

        if self.state == "COLLAPSED":
            self.collapsed_weeks += 1
        else:
            self.collapsed_weeks = 0

        return self.state

    def _determine_state(self, week: int) -> str:
        # Not enough data
        if len(self.val_sharpes) < 5:
            return "WARMING_UP"

        # Check action collapse
        if self._check_collapsed():
            return "COLLAPSED"

        # Check overfitting (train improving, val degrading — classic overfitting)
        if self._check_overfitting():
            return "OVERFITTING"

        # Check val regression (val significantly below best, regardless of train) (FIX-B2/B5)
        # Catches regime-shift where BOTH train AND val decline — missed by classic check
        if self._check_val_declining():
            return "VAL_DECLINING"

        # Check plateau (no improvement for too long)
        if self.weeks_since_best >= self.weeks_since_best_limit:
            return "PLATEAU"

        # Check convergence/stability
        recent_sharpes = self.val_sharpes[-self.window:]
        if len(recent_sharpes) >= 10:
            sharpe_std = np.std(recent_sharpes[-10:])
            sharpe_mean = np.mean(recent_sharpes[-10:])

            if sharpe_std < self.sharpe_threshold and sharpe_mean > 0:
                if self.weeks_since_best < 10:
                    return "STABLE"
                return "CONVERGED"

        # Default: improving
        return "IMPROVING"

    def _check_collapsed(self) -> bool:
        """Check if action std has been below threshold for too long."""
        if len(self.action_stds) < self.collapsed_duration:
            return False
        recent = self.action_stds[-self.collapsed_duration:]
        return all(s < self.collapsed_std for s in recent)

    def _check_overfitting(self) -> bool:
        """Check if train improves but val degrades (classic overfitting).

        FIX-15: Uses time-aligned train_sharpes_at_val (sampled at val points only).
        FIX-B7: Window reduced from 5→3 val points (was 50 training week lag, now 30).
        FIX-B8: Uses composite scores (Sharpe * sqrt(1+return)) instead of raw
        Sharpe.  Raw Sharpe falsely detected overfitting when the model was
        actually doubling its profit — it just had more variable per-week
        returns.  This caused a false-positive rollback that destroyed a
        strong model (W60-W89 incident).
        """
        if len(self.val_sharpes) < 6 or len(self.train_sharpes_at_val) < 6:
            return False
        if len(self.val_returns) < 6:
            return False

        # Use composite scores for val assessment (same as checkpoint selection)
        val_scores_recent = [
            self.compute_val_score(s, r)
            for s, r in zip(self.val_sharpes[-3:], self.val_returns[-3:])
        ]
        val_scores_prev = [
            self.compute_val_score(s, r)
            for s, r in zip(self.val_sharpes[-6:-3], self.val_returns[-6:-3])
        ]
        val_recent = np.mean(val_scores_recent)
        val_prev = np.mean(val_scores_prev)

        train_recent = np.mean(self.train_sharpes_at_val[-3:])
        train_prev = np.mean(self.train_sharpes_at_val[-6:-3])

        val_declining = val_recent < val_prev * 0.85  # 15% composite drop (not absolute 0.01)
        train_improving = train_recent > train_prev + 0.01

        return val_declining and train_improving

    def _check_val_declining(self) -> bool:
        """Detect significant validation regression (FIX-B2/B5/B8).

        FIX-VAL: Uses composite scores (Sharpe * sqrt(1+return)).
        FIX-B8: Compares recent 3 against previous 3 (trailing window),
        NOT against single best.  The old comparison (recent avg vs best peak)
        triggered immediately after any sharp improvement because
        avg(old, old, new_best) < new_best always.  This caused permanent
        VAL_DECLINING state after the W80 peak, leading to a false rollback.

        Triggers when: recent composite drops >30% below previous window,
        AND we have enough data (6+ val points).
        """
        if len(self.val_sharpes) < 6 or len(self.val_returns) < 6:
            return False
        if self.best_val_score <= 0.5:
            return False

        # Compare recent 3 composite scores vs previous 3 (trailing window)
        recent_scores = [
            self.compute_val_score(s, r)
            for s, r in zip(self.val_sharpes[-3:], self.val_returns[-3:])
        ]
        prev_scores = [
            self.compute_val_score(s, r)
            for s, r in zip(self.val_sharpes[-6:-3], self.val_returns[-6:-3])
        ]
        recent_avg = np.mean(recent_scores)
        prev_avg = np.mean(prev_scores)

        # 30% decline from previous window (not single best)
        return prev_avg > 1.0 and recent_avg < prev_avg * 0.70

    def get_summary(self) -> Dict:
        return {
            "state": self.state,
            "best_val_sharpe": self.best_val_sharpe,
            "best_val_score": self.best_val_score,
            "weeks_since_best": self.weeks_since_best,
            "overfitting_weeks": self.overfitting_weeks,
            "collapsed_weeks": self.collapsed_weeks,
            "val_sharpe_trend": (
                np.mean(self.val_sharpes[-5:]) - np.mean(self.val_sharpes[-10:-5])
                if len(self.val_sharpes) >= 10 else 0.0
            ),
            "n_val_points": len(self.val_sharpes),
            # History lists for resume — without these, convergence detection
            # resets to WARMING_UP for ~100 weeks after any resume
            "val_sharpes": list(self.val_sharpes),
            "val_returns": list(self.val_returns),
            "train_sharpes": list(self.train_sharpes),
            "train_sharpes_at_val": list(self.train_sharpes_at_val),
            "action_stds": list(self.action_stds),
        }

    def restore_from_summary(self, summary: Dict):
        """Restore full state from a saved summary (for resume)."""
        self.state = summary.get("state", "WARMING_UP")
        bvs = summary.get("best_val_sharpe", -np.inf)
        if isinstance(bvs, str):
            bvs = -np.inf
        self.best_val_sharpe = bvs
        self.weeks_since_best = summary.get("weeks_since_best", 0)
        self.overfitting_weeks = summary.get("overfitting_weeks", 0)
        self.collapsed_weeks = summary.get("collapsed_weeks", 0)
        self.val_sharpes = summary.get("val_sharpes", [])
        self.val_returns = summary.get("val_returns", [])
        self.train_sharpes = summary.get("train_sharpes", [])
        self.train_sharpes_at_val = summary.get("train_sharpes_at_val", [])
        self.action_stds = summary.get("action_stds", [])

        # FIX-VAL: Restore or recompute best_val_score
        bvsc = summary.get("best_val_score", None)
        if bvsc is not None and bvsc != -np.inf:
            self.best_val_score = bvsc
        elif self.val_sharpes and self.val_returns:
            # Recompute from history (upgrade path from old state files)
            scores = [
                self.compute_val_score(s, r)
                for s, r in zip(self.val_sharpes, self.val_returns)
            ]
            self.best_val_score = max(scores) if scores else -np.inf
        elif self.val_sharpes:
            # No val_returns saved yet — approximate using Sharpe alone
            self.best_val_score = max(self.val_sharpes)
        else:
            self.best_val_score = -np.inf


class LiveConvergenceDetector:
    """Convergence detection for live fine-tuning episodes.

    Unlike the training ConvergenceDetector (which tracks weeks of offline
    training), this tracks per-episode performance of the fine-tuning loop:
      - Each "episode" = one TradeEnv rollout on live bar data (~4 hours)
      - Metrics arrive infrequently (every few hours, not every step)
      - KL divergence tracks drift from the frozen baseline policy

    States:
        FT_WARMING_UP  → Fewer than 3 episodes completed
        FT_ADAPTING    → Sharpe improving across episodes
        FT_STABLE      → Sharpe stable and above baseline, KL controlled
        FT_PLATEAUED   → No Sharpe improvement for N episodes
        FT_DRIFTING    → KL divergence exceeded safe threshold
        FT_COLLAPSED   → Action std below collapse threshold
    """

    STATES = [
        "FT_WARMING_UP", "FT_ADAPTING", "FT_STABLE",
        "FT_PLATEAUED", "FT_DRIFTING", "FT_COLLAPSED",
    ]

    # Thresholds
    MIN_EPISODES_FOR_ASSESSMENT = 3
    PLATEAU_EPISODES = 5          # No improvement for this many episodes = plateau
    KL_DRIFT_THRESHOLD = 0.30     # matches finetune_kl_emergency_threshold in config
    COLLAPSED_STD = 0.30          # matches finetune_action_std_min in config
    IMPROVING_SLOPE = 0.005       # min per-episode Sharpe slope to count as adapting

    def __init__(self, config=None):
        cfg = config
        if cfg is not None:
            self.PLATEAU_EPISODES = getattr(cfg, "finetune_auto_rollback_failures", 5)
            self.KL_DRIFT_THRESHOLD = getattr(cfg, "finetune_kl_emergency_threshold", 0.30)
            self.COLLAPSED_STD = getattr(cfg, "finetune_action_std_min", 0.30)

        self.episode_sharpes: List[float] = []
        self.kl_divergences: List[float] = []
        self.action_stds: List[float] = []

        self.best_sharpe: float = -np.inf
        self.episodes_since_best: int = 0
        self.state: str = "FT_WARMING_UP"

    def update(
        self,
        episode_sharpe: float,
        kl_divergence: float = None,
        action_std: float = None,
    ) -> str:
        """Record episode results and return current convergence state.

        Args:
            episode_sharpe: Sharpe ratio from the completed fine-tune episode.
            kl_divergence: KL(π_new || π_frozen) measured after the episode.
            action_std: Mean action std from the episode's replay buffer sample.

        Returns:
            Current state string (one of STATES).
        """
        # Clamp to prevent contamination
        episode_sharpe = max(-10.0, min(10.0, episode_sharpe))
        self.episode_sharpes.append(episode_sharpe)

        if kl_divergence is not None:
            self.kl_divergences.append(float(kl_divergence))
        if action_std is not None:
            self.action_stds.append(float(action_std))

        if episode_sharpe > self.best_sharpe:
            self.best_sharpe = episode_sharpe
            self.episodes_since_best = 0
        else:
            self.episodes_since_best += 1

        self.state = self._determine_state()
        return self.state

    def _determine_state(self) -> str:
        n = len(self.episode_sharpes)

        if n < self.MIN_EPISODES_FOR_ASSESSMENT:
            return "FT_WARMING_UP"

        # Collapse check (highest priority — policy has broken)
        if self.action_stds:
            recent_stds = self.action_stds[-3:]
            if all(s < self.COLLAPSED_STD for s in recent_stds):
                return "FT_COLLAPSED"

        # KL drift check (second priority — too far from base)
        if self.kl_divergences:
            recent_kl = self.kl_divergences[-1]
            if recent_kl >= self.KL_DRIFT_THRESHOLD:
                return "FT_DRIFTING"

        # Plateau check
        if self.episodes_since_best >= self.PLATEAU_EPISODES:
            return "FT_PLATEAUED"

        # Trend check
        recent = self.episode_sharpes[-min(5, n):]
        if len(recent) >= 2:
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
            if slope > self.IMPROVING_SLOPE:
                return "FT_ADAPTING"

        # Stable: not improving strongly, but not plateaued/drifted either
        return "FT_STABLE"

    @property
    def n_episodes(self) -> int:
        return len(self.episode_sharpes)

    @property
    def last_sharpe(self) -> float:
        return self.episode_sharpes[-1] if self.episode_sharpes else 0.0

    @property
    def last_kl(self) -> float:
        return self.kl_divergences[-1] if self.kl_divergences else 0.0

    @property
    def sharpe_trend_5(self) -> float:
        """Per-episode Sharpe slope over last 5 episodes."""
        if len(self.episode_sharpes) < 2:
            return 0.0
        recent = self.episode_sharpes[-5:]
        return float(np.polyfit(range(len(recent)), recent, 1)[0])

    def get_summary(self) -> Dict:
        return {
            "state": self.state,
            "n_episodes": self.n_episodes,
            "best_sharpe": self.best_sharpe,
            "last_sharpe": self.last_sharpe,
            "last_kl": self.last_kl,
            "episodes_since_best": self.episodes_since_best,
            "sharpe_trend_5": self.sharpe_trend_5,
            "episode_sharpes": list(self.episode_sharpes),
            "kl_divergences": list(self.kl_divergences),
            "action_stds": list(self.action_stds),
        }

    def reset(self):
        """Reset all history (called when starting a new fine-tune session)."""
        self.episode_sharpes.clear()
        self.kl_divergences.clear()
        self.action_stds.clear()
        self.best_sharpe = -np.inf
        self.episodes_since_best = 0
        self.state = "FT_WARMING_UP"


class EraPerformanceTracker:
    """Track per-year trading performance to detect catastrophic forgetting.

    Records episode Sharpe per training era (data year) and surfaces
    eras where the model has significantly degraded since its personal best.
    Feeds into CuratedTrainingBuffer to upweight underperforming eras.

    Usage
    -----
    Called by the trainer after each training episode::

        era_tracker.record(data_year=2016, sharpe=1.23)
        weak = era_tracker.get_weak_eras()   # [{year, recent_avg, best, severity}]
    """

    FORGETTING_THRESHOLD = 0.40   # Sharpe drop from best before flagging
    MIN_EPISODES_TO_ASSESS = 3    # Need at least this many episodes per year
    MIN_BEST_TO_FLAG = 0.25       # Only flag if the era ever achieved > this

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self._year_sharpes: Dict[int, List[float]] = {}
        self._year_best: Dict[int, float] = {}

    # ──────────────────────────────────────────────────────────────────────

    def record(self, year: int, sharpe: float) -> None:
        """Record one training episode result for a given data year."""
        year = int(year)
        if year not in self._year_sharpes:
            self._year_sharpes[year] = []
            self._year_best[year] = sharpe
        buf = self._year_sharpes[year]
        buf.append(float(sharpe))
        if len(buf) > self._window:
            buf.pop(0)
        if sharpe > self._year_best[year]:
            self._year_best[year] = sharpe

    def get_weak_eras(self) -> List[Dict]:
        """Return eras where recent performance has dropped significantly.

        Returns list of dicts sorted by severity (worst first)::

            [{"year": 2016, "recent_avg": 0.4, "best": 1.8, "severity": 0.78}, ...]
        """
        weak = []
        for year, sharpes in self._year_sharpes.items():
            if len(sharpes) < self.MIN_EPISODES_TO_ASSESS:
                continue
            best = self._year_best.get(year, 0.0)
            if best < self.MIN_BEST_TO_FLAG:
                continue
            recent_avg = float(np.mean(sharpes[-3:]))
            drop = best - recent_avg
            if drop >= self.FORGETTING_THRESHOLD:
                severity = drop / best  # 0-1 scale
                weak.append({
                    "year": year,
                    "recent_avg": round(recent_avg, 3),
                    "best": round(best, 3),
                    "drop": round(drop, 3),
                    "severity": round(severity, 3),
                })
        return sorted(weak, key=lambda x: x["severity"], reverse=True)

    def get_all_years_summary(self) -> Dict[int, Dict]:
        """Return full per-year stats for logging/monitoring."""
        result = {}
        for year, sharpes in self._year_sharpes.items():
            if not sharpes:
                continue
            recent = sharpes[-3:] if len(sharpes) >= 3 else sharpes
            result[year] = {
                "n_episodes": len(sharpes),
                "recent_avg": round(float(np.mean(recent)), 3),
                "best": round(self._year_best.get(year, 0.0), 3),
                "latest": round(sharpes[-1], 3),
            }
        return result

    def to_dict(self) -> Dict:
        """Serialise for training_state.json persistence."""
        return {
            "year_sharpes": {str(k): v for k, v in self._year_sharpes.items()},
            "year_best": {str(k): v for k, v in self._year_best.items()},
            "window": self._window,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EraPerformanceTracker":
        """Restore from serialised dict (JSON string-keys → int)."""
        tracker = cls(window=d.get("window", 10))
        tracker._year_sharpes = {
            int(k): list(v) for k, v in d.get("year_sharpes", {}).items()
        }
        tracker._year_best = {
            int(k): float(v) for k, v in d.get("year_best", {}).items()
        }
        return tracker
