"""LiveNormalizer -- Per-feature z-score normalization for live inference.

Supports two modes:
    "adaptive"  -- Rolling 200-bar deque buffers (original).
    "frozen"    -- Static mean/std from training baseline (recommended).

In frozen mode the model sees EXACTLY the same distribution it was trained
on.  This eliminates the entire class of conviction-collapse failures
caused by violent market moves poisoning rolling statistics.

In adaptive mode, additional safeguards are active:
    * Outlier clamping: values beyond N sigma are clamped before appending.
    * Baseline std cap: clamp uses min(current_std, baseline_std * 2) so
      inflated std doesn't weaken protection.
    * Auto-reset detection: if z-score distribution deviates too far
      (mean |z| > 3.5 or >20% features above 4 sigma), the normalizer
      triggers an automatic reset and re-warm.

Only the 38 MARKET features (Groups A-E + Upgrade 1 + Upgrade 4) are
z-score normalised.  All other features pass through unchanged.

Usage:
    from config.live_config import LiveConfig
    from core.live_normalizer import LiveNormalizer

    cfg = LiveConfig()
    normalizer = LiveNormalizer(cfg.market_feature_names, cfg.norm_window, cfg.norm_clip)

    # For frozen mode, load baseline from model package:
    normalizer.set_baseline(feature_baseline_dict)

    # Per-bar:
    normalized = normalizer.normalize_batch(raw_features_dict)
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum number of observations before z-score is meaningful
_MIN_PERIODS = 50


class LiveNormalizer:
    """Per-feature z-score normalization for live inference.

    Supports two modes controlled by ``mode``:

    **adaptive** (original):
        Rolling 200-bar deque buffers compute mean/std per feature.
        Protected by outlier clamping and auto-reset detection.

    **frozen** (recommended for deployment):
        Uses static mean/std from the training feature baseline.
        Immune to market crashes.  The model sees exactly the same
        distribution it was trained on.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        market_features: List[str],
        window: int = 200,
        clip: float = 5.0,
        outlier_sigma: float = 6.0,
        mode: str = "frozen",
    ) -> None:
        self.market_features: List[str] = list(market_features)
        self._market_set: set = set(market_features)
        self.window: int = window
        self.clip: float = clip
        self.min_periods: int = _MIN_PERIODS
        self._outlier_sigma: float = outlier_sigma
        self._mode: str = mode  # "adaptive" or "frozen"

        # Per-feature rolling buffers (used in adaptive mode)
        self._buffers: Dict[str, deque] = {
            name: deque(maxlen=window) for name in market_features
        }

        # Training baseline (used in frozen mode, also for clamp cap)
        # {feature_name: {"mean": float, "std": float}}
        self._baseline: Dict[str, Dict[str, float]] = {}
        self._has_baseline: bool = False

        # Stats cache (recomputed only when buffers change)
        self._stats_dirty: bool = True
        self._cached_stats: Dict[str, Dict[str, float]] = {}

        # Outlier clamping stats (for diagnostics)
        self._outlier_clamp_count: int = 0

        # Auto-reset detection state
        self._last_z_scores: Dict[str, float] = {}
        self._bars_since_reset: int = 0
        self._auto_reset_count: int = 0

        log.info(
            "LiveNormalizer initialised: mode=%s, %d market features, "
            "window=%d, clip=+/-%.1f, outlier_sigma=%.1f",
            mode,
            len(market_features),
            window,
            clip,
            outlier_sigma,
        )

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(self, baseline: Dict[str, Dict[str, float]]) -> None:
        """Load the training feature baseline for frozen mode.

        Args:
            baseline: Dict from feature_baseline.json in model package.
                Keys are feature names, values have "mean" and "std".
        """
        loaded = 0
        for name in self.market_features:
            if name in baseline:
                entry = baseline[name]
                mean = entry.get("mean", 0.0)
                std = entry.get("std", 1.0)
                if std < 1e-8:
                    std = 1.0  # prevent division by zero
                self._baseline[name] = {"mean": mean, "std": std}
                loaded += 1
            else:
                # Feature not in baseline -- use neutral defaults
                self._baseline[name] = {"mean": 0.0, "std": 1.0}

        self._has_baseline = loaded > 0
        log.info(
            "Feature baseline loaded: %d/%d market features matched",
            loaded,
            len(self.market_features),
        )

    @property
    def mode(self) -> str:
        """Current normalization mode: 'adaptive' or 'frozen'."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("adaptive", "frozen"):
            raise ValueError(f"Invalid mode: {value!r} (expected 'adaptive' or 'frozen')")
        if value == "frozen" and not self._has_baseline:
            log.warning(
                "Switching to frozen mode without a baseline loaded. "
                "Z-scores will use neutral defaults (mean=0, std=1) "
                "until set_baseline() is called."
            )
        old = self._mode
        self._mode = value
        if old != value:
            log.info("Normalization mode changed: %s -> %s", old, value)

    # ------------------------------------------------------------------
    # Single-feature normalisation
    # ------------------------------------------------------------------

    def normalize(self, feature_name: str, raw_value: float) -> float:
        """Normalise a single feature value.

        In **frozen** mode: uses baseline mean/std directly.
        In **adaptive** mode: appends to rolling buffer and computes z-score,
        with outlier clamping (using baseline std cap if available).

        Args:
            feature_name: One of the 38 market feature names.
            raw_value: Raw (un-normalised) feature value.

        Returns:
            Z-score clipped to [-clip, +clip], or 0.0 if insufficient data.
        """
        # Sanitise input
        if np.isnan(raw_value) or np.isinf(raw_value):
            raw_value = 0.0

        # ---- Frozen mode: deterministic, no buffer updates ----
        if self._mode == "frozen":
            bl = self._baseline.get(feature_name)
            if bl is None:
                return raw_value
            z = (raw_value - bl["mean"]) / (bl["std"] + 1e-8)
            z_clipped = float(np.clip(z, -self.clip, self.clip))
            self._last_z_scores[feature_name] = z_clipped
            return z_clipped

        # ---- Adaptive mode: rolling buffer with outlier protection ----
        buf = self._buffers.get(feature_name)
        if buf is None:
            return raw_value

        # Outlier clamping with baseline std cap
        if self._outlier_sigma > 0 and len(buf) >= self.min_periods:
            arr = np.array(buf, dtype=np.float64)
            mean = arr.mean()
            std = arr.std(ddof=0)

            # Use min(current_std, baseline_std * 2) for stronger protection
            # when current std is inflated by crash data
            bl = self._baseline.get(feature_name)
            if bl is not None and bl["std"] > 1e-8:
                clamp_std = min(std, bl["std"] * 2.0)
            else:
                clamp_std = std

            if clamp_std > 1e-8:
                deviation = abs(raw_value - mean) / clamp_std
                if deviation > self._outlier_sigma:
                    clamped = mean + np.sign(raw_value - mean) * self._outlier_sigma * clamp_std
                    self._outlier_clamp_count += 1
                    if self._outlier_clamp_count <= 20 or self._outlier_clamp_count % 100 == 0:
                        log.warning(
                            "Outlier clamped [%s]: raw=%.4f -> %.4f "
                            "(%.1fσ, mean=%.4f clamp_std=%.4f) [#%d]",
                            feature_name, raw_value, clamped,
                            deviation, mean, clamp_std,
                            self._outlier_clamp_count,
                        )
                    raw_value = float(clamped)

        buf.append(raw_value)
        self._stats_dirty = True

        if len(buf) < self.min_periods:
            return 0.0

        arr = np.array(buf, dtype=np.float64)
        mean = arr.mean()
        std = arr.std(ddof=0)
        z = (raw_value - mean) / (std + 1e-8)
        z_clipped = float(np.clip(z, -self.clip, self.clip))
        self._last_z_scores[feature_name] = z_clipped
        return z_clipped

    # ------------------------------------------------------------------
    # Batch normalisation
    # ------------------------------------------------------------------

    def normalize_batch(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalise all features in a dict.

        Market features are z-score normalised (frozen or adaptive).
        All other features pass through unchanged.
        """
        result: Dict[str, float] = {}
        for name, value in features.items():
            if name in self._market_set:
                result[name] = self.normalize(name, value)
            else:
                result[name] = value

        self._bars_since_reset += 1
        return result

    # ------------------------------------------------------------------
    # Auto-reset detection (adaptive mode only)
    # ------------------------------------------------------------------

    def check_distribution_health(
        self,
        z_abs_mean_threshold: float = 3.5,
        pct_over_4sigma: float = 0.20,
        cooldown_bars: int = 50,
    ) -> Tuple[bool, str]:
        """Check if the z-score distribution has deviated dangerously.

        Should be called after each normalize_batch() in adaptive mode.
        In frozen mode, always returns (True, "ok").

        Args:
            z_abs_mean_threshold: Trigger if mean(|z|) exceeds this.
            pct_over_4sigma: Trigger if this fraction of features > 4 sigma.
            cooldown_bars: Min bars between auto-resets.

        Returns:
            (healthy, reason): True if OK, False if auto-reset needed.
        """
        if self._mode == "frozen":
            return True, "ok (frozen mode)"

        if not self._last_z_scores:
            return True, "ok (no z-scores yet)"

        if self._bars_since_reset < cooldown_bars:
            return True, f"ok (cooldown: {self._bars_since_reset}/{cooldown_bars})"

        z_values = list(self._last_z_scores.values())
        if not z_values:
            return True, "ok"

        z_abs = [abs(z) for z in z_values]
        z_abs_mean = sum(z_abs) / len(z_abs)
        n_over_4 = sum(1 for z in z_abs if z > 4.0)
        pct_over = n_over_4 / len(z_abs)

        # Check triggers
        if z_abs_mean > z_abs_mean_threshold:
            reason = (
                f"z_abs_mean={z_abs_mean:.2f} > {z_abs_mean_threshold} "
                f"(distribution shift detected)"
            )
            log.warning("AUTO-RESET TRIGGER: %s", reason)
            return False, reason

        if pct_over > pct_over_4sigma:
            reason = (
                f"{n_over_4}/{len(z_abs)} features ({pct_over:.0%}) > 4σ "
                f"(threshold {pct_over_4sigma:.0%})"
            )
            log.warning("AUTO-RESET TRIGGER: %s", reason)
            return False, reason

        return True, "ok"

    def on_auto_reset(self) -> None:
        """Called after an auto-reset to update internal counters."""
        self._bars_since_reset = 0
        self._auto_reset_count += 1
        log.warning(
            "Normalizer auto-reset #%d triggered", self._auto_reset_count,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all rolling buffers, forcing a fresh warmup."""
        for name in self.market_features:
            self._buffers[name] = deque(maxlen=self.window)
        self._stats_dirty = True
        self._cached_stats.clear()
        self._outlier_clamp_count = 0
        self._last_z_scores.clear()
        self._bars_since_reset = 0
        log.warning("LiveNormalizer RESET: all %d buffers cleared", len(self.market_features))

    # ------------------------------------------------------------------
    # Bulk warmup
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> None:
        """Bulk-load historical data to initialise buffers.

        Only relevant in adaptive mode, but harmless in frozen mode
        (buffers are populated but not used for z-scoring).
        """
        available = [c for c in self.market_features if c in features_df.columns]
        n_rows = len(features_df)

        if n_rows == 0:
            log.warning("LiveNormalizer.fit: empty DataFrame -- no warmup performed")
            return

        start = max(0, n_rows - self.window)
        subset = features_df.iloc[start:]

        for col in available:
            buf = self._buffers[col]
            vals = subset[col].values
            for v in vals:
                if np.isnan(v) or np.isinf(v):
                    buf.append(0.0)
                else:
                    buf.append(float(v))

        self._stats_dirty = True
        filled = sum(1 for b in self._buffers.values() if len(b) >= self.min_periods)
        log.info(
            "LiveNormalizer.fit: loaded %d rows, %d/%d features above min_periods (%d)",
            len(subset),
            filled,
            len(self.market_features),
            self.min_periods,
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Export buffer state for persistence."""
        return {
            "buffers": {name: list(buf) for name, buf in self._buffers.items()},
            "window": self.window,
            "clip": self.clip,
            "mode": self._mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore buffer state from a previously exported dict."""
        buffers_data = state.get("buffers", {})
        restored = 0
        for name in self.market_features:
            if name in buffers_data:
                vals = buffers_data[name]
                buf = deque(maxlen=self.window)
                for v in vals:
                    buf.append(float(v))
                self._buffers[name] = buf
                restored += 1

        self._stats_dirty = True
        log.info(
            "LiveNormalizer.set_state: restored %d/%d feature buffers",
            restored,
            len(self.market_features),
        )

    def save_state(self, path: str) -> None:
        """Save normaliser state to a JSON file."""
        state = self.get_state()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

        log.info("LiveNormalizer state saved to %s", path)

    def load_state(self, path: str) -> bool:
        """Restore normaliser state from a JSON file.

        Only restores if the saved state is less than 1 hour old.
        """
        p = Path(path)
        if not p.exists():
            log.info("LiveNormalizer.load_state: file not found: %s", path)
            return False

        try:
            with open(p, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("LiveNormalizer.load_state: failed to read %s: %s", path, exc)
            return False

        # Age check -- reject if older than 1 hour
        ts_str = state.get("timestamp")
        if ts_str:
            try:
                saved_at = datetime.fromisoformat(ts_str)
                if saved_at.tzinfo is None:
                    saved_at = saved_at.replace(tzinfo=timezone.utc)
                age_seconds = (datetime.now(timezone.utc) - saved_at).total_seconds()
                if age_seconds > 3600:
                    log.info(
                        "LiveNormalizer.load_state: state is %.0f seconds old "
                        "(> 3600s) -- discarding",
                        age_seconds,
                    )
                    return False
            except (ValueError, TypeError) as exc:
                log.warning(
                    "LiveNormalizer.load_state: could not parse timestamp: %s", exc
                )
                return False

        self.set_state(state)
        return True

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------

    def get_buffer_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-feature mean, std, and count for health monitoring."""
        if self._stats_dirty:
            stats: Dict[str, Dict[str, float]] = {}
            for name, buf in self._buffers.items():
                count = len(buf)
                if count > 0:
                    arr = np.array(buf, dtype=np.float64)
                    stats[name] = {
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                        "count": float(count),
                    }
                else:
                    stats[name] = {"mean": 0.0, "std": 0.0, "count": 0.0}
            self._cached_stats = stats
            self._stats_dirty = False
        return self._cached_stats

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return full diagnostic state for dashboard display."""
        return {
            "mode": self._mode,
            "has_baseline": self._has_baseline,
            "baseline_features": len(self._baseline),
            "outlier_clamp_count": self._outlier_clamp_count,
            "auto_reset_count": self._auto_reset_count,
            "bars_since_reset": self._bars_since_reset,
            "last_z_abs_mean": (
                sum(abs(z) for z in self._last_z_scores.values()) / len(self._last_z_scores)
                if self._last_z_scores else 0.0
            ),
        }
