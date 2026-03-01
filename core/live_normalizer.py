"""LiveNormalizer -- Per-feature rolling z-score normalization for live inference.

Maintains per-feature rolling buffers for z-score normalization.
Matches training's ExpandingWindowNormalizer exactly:
    - Window: 200 bars
    - Min periods: 50
    - Clip: [-5, +5]

Only the 38 MARKET features (Groups A-E + Upgrade 1 + Upgrade 4) are
z-score normalised.  All other features (Group F, G, H, Upgrades 2/3/5)
pass through unchanged.

Usage:
    from config.live_config import LiveConfig
    from core.live_normalizer import LiveNormalizer

    cfg = LiveConfig()
    normalizer = LiveNormalizer(cfg.market_feature_names, cfg.norm_window, cfg.norm_clip)
    normalizer.warmup(historical_features_df)

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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum number of observations before z-score is meaningful
_MIN_PERIODS = 50


class LiveNormalizer:
    """Maintains per-feature rolling buffers for z-score normalization.

    Matches training's ExpandingWindowNormalizer exactly:
        - Window: 200 bars (configurable)
        - Min periods: 50
        - Clip: [-5, +5] (configurable)

    Uses ``collections.deque(maxlen=window)`` per feature so only
    the most recent ``window`` observations contribute to the mean
    and standard deviation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        market_features: List[str],
        window: int = 200,
        clip: float = 5.0,
    ) -> None:
        """Create per-feature rolling buffers.

        Args:
            market_features: Names of the features that receive z-score
                normalisation (the 38 market features).
            window: Rolling window size (default 200, matching training).
            clip: Symmetric clip bound (default 5.0, matching training).
        """
        self.market_features: List[str] = list(market_features)
        self._market_set: set = set(market_features)
        self.window: int = window
        self.clip: float = clip
        self.min_periods: int = _MIN_PERIODS

        # Per-feature rolling buffers
        self._buffers: Dict[str, deque] = {
            name: deque(maxlen=window) for name in market_features
        }

        # Stats cache (recomputed only when buffers change)
        self._stats_dirty: bool = True
        self._cached_stats: Dict[str, Dict[str, float]] = {}

        log.info(
            "LiveNormalizer initialised: %d market features, window=%d, clip=+/-%.1f",
            len(market_features),
            window,
            clip,
        )

    # ------------------------------------------------------------------
    # Single-feature normalisation
    # ------------------------------------------------------------------

    def normalize(self, feature_name: str, raw_value: float) -> float:
        """Append *raw_value* to the feature's rolling buffer and return
        the z-score normalised value.

        If fewer than ``min_periods`` (50) samples are in the buffer,
        returns 0.0 (uninformative prior).

        Args:
            feature_name: One of the 38 market feature names.
            raw_value: Raw (un-normalised) feature value.

        Returns:
            Z-score clipped to [-clip, +clip], or 0.0 if insufficient data.
        """
        buf = self._buffers.get(feature_name)
        if buf is None:
            # Not a market feature -- should not happen if caller uses
            # normalize_batch, but handle gracefully.
            return raw_value

        # Sanitise input
        if np.isnan(raw_value) or np.isinf(raw_value):
            raw_value = 0.0

        buf.append(raw_value)
        self._stats_dirty = True

        if len(buf) < self.min_periods:
            return 0.0

        arr = np.array(buf, dtype=np.float64)
        mean = arr.mean()
        std = arr.std(ddof=0)  # population std, matching pd.rolling().std() default
        z = (raw_value - mean) / (std + 1e-8)
        return float(np.clip(z, -self.clip, self.clip))

    # ------------------------------------------------------------------
    # Batch normalisation
    # ------------------------------------------------------------------

    def normalize_batch(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalise all features in a dict.

        Market features are z-score normalised via their rolling buffers.
        All other features pass through unchanged.

        Args:
            features: Dict mapping feature_name -> raw float value.

        Returns:
            New dict with the same keys, market features normalised.
        """
        result: Dict[str, float] = {}
        for name, value in features.items():
            if name in self._market_set:
                result[name] = self.normalize(name, value)
            else:
                result[name] = value
        return result

    # ------------------------------------------------------------------
    # Bulk warmup
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> None:
        """Bulk-load historical data to initialise buffers.

        Iterates through the DataFrame rows chronologically and feeds
        each market feature value into its rolling buffer.  After calling
        this method, the normaliser is ready for live inference.

        Args:
            features_df: DataFrame with columns matching market feature
                names.  Should contain at least ``window`` rows for full
                initialisation.
        """
        available = [c for c in self.market_features if c in features_df.columns]
        n_rows = len(features_df)

        if n_rows == 0:
            log.warning("LiveNormalizer.fit: empty DataFrame -- no warmup performed")
            return

        # For efficiency, feed only the last `window` rows (older rows
        # would fall out of the deque anyway).
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
        """Export buffer state for persistence.

        Returns:
            Dict with ``buffers`` (feature_name -> list of floats),
            ``window``, ``clip``, and ``timestamp``.
        """
        return {
            "buffers": {name: list(buf) for name, buf in self._buffers.items()},
            "window": self.window,
            "clip": self.clip,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore buffer state from a previously exported dict.

        Args:
            state: Dict produced by :meth:`get_state`.
        """
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
        """Save normaliser state to a JSON file.

        Persists all rolling buffers plus a timestamp so that the state
        can be validated for freshness on reload.

        Args:
            path: File path for the JSON output.
        """
        state = self.get_state()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

        log.info("LiveNormalizer state saved to %s", path)

    def load_state(self, path: str) -> bool:
        """Restore normaliser state from a JSON file.

        Only restores if the saved state is less than 1 hour old.
        This prevents using stale statistics after a long downtime
        (buffers would be out-of-date and produce bad z-scores).

        Args:
            path: File path of the JSON state file.

        Returns:
            True if the state was loaded successfully, False if the
            file does not exist, is too old, or is corrupt.
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
        """Return per-feature mean, std, and count for health monitoring.

        Cached internally -- recomputed at most once per ``normalize()``
        call (i.e. once per bar), not every dashboard tick.

        Returns:
            Dict mapping feature_name -> {"mean", "std", "count"}.
        """
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
