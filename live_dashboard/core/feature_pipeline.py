"""LiveFeaturePipeline -- Compute all 68 features in real-time from MT5 data.

Tier 1 upgrade: +1 regime_label (Upgrade 6) → 68 features / 680 obs_dim.
Maintains rolling buffers for M5, H1, H4, D1 bars and correlated instruments.
Produces identical 680-dim observations as training.

Feature ordering MUST match training exactly (68 features):
    1-7:   Group A  (close_frac_diff, returns_1bar, returns_5bar, returns_20bar,
                      bar_range, close_position, body_ratio)
    8-11:  Group B  (atr_14_norm, atr_ratio, bb_width, bb_position)
    12-17: Group C  (rsi_14, macd_signal, adx_14, ema_cross, price_vs_ema200, stoch_k)
    18-19: Group D  (volume_ratio, obv_slope)
    20-25: Group E  (h1_trend_dir, h4_trend_dir, d1_trend_dir, h1_rsi,
                      mtf_alignment, htf_momentum)
    26-29: Group F  (hour_sin, hour_cos, day_of_week, session_quality)
    30-40: Upgrade 1 (eurusd_returns_20, eurusd_rsi_14, eurusd_trend,
                       xagusd_returns_20, xagusd_rsi_14,
                       usdjpy_returns_20, usdjpy_trend,
                       us500_returns_20, us500_rsi_14,
                       usoil_returns_20, gold_silver_ratio_z)
    41-46: Upgrade 2 (hours_to_next_high_impact, hours_to_next_nfp_fomc,
                       in_event_window, daily_event_density,
                       london_fix_proximity, comex_session_active)
    47-48: Upgrade 3 (spread_estimate_norm, volume_spike)
    49-50: Upgrade 4 (corr_gold_usd_100, corr_gold_spx_100)
    51-54: Upgrade 5 (asian_range_norm, asian_range_position,
                       session_momentum, london_ny_overlap)
    55:    Upgrade 6 (regime_label — ADX>25=+1, ATR_ratio>1.2=-1, else 0)
    56-63: Group G   account features (8)
    64-68: Group H   memory features (5)

MARKET features to z-score: Groups A-E (25) + Upgrade 1 (11) + Upgrade 4 (2) = 38.
EXEMPT features (pass-through): Group F (4) + G-H (13) + Upgrades 2 (6), 3 (2), 5 (4) + Upgrade 6 (1) = 30.

Usage:
    from config.live_config import LiveConfig
    from core.feature_pipeline import LiveFeaturePipeline
    from core.mt5_bridge import MT5Bridge

    cfg = LiveConfig()
    pipeline = LiveFeaturePipeline(cfg)
    pipeline.warmup(mt5_bridge)

    # On each new M5 bar:
    obs = pipeline.on_new_bar(mt5_bridge, account_state, memory_features)
    # obs is a 670-dim numpy array ready for inference
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Route through the bridge so native (Windows) and mt5linux (Linux/Wine)
# transports are used consistently. Direct `import MetaTrader5` would crash
# at module load on Linux even when the bridge is providing access.
from core.mt5_bridge import mt5  # type: ignore[no-redef]

from config.live_config import LiveConfig
from core.live_normalizer import LiveNormalizer, _MIN_PERIODS

# Feature modules
from features.technical import (
    compute_technical_features,
    get_atr_14,
)
from features.time_session import compute_time_session_features
from features.correlation import compute_correlation_features
from features.calendar import compute_calendar_features
from features.spread_liquidity import compute_spread_liquidity_features
from features.regime import compute_regime_features
from features.session_micro import compute_session_micro_features

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature ordering (67 features)
# ---------------------------------------------------------------------------

# The 54 precomputed features in EXACT training order
_PRECOMPUTED_FEATURE_ORDER: List[str] = [
    # Group A: Price & Returns (7)
    "close_frac_diff", "returns_1bar", "returns_5bar", "returns_20bar",
    "bar_range", "close_position", "body_ratio",
    # Group B: Volatility (4)
    "atr_14_norm", "atr_ratio", "bb_width", "bb_position",
    # Group C: Momentum & Trend (6)
    "rsi_14", "macd_signal", "adx_14", "ema_cross", "price_vs_ema200", "stoch_k",
    # Group D: Volume (2)
    "volume_ratio", "obv_slope",
    # Group E: Multi-Timeframe Context (6)
    "h1_trend_dir", "h4_trend_dir", "d1_trend_dir",
    "h1_rsi", "mtf_alignment", "htf_momentum",
    # Group F: Time & Session (4)
    "hour_sin", "hour_cos", "day_of_week", "session_quality",
    # Upgrade 1: Correlated Instruments (11)
    "eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend",
    "xagusd_returns_20", "xagusd_rsi_14",
    "usdjpy_returns_20", "usdjpy_trend",
    "us500_returns_20", "us500_rsi_14",
    "usoil_returns_20",
    "gold_silver_ratio_z",
    # Upgrade 2: Calendar & Events (6)
    "hours_to_next_high_impact", "hours_to_next_nfp_fomc",
    "in_event_window", "daily_event_density",
    "london_fix_proximity", "comex_session_active",
    # Upgrade 3: Spread & Liquidity (2)
    "spread_estimate_norm", "volume_spike",
    # Upgrade 4: Regime Detection (2)
    "corr_gold_usd_100", "corr_gold_spx_100",
    # Upgrade 5: Session Microstructure (4)
    "asian_range_norm", "asian_range_position",
    "session_momentum", "london_ny_overlap",
    # Upgrade 6: Regime Label (1) — ADX>25=trending(+1), ATR_ratio>1.2=volatile(-1), else 0
    "regime_label",
]

# Account features (Group G, 8 features) -- order matches training
_ACCOUNT_FEATURE_ORDER: List[str] = [
    "has_position", "position_side", "unrealized_pnl", "position_duration",
    "current_drawdown", "equity_ratio", "sl_distance_ratio", "profit_locked_pct",
]

# Memory features (Group H, 5 features) -- order matches training
_MEMORY_FEATURE_ORDER: List[str] = [
    "recent_win_rate", "similar_pattern_winrate",
    "trend_prediction_accuracy", "tp_hit_rate", "avg_sl_trail_profit",
]

# Full 68-feature order (Tier 1: +1 regime_label)
_FULL_FEATURE_ORDER: List[str] = (
    _PRECOMPUTED_FEATURE_ORDER + _ACCOUNT_FEATURE_ORDER + _MEMORY_FEATURE_ORDER
)

assert len(_PRECOMPUTED_FEATURE_ORDER) == 55, (
    f"Expected 55 precomputed features, got {len(_PRECOMPUTED_FEATURE_ORDER)}"
)
assert len(_FULL_FEATURE_ORDER) == 68, (
    f"Expected 68 total features, got {len(_FULL_FEATURE_ORDER)}"
)

# Correlated instruments -- canonical names
_CORRELATED_SYMBOLS: List[str] = ["EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"]

# MT5 timeframe constants (guard for missing import)
_TF_M5 = getattr(mt5, "TIMEFRAME_M5", 5)
_TF_H1 = getattr(mt5, "TIMEFRAME_H1", 16385)
_TF_H4 = getattr(mt5, "TIMEFRAME_H4", 16388)
_TF_D1 = getattr(mt5, "TIMEFRAME_D1", 16408)


class LiveFeaturePipeline:
    """Computes all 67 features in real-time from MT5 data.

    Maintains rolling DataFrames for M5, H1, H4, D1 bars and correlated
    instruments.  Uses all feature modules from features/ to compute
    54 precomputed features.  Uses LiveNormalizer to z-score the 38
    market features.  Combines with 8 account + 5 memory features = 67
    per frame.  Maintains a frame buffer (deque maxlen=10) for frame
    stacking and returns a 670-dim observation vector.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: LiveConfig) -> None:
        """Initialise the pipeline.

        Args:
            config: LiveConfig with all runtime parameters.
        """
        self._config = config

        # Rolling OHLCV buffers (DataFrames)
        self._m5: pd.DataFrame = _empty_ohlcv()
        self._h1: pd.DataFrame = _empty_ohlcv()
        self._h4: pd.DataFrame = _empty_ohlcv()
        self._d1: pd.DataFrame = _empty_ohlcv()

        # Rolling correlated instrument M5 buffers
        self._correlated_m5: Dict[str, pd.DataFrame] = {
            sym: _empty_ohlcv() for sym in _CORRELATED_SYMBOLS
        }

        # Maximum buffer sizes (rows to retain)
        self._max_m5 = max(config.warmup_bars, 500)
        self._max_h1 = 300
        self._max_h4 = 120
        self._max_d1 = 300
        self._max_corr = 300

        # Live normaliser (38 market features)
        self._normalizer = LiveNormalizer(
            market_features=config.market_feature_names,
            window=config.norm_window,
            clip=config.norm_clip,
            mode=config.normalization_mode,
        )

        # Frame stacking buffer: 10 most recent 67-dim frames
        self._frame_buffer: deque = deque(maxlen=config.frame_stack)

        # Duplicate bar guard
        self._last_bar_time: Optional[datetime] = None

        # Warmup state
        self._warmed_up: bool = False

        # Calendar data (loaded once during warmup)
        self._calendar_events: Optional[list] = None

        # Step counter (for logging cadence)
        self._step_count: int = 0

        # Latest ATR(14) cache
        self._latest_atr: float = 1.0

        # Latest raw feature snapshot (for logging / diagnostics)
        self._latest_features: Dict[str, float] = {}

        log.info(
            "LiveFeaturePipeline initialised: %d features, frame_stack=%d, obs_dim=%d",
            config.n_features,
            config.frame_stack,
            config.obs_dim,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, mt5_bridge) -> bool:
        """Load historical data and initialise everything.

        Fetches M5 (500 bars), H1 (300), H4 (120), D1 (300), and
        correlated instrument M5 (300 each) from the MT5 bridge.
        Processes all bars through feature computation and initialises
        the normaliser with the historical data.  Fills the frame
        buffer with the last 10 frames.

        Args:
            mt5_bridge: Connected MT5Bridge instance.

        Returns:
            True if warmup completed successfully.
        """
        log.info("LiveFeaturePipeline: starting warmup...")
        t0 = time.monotonic()

        symbol = self._config.mt5_symbol

        # ---- Fetch historical bars ----
        self._m5 = mt5_bridge.get_latest_bars(symbol, _TF_M5, self._max_m5)
        self._h1 = mt5_bridge.get_latest_bars(symbol, _TF_H1, self._max_h1)
        self._h4 = mt5_bridge.get_latest_bars(symbol, _TF_H4, self._max_h4)
        self._d1 = mt5_bridge.get_latest_bars(symbol, _TF_D1, self._max_d1)

        if self._m5.empty or len(self._m5) < 200:
            log.error(
                "LiveFeaturePipeline warmup FAILED: insufficient M5 data (%d bars)",
                len(self._m5),
            )
            return False

        log.info(
            "Historical bars loaded: M5=%d, H1=%d, H4=%d, D1=%d",
            len(self._m5),
            len(self._h1),
            len(self._h4),
            len(self._d1),
        )

        # ---- Fetch correlated instrument data ----
        for sym in _CORRELATED_SYMBOLS:
            df = mt5_bridge.get_latest_bars(sym, _TF_M5, self._max_corr)
            if not df.empty:
                self._correlated_m5[sym] = df
                log.info("Correlated %s: %d M5 bars loaded", sym, len(df))
            else:
                log.warning("Correlated %s: no data available", sym)

        # ---- Load calendar events ----
        self._load_calendar_events()

        # ---- Compute features for the last N bars to warm up normalizer ----
        # Process the last (norm_window + 10) bars to seed normaliser buffers
        # and fill the frame buffer.
        warmup_count = self._config.norm_window + self._config.frame_stack
        warmup_count = min(warmup_count, len(self._m5))

        log.info("Processing %d bars for normalizer warmup + frame buffer...", warmup_count)

        # We compute features for each of the last warmup_count bars by
        # sliding through the buffer.  This is more expensive than batch
        # computation but ensures the normaliser sees exactly the same
        # incremental data it will see in live mode.
        start_idx = len(self._m5) - warmup_count

        for i in range(start_idx, len(self._m5)):
            # Slice M5 buffer up to bar i (inclusive)
            m5_slice = self._m5.iloc[: i + 1].copy()

            # Compute raw precomputed features
            raw_features = self._compute_precomputed_features_from(m5_slice)

            # Normalise (this also feeds the normaliser buffers)
            normed = self._normalizer.normalize_batch(raw_features)

            # Build dummy account/memory for warmup (neutral defaults)
            account_feats = np.zeros(8, dtype=np.float32)
            account_feats[5] = 1.0  # equity_ratio = 1.0 (no position)
            memory_feats = np.full(5, 0.5, dtype=np.float32)

            frame = self._build_frame(normed, account_feats, memory_feats)
            self._frame_buffer.append(frame)

        # Update ATR cache
        self._latest_atr = get_atr_14(self._m5)

        # Set last bar time
        self._last_bar_time = self._m5["time"].iloc[-1]
        if hasattr(self._last_bar_time, "to_pydatetime"):
            self._last_bar_time = self._last_bar_time.to_pydatetime()

        self._warmed_up = True
        elapsed = time.monotonic() - t0
        log.info(
            "LiveFeaturePipeline warmup COMPLETE in %.1fs "
            "(normalizer buffers: %d/%d above min_periods, "
            "frame buffer: %d frames)",
            elapsed,
            sum(
                1 for s in self._normalizer.get_buffer_stats().values()
                if s["count"] >= _MIN_PERIODS
            ),
            len(self._config.market_feature_names),
            len(self._frame_buffer),
        )

        return True

    # ------------------------------------------------------------------
    # Per-bar update
    # ------------------------------------------------------------------

    def on_new_bar(
        self,
        mt5_bridge,
        account_state: Dict[str, float],
        memory_features: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Process a new M5 bar and return the 670-dim stacked observation.

        Args:
            mt5_bridge: Connected MT5Bridge instance.
            account_state: Dict with 8 account feature values (keyed by
                name from _ACCOUNT_FEATURE_ORDER).
            memory_features: 5-element numpy array of memory features
                (in _MEMORY_FEATURE_ORDER).

        Returns:
            670-dim numpy array (10 stacked 67-dim frames), or None if
            the pipeline is not warmed up or the bar is a duplicate.
        """
        if not self._warmed_up:
            log.warning("on_new_bar called before warmup -- returning None")
            return None

        symbol = self._config.mt5_symbol

        # ---- Fetch latest M5 bar ----
        latest_m5 = mt5_bridge.get_latest_bars(symbol, _TF_M5, 1)
        if latest_m5.empty:
            log.warning("on_new_bar: no M5 data returned")
            return None

        bar_time = latest_m5["time"].iloc[-1]
        if hasattr(bar_time, "to_pydatetime"):
            bar_time = bar_time.to_pydatetime()

        # ---- Duplicate bar guard ----
        if self._last_bar_time is not None and bar_time <= self._last_bar_time:
            log.debug(
                "on_new_bar: duplicate bar (time=%s <= last=%s) -- skipping",
                bar_time,
                self._last_bar_time,
            )
            return None

        self._last_bar_time = bar_time
        self._step_count += 1

        # ---- Append to M5 rolling buffer ----
        # Use loc append + deferred trim (only trim when 10% over max)
        self._m5 = pd.concat([self._m5, latest_m5], ignore_index=True)
        if len(self._m5) > self._max_m5 + 50:  # trim in bulk (every ~50 bars)
            self._m5 = self._m5.iloc[-self._max_m5:].reset_index(drop=True)

        # ---- Check if new H1/H4/D1 bars closed, fetch if so ----
        self._update_htf_if_needed(mt5_bridge, bar_time)

        # ---- Fetch latest correlated instrument bars ----
        self._update_correlated(mt5_bridge)

        # ---- Compute all 54 precomputed features ----
        raw_features = self._compute_precomputed_features()

        # ---- Normalise 38 market features ----
        normed_features = self._normalizer.normalize_batch(raw_features)

        # ---- Auto-reset detection (adaptive mode only) ----
        if self._normalizer.mode == "adaptive":
            healthy, reason = self._normalizer.check_distribution_health(
                z_abs_mean_threshold=self._config.auto_reset_z_abs_mean_threshold,
                pct_over_4sigma=self._config.auto_reset_pct_features_over_4sigma,
                cooldown_bars=self._config.auto_reset_cooldown_bars,
            )
            if not healthy:
                log.warning("AUTO-RESET: %s -- resetting normalizer", reason)
                self._normalizer.on_auto_reset()
                self.reset_normalizer()
                # Recompute with fresh normalizer
                normed_features = self._normalizer.normalize_batch(raw_features)

        # Store for diagnostics
        self._latest_features = {**raw_features}  # shallow copy of raw
        self._latest_features["_normalized"] = {
            k: normed_features[k]
            for k in self._config.market_feature_names
            if k in normed_features
        }

        # ---- Build account features array (in canonical order) ----
        account_arr = np.array(
            [account_state.get(name, 0.0) for name in _ACCOUNT_FEATURE_ORDER],
            dtype=np.float32,
        )

        # ---- Ensure memory features are the right shape ----
        mem_arr = np.asarray(memory_features, dtype=np.float32)
        if mem_arr.shape != (5,):
            log.warning(
                "on_new_bar: memory_features shape %s, expected (5,) -- using defaults",
                mem_arr.shape,
            )
            mem_arr = np.full(5, 0.5, dtype=np.float32)

        # ---- Build frame (n_features dim, matches loaded model) ----
        frame = self._build_frame(normed_features, account_arr, mem_arr)

        # ---- Feature health check ----
        health = self._check_feature_health(frame)
        if health.get("has_issues"):
            log.warning("Feature health issues: %s", health.get("issues"))

        # ---- Append to frame buffer ----
        self._frame_buffer.append(frame)

        # ---- Update ATR cache ----
        self._latest_atr = get_atr_14(self._m5)

        # ---- Return stacked observation (obs_dim = n_features × frame_stack) ----
        return self._get_stacked_observation()

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_precomputed_features(self) -> Dict[str, float]:
        """Compute all 54 precomputed features from current rolling buffers.

        Returns:
            Dict mapping feature_name -> float value.
        """
        return self._compute_precomputed_features_from(self._m5)

    def _compute_precomputed_features_from(
        self, m5: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute all 54 precomputed features from a given M5 slice.

        Uses the current HTF and correlated buffers alongside the
        provided M5 slice.

        Args:
            m5: M5 OHLCV DataFrame to use for feature computation.

        Returns:
            Dict mapping feature_name -> float value.
        """
        features: Dict[str, float] = {}

        # Groups A-E: Technical indicators (25 features)
        tech_feats = compute_technical_features(
            m5=m5,
            h1=self._h1,
            h4=self._h4,
            d1=self._d1,
            frac_diff_d=0.35,
        )
        features.update(tech_feats)

        # Group F: Time & Session (4 features)
        current_time = m5["time"].iloc[-1]
        if hasattr(current_time, "to_pydatetime"):
            current_time = current_time.to_pydatetime()
        # Ensure tz-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        time_feats = compute_time_session_features(current_time)
        features.update(time_feats)

        # Upgrade 1: Correlated Instruments (11 features)
        # Filter out empty correlated DataFrames to avoid NaN propagation
        valid_correlated = {
            sym: df for sym, df in self._correlated_m5.items()
            if not df.empty and len(df) >= 20
        }
        corr_feats = compute_correlation_features(
            xau_m5=m5,
            correlated_m5=valid_correlated,
        )
        features.update(corr_feats)

        # Upgrade 2: Calendar & Events (6 features)
        cal_feats = compute_calendar_features(
            timestamp=current_time,
            calendar_events=self._calendar_events,
            mql5_bridge_path=self._resolve_path(self._config.calendar_bridge_path),
            calendar_csv_path=self._resolve_path(self._config.calendar_csv_path),
        )
        features.update(cal_feats)

        # Upgrade 3: Spread & Liquidity (2 features)
        atr = get_atr_14(m5)
        spread_feats = compute_spread_liquidity_features(
            m5=m5,
            atr_14=atr,
            live_spread=None,  # Will be set by caller if available
            pip_price=0.10,
        )
        features.update(spread_feats)

        # Upgrade 4: Regime Detection (2 features)
        # Only pass correlated data if we have enough bars for the 100-bar window
        eurusd_df = self._correlated_m5.get("EURUSD")
        us500_df = self._correlated_m5.get("US500")
        regime_feats = compute_regime_features(
            xau_m5=m5,
            eurusd_m5=eurusd_df if eurusd_df is not None and len(eurusd_df) >= 100 else None,
            us500_m5=us500_df if us500_df is not None and len(us500_df) >= 100 else None,
            regime_corr_window=100,
        )
        features.update(regime_feats)

        # Upgrade 5: Session Microstructure (4 features)
        session_feats = compute_session_micro_features(
            m5=m5,
            atr_14=atr,
        )
        features.update(session_feats)

        # Upgrade 6: Regime Label (1 feature) — matches training feature_builder.py
        # ADX>25 = trending (+1), ATR_ratio>1.2 = volatile (-1), else ranging (0)
        adx = features.get("adx_14", 0.0)
        atr_ratio = features.get("atr_ratio", 1.0)
        if adx > 0.25:
            features["regime_label"] = 1.0
        elif atr_ratio > 1.2:
            features["regime_label"] = -1.0
        else:
            features["regime_label"] = 0.0

        # Validate we have all 55 precomputed features
        missing = [f for f in _PRECOMPUTED_FEATURE_ORDER if f not in features]
        if missing:
            log.warning(
                "_compute_precomputed_features: %d missing features filled with 0.0: %s",
                len(missing),
                missing[:5],
            )
            for f in missing:
                features[f] = 0.0

        return features

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def _build_frame(
        self,
        precomputed: Dict[str, float],
        account_features: np.ndarray,
        memory_features: np.ndarray,
    ) -> np.ndarray:
        """Build a frame from feature components, sized to match the loaded model.

        Supports both 67-feature models (pre-Tier-1, no regime_label) and
        68-feature models (Tier 1+, includes regime_label).

        Assembles features in the exact training order:
        - 67-feature model: 54 precomputed + 8 account + 5 memory = 67
        - 68-feature model: 55 precomputed + 8 account + 5 memory = 68

        Args:
            precomputed: Dict of normalised precomputed features.
            account_features: 8-element array (Group G).
            memory_features: 5-element array (Group H).

        Returns:
            n_features-dim numpy array.
        """
        n_features = self._config.n_features  # 67 for pre-Tier-1, 68 for Tier-1+
        n_precomp = n_features - 13  # subtract 8 account + 5 memory

        # Precomputed features in canonical order, trimmed to match model
        precomp_order = _PRECOMPUTED_FEATURE_ORDER[:n_precomp]
        precomp_arr = np.array(
            [precomputed.get(name, 0.0) for name in precomp_order],
            dtype=np.float32,
        )

        # Concatenate: n_precomp + 8 + 5 = n_features
        frame = np.concatenate([precomp_arr, account_features, memory_features])

        # Sanitise NaN/Inf
        nan_mask = np.isnan(frame)
        inf_mask = np.isinf(frame)
        if nan_mask.any() or inf_mask.any():
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

        assert frame.shape == (n_features,), f"Frame shape {frame.shape} != ({n_features},)"
        return frame

    def _get_stacked_observation(self) -> np.ndarray:
        """Flatten the frame buffer to an obs_dim observation vector.

        If the frame buffer has fewer than ``frame_stack`` (10) frames,
        pads with zero frames at the front (oldest positions).

        Returns:
            obs_dim numpy array (frame_stack stacked n_features-dim frames).
        """
        n_frames = self._config.frame_stack  # 10
        n_features = self._config.n_features  # 68

        frames = list(self._frame_buffer)

        # Pad with zeros if we don't have enough frames yet
        while len(frames) < n_frames:
            frames.insert(0, np.zeros(n_features, dtype=np.float32))

        # Stack: oldest first, newest last (matches training)
        stacked = np.concatenate(frames[-n_frames:])

        # Clip to training bounds [-10, +10]
        stacked = np.clip(stacked, -10.0, 10.0)

        assert stacked.shape == (self._config.obs_dim,), (
            f"Stacked observation shape {stacked.shape} != ({self._config.obs_dim},)"
        )
        return stacked

    # ------------------------------------------------------------------
    # Feature health checking
    # ------------------------------------------------------------------

    def _check_feature_health(self, frame: np.ndarray) -> Dict[str, Any]:
        """Check a 67-dim frame for NaN, Inf, and constant features.

        Args:
            frame: 67-dim numpy array.

        Returns:
            Dict with keys: has_issues (bool), issues (list of str),
            nan_count (int), inf_count (int), constant_count (int).
        """
        issues: List[str] = []

        nan_count = int(np.isnan(frame).sum())
        inf_count = int(np.isinf(frame).sum())

        if nan_count > 0:
            nan_indices = np.where(np.isnan(frame))[0]
            nan_names = [
                _FULL_FEATURE_ORDER[i] if i < len(_FULL_FEATURE_ORDER) else f"idx_{i}"
                for i in nan_indices[:5]
            ]
            issues.append(f"NaN in {nan_count} features: {nan_names}")

        if inf_count > 0:
            inf_indices = np.where(np.isinf(frame))[0]
            inf_names = [
                _FULL_FEATURE_ORDER[i] if i < len(_FULL_FEATURE_ORDER) else f"idx_{i}"
                for i in inf_indices[:5]
            ]
            issues.append(f"Inf in {inf_count} features: {inf_names}")

        # Check for all-zero precomputed section (first 54 features)
        precomp = frame[:54]
        if np.all(precomp == 0.0):
            issues.append("ALL precomputed features are zero (warmup issue?)")

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

    # ------------------------------------------------------------------
    # HTF bar management
    # ------------------------------------------------------------------

    def _update_htf_if_needed(self, mt5_bridge, current_bar_time: datetime) -> None:
        """Check if new H1/H4/D1 bars have closed and fetch them.

        Compares the latest bar timestamp in each HTF buffer against
        the current M5 bar time.  If a new HTF bar should have closed,
        fetches the latest bars to capture it.

        Args:
            mt5_bridge: Connected MT5Bridge instance.
            current_bar_time: Timestamp of the current M5 bar.
        """
        symbol = self._config.mt5_symbol

        # Ensure we have a proper datetime for comparison
        if hasattr(current_bar_time, "to_pydatetime"):
            current_bar_time = current_bar_time.to_pydatetime()

        hour = current_bar_time.hour if hasattr(current_bar_time, "hour") else 0
        minute = current_bar_time.minute if hasattr(current_bar_time, "minute") else 0

        # H1: new bar every hour (minute == 0)
        if minute == 0:
            new_h1 = mt5_bridge.get_latest_bars(symbol, _TF_H1, 2)
            if not new_h1.empty:
                last_h1_time = self._h1["time"].iloc[-1] if not self._h1.empty else None
                new_time = new_h1["time"].iloc[-1]
                if last_h1_time is None or new_time > last_h1_time:
                    self._h1 = pd.concat([self._h1, new_h1.tail(1)], ignore_index=True)
                    if len(self._h1) > self._max_h1 + 20:
                        self._h1 = self._h1.iloc[-self._max_h1:].reset_index(drop=True)

        # H4: new bar at 0:00, 4:00, 8:00, 12:00, 16:00, 20:00 (minute == 0)
        if minute == 0 and hour % 4 == 0:
            new_h4 = mt5_bridge.get_latest_bars(symbol, _TF_H4, 2)
            if not new_h4.empty:
                last_h4_time = self._h4["time"].iloc[-1] if not self._h4.empty else None
                new_time = new_h4["time"].iloc[-1]
                if last_h4_time is None or new_time > last_h4_time:
                    self._h4 = pd.concat([self._h4, new_h4.tail(1)], ignore_index=True)
                    if len(self._h4) > self._max_h4 + 10:
                        self._h4 = self._h4.iloc[-self._max_h4:].reset_index(drop=True)

        # D1: new bar at 0:00 UTC
        if hour == 0 and minute == 0:
            new_d1 = mt5_bridge.get_latest_bars(symbol, _TF_D1, 2)
            if not new_d1.empty:
                last_d1_time = self._d1["time"].iloc[-1] if not self._d1.empty else None
                new_time = new_d1["time"].iloc[-1]
                if last_d1_time is None or new_time > last_d1_time:
                    self._d1 = pd.concat([self._d1, new_d1.tail(1)], ignore_index=True)
                    if len(self._d1) > self._max_d1 + 10:
                        self._d1 = self._d1.iloc[-self._max_d1:].reset_index(drop=True)

    def _update_correlated(self, mt5_bridge, bar_time: Optional[datetime] = None) -> None:
        """Fetch the latest M5 bar for each correlated instrument.

        Appends new bars to their respective rolling buffers and
        trims to max size.  Batches all 5 calls but skips individual
        symbols where the last bar time hasn't changed.

        Args:
            mt5_bridge: Connected MT5Bridge instance.
            bar_time: Current XAUUSD bar time (unused, kept for API compat).
        """
        for sym in _CORRELATED_SYMBOLS:
            try:
                new_bar = mt5_bridge.get_latest_bars(sym, _TF_M5, 1)
                if new_bar.empty:
                    continue

                existing = self._correlated_m5[sym]
                if not existing.empty:
                    last_time = existing["time"].iloc[-1]
                    new_time = new_bar["time"].iloc[-1]
                    if new_time <= last_time:
                        continue  # Duplicate

                self._correlated_m5[sym] = pd.concat(
                    [existing, new_bar], ignore_index=True
                )
                if len(self._correlated_m5[sym]) > self._max_corr + 50:
                    self._correlated_m5[sym] = (
                        self._correlated_m5[sym]
                        .iloc[-self._max_corr:]
                        .reset_index(drop=True)
                    )
            except Exception as exc:
                log.debug("Failed to update correlated %s: %s", sym, exc)

    # ------------------------------------------------------------------
    # Calendar loading
    # ------------------------------------------------------------------

    def _load_calendar_events(self) -> None:
        """Load calendar events from configured sources.

        Priority:
            1. Auto-generated events (NFP, CPI, FOMC, ECB, ISM — rule-based)
            2. CSV (user-supplied / bridge-persisted history)
            3. known_events.json (static fallback)
        All sources are merged and deduplicated.
        The MQL5 bridge JSON (if present) is still read per-bar in
        compute_calendar_features() and new events are auto-persisted to CSV.
        """
        import json as _json
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        _utc = _ZI("UTC")

        # --- Source 1: Auto-generated events (always available, no deps) ---
        auto_events: list = []
        try:
            from features.calendar_generator import generate_upcoming_events
            auto_events = generate_upcoming_events(months_ahead=6)
            log.info(
                "Calendar auto-gen: %d events (next 6 months)",
                len(auto_events),
            )
        except Exception as exc:
            log.warning("Calendar auto-gen failed: %s", exc)

        # --- Source 2: CSV (historical + bridge-persisted events) ---
        csv_events: list = []
        csv_path = self._resolve_path(self._config.calendar_csv_path)
        if csv_path.exists():
            try:
                from features.calendar import load_calendar_csv
                df = load_calendar_csv(csv_path)
                for row in df.itertuples(index=False):
                    dt = row.datetime_utc
                    if hasattr(dt, "to_pydatetime"):
                        dt = dt.to_pydatetime()
                    csv_events.append({
                        "datetime_utc": dt,
                        "event_name": getattr(row, "event_name", ""),
                        "impact": "HIGH",
                    })
                log.info(
                    "Calendar CSV: %d events (%s to %s)",
                    len(csv_events),
                    csv_events[0]["datetime_utc"].date() if csv_events else "?",
                    csv_events[-1]["datetime_utc"].date() if csv_events else "?",
                )
            except Exception as exc:
                log.warning("Failed to load calendar CSV: %s", exc)

        # --- Source 3: Static JSON (legacy fallback) ---
        static_events: list = []
        static_path = self._resolve_path(self._config.calendar_static_path)
        if static_path.exists():
            try:
                with open(static_path, "r", encoding="utf-8") as fh:
                    data = _json.load(fh)
                for ev in data.get("events", []):
                    date_str = ev.get("date", "")
                    time_str = ev.get("time_utc", "00:00")
                    if not date_str:
                        continue
                    dt = pd.to_datetime(
                        f"{date_str} {time_str}", utc=True
                    ).to_pydatetime()
                    static_events.append({
                        "datetime_utc": dt,
                        "event_name": ev.get("name", ""),
                        "impact": ev.get("importance", "HIGH").upper(),
                    })
            except Exception as exc:
                log.warning("Failed to load static calendar: %s", exc)

        # --- Merge all sources (dedup by date + event_name) ---
        # Priority: auto-gen first, then CSV, then static
        seen: set = set()
        merged: list = []
        sources = [
            (auto_events, "auto"),
            (csv_events, "csv"),
            (static_events, "static"),
        ]
        counts = {"auto": 0, "csv": 0, "static": 0}
        for events, label in sources:
            for e in events:
                key = (e["datetime_utc"].date(), e["event_name"])
                if key not in seen:
                    seen.add(key)
                    merged.append(e)
                    counts[label] += 1

        merged.sort(key=lambda e: e["datetime_utc"])
        self._calendar_events = merged

        now_utc = _dt.now(_utc)
        future_count = sum(1 for e in merged if e["datetime_utc"] > now_utc)
        log.info(
            "Calendar ready: %d events (%d auto-gen, %d csv, %d static), "
            "%d upcoming",
            len(merged), counts["auto"], counts["csv"], counts["static"],
            future_count,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_current_atr(self) -> float:
        """Return the latest ATR(14) value for the risk manager.

        Returns:
            ATR(14) in price units, or 1.0 if not available.
        """
        return self._latest_atr

    def get_feature_snapshot(self) -> Dict[str, float]:
        """Return the current raw feature values for logging.

        Returns:
            Dict mapping feature_name -> raw float value.
        """
        return dict(self._latest_features)

    def get_normalizer_stats(self) -> Dict[str, Dict[str, float]]:
        """Return normaliser buffer statistics for health monitoring.

        Returns:
            Dict mapping feature_name -> {mean, std, count}.
        """
        return self._normalizer.get_buffer_stats()

    def is_warmed_up(self) -> bool:
        """Check whether the pipeline has been warmed up.

        Returns:
            True if warmup() has completed successfully.
        """
        return self._warmed_up

    def get_normalizer(self) -> LiveNormalizer:
        """Return the internal LiveNormalizer instance.

        Useful for persisting / restoring normaliser state externally.
        """
        return self._normalizer

    def get_frame_buffer_depth(self) -> int:
        """Return the current number of frames in the frame buffer."""
        return len(self._frame_buffer)

    def get_bar_count(self) -> Dict[str, int]:
        """Return the number of bars in each rolling buffer."""
        counts = {
            "M5": len(self._m5),
            "H1": len(self._h1),
            "H4": len(self._h4),
            "D1": len(self._d1),
        }
        for sym in _CORRELATED_SYMBOLS:
            counts[f"corr_{sym}"] = len(self._correlated_m5[sym])
        return counts

    def get_last_bar_time(self) -> Optional[datetime]:
        """Return the timestamp of the last processed M5 bar."""
        return self._last_bar_time

    def get_calendar_info(self) -> Dict[str, Any]:
        """Return calendar summary for the dashboard.

        Returns:
            Dict with:
                events_today (int): Number of high-impact events today.
                next_event_time (str or None): ISO time of next event.
                next_event_name (str or None): Name of next event.
        """
        now_utc = datetime.now(timezone.utc)
        today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        events_today = 0
        next_event_time = None
        next_event_name = None

        if self._calendar_events:
            for ev in self._calendar_events:
                dt = ev.get("datetime_utc")
                if dt is None:
                    continue
                # Ensure tz-aware comparison
                if hasattr(dt, "tzinfo") and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if today_start <= dt < today_end:
                    events_today += 1
                if dt > now_utc and next_event_time is None:
                    next_event_time = dt.strftime("%H:%M UTC")
                    next_event_name = ev.get("event_name", "Unknown")

        return {
            "events_today": events_today,
            "next_event_time": next_event_time,
            "next_event_name": next_event_name,
        }

    # ------------------------------------------------------------------
    # Normalizer persistence (convenience wrappers)
    # ------------------------------------------------------------------

    def set_feature_baseline(self, baseline: Dict[str, Dict[str, float]]) -> None:
        """Load training feature baseline into the normalizer.

        Must be called before first tick in frozen mode.

        Args:
            baseline: Dict from feature_baseline.json in model package.
                Keys are feature names, values have "mean" and "std".
        """
        self._normalizer.set_baseline(baseline)
        log.info(
            "Feature baseline loaded into normalizer (mode=%s)",
            self._normalizer.mode,
        )

    def save_normalizer_state(self) -> None:
        """Save the normaliser state to the configured path."""
        path = self._resolve_path(self._config.normalizer_state_path)
        self._normalizer.save_state(str(path))

    def load_normalizer_state(self) -> bool:
        """Load the normaliser state from the configured path.

        Returns:
            True if the state was loaded successfully.
        """
        path = self._resolve_path(self._config.normalizer_state_path)
        return self._normalizer.load_state(str(path))

    def reset_normalizer(self, mt5_bridge=None) -> None:
        """Reset the normalizer and optionally re-warm from live data.

        Clears all rolling buffers, deletes the saved state file, and
        optionally re-warms from the current M5 buffer (if mt5_bridge
        is not needed because we already have bars in self._m5).

        This is the "un-stuck" button for conviction collapse caused
        by normalizer contamination after violent market moves.

        Args:
            mt5_bridge: If provided, fetches fresh M5 data and re-warms.
                If None, re-warms from the existing M5 buffer.
        """
        # 1. Reset normalizer buffers
        self._normalizer.reset()

        # 2. Delete saved state file
        state_path = self._resolve_path(self._config.normalizer_state_path)
        if state_path.exists():
            state_path.unlink()
            log.info("Deleted normalizer state file: %s", state_path)

        # 3. Clear frame buffer (old frames have contaminated z-scores)
        self._frame_buffer.clear()

        # 4. Re-warm from existing M5 buffer if we have data
        if len(self._m5) >= self._config.norm_window:
            warmup_count = self._config.norm_window + self._config.frame_stack
            warmup_count = min(warmup_count, len(self._m5))
            start_idx = len(self._m5) - warmup_count

            log.info(
                "Re-warming normalizer from %d existing M5 bars...",
                warmup_count,
            )
            for i in range(start_idx, len(self._m5)):
                m5_slice = self._m5.iloc[: i + 1].copy()
                raw_features = self._compute_precomputed_features_from(m5_slice)
                normed = self._normalizer.normalize_batch(raw_features)

                account_feats = np.zeros(8, dtype=np.float32)
                account_feats[5] = 1.0  # equity_ratio = 1.0
                memory_feats = np.full(5, 0.5, dtype=np.float32)
                frame = self._build_frame(normed, account_feats, memory_feats)
                self._frame_buffer.append(frame)

            log.info("Normalizer re-warm complete (%d frames in buffer)", len(self._frame_buffer))
        else:
            log.warning(
                "Not enough M5 data for re-warm (%d bars, need %d). "
                "Normalizer will produce 0.0 z-scores until %d bars accumulated.",
                len(self._m5),
                self._config.norm_window,
                self.min_periods if hasattr(self, 'min_periods') else 50,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the base directory.

        Args:
            relative: Relative or absolute path string.

        Returns:
            Resolved absolute Path.
        """
        p = Path(relative)
        if p.is_absolute():
            return p
        return self._config.get_base_dir() / p


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empty_ohlcv() -> pd.DataFrame:
    """Return an empty OHLCV DataFrame with the correct columns."""
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
