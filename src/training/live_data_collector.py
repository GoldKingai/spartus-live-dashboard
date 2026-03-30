"""Live Data Collector — accumulates MT5 M5 bars for fine-tuning episodes.

Connects to MetaTrader 5, pulls XAUUSD + 5 correlated instrument M5 bars,
runs the exact same FeatureBuilder as training (ensuring feature parity),
and maintains a rolling buffer ready for TradeEnv episodes.

Key design decision (from design plan §3.1):
    Reuse FeatureBuilder from training, NOT the live dashboard's FeaturePipeline.
    This ensures features are computed identically to training, preventing the
    model from seeing a "different" input distribution at inference time.

Data flow:
    MT5 → XAUUSD M5 + 5x correlated M5 + H1/H4/D1
        → FeatureBuilder.build_features()
        → 54 precomputed feature columns
        → Rolling 2000-bar buffer
        → features_df ready for TradeEnv

Thread-safety: All MT5 calls and buffer updates are protected by a threading.Lock.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# MT5 timeframe constants (duplicated to avoid import failures when MT5 unavailable)
_TF_M5 = 5
_TF_H1 = 16385
_TF_H4 = 16388
_TF_D1 = 16408


def _try_import_mt5():
    """Import MetaTrader5. Returns module or None."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        return None


class LiveDataCollector:
    """Accumulates live M5 bars from MT5 and builds features for fine-tuning.

    Usage:
        collector = LiveDataCollector(config)
        collector.connect()  # Initializes MT5

        # Poll on timer (e.g., every 5 minutes):
        if collector.update():
            if collector.is_ready():
                features_df = collector.get_features_df()
                # Pass to TradeEnv for a training episode
    """

    def __init__(self, config):
        self.cfg = config
        self._lock = threading.Lock()

        # Rolling bar buffer: {symbol: deque of OHLCV dicts}
        max_bars = getattr(config, "finetune_bar_buffer_size", 2000)
        self._max_bars = max_bars
        self._bars: Dict[str, deque] = {
            "XAUUSD": deque(maxlen=max_bars),
            "H1": deque(maxlen=max_bars),
            "H4": deque(maxlen=max_bars),
            "D1": deque(maxlen=max_bars),
        }
        for sym in getattr(config, "correlated_symbols", ("EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL")):
            self._bars[sym] = deque(maxlen=max_bars)

        self._correlated_symbols = list(
            getattr(config, "correlated_symbols", ("EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"))
        )
        self._primary_symbol = getattr(config, "mt5_symbol", "XAUUSD")

        # Cached features (rebuilt when new bars arrive)
        self._features_df: Optional[pd.DataFrame] = None
        self._features_dirty: bool = True
        self._last_update: float = 0.0
        self._last_bar_time = None

        # Symbol map for broker name resolution (e.g., US500 → USA500IDXUSD)
        self._symbol_map = getattr(config, "symbol_map", {})

        # MT5 connection handle (MT5Handle from MT5Connection manager)
        self._mt5 = None        # MT5Handle — set by connect()
        self._connected: bool = False

        # FeatureBuilder (lazy init)
        self._feature_builder = None

        # Stats
        self.total_bars_collected: int = 0
        self.total_updates: int = 0
        self.last_error: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self, terminal_path: Optional[str] = None) -> bool:
        """Acquire a shared MT5 connection handle.

        Uses MT5Connection manager so training and fine-tuning never shut
        down each other's connection — reference counting ensures the
        underlying mt5.shutdown() only fires when ALL holders release.

        Args:
            terminal_path: Path to terminal64.exe. None = use default.

        Returns:
            True if connected successfully.
        """
        from src.data.mt5_connection import MT5Connection

        handle = MT5Connection.acquire(
            terminal_path=terminal_path,
            symbol_map=self._symbol_map,
        )
        if handle is None:
            log.error("LiveDataCollector: MT5 unavailable (not installed or init failed)")
            self.last_error = "MT5 unavailable"
            return False

        self._mt5 = handle
        self._connected = True
        log.info("LiveDataCollector: MT5 handle acquired")
        return True

    def disconnect(self) -> None:
        """Release the MT5 connection handle.

        Decrements the shared ref count. The underlying mt5.shutdown() only
        fires when the last holder (training or fine-tuner) releases.
        """
        if self._mt5 is not None and self._connected:
            self._mt5.release()
            self._mt5 = None
            self._connected = False
            log.info("LiveDataCollector: MT5 handle released")

    # ─────────────────────────────────────────────────────────────────────────
    # Bar Collection
    # ─────────────────────────────────────────────────────────────────────────

    def prefill_from_date(self, start_date=None) -> int:
        """Pull all M5 bars from start_date to now and fill the buffer.

        Called once at fine-tune startup to seed the buffer with the full
        year's data. After this, update() adds new bars bar-by-bar as the
        market moves.

        Args:
            start_date: datetime (UTC) to pull from. Defaults to Jan 1 of
                        the current year so the AI sees the full year before
                        transitioning to live bar-by-bar mode.

        Returns:
            Number of M5 bars loaded into the primary buffer.
        """
        if not self._connected or self._mt5 is None:
            return 0

        from datetime import datetime, timezone

        if start_date is None:
            now = datetime.now(timezone.utc)
            start_date = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        elif start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)

        end_date = datetime.now(timezone.utc)
        log.info(
            f"LiveDataCollector: prefilling from {start_date.date()} "
            f"to {end_date.date()} ..."
        )

        try:
            mt5 = self._mt5
            broker_sym = self._resolve_symbol(self._primary_symbol)

            rates = mt5.copy_rates_range(broker_sym, _TF_M5, start_date, end_date)
            if rates is None or len(rates) == 0:
                log.warning("LiveDataCollector: prefill returned no bars")
                return 0

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            if "tick_volume" in df.columns:
                df.rename(columns={"tick_volume": "volume"}, inplace=True)

            n_bars = len(df)

            # Expand buffer under the lock if year's data exceeds current capacity
            with self._lock:
                if n_bars > self._max_bars:
                    self._max_bars = n_bars + 500
                    for key in self._bars:
                        old = list(self._bars[key])
                        self._bars[key] = deque(old, maxlen=self._max_bars)

            with self._lock:
                self._bars["XAUUSD"].clear()
                for _, row in df.iterrows():
                    self._bars["XAUUSD"].append({
                        "time": row["time"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row.get("volume", 0),
                    })
                if df["time"].iloc[-1] is not None:
                    self._last_bar_time = df["time"].iloc[-1]
                self.total_bars_collected += n_bars

            log.info(f"LiveDataCollector: prefilled {n_bars} XAUUSD M5 bars")

            # Also prefill HTF and correlated
            self._fetch_higher_timeframes()
            self._prefill_correlated(start_date, end_date)

            self._features_dirty = True
            return n_bars

        except Exception as e:
            log.error(f"LiveDataCollector: prefill failed: {e}", exc_info=True)
            return 0

    def _prefill_correlated(self, start_date, end_date) -> None:
        """Pull correlated instrument M5 bars for the prefill date range."""
        mt5 = self._mt5
        for sym in self._correlated_symbols:
            try:
                broker_sym = self._resolve_symbol(sym)
                rates = mt5.copy_rates_range(broker_sym, _TF_M5, start_date, end_date)
                if rates is None or len(rates) == 0:
                    continue
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                if "tick_volume" in df.columns:
                    df.rename(columns={"tick_volume": "volume"}, inplace=True)

                with self._lock:
                    self._bars[sym].clear()
                    for _, row in df.iterrows():
                        self._bars[sym].append({
                            "time": row["time"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row.get("volume", 0),
                        })
                log.info(f"LiveDataCollector: prefilled {len(df)} {sym} bars")
            except Exception as e:
                log.debug(f"LiveDataCollector: {sym} prefill failed: {e}")

    def update(self) -> bool:
        """Fetch latest bars from MT5 and update the buffer.

        Should be called every 5 minutes (on the M5 bar close).

        Returns:
            True if new bars were added.
        """
        if not self._connected or self._mt5 is None:
            return False

        try:
            new_bars = self._fetch_primary()
            if not new_bars:
                return False

            self._fetch_higher_timeframes()
            self._fetch_correlated()

            self._features_dirty = True
            self._last_update = time.time()
            self.total_updates += 1
            return True
        except Exception as e:
            log.warning(f"LiveDataCollector: update failed: {e}")
            self.last_error = str(e)
            return False

    def _fetch_primary(self) -> bool:
        """Fetch XAUUSD M5 bars. Returns True if new bars added."""
        mt5 = self._mt5
        broker_sym = self._resolve_symbol(self._primary_symbol)
        count = min(1000, self._max_bars)

        rates = mt5.copy_rates_from_pos(broker_sym, _TF_M5, 0, count)
        if rates is None or len(rates) == 0:
            return False

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "tick_volume" in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)

        with self._lock:
            for _, row in df.iterrows():
                t = row["time"]
                if self._last_bar_time is None or t > self._last_bar_time:
                    self._bars["XAUUSD"].append({
                        "time": t,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row.get("volume", 0),
                    })
                    self._last_bar_time = t
                    self.total_bars_collected += 1

        return True

    def _fetch_higher_timeframes(self) -> None:
        """Fetch H1, H4, D1 bars for the primary symbol."""
        mt5 = self._mt5
        broker_sym = self._resolve_symbol(self._primary_symbol)

        tf_map = {
            "H1": (_TF_H1, 200),
            "H4": (_TF_H4, 100),
            "D1": (_TF_D1, 50),
        }

        for key, (tf, count) in tf_map.items():
            try:
                rates = mt5.copy_rates_from_pos(broker_sym, tf, 0, count)
                if rates is None or len(rates) == 0:
                    continue
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                if "tick_volume" in df.columns:
                    df.rename(columns={"tick_volume": "volume"}, inplace=True)

                with self._lock:
                    self._bars[key].clear()
                    for _, row in df.iterrows():
                        self._bars[key].append({
                            "time": row["time"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row.get("volume", 0),
                        })
            except Exception as e:
                log.debug(f"LiveDataCollector: {key} fetch failed: {e}")

    def _fetch_correlated(self) -> None:
        """Fetch M5 bars for all correlated instruments."""
        mt5 = self._mt5
        count = min(500, self._max_bars)

        for sym in self._correlated_symbols:
            try:
                broker_sym = self._resolve_symbol(sym)
                rates = mt5.copy_rates_from_pos(broker_sym, _TF_M5, 0, count)
                if rates is None or len(rates) == 0:
                    continue
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                if "tick_volume" in df.columns:
                    df.rename(columns={"tick_volume": "volume"}, inplace=True)

                with self._lock:
                    self._bars[sym].clear()
                    for _, row in df.iterrows():
                        self._bars[sym].append({
                            "time": row["time"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row.get("volume", 0),
                        })
            except Exception as e:
                log.debug(f"LiveDataCollector: {sym} fetch failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Building
    # ─────────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True if enough bars have been collected for a training episode."""
        min_bars = getattr(self.cfg, "finetune_min_bars", 500)
        with self._lock:
            return len(self._bars["XAUUSD"]) >= min_bars

    def get_bar_count(self) -> int:
        """Return current number of buffered M5 bars."""
        with self._lock:
            return len(self._bars["XAUUSD"])

    def get_latest_bar_time(self):
        """Return the timestamp of the most recent XAUUSD M5 bar, or None.

        Returns a timezone-aware datetime (UTC), or None if buffer is empty.
        """
        from datetime import timezone
        with self._lock:
            bars = self._bars.get("XAUUSD")
            if not bars:
                return None
            last = bars[-1]
            t = last.get("time")
            if t is None:
                return None
            if hasattr(t, "tzinfo") and t.tzinfo is not None:
                return t
            # Numeric Unix timestamp
            try:
                from datetime import datetime
                return datetime.fromtimestamp(float(t), tz=timezone.utc)
            except Exception:
                return None

    def get_features_df(self, force_rebuild: bool = False) -> Optional[pd.DataFrame]:
        """Build and return a features DataFrame ready for TradeEnv.

        Features are cached until new bars arrive (dirty flag).

        Args:
            force_rebuild: If True, always rebuild even if clean.

        Returns:
            DataFrame with 54 precomputed features + OHLCV + time columns.
            None if insufficient data.
        """
        if not self.is_ready():
            return None

        if not self._features_dirty and not force_rebuild and self._features_df is not None:
            return self._features_df

        with self._lock:
            m5_bars = list(self._bars["XAUUSD"])
            h1_bars = list(self._bars["H1"])
            h4_bars = list(self._bars["H4"])
            d1_bars = list(self._bars["D1"])
            corr_bars = {sym: list(self._bars[sym]) for sym in self._correlated_symbols}

        try:
            m5_df = self._bars_to_df(m5_bars)
            h1_df = self._bars_to_df(h1_bars) if h1_bars else self._make_stub_htf(m5_df, "1h")
            h4_df = self._bars_to_df(h4_bars) if h4_bars else self._make_stub_htf(m5_df, "4h")
            d1_df = self._bars_to_df(d1_bars) if d1_bars else self._make_stub_htf(m5_df, "1D")

            corr_dfs = {}
            for sym in self._correlated_symbols:
                if corr_bars.get(sym):
                    corr_dfs[sym] = self._bars_to_df(corr_bars[sym])

            fb = self._get_feature_builder()
            features_df = fb.build_features(
                m5=m5_df,
                h1=h1_df,
                h4=h4_df,
                d1=d1_df,
                correlated_m5=corr_dfs if corr_dfs else None,
            )

            # Drop warmup rows (first 200 rows will have NaN features)
            warmup = getattr(self.cfg, "lookback", 200)
            features_df = features_df.iloc[warmup:].reset_index(drop=True)

            if len(features_df) < 50:
                log.warning(f"LiveDataCollector: too few rows after warmup: {len(features_df)}")
                return None

            self._features_df = features_df
            self._features_dirty = False
            return features_df

        except Exception as e:
            log.warning(f"LiveDataCollector: feature build failed: {e}")
            self.last_error = str(e)
            return None

    def _bars_to_df(self, bars: list) -> pd.DataFrame:
        """Convert list of bar dicts to DataFrame."""
        df = pd.DataFrame(bars)
        # Ensure 'time' is timezone-aware UTC
        if "time" in df.columns:
            if df["time"].dtype == "object" or not hasattr(df["time"].iloc[0], "tzinfo"):
                df["time"] = pd.to_datetime(df["time"], utc=True)
            elif df["time"].dt.tz is None:
                df["time"] = df["time"].dt.tz_localize("UTC")
        return df

    def _make_stub_htf(self, m5_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample M5 data to higher timeframe as fallback when MT5 HTF unavailable."""
        try:
            df = m5_df.copy()
            df = df.set_index("time")
            df.index = pd.DatetimeIndex(df.index)
            resampled = df.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
            return resampled.reset_index()
        except Exception:
            # Last resort: return minimal stub
            return m5_df.iloc[::12].copy().reset_index(drop=True)

    def _get_feature_builder(self):
        """Lazy-init FeatureBuilder (same as training)."""
        if self._feature_builder is None:
            from src.data.feature_builder import FeatureBuilder
            self._feature_builder = FeatureBuilder(config=self.cfg)
        return self._feature_builder

    # ─────────────────────────────────────────────────────────────────────────
    # Symbol Resolution
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_symbol(self, canonical: str) -> str:
        """Resolve canonical symbol name to broker-specific name."""
        return self._symbol_map.get(canonical, canonical)

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return status dict for dashboard display."""
        bar_count = self.get_bar_count()
        min_bars = getattr(self.cfg, "finetune_min_bars", 500)
        max_bars = self._max_bars
        hours_to_ready = max(0, (min_bars - bar_count) * 5 / 60) if bar_count < min_bars else 0

        return {
            "connected": self._connected,
            "bar_count": bar_count,
            "max_bars": max_bars,
            "buffer_pct": bar_count / max_bars * 100,
            "is_ready": bar_count >= min_bars,
            "hours_to_ready": hours_to_ready,
            "last_update": self._last_update,
            "total_bars_collected": self.total_bars_collected,
            "total_updates": self.total_updates,
            "last_error": self.last_error,
            "correlated_counts": {
                sym: len(self._bars.get(sym, [])) for sym in self._correlated_symbols
            },
        }
