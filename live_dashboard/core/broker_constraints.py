"""BrokerConstraints -- Cached MT5 symbol metadata and spread analytics.

Provides two-tier refresh of broker data:
  - Heavy refresh (every 10 min or on reconnect): full symbol_info()
  - Light refresh (every 10s or every new bar): spread + tick data

Also computes spread EMA for anomaly detection and enforces
min SL distance based on broker stops_level.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class BrokerSnapshot:
    """Immutable snapshot of current broker constraints for dashboard display."""

    # From symbol_info (heavy refresh)
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 0        # trade_stops_level (integer points)
    freeze_level: int = 0       # trade_freeze_level (integer points)
    tick_value: float = 0.745   # trade_tick_value (GBP account fallback)
    tick_size: float = 0.01     # trade_tick_size
    contract_size: float = 100.0
    point: float = 0.01
    digits: int = 2

    # Computed
    value_per_point: float = 74.5   # tick_value / tick_size (GBP)

    # From light refresh
    spread_current: float = 0.0         # Current spread in price (e.g. 0.21)
    spread_current_points: int = 0      # Current spread in integer points (e.g. 21)
    spread_ema: float = 0.0             # EMA of spread in price
    spread_max_1h: float = 0.0          # Max spread in last hour (price)

    # Min SL distance in price (computed)
    min_sl_distance: float = 0.0

    # Timestamps (monotonic)
    last_heavy_refresh: float = 0.0
    last_light_refresh: float = 0.0


class BrokerConstraints:
    """Cached MT5 symbol metadata with two-tier refresh.

    No threads -- call ``heavy_refresh()`` and ``light_refresh()``
    from the main trading loop.  They self-throttle via timestamps.
    """

    # Spread EMA smoothing factor
    SPREAD_EMA_ALPHA = 0.05

    # Spread history for max-1h (360 entries at 10s = 1 hour)
    SPREAD_HISTORY_SIZE = 360

    def __init__(self, config, mt5_bridge) -> None:
        self._config = config
        self._bridge = mt5_bridge
        self._symbol: str = config.mt5_symbol

        # Refresh intervals from config (with sensible defaults)
        self._heavy_interval: int = getattr(config, "broker_heavy_refresh_s", 600)
        self._light_interval: int = getattr(config, "broker_light_refresh_s", 10)

        # Spread gate config
        self._spread_hard_max_points: int = getattr(config, "spread_hard_max_points", 50)
        self._spread_spike_mult: float = getattr(config, "spread_spike_multiplier", 2.5)
        self._min_sl_buffer_points: int = getattr(config, "min_sl_buffer_points", 5)

        # --- Internal state ---
        self._spread_ema: float = 0.0
        self._spread_history: deque = deque(maxlen=self.SPREAD_HISTORY_SIZE)
        self._ema_initialized: bool = False

        # Cached raw values from MT5
        self._stops_level: int = 0
        self._freeze_level: int = 0
        self._volume_min: float = 0.01
        self._volume_max: float = 100.0
        self._volume_step: float = 0.01
        self._tick_value: float = 0.745
        self._tick_size: float = 0.01
        self._contract_size: float = 100.0
        self._point: float = 0.01
        self._digits: int = 2

        # Spread
        self._spread_current: float = 0.0
        self._spread_current_points: int = 0

        # Timestamps (monotonic)
        self._last_heavy: float = 0.0
        self._last_light: float = 0.0

    # ------------------------------------------------------------------
    # Refresh methods (called from main loop, no threads)
    # ------------------------------------------------------------------

    def heavy_refresh(self, force: bool = False) -> bool:
        """Full symbol_info refresh.  Call on connect and every 10 min.

        Returns True if refresh was performed.
        """
        now = time.monotonic()
        if not force and (now - self._last_heavy) < self._heavy_interval:
            return False

        try:
            # Route through the bridge so native (Windows) and mt5linux
            # (Linux/Wine) transports both work. Direct `import MetaTrader5`
            # fails on Linux even when the bridge is providing access.
            from core.mt5_bridge import mt5
            if mt5 is None:
                return False  # Offline mode — no constraints to refresh

            broker_sym = self._bridge._broker_name(self._symbol)
            sym = mt5.symbol_info(broker_sym)
            if sym is None:
                log.warning(
                    "BrokerConstraints heavy_refresh: symbol_info(%s) returned None",
                    broker_sym,
                )
                return False

            old_stops = self._stops_level
            old_tv = self._tick_value

            self._stops_level = sym.trade_stops_level
            self._freeze_level = sym.trade_freeze_level
            self._volume_min = sym.volume_min
            self._volume_max = sym.volume_max
            self._volume_step = sym.volume_step
            self._tick_value = sym.trade_tick_value
            self._tick_size = sym.trade_tick_size
            self._contract_size = sym.trade_contract_size
            self._point = sym.point
            self._digits = sym.digits

            self._last_heavy = now

            # Log changes
            if old_stops != self._stops_level and old_stops != 0:
                log.warning(
                    "Broker stops_level CHANGED: %d -> %d",
                    old_stops, self._stops_level,
                )
            if abs(old_tv - self._tick_value) > 0.001 and old_tv != 0.745:
                log.info(
                    "Broker tick_value updated: %.5f -> %.5f",
                    old_tv, self._tick_value,
                )

            vpp = self._tick_value / self._tick_size if self._tick_size > 0 else 0
            log.info(
                "BrokerConstraints heavy_refresh: stops_level=%d  freeze=%d  "
                "tick_value=%.5f  VPP=%.4f  vol_min=%.3f  vol_step=%.3f",
                self._stops_level, self._freeze_level,
                self._tick_value, vpp,
                self._volume_min, self._volume_step,
            )
            return True

        except Exception:
            log.exception("BrokerConstraints heavy_refresh failed")
            return False

    def light_refresh(self, force: bool = False) -> bool:
        """Spread/tick refresh.  Call every 10s or every new bar.

        Returns True if refresh was performed.
        """
        now = time.monotonic()
        if not force and (now - self._last_light) < self._light_interval:
            return False

        try:
            # Route through the bridge — see heavy_refresh comment above.
            from core.mt5_bridge import mt5
            if mt5 is None:
                return False  # Offline mode

            broker_sym = self._bridge._broker_name(self._symbol)
            sym = mt5.symbol_info(broker_sym)
            if sym is None:
                return False

            # Spread in integer points and price
            self._spread_current_points = sym.spread
            self._spread_current = sym.spread * sym.point

            # Also update tick_value (can shift with FX rates on non-USD accounts)
            new_tv = sym.trade_tick_value
            if abs(new_tv - self._tick_value) > 0.0001:
                log.info(
                    "Tick value drift: %.5f -> %.5f", self._tick_value, new_tv,
                )
                self._tick_value = new_tv

            # Update spread EMA
            if not self._ema_initialized:
                self._spread_ema = self._spread_current
                self._ema_initialized = True
            else:
                a = self.SPREAD_EMA_ALPHA
                self._spread_ema = a * self._spread_current + (1 - a) * self._spread_ema

            # Track spread history for max-1h
            self._spread_history.append(self._spread_current)

            self._last_light = now
            return True

        except Exception:
            log.exception("BrokerConstraints light_refresh failed")
            return False

    # ------------------------------------------------------------------
    # Spread gate
    # ------------------------------------------------------------------

    def check_spread_gate(self) -> tuple:
        """Check if current spread allows entry.

        Returns:
            (allowed, reason) -- reason is ``"ok"`` when allowed.
        """
        if self._spread_current <= 0:
            return True, "ok"  # No spread data yet, fail-open

        # Hard max (in integer points)
        if self._spread_current_points > self._spread_hard_max_points:
            return False, (
                f"high_spread ({self._spread_current_points}pts "
                f"> {self._spread_hard_max_points}pts)"
            )

        # Spike detection (current vs EMA)
        if self._ema_initialized and self._spread_ema > 0:
            ratio = self._spread_current / self._spread_ema
            if ratio > self._spread_spike_mult:
                return False, (
                    f"spread_spike ({self._spread_current:.2f} = "
                    f"{ratio:.1f}x EMA {self._spread_ema:.2f})"
                )

        return True, "ok"

    # ------------------------------------------------------------------
    # Min SL enforcement
    # ------------------------------------------------------------------

    def enforce_min_sl(
        self,
        side: str,
        entry_price: float,
        sl_price: float,
    ) -> float:
        """Widen SL if it violates broker minimum distance.

        min_distance = (stops_level + spread + buffer) * point

        Args:
            side: ``"LONG"`` or ``"SHORT"``
            entry_price: The entry price.
            sl_price: The proposed SL price.

        Returns:
            Adjusted SL price (widened if necessary, otherwise unchanged).
        """
        min_distance_points = (
            self._stops_level
            + self._spread_current_points
            + self._min_sl_buffer_points
        )
        min_distance_price = min_distance_points * self._point

        if side == "LONG":
            sl_distance = entry_price - sl_price
            if sl_distance < min_distance_price:
                new_sl = entry_price - min_distance_price
                log.warning(
                    "Min SL enforcement: LONG SL %.2f -> %.2f  "
                    "(dist %.1f pts < min %d pts "
                    "[stops=%d + spread=%d + buf=%d])",
                    sl_price, new_sl,
                    sl_distance / self._point, min_distance_points,
                    self._stops_level, self._spread_current_points,
                    self._min_sl_buffer_points,
                )
                return new_sl
        else:  # SHORT
            sl_distance = sl_price - entry_price
            if sl_distance < min_distance_price:
                new_sl = entry_price + min_distance_price
                log.warning(
                    "Min SL enforcement: SHORT SL %.2f -> %.2f  "
                    "(dist %.1f pts < min %d pts "
                    "[stops=%d + spread=%d + buf=%d])",
                    sl_price, new_sl,
                    sl_distance / self._point, min_distance_points,
                    self._stops_level, self._spread_current_points,
                    self._min_sl_buffer_points,
                )
                return new_sl

        return sl_price  # No adjustment needed

    # ------------------------------------------------------------------
    # Snapshot for dashboard / external reads
    # ------------------------------------------------------------------

    def get_snapshot(self) -> BrokerSnapshot:
        """Return a frozen snapshot of all constraint values."""
        spread_max_1h = max(self._spread_history) if self._spread_history else 0.0
        vpp = self._tick_value / self._tick_size if self._tick_size > 0 else 0.0

        min_sl_points = (
            self._stops_level
            + self._spread_current_points
            + self._min_sl_buffer_points
        )

        return BrokerSnapshot(
            volume_min=self._volume_min,
            volume_max=self._volume_max,
            volume_step=self._volume_step,
            stops_level=self._stops_level,
            freeze_level=self._freeze_level,
            tick_value=self._tick_value,
            tick_size=self._tick_size,
            contract_size=self._contract_size,
            point=self._point,
            digits=self._digits,
            value_per_point=vpp,
            spread_current=self._spread_current,
            spread_current_points=self._spread_current_points,
            spread_ema=self._spread_ema,
            spread_max_1h=spread_max_1h,
            min_sl_distance=min_sl_points * self._point,
            last_heavy_refresh=self._last_heavy,
            last_light_refresh=self._last_light,
        )

    # ------------------------------------------------------------------
    # Properties for direct access
    # ------------------------------------------------------------------

    @property
    def tick_value(self) -> float:
        return self._tick_value

    @property
    def point(self) -> float:
        return self._point

    @property
    def stops_level(self) -> int:
        return self._stops_level

    @property
    def spread_current_points(self) -> int:
        return self._spread_current_points

    @property
    def volume_min(self) -> float:
        return self._volume_min
