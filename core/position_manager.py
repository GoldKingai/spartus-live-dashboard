"""Position tracking and management for live trading."""
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class PositionManager:
    """Track open positions, trailing SL, recovery on restart."""

    def __init__(self, mt5_bridge, config):
        self.mt5 = mt5_bridge
        self.cfg = config
        self.position = None  # Current tracked position dict or None
        self._entry_step = 0

    def get_current_position(self):
        """Return current position dict or None."""
        return self.position

    def set_position(self, position_data: dict):
        """Set tracked position after opening."""
        self.position = position_data
        self._entry_step = position_data.get('entry_step', 0)
        log.info(f"Position set: {position_data.get('side')} "
                 f"{position_data.get('lots')} lots @ {position_data.get('entry_price')}")

    def clear_position(self):
        """Mark no position (after close)."""
        self.position = None
        log.info("Position cleared")

    def update_from_mt5(self):
        """Sync with MT5 positions for recovery on restart.
        If MT5 has an open XAUUSD position we don't know about, adopt it."""
        try:
            positions = self.mt5.get_open_positions(self.cfg.mt5_symbol)
            if positions and not self.position:
                pos = positions[0]  # Take first XAUUSD position
                # MT5 bridge returns 'type' as int (0=BUY, 1=SELL)
                # Convert to "LONG"/"SHORT" strings for consistency
                pos_type = pos.get('type', 0)
                side_str = "LONG" if pos_type == 0 else "SHORT"
                self.position = {
                    'ticket': pos['ticket'],
                    'side': side_str,
                    'entry_price': pos['price_open'],
                    'lots': pos['volume'],
                    'stop_loss': pos.get('sl', 0),
                    'take_profit': pos.get('tp', 0),
                    'conviction': 0.5,  # Unknown for recovered positions
                    'entry_step': 0,
                    'entry_time': datetime.now(timezone.utc),
                }
                log.warning(f"Recovered position from MT5: {self.position}")
            elif not positions and self.position:
                # Position was closed externally (SL/TP hit by broker)
                log.warning("Position closed externally (SL/TP hit by broker)")
                self.position = None
        except Exception as e:
            log.error(f"Failed to sync with MT5 positions: {e}")

    def get_bars_held(self, current_step: int) -> int:
        if not self.position:
            return 0
        return current_step - self.position.get('entry_step', current_step)

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized P/L using tick_value math."""
        if not self.position:
            return 0.0
        entry = self.position['entry_price']
        lots = self.position['lots']
        side = self.position['side']

        sym_info = self.mt5.get_symbol_info()
        tick_value = sym_info.get('tick_value', 1.0)
        tick_size = sym_info.get('tick_size', 0.01)
        value_per_point = tick_value / tick_size

        if side == "LONG":
            price_move = current_price - entry
        else:
            price_move = entry - current_price

        ticks = price_move / tick_size
        pnl = ticks * tick_value * lots
        return pnl

    def has_position(self) -> bool:
        return self.position is not None

    def get_position_duration_minutes(self) -> float:
        if not self.position:
            return 0.0
        entry_time = self.position.get('entry_time')
        if entry_time:
            delta = datetime.now(timezone.utc) - entry_time
            return delta.total_seconds() / 60.0
        return 0.0
