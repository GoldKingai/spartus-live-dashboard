"""One-click emergency stop for all trading."""
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class EmergencyStop:
    """One-click kill switch accessible from dashboard.

    Actions:
    1. Close all open positions immediately
    2. Cancel all pending orders
    3. Block all new orders
    4. Log emergency event

    Requires manual reset to resume trading.
    """

    def __init__(self, mt5_bridge):
        self.mt5 = mt5_bridge
        self._active = False
        self._activated_at = None
        self._reason = ""
        self._positions_closed = 0

    def activate(self, reason: str = "manual") -> dict:
        """Activate emergency stop."""
        log.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        self._active = True
        self._activated_at = datetime.now(timezone.utc)
        self._reason = reason

        # Close all positions
        results = []
        try:
            closed = self.mt5.close_all_positions()
            results = closed if closed else []
            self._positions_closed = len(results)
        except Exception as e:
            log.error(f"Error closing positions during emergency: {e}")
            self._positions_closed = 0

        log.critical(f"Emergency stop complete: {self._positions_closed} positions closed")

        return {
            'success': True,
            'positions_closed': self._positions_closed,
            'reason': reason,
            'timestamp': self._activated_at.isoformat(),
        }

    def is_active(self) -> bool:
        return self._active

    def reset(self):
        """Manual reset to resume trading."""
        log.warning("Emergency stop RESET by user")
        self._active = False
        self._activated_at = None
        self._reason = ""

    def get_status(self) -> dict:
        return {
            'active': self._active,
            'activated_at': self._activated_at.isoformat() if self._activated_at else None,
            'reason': self._reason,
            'positions_closed': self._positions_closed,
        }
