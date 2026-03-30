"""Monitor MT5 connection health."""
import logging
import time

log = logging.getLogger(__name__)


class ConnectionMonitor:
    """Monitor MT5 connection health and latency."""

    def __init__(self, mt5_bridge):
        self.mt5 = mt5_bridge
        self._last_check = 0
        self._latency_ms = 0.0
        self._connected = False

    def check_connection(self) -> dict:
        """Return connection health status."""
        try:
            start = time.perf_counter()
            info = self.mt5.get_account_info()
            elapsed = (time.perf_counter() - start) * 1000

            self._connected = info is not None
            self._latency_ms = elapsed
            self._last_check = time.time()

            return {
                'connected': self._connected,
                'latency_ms': round(self._latency_ms, 1),
                'reconnect_attempts': getattr(self.mt5, '_reconnect_attempts', 0),
                'last_check': self._last_check,
            }
        except Exception as e:
            log.error(f"Connection check failed: {e}")
            self._connected = False
            return {
                'connected': False,
                'latency_ms': 0,
                'reconnect_attempts': getattr(self.mt5, '_reconnect_attempts', 0),
                'last_check': self._last_check,
                'error': str(e),
            }

    def get_latency(self) -> float:
        return self._latency_ms

    def is_connected(self) -> bool:
        return self._connected
