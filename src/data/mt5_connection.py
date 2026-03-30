"""MT5 Connection Manager — process-level singleton with reference counting.

Ensures that training, live fine-tuning, and any future component can all
share one MT5 connection without stepping on each other:

  - Each component calls MT5Connection.acquire() to get a handle.
  - The first acquire() calls mt5.initialize().
  - Each release() decrements the ref count; the last release() calls mt5.shutdown().
  - If a component crashes without releasing, the connection stays open (safe).

Thread-safe: all operations are protected by a module-level lock.

Usage:
    handle = MT5Connection.acquire(terminal_path=None, symbol_map=None)
    try:
        rates = handle.copy_rates_range(...)
    finally:
        handle.release()

Or as context manager:
    with MT5Connection.acquire() as conn:
        rates = conn.copy_rates_range(...)
"""

import logging
import threading
from typing import Dict, Optional

log = logging.getLogger(__name__)

_lock = threading.Lock()
_ref_count: int = 0
_initialized: bool = False
_mt5 = None   # The mt5 module, loaded once


def _load_mt5():
    """Import MetaTrader5 module. Returns it or None."""
    global _mt5
    if _mt5 is not None:
        return _mt5
    try:
        import MetaTrader5 as mt5
        _mt5 = mt5
        return mt5
    except ImportError:
        return None


class MT5Handle:
    """Thin wrapper returned by acquire(). Delegates all calls to global mt5 module."""

    def __init__(self, mt5_mod, symbol_map: Dict[str, str]):
        self._mt5 = mt5_mod
        self._symbol_map = symbol_map or {}
        self._released = False

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    # ── Connection lifecycle ────────────────────────────────────────────────

    def release(self):
        """Decrement ref count. Shuts down MT5 when the last handle is released."""
        if self._released:
            return
        self._released = True
        MT5Connection.release()

    # ── Symbol resolution ──────────────────────────────────────────────────

    def resolve(self, canonical: str) -> str:
        """Map canonical symbol name to broker name (e.g. US500 → SP500)."""
        return self._symbol_map.get(canonical, canonical)

    # ── MT5 passthrough ────────────────────────────────────────────────────

    def copy_rates_range(self, symbol, tf, start, end):
        canonical = self.resolve(symbol)
        self._mt5.symbol_select(canonical, True)
        return self._mt5.copy_rates_range(canonical, tf, start, end)

    def copy_rates_from(self, symbol, tf, start, count):
        canonical = self.resolve(symbol)
        self._mt5.symbol_select(canonical, True)
        return self._mt5.copy_rates_from(canonical, tf, start, count)

    def copy_rates_from_pos(self, symbol, tf, pos, count):
        canonical = self.resolve(symbol)
        self._mt5.symbol_select(canonical, True)
        return self._mt5.copy_rates_from_pos(canonical, tf, pos, count)

    def symbol_info(self, symbol):
        return self._mt5.symbol_info(self.resolve(symbol))

    def last_error(self):
        return self._mt5.last_error()

    def terminal_info(self):
        return self._mt5.terminal_info()


class MT5Connection:
    """Process-level MT5 connection singleton with reference counting."""

    @staticmethod
    def acquire(
        terminal_path: Optional[str] = None,
        symbol_map: Optional[Dict[str, str]] = None,
    ) -> Optional[MT5Handle]:
        """Get a handle to the shared MT5 connection.

        Initializes MT5 on first call. Subsequent calls reuse the existing
        connection. Returns None if MT5 is unavailable.

        Args:
            terminal_path: Path to terminal64.exe. None = use default.
            symbol_map: Broker name overrides (canonical → broker symbol).
        """
        global _ref_count, _initialized

        mt5 = _load_mt5()
        if mt5 is None:
            return None

        with _lock:
            if not _initialized:
                kwargs = {}
                if terminal_path:
                    kwargs["path"] = terminal_path
                if not mt5.initialize(**kwargs):
                    err = mt5.last_error()
                    log.error(f"MT5Connection: initialize failed: {err}")
                    return None
                _initialized = True
                log.info("MT5Connection: initialized (ref=1)")
            else:
                log.debug(f"MT5Connection: reusing existing connection (ref={_ref_count + 1})")

            _ref_count += 1

        return MT5Handle(mt5, symbol_map or {})

    @staticmethod
    def release():
        """Decrement ref count. Calls mt5.shutdown() when last handle released."""
        global _ref_count, _initialized

        mt5 = _load_mt5()

        with _lock:
            if _ref_count <= 0:
                return
            _ref_count -= 1
            log.debug(f"MT5Connection: released (ref={_ref_count})")
            if _ref_count == 0 and _initialized and mt5 is not None:
                mt5.shutdown()
                _initialized = False
                log.info("MT5Connection: shutdown (no more holders)")

    @staticmethod
    def is_connected() -> bool:
        """Return True if MT5 is currently initialized."""
        with _lock:
            return _initialized

    @staticmethod
    def ref_count() -> int:
        """Current number of active handles."""
        with _lock:
            return _ref_count
