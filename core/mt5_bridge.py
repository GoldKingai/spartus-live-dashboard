"""MT5Bridge -- MetaTrader 5 connection, data retrieval, and order execution.

Provides a single entry-point for the Live Dashboard to interact with
the MT5 terminal.  Handles:

* Connection / reconnection with heartbeat monitoring
* Account and symbol introspection (auto-detects currency, tick_value, etc.)
* Market-data retrieval (bars, spread, tick_value)
* Order execution (market orders, modify SL/TP, close positions)
* Emergency shutdown via callback

Usage:
    from config.live_config import LiveConfig
    from core.mt5_bridge import MT5Bridge

    bridge = MT5Bridge(LiveConfig())
    if bridge.connect():
        info = bridge.get_account_info()
        bars = bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_M5, 500)
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import MetaTrader5 as mt5
import pandas as pd

from config.live_config import LiveConfig
from utils.symbol_mapper import build_resolved_map, resolve_symbol

log = logging.getLogger(__name__)

# Magic number used to tag all orders placed by this system
MAGIC_NUMBER = 234000

# Heartbeat cadence
_HEARTBEAT_INTERVAL_S = 5
_MAX_RECONNECT_ATTEMPTS = 6


class MT5Bridge:
    """Manages the full lifecycle of the MT5 terminal connection."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: LiveConfig) -> None:
        self._config = config

        # Connection state
        self._initialized: bool = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()

        # Resolved symbol map  (canonical -> broker name)
        self._resolved_symbols: Dict[str, str] = {}

        # Account / symbol metadata populated by _detect_account_setup
        self.account_currency: str = "USD"
        self.tick_value: float = 1.0
        self.tick_size: float = 0.01
        self.contract_size: float = 100.0
        self.value_per_point: float = 100.0  # tick_value / tick_size

        # Emergency callback -- set externally (e.g. by the safety module)
        self.on_emergency_stop: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the MT5 terminal and validate the environment.

        Returns:
            True if the terminal is connected, account detected, and
            all required symbols are available (or resolved via
            alternatives).
        """
        kwargs: Dict[str, Any] = {}
        if self._config.mt5_terminal_path:
            kwargs["path"] = self._config.mt5_terminal_path

        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            log.error("MT5 initialize failed: %s", error)
            return False

        self._initialized = True
        log.info("MT5 terminal initialised successfully")

        # Detect account parameters
        self._detect_account_setup()

        # Validate and resolve symbols
        self._validate_symbols()

        # Start background heartbeat
        self._start_heartbeat()

        return True

    def _detect_account_setup(self) -> None:
        """Auto-detect account currency, tick_value, tick_size, etc.

        Reads live values from the MT5 terminal so the system
        automatically adapts to the connected broker / account.
        """
        # ---- Account currency ----
        acct = mt5.account_info()
        if acct is not None:
            self.account_currency = acct.currency
            log.info(
                "Account detected: %s | currency=%s | leverage=%d | server=%s",
                acct.name,
                acct.currency,
                acct.leverage,
                acct.server,
            )
        else:
            log.warning("Could not read account_info; defaulting to USD")

        # ---- Symbol tick info for the primary symbol ----
        primary = self._config.mt5_symbol
        sym = mt5.symbol_info(primary)
        if sym is None:
            # Try to select the symbol first (some brokers need this)
            mt5.symbol_select(primary, True)
            sym = mt5.symbol_info(primary)

        if sym is not None:
            self.tick_value = sym.trade_tick_value
            self.tick_size = sym.trade_tick_size
            self.contract_size = sym.trade_contract_size

            if self.tick_size > 0:
                self.value_per_point = self.tick_value / self.tick_size
            else:
                self.value_per_point = self.tick_value
                log.warning("tick_size is zero for %s; value_per_point may be wrong", primary)

            log.info(
                "Symbol %s: tick_value=%.5f tick_size=%.5f contract_size=%.1f "
                "value_per_point=%.4f",
                primary,
                self.tick_value,
                self.tick_size,
                self.contract_size,
                self.value_per_point,
            )
        else:
            log.warning(
                "Cannot retrieve symbol_info for %s; using defaults "
                "(tick_value=1.0, tick_size=0.01)",
                primary,
            )

    def _validate_symbols(self) -> None:
        """Check that all required symbols are available on this broker.

        Uses the symbol_mapper to resolve alternative names when a
        canonical symbol is not found.
        """
        # Collect every symbol the terminal knows about
        all_symbols_raw = mt5.symbols_get()
        if all_symbols_raw is None:
            log.warning("Could not retrieve symbol list from MT5")
            return

        available: Set[str] = {s.name for s in all_symbols_raw}
        log.info("MT5 terminal reports %d symbols", len(available))

        # Ensure every symbol is selected (visible) in Market Watch
        # Build resolved map
        self._resolved_symbols = build_resolved_map(
            available, self._config.symbol_map
        )

        # Also ensure the primary symbol is in the resolved map
        primary = self._config.mt5_symbol
        if primary not in self._resolved_symbols:
            broker_name = resolve_symbol(primary, available, self._config.symbol_map)
            if broker_name:
                self._resolved_symbols[primary] = broker_name

        # Select all resolved symbols in Market Watch
        for canonical, broker_name in self._resolved_symbols.items():
            if not mt5.symbol_select(broker_name, True):
                log.warning(
                    "Failed to select %s (%s) in Market Watch",
                    canonical,
                    broker_name,
                )
            else:
                log.info("Symbol resolved: %s -> %s", canonical, broker_name)

        # Report any symbols that could NOT be resolved
        from utils.symbol_mapper import SYMBOL_MAP_DEFAULT

        for canonical in SYMBOL_MAP_DEFAULT:
            if canonical not in self._resolved_symbols:
                log.warning(
                    "Symbol %s could not be resolved on this broker "
                    "(checked alternatives too)",
                    canonical,
                )

    # ------------------------------------------------------------------
    # Broker name resolution helper
    # ------------------------------------------------------------------

    def _broker_name(self, canonical: str) -> str:
        """Return the broker-specific symbol name for a canonical name.

        Falls back to the canonical name itself if no mapping exists.
        """
        return self._resolved_symbols.get(canonical, canonical)

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, Any]:
        """Return key account fields as a plain dict.

        Keys: currency, balance, equity, margin, free_margin, leverage,
        server, name.
        """
        acct = mt5.account_info()
        if acct is None:
            log.error("get_account_info: account_info() returned None")
            return {}

        return {
            "currency": acct.currency,
            "balance": acct.balance,
            "equity": acct.equity,
            "margin": acct.margin,
            "free_margin": acct.margin_free,
            "leverage": acct.leverage,
            "server": acct.server,
            "name": acct.name,
        }

    # ------------------------------------------------------------------
    # Symbol information
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Return symbol specification fields as a plain dict.

        Keys: tick_value, tick_size, contract_size, volume_min,
        volume_max, volume_step, point, spread, digits.
        """
        broker_sym = self._broker_name(symbol)
        sym = mt5.symbol_info(broker_sym)
        if sym is None:
            log.error("get_symbol_info: symbol_info(%s) returned None", broker_sym)
            return {}

        return {
            "tick_value": sym.trade_tick_value,
            "tick_size": sym.trade_tick_size,
            "contract_size": sym.trade_contract_size,
            "volume_min": sym.volume_min,
            "volume_max": sym.volume_max,
            "volume_step": sym.volume_step,
            "point": sym.point,
            "spread": sym.spread,
            "digits": sym.digits,
        }

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_latest_bars(
        self,
        symbol: str,
        timeframe: int,
        count: int,
    ) -> pd.DataFrame:
        """Fetch the most recent *count* bars for *symbol* / *timeframe*.

        Args:
            symbol:    Canonical symbol name (e.g. "XAUUSD").
            timeframe: MT5 timeframe constant (e.g. mt5.TIMEFRAME_M5).
            count:     Number of bars to retrieve.

        Returns:
            DataFrame with columns: time, open, high, low, close, volume.
            ``time`` is a UTC datetime.  Returns an empty DataFrame on
            failure.
        """
        broker_sym = self._broker_name(symbol)
        rates = mt5.copy_rates_from_pos(broker_sym, timeframe, 0, count)

        if rates is None or len(rates) == 0:
            log.warning(
                "get_latest_bars: no data for %s (%s) tf=%d count=%d",
                symbol,
                broker_sym,
                timeframe,
                count,
            )
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # MT5 returns tick_volume -- rename to the canonical 'volume'
        if "tick_volume" in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Keep only the columns we need (drop spread, real_volume if present)
        keep = ["time", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]

        return df

    # ------------------------------------------------------------------
    # Spread & tick value
    # ------------------------------------------------------------------

    def get_current_spread(self, symbol: str = "XAUUSD") -> float:
        """Return the current spread for *symbol* in price points.

        E.g. for XAUUSD with digits=2 and spread=21, this returns 0.21.
        """
        broker_sym = self._broker_name(symbol)
        sym = mt5.symbol_info(broker_sym)
        if sym is None:
            log.warning("get_current_spread: no symbol_info for %s", broker_sym)
            return 0.0

        # sym.spread is in integer "points" (smallest price increment units)
        return sym.spread * sym.point

    def get_tick_value(self, symbol: str = "XAUUSD") -> float:
        """Return the live trade_tick_value in account currency."""
        broker_sym = self._broker_name(symbol)
        sym = mt5.symbol_info(broker_sym)
        if sym is None:
            log.warning("get_tick_value: no symbol_info for %s", broker_sym)
            return self.tick_value  # fallback to cached
        return sym.trade_tick_value

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_open_positions(self, symbol: str = "XAUUSD") -> List[Dict[str, Any]]:
        """Return all open positions for *symbol* as a list of dicts.

        Each dict contains: ticket, type (0=BUY,1=SELL), volume, price_open,
        sl, tp, profit, swap, time, magic, comment.
        """
        broker_sym = self._broker_name(symbol)
        positions = mt5.positions_get(symbol=broker_sym)

        if positions is None:
            return []

        result: List[Dict[str, Any]] = []
        for pos in positions:
            result.append(
                {
                    "ticket": pos.ticket,
                    "type": pos.type,  # 0 = BUY, 1 = SELL
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
                    "magic": pos.magic,
                    "comment": pos.comment,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def send_market_order(
        self,
        symbol: str,
        side: str,
        lots: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> Dict[str, Any]:
        """Place a market order.

        Args:
            symbol:  Canonical symbol name.
            side:    "BUY" or "SELL".
            lots:    Volume in lots.
            sl:      Stop-loss price (0.0 = none).
            tp:      Take-profit price (0.0 = none).
            comment: Order comment string.

        Returns:
            Dict with keys: success (bool), ticket (int or None),
            fill_price (float or None), error (str or None).
        """
        broker_sym = self._broker_name(symbol)
        sym = mt5.symbol_info(broker_sym)
        if sym is None:
            msg = f"Symbol {broker_sym} not found"
            log.error("send_market_order: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        # Determine order type and price
        if side.upper() == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = sym.ask
        elif side.upper() == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = sym.bid
        else:
            msg = f"Invalid side: {side}"
            log.error("send_market_order: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        log.info(
            "Sending %s %s %.3f lots @ %.2f  SL=%.2f  TP=%.2f",
            side.upper(),
            broker_sym,
            lots,
            price,
            sl,
            tp,
        )

        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            msg = f"order_send returned None: {error}"
            log.error("send_market_order: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"Order rejected: retcode={result.retcode} comment={result.comment}"
            log.error("send_market_order: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        log.info(
            "Order filled: ticket=%d fill_price=%.2f",
            result.order,
            result.price,
        )
        return {
            "success": True,
            "ticket": result.order,
            "fill_price": result.price,
            "error": None,
        }

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Modify the SL and/or TP of an existing position.

        Args:
            ticket: Position ticket number.
            sl:     New stop-loss price (None = keep current).
            tp:     New take-profit price (None = keep current).

        Returns:
            True if the modification was accepted by the server.
        """
        # Look up the current position to get symbol and current SL/TP
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            log.error("modify_position: position %d not found", ticket)
            return False

        pos = positions[0]
        new_sl = float(sl) if sl is not None else pos.sl
        new_tp = float(tp) if tp is not None else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            log.error("modify_position: order_send returned None: %s", error)
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                "modify_position: rejected retcode=%d comment=%s",
                result.retcode,
                result.comment,
            )
            return False

        log.info(
            "Position %d modified: SL=%.2f TP=%.2f",
            ticket,
            new_sl,
            new_tp,
        )
        return True

    def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close a position by ticket number.

        Returns:
            Dict with keys: success (bool), ticket (int or None),
            fill_price (float or None), error (str or None).
        """
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            msg = f"Position {ticket} not found"
            log.error("close_position: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        pos = positions[0]
        sym = mt5.symbol_info(pos.symbol)
        if sym is None:
            msg = f"Symbol {pos.symbol} info unavailable"
            log.error("close_position: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        # Close = opposite direction deal
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = sym.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = sym.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "position": ticket,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "spartus_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        log.info(
            "Closing position %d: %s %.3f lots @ %.2f",
            ticket,
            "SELL" if close_type == mt5.ORDER_TYPE_SELL else "BUY",
            pos.volume,
            price,
        )

        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            msg = f"order_send returned None: {error}"
            log.error("close_position: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"Close rejected: retcode={result.retcode} comment={result.comment}"
            log.error("close_position: %s", msg)
            return {"success": False, "ticket": None, "fill_price": None, "error": msg}

        log.info("Position %d closed at %.2f", ticket, result.price)
        return {
            "success": True,
            "ticket": result.order,
            "fill_price": result.price,
            "error": None,
        }

    def close_all_positions(self, symbol: str = "XAUUSD") -> List[Dict[str, Any]]:
        """Emergency close all open positions for *symbol*.

        Returns:
            List of result dicts from each close_position call.
        """
        positions = self.get_open_positions(symbol)
        if not positions:
            log.info("close_all_positions: no open positions for %s", symbol)
            return []

        log.warning(
            "EMERGENCY CLOSE: closing %d positions for %s",
            len(positions),
            symbol,
        )

        results: List[Dict[str, Any]] = []
        for pos in positions:
            res = self.close_position(pos["ticket"])
            results.append(res)

        return results

    # ------------------------------------------------------------------
    # Market status
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Check if the primary symbol is currently tradeable.

        Uses the symbol's session info and trade mode to determine
        whether orders can be placed right now.
        """
        broker_sym = self._broker_name(self._config.mt5_symbol)
        sym = mt5.symbol_info(broker_sym)
        if sym is None:
            log.warning("is_market_open: cannot get symbol_info for %s", broker_sym)
            return False

        # trade_mode: 0 = disabled, 2 = close_only, 4 = full
        # SYMBOL_TRADE_MODE_FULL = 4
        return sym.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

    # ------------------------------------------------------------------
    # Heartbeat / reconnection
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        """Start the background heartbeat thread."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return  # already running

        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="MT5Bridge-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        log.info("Heartbeat thread started")

    def _heartbeat_loop(self) -> None:
        """Periodically verify the MT5 connection.

        If the connection drops, attempts up to _MAX_RECONNECT_ATTEMPTS
        reconnections (with progressive back-off).  If all fail,
        triggers the emergency stop callback.
        """
        consecutive_failures = 0

        while not self._stop_heartbeat.is_set():
            self._stop_heartbeat.wait(timeout=_HEARTBEAT_INTERVAL_S)
            if self._stop_heartbeat.is_set():
                break

            # Check connection health
            if self._connected:
                consecutive_failures = 0
                continue

            # Connection lost -- attempt reconnect
            consecutive_failures += 1
            log.warning(
                "MT5 heartbeat: connection lost (attempt %d/%d)",
                consecutive_failures,
                _MAX_RECONNECT_ATTEMPTS,
            )

            # Progressive back-off: wait longer between retries
            backoff = min(consecutive_failures * 2, 10)
            time.sleep(backoff)

            # Try to reconnect
            kwargs: Dict[str, Any] = {}
            if self._config.mt5_terminal_path:
                kwargs["path"] = self._config.mt5_terminal_path

            if mt5.initialize(**kwargs):
                self._initialized = True
                log.info("MT5 heartbeat: reconnected successfully")
                consecutive_failures = 0
                continue

            if consecutive_failures >= _MAX_RECONNECT_ATTEMPTS:
                log.critical(
                    "MT5 heartbeat: %d reconnect attempts failed -- "
                    "triggering emergency stop",
                    _MAX_RECONNECT_ATTEMPTS,
                )
                if self.on_emergency_stop is not None:
                    try:
                        self.on_emergency_stop()
                    except Exception:
                        log.exception("Emergency stop callback raised an exception")
                break  # stop the heartbeat loop

    @property
    def _connected(self) -> bool:
        """Check if the MT5 terminal is currently reachable."""
        if not self._initialized:
            return False
        try:
            info = mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Cleanly shut down the MT5 connection and heartbeat thread."""
        log.info("MT5Bridge: disconnecting...")

        # Signal heartbeat to stop
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=10)
            self._heartbeat_thread = None

        # Shut down MT5
        if self._initialized:
            try:
                mt5.shutdown()
            except Exception:
                log.exception("Exception during mt5.shutdown()")
            self._initialized = False

        log.info("MT5Bridge: disconnected")
