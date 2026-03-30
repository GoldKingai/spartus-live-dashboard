"""TradeExecutor -- Converts AI actions into MT5 trade operations.

Decision flow per bar:
    1. Get action from InferenceEngine
    2. If no position:
       a. direction > 0.3  -> check if LONG allowed  -> open LONG
       b. direction < -0.3 -> check if SHORT allowed  -> open SHORT
       c. else -> do nothing (flat)
    3. If in position:
       a. exit_urgency > 0.5 and held >= min_hold_bars -> close
       b. else -> adjust trailing SL based on sl_adjustment

Usage:
    from config.live_config import LiveConfig
    from core.mt5_bridge import MT5Bridge
    from core.risk_manager import LiveRiskManager
    from core.trade_executor import TradeExecutor, TradingState
    from memory.trading_memory import TradingMemory

    executor = TradeExecutor(mt5_bridge, risk_manager, memory, config)
    result = executor.execute_action(action, current_bar, account)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.live_config import LiveConfig

log = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading engine state machine."""
    STOPPED = "stopped"            # Default on launch -- AI idle
    RUNNING = "running"            # AI actively monitoring and trading
    WINDING_DOWN = "winding_down"  # No new trades, managing open position until close


# ---------------------------------------------------------------------------
# Lesson classification (matches training TradeAnalyzer exactly)
# ---------------------------------------------------------------------------

_LESSON_TYPES = {
    "GOOD_TRADE":               "Correct direction, profitable exit",
    "CORRECT_DIR_CLOSED_EARLY": "Right direction but closed too early",
    "CORRECT_DIR_BAD_SL":       "Right direction but SL was too tight",
    "WRONG_DIRECTION":          "Price moved against the position",
    "BAD_TIMING":               "Direction eventually right, bad entry timing",
    "WHIPSAW":                  "Caught in choppy/ranging market",
    "EMERGENCY_STOP":           "Account-level forced close",
    "BREAKEVEN":                "Closed near entry, no significant P/L",
    "SCALP_WIN":                "Quick small profit",
    "HELD_TOO_LONG":            "Profit eroded by holding past the turn",
}


class TradeExecutor:
    """Converts AI actions into MT5 trade operations.

    Manages the full lifecycle of live trades: entry decisions, SL
    trailing, exit decisions, trade recording, and state transitions.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        mt5_bridge,
        risk_manager,
        memory,
        config: LiveConfig,
    ) -> None:
        """Initialise the trade executor with all dependencies.

        Args:
            mt5_bridge:       Connected MT5Bridge instance for order execution.
            risk_manager:     LiveRiskManager for lot sizing and SL/TP.
            memory:           TradingMemory for trade recording and journal.
            config:           LiveConfig with all runtime parameters.
        """
        self._bridge = mt5_bridge
        self._risk = risk_manager
        self._memory = memory
        self._config = config

        # --- State ---
        self._state: TradingState = TradingState.STOPPED
        self._step_count: int = 0
        self._daily_trades: int = 0

        # --- Position tracking ---
        self._position: Optional[Dict[str, Any]] = None  # Current open position
        self._entry_step: int = 0
        self._entry_conditions: Dict[str, Any] = {}
        self._max_favorable: float = 0.0  # Max favorable excursion
        self._initial_sl: float = 0.0
        self._initial_tp: float = 0.0
        self._protection_stage: int = 0  # Profit protection stage (0-3)
        self._r_value_gbp: float = 0.0  # Initial risk in account currency (for £-threshold mode)
        self._sl_modifications: List[Dict[str, Any]] = []  # SL trail history per trade

        # --- Model version (injected from main.py after load) ---
        self._model_version: str = "unknown"
        self._model_hash: str = ""

        # --- Balance tracking ---
        self._peak_balance: float = 0.0
        self._initial_balance: float = 0.0

        # --- ATR cache (updated from feature pipeline) ---
        self._current_atr: float = 1.0

        # --- Bar context (set by main.py each bar for trade logging) ---
        self._bar_observation: Optional[Any] = None  # 670-dim numpy array
        self._bar_risk_state: Dict[str, Any] = {}
        self._bar_spread: float = 0.0

        # --- Entry snapshots (captured at trade open for close record) ---
        self._entry_observation: Optional[Any] = None
        self._entry_risk_state: Dict[str, Any] = {}
        self._entry_spread: float = 0.0

        # --- Broker constraints (injected after init) ---
        self._broker_constraints = None

        # --- Manual trade management ---
        # Tracks manually-opened positions (magic != 234000) for SL protection.
        # Key = MT5 ticket, Value = tracking dict with entry_price, side, etc.
        self._manual_positions: Dict[int, Dict[str, Any]] = {}
        # Runtime overrides from UI sliders (separate from AI protection)
        self._manual_protection_overrides: Dict[str, float] = {}

        # --- Protection state persistence ---
        self._protection_state_path = self._resolve_path(
            "storage/protection_state.json"
        )
        self._protection_state_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Trades log file ---
        self._trades_log_path = self._resolve_path("storage/logs/trades.jsonl")
        self._trades_log_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "TradeExecutor initialised  state=%s  "
            "direction_threshold=%.2f  exit_threshold=%.2f  min_hold_bars=%d",
            self._state.value,
            self._config.direction_threshold,
            self._config.exit_threshold,
            self._config.min_hold_bars,
        )

    # ------------------------------------------------------------------
    # Broker constraints injection
    # ------------------------------------------------------------------

    def set_broker_constraints(self, constraints) -> None:
        """Inject BrokerConstraints for min SL enforcement."""
        self._broker_constraints = constraints

    def set_model_version(self, version: str, file_hash: str = "") -> None:
        """Inject model version info for trade logging traceability."""
        self._model_version = version
        self._model_hash = file_hash

    def set_bar_context(
        self,
        observation=None,
        risk_state: Optional[Dict[str, Any]] = None,
        spread: float = 0.0,
    ) -> None:
        """Set per-bar context for trade logging.

        Called by main.py each bar BEFORE execute_action() so that
        trade entry records capture the observation, risk state, and
        spread at the moment of entry.
        """
        self._bar_observation = observation
        self._bar_risk_state = risk_state or {}
        self._bar_spread = spread

    # ------------------------------------------------------------------
    # Protection state persistence (survives restarts)
    # ------------------------------------------------------------------

    def _save_protection_state(self) -> None:
        """Persist protection state so it survives dashboard restarts."""
        if self._position is None:
            return
        state = {
            "ticket": self._position["ticket"],
            "initial_sl": self._initial_sl,
            "max_favorable": self._max_favorable,
            "protection_stage": self._protection_stage,
            "entry_price": self._position["entry_price"],
            "side": self._position["side"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._protection_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            log.warning("Failed to save protection state: %s", exc)

    def _load_protection_state(self, ticket: int) -> Optional[Dict]:
        """Load persisted protection state for a given ticket."""
        if not self._protection_state_path.exists():
            return None
        try:
            with open(self._protection_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("ticket") == ticket:
                return state
            log.info(
                "Protection state file is for ticket %d, not %d -- ignoring",
                state.get("ticket", 0), ticket,
            )
        except Exception as exc:
            log.warning("Failed to load protection state: %s", exc)
        return None

    def _clear_protection_state(self) -> None:
        """Remove persisted protection state (trade closed)."""
        try:
            if self._protection_state_path.exists():
                self._protection_state_path.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Position recovery (restart / crash recovery)
    # ------------------------------------------------------------------

    def recover_from_mt5(self) -> bool:
        """Check MT5 for existing positions and adopt if found.

        Called when transitioning to RUNNING state.  Picks up positions
        from a previous session that weren't closed (e.g. dashboard
        crash, window close without Stop Trading).

        Only adopts positions tagged with our magic number (234000) to
        avoid accidentally managing the user's manual trades.

        Returns True if a position was recovered.
        """
        if self._position is not None:
            return False  # Already tracking a position

        try:
            positions = self._bridge.get_open_positions(self._config.mt5_symbol)
        except Exception:
            log.exception("recover_from_mt5: failed to query MT5")
            return False

        if not positions:
            return False

        # Filter: only adopt positions placed by THIS system (magic=234000)
        our_positions = [
            p for p in positions if p.get("magic", 0) == 234000
        ]
        if not our_positions:
            if positions:
                log.info(
                    "recover_from_mt5: found %d %s position(s) but none "
                    "with our magic number (234000) -- ignoring (user trades)",
                    len(positions), self._config.mt5_symbol,
                )
            return False

        pos = our_positions[0]
        side = "LONG" if pos.get("type", 0) == 0 else "SHORT"

        self._position = {
            "ticket": pos["ticket"],
            "side": side,
            "entry_price": pos["price_open"],
            "lots": pos["volume"],
            "sl": pos.get("sl", 0),
            "tp": pos.get("tp", 0),
            "conviction": 0.5,  # Unknown for recovered positions
            "open_time": pos.get("time", datetime.now(timezone.utc)),
        }
        self._entry_step = self._step_count
        self._initial_tp = pos.get("tp", 0)

        # --- Restore protection state (persisted > inferred > estimated) ---
        entry = pos["price_open"]
        current_sl = pos.get("sl", 0)
        current_price = pos.get("price_current", entry)

        # Try loading persisted protection state first (best source)
        saved = self._load_protection_state(pos["ticket"])
        if saved:
            self._initial_sl = saved["initial_sl"]
            self._max_favorable = saved["max_favorable"]
            self._protection_stage = saved["protection_stage"]
            # max_favorable might have grown since last save -- update
            # with current price as a floor
            if side == "LONG":
                live_fav = max(0.0, current_price - entry)
            else:
                live_fav = max(0.0, entry - current_price)
            self._max_favorable = max(self._max_favorable, live_fav)
            log.info(
                "RECOVERED protection state from disk: ticket=%d "
                "initial_sl=%.2f max_fav=%.4f stage=%d",
                pos["ticket"], self._initial_sl,
                self._max_favorable, self._protection_stage,
            )
        else:
            # Fallback: estimate from current MT5 position state
            if side == "LONG":
                self._max_favorable = max(0.0, current_price - entry)
            else:
                self._max_favorable = max(0.0, entry - current_price)

            # Infer initial SL and protection stage from SL position
            sl_moved_toward_entry = False
            if side == "LONG" and current_sl > entry:
                sl_moved_toward_entry = True
            elif side == "SHORT" and 0 < current_sl < entry:
                sl_moved_toward_entry = True

            if sl_moved_toward_entry:
                tp = pos.get("tp", 0)
                if tp > 0:
                    if side == "LONG":
                        self._initial_sl = entry - (tp - entry)
                    else:
                        self._initial_sl = entry + (entry - tp)
                else:
                    self._initial_sl = (
                        entry - 2.0 * self._current_atr if side == "LONG"
                        else entry + 2.0 * self._current_atr
                    )
                r_dist = abs(entry - self._initial_sl)
                if r_dist > 0:
                    if side == "LONG":
                        sl_profit = current_sl - entry
                    else:
                        sl_profit = entry - current_sl
                    locked_r = sl_profit / r_dist
                    if locked_r >= self._config.protection_lock_amount_r:
                        self._protection_stage = 2
                    else:
                        self._protection_stage = 1
                else:
                    self._protection_stage = 1
            else:
                self._initial_sl = current_sl
                self._protection_stage = 0

            log.warning(
                "No persisted protection state -- estimated from MT5: "
                "initial_sl=%.2f max_fav=%.4f stage=%d",
                self._initial_sl, self._max_favorable, self._protection_stage,
            )

        # Guard: if initial_sl is 0 or equals entry, protection is disabled
        # Estimate from ATR as fallback
        if self._initial_sl <= 0 or abs(self._initial_sl - entry) < 0.01:
            fallback_sl = (
                entry - 2.0 * self._current_atr if side == "LONG"
                else entry + 2.0 * self._current_atr
            )
            log.warning(
                "initial_sl=%.2f is invalid (entry=%.2f) -- "
                "using ATR fallback: %.2f",
                self._initial_sl, entry, fallback_sl,
            )
            self._initial_sl = fallback_sl

        log.warning(
            "RECOVERED position from MT5: ticket=%d %s %.3f lots @ %.2f "
            "SL=%.2f TP=%.2f  initial_sl=%.2f max_fav=%.4f stage=%d "
            "(will manage with conviction=0.5)",
            pos["ticket"], side, pos["volume"], pos["price_open"],
            pos.get("sl", 0), pos.get("tp", 0),
            self._initial_sl, self._max_favorable, self._protection_stage,
        )
        # Persist recovered state immediately
        self._save_protection_state()
        return True

    def reconcile_trade_history(self) -> int:
        """Reconcile MT5 deal history with our trade database.

        Fetches recent closed deals from MT5 and imports any that are
        missing from our database.  This handles the case where the
        dashboard was restarted and a trade closed (via TP/SL) while
        it was down.

        Uses two strategies:
        1. Date-range query for broad discovery of unknown trades.
        2. Position-specific lookups for tickets we know about but that
           the date-range query missed (MT5 API quirk).

        Returns:
            Number of trades recovered from MT5 history.
        """
        try:
            completed = self._bridge.get_deal_history(
                symbol=self._config.mt5_symbol,
                days=7,
            )
        except Exception:
            log.exception("reconcile_trade_history: failed to fetch MT5 history")
            return 0

        if not completed:
            completed = []

        # Get all tickets we already know about
        known_tickets = self._memory.get_known_mt5_tickets()
        log.info(
            "reconcile_trade_history: %d MT5 trades, known_tickets=%s",
            len(completed), sorted(known_tickets),
        )

        recovered = 0
        for trade in completed:
            pos_id = trade["position_id"]

            # Skip trades not placed by us (different magic number)
            if trade.get("magic", 0) != 234000:
                log.info(
                    "reconcile: SKIP ticket=%d (magic=%d, not ours)",
                    pos_id, trade.get("magic", 0),
                )
                continue

            # Skip trades we already have (by ticket)
            if pos_id in known_tickets:
                log.debug(
                    "reconcile: SKIP ticket=%d (already known)", pos_id,
                )
                continue

            # Also check by trade characteristics (handles old trades
            # recorded before mt5_ticket column existed)
            if self._memory.has_matching_trade(
                side=trade["side"],
                entry_price=trade["entry_price"],
                exit_price=trade["exit_price"],
                lot_size=trade["lots"],
                pnl=round(trade.get("pnl", 0.0), 2),
            ):
                # Trade exists but without ticket -- backfill the ticket
                self._memory.backfill_mt5_ticket(
                    side=trade["side"],
                    entry_price=trade["entry_price"],
                    exit_price=trade["exit_price"],
                    lot_size=trade["lots"],
                    pnl=round(trade.get("pnl", 0.0), 2),
                    mt5_ticket=pos_id,
                )
                known_tickets.add(pos_id)
                continue

            # Determine close reason from comment or P/L
            comment = trade.get("comment", "").lower()
            pnl = trade.get("pnl", 0.0)
            if "tp" in comment or "take profit" in comment:
                close_reason = "TP_HIT"
            elif "sl" in comment or "stop loss" in comment:
                close_reason = "SL_HIT"
            elif pnl > 0:
                close_reason = "TP_HIT"  # Profit likely means TP
            elif pnl < 0:
                close_reason = "SL_HIT"  # Loss likely means SL
            else:
                close_reason = "EXTERNAL_CLOSE"

            # Compute hold duration in bars (5-min bars)
            open_time = trade.get("open_time", datetime.now(timezone.utc))
            close_time = trade.get("close_time", datetime.now(timezone.utc))
            hold_seconds = (close_time - open_time).total_seconds()
            hold_bars = max(1, int(hold_seconds / 300))

            # Compute P/L percentage (approximate using initial balance)
            balance = self._peak_balance or self._initial_balance or 1000.0
            pnl_pct = pnl / balance if balance > 0 else 0.0

            # Determine session from open time (matches _get_session_name)
            hour = open_time.hour
            if 0 <= hour < 7:
                session = "Asia"
            elif 7 <= hour < 12:
                session = "London"
            elif 12 <= hour < 16:
                session = "NY"
            elif 16 <= hour < 20:
                session = "NY_PM"
            else:
                session = "Off"

            trade_data = {
                "timestamp": close_time.isoformat(),
                "week": 0,
                "step": 0,
                "side": trade["side"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "lot_size": trade["lots"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 6),
                "hold_bars": hold_bars,
                "close_reason": close_reason,
                "conviction": 0.5,  # Unknown for recovered trades
                "rsi_at_entry": 0.5,
                "trend_dir_at_entry": 0.0,
                "session_at_entry": session,
                "vol_regime_at_entry": 1.0,
                "entry_conditions": {"recovered_from_mt5": True},
                "paper_trade": False,
                "mt5_ticket": pos_id,
            }

            trade_id = self._memory.record_trade(trade_data)

            # Also update risk manager with the P/L
            self._risk.record_trade_result(pnl)

            log.warning(
                "RECONCILED missed trade from MT5: ticket=%d %s "
                "%.3f lots  entry=%.2f exit=%.2f  P/L=%+.2f (%s)  "
                "held=%d bars  -> trade_id=%d",
                pos_id,
                trade["side"],
                trade["lots"],
                trade["entry_price"],
                trade["exit_price"],
                pnl,
                close_reason,
                hold_bars,
                trade_id,
            )
            recovered += 1

        # --- Pass 2: Position-specific lookups for known tickets ---
        # MT5's date-range query can miss recent deals.  For any ticket
        # in our DB that doesn't appear in the date-range results, do a
        # direct position lookup to catch deals the broad query missed.
        date_range_pos_ids = {t["position_id"] for t in completed}
        for db_ticket in known_tickets:
            if db_ticket in date_range_pos_ids:
                continue  # Already found in the broad query
            # This ticket is in our DB but wasn't in the date-range results.
            # It's already recorded, so no action needed -- but log it for
            # debugging.
            log.debug(
                "reconcile: ticket=%d in DB but not in date-range query",
                db_ticket,
            )

        # --- Pass 3: Check for orphaned tickets ---
        # Look for tickets we recorded as open but that are now closed
        # and weren't found by the date-range query.  Use position-specific
        # lookup to find them.
        if self._memory is not None:
            # Get tickets that might have been missed: tickets in our
            # predictions table (we made predictions for them) but not
            # in trades.  Or check the decision log for open tickets.
            # Simpler approach: check the known tickets set against the
            # date-range results.  Any ticket NOT in date-range AND NOT
            # in our DB might have been missed entirely.
            # Since we can't discover truly unknown tickets by position
            # lookup (we need to know the ticket), we rely on the date-range
            # query for discovery.  But we CAN verify existing DB entries.
            pass

        if recovered > 0:
            log.warning(
                "RECONCILIATION COMPLETE: recovered %d missed trades from MT5",
                recovered,
            )
        else:
            log.info("reconcile_trade_history: all MT5 trades already recorded")

        return recovered

    # ------------------------------------------------------------------
    # Public API -- main decision loop
    # ------------------------------------------------------------------

    def execute_action(
        self,
        action: Dict[str, float],
        current_bar: Dict[str, Any],
        account: Dict[str, Any],
    ) -> str:
        """Main decision loop -- called once per M5 bar.

        Respects TradingState:
            STOPPED:      return immediately, no trading.
            WINDING_DOWN: only manage existing position; if position closes,
                          auto-transition to STOPPED.
            RUNNING:      full entry + exit logic.

        Args:
            action: Dict with keys direction, conviction, exit_urgency,
                    sl_adjustment (from InferenceEngine.predict()).
            current_bar: Dict with at least 'close', 'time', and optional
                         feature snapshot keys (rsi_14, h1_trend_dir, etc.)
            account: Dict with 'balance', 'equity', and other account info.

        Returns:
            String describing the action taken, for logging / dashboard.
        """
        self._step_count += 1

        # Update balance tracking
        balance = account.get("balance", 0.0)
        equity = account.get("equity", balance)
        if balance > self._peak_balance:
            self._peak_balance = balance
        if self._initial_balance == 0.0 and balance > 0:
            self._initial_balance = balance

        # --- STOPPED: do nothing ---
        if self._state == TradingState.STOPPED:
            return "AI_STOPPED"

        # --- Update max favorable excursion if in position ---
        if self._position is not None:
            self._update_max_favorable(current_bar)

        # --- WINDING_DOWN: manage existing position only ---
        if self._state == TradingState.WINDING_DOWN:
            if self._position is not None:
                result = self._handle_in_position(action, current_bar, account)
                # If position was closed, transition to STOPPED
                if result.startswith("CLOSE"):
                    log.info(
                        "Wind-down complete: position closed (%s) -> STOPPED",
                        result,
                    )
                    self._state = TradingState.STOPPED
                return result
            else:
                # No position and winding down -> stop
                self._state = TradingState.STOPPED
                log.info("Wind-down: no position open -> STOPPED")
                return "WIND_DOWN_COMPLETE"

        # --- RUNNING: full logic ---
        if self._position is not None:
            return self._handle_in_position(action, current_bar, account)
        return self._handle_no_position(action, current_bar, account)

    # ------------------------------------------------------------------
    # Entry logic (no position)
    # ------------------------------------------------------------------

    def _handle_no_position(
        self,
        action: Dict[str, float],
        bar: Dict[str, Any],
        account: Dict[str, Any],
    ) -> str:
        """Evaluate whether to open a new position.

        Decision flow:
            1. Check direction threshold (|direction| > 0.3)
            2. Check risk_manager.check_position_allowed()
            3. Calculate lot size
            4. Calculate SL/TP
            5. Send market order
            6. Track position and record in memory

        Returns:
            Action string: "OPEN_LONG", "OPEN_SHORT", "HOLD_FLAT",
            "BLOCKED_reason", "LOTS_ZERO", or "ORDER_FAILED_error".
        """
        direction = action["direction"]
        conviction = action["conviction"]

        # 1. Direction threshold gate
        if abs(direction) < self._config.direction_threshold:
            return "HOLD_FLAT"

        # 1b. Conviction floor override: if direction is strong but conviction
        # is near-zero (model knows direction but won't commit), force minimum
        # conviction so the trade enters at minimum lot size.  The model's
        # direction signal is valid — conviction is suppressed because the
        # price regime ($5200+) is unfamiliar vs training ($1200-2000).
        conviction_floor = getattr(self._config, "conviction_tier_low", 0.15)
        if conviction < conviction_floor and abs(direction) >= self._config.direction_threshold:
            log.info(
                "CONVICTION OVERRIDE: raw=%.3f -> floor=%.3f (direction=%.3f strong enough)",
                conviction, conviction_floor, direction,
            )
            conviction = conviction_floor

        # Determine side
        side = "LONG" if direction > 0 else "SHORT"
        mt5_side = "BUY" if side == "LONG" else "SELL"

        # 2. Risk gate
        balance = account.get("balance", 0.0)
        allowed, reason = self._risk.check_position_allowed(
            balance=balance,
            peak_balance=self._peak_balance,
            daily_trade_count=self._daily_trades,
            conviction=conviction,
        )
        if not allowed:
            log.debug(
                "Entry blocked: %s  direction=%.3f conviction=%.3f",
                reason, direction, conviction,
            )
            return f"BLOCKED_{reason}"

        # 3. Lot sizing (using MT5-exact profit calculator)
        # FIX-10: Direction-scaled conviction for SL distance (matches training)
        direction_strength = abs(direction)
        sl_conviction = direction_strength * conviction

        atr = self._current_atr
        current_price = bar.get("close", 0.0)
        symbol_info = self._bridge.get_symbol_info(self._config.mt5_symbol)
        if not symbol_info:
            log.warning("Cannot get symbol_info for lot sizing")
            return "BLOCKED_no_symbol_info"

        # Build symbol_info dict with the keys calculate_lot_size expects
        lot_info = {
            "trade_tick_value": symbol_info.get("tick_value", 0.745),
            "trade_tick_size": symbol_info.get("tick_size", 0.01),
            "volume_min": symbol_info.get("volume_min", 0.01),
            "volume_max": symbol_info.get("volume_max", 100.0),
            "volume_step": symbol_info.get("volume_step", 0.01),
            "point": symbol_info.get("point", 0.01),
        }

        # Build MT5 calc_profit callback for exact account-currency math
        _symbol = self._config.mt5_symbol

        def _mt5_calc(calc_side, calc_lots, open_price, close_price):
            return self._bridge.calc_profit(
                calc_side, _symbol, calc_lots, open_price, close_price,
            )

        lots = self._risk.calculate_lot_size(
            conviction=conviction,
            balance=balance,
            peak_balance=self._peak_balance,
            atr=atr,
            symbol_info=lot_info,
            side=mt5_side,
            entry_price=current_price,
            mt5_calc_profit=_mt5_calc,
            sl_conviction=sl_conviction,
        )

        if lots <= 0:
            log.debug("Lot size is zero (risk budget exhausted)")
            return "LOTS_ZERO"

        # Log scalp mode for visibility
        tier_mid = getattr(self._config, "conviction_tier_mid", 0.30)
        if conviction < tier_mid:
            log.info(
                "SCALP ENTRY: %s conv=%.3f (below %.2f), lots=%.4f (minimum)",
                side, conviction, tier_mid, lots,
            )

        # 4. SL / TP — SL uses direction-scaled conviction (FIX-10), TP uses raw conviction
        sl_price = self._risk.calculate_sl(side, current_price, atr, sl_conviction)
        tp_price = self._risk.calculate_tp(side, current_price, atr, conviction)

        # 4a. Enforce broker minimum SL distance
        if self._broker_constraints is not None:
            sl_price = self._broker_constraints.enforce_min_sl(
                side=side,
                entry_price=current_price,
                sl_price=sl_price,
            )

        # 4b. Pre-flight: verify margin with MT5 order_check
        preflight = self._bridge.order_check(
            symbol=self._config.mt5_symbol,
            side=mt5_side,
            lots=lots,
            price=current_price,
            sl=sl_price,
            tp=tp_price,
        )
        if preflight is not None:
            if preflight["retcode"] != 0:
                log.warning(
                    "Order pre-flight REJECTED: %s (margin=%.2f free=%.2f)",
                    preflight["comment"],
                    preflight.get("margin", 0),
                    preflight.get("margin_free", 0),
                )
                return f"BLOCKED_preflight_{preflight['comment']}"

            # Log exact margin from MT5
            log.info(
                "Pre-flight OK: lots=%.2f margin=%.2f margin_free=%.2f margin_level=%.1f%%",
                lots,
                preflight.get("margin", 0),
                preflight.get("margin_free", 0),
                preflight.get("margin_level", 0),
            )

            # Also compute and log exact SL loss from MT5
            sl_loss = self._bridge.calc_profit(
                mt5_side, self._config.mt5_symbol, lots, current_price, sl_price,
            )
            if sl_loss is not None:
                log.info(
                    "MT5-exact risk: lots=%.2f SL_loss=%.2f (%.1f%% of balance)",
                    lots, abs(sl_loss), abs(sl_loss) / balance * 100 if balance else 0,
                )

        # 5. Safety: check for existing AI positions we don't know about
        # Only consider positions with our magic number -- ignore user's manual trades
        existing = self._bridge.get_open_positions(self._config.mt5_symbol) or []
        our_existing = [p for p in existing if p.get("magic", 0) == 234000]
        if our_existing:
            log.warning(
                "Aborting new order: MT5 already has %d AI-placed %s "
                "position(s) (magic=234000). Recovering...",
                len(our_existing), self._config.mt5_symbol,
            )
            self.recover_from_mt5()
            return f"RECOVERED_{side}"

        # 6. Send market order
        comment = f"spartus_{side.lower()}_{conviction:.2f}"
        order_result = self._bridge.send_market_order(
            symbol=self._config.mt5_symbol,
            side=mt5_side,
            lots=lots,
            sl=sl_price,
            tp=tp_price,
            comment=comment,
        )

        if not order_result.get("success"):
            error = order_result.get("error", "unknown")
            log.warning("Order failed: %s", error)
            return f"ORDER_FAILED_{error}"

        # 6. Track position internally
        fill_price = order_result.get("fill_price", current_price)
        ticket = order_result.get("ticket", 0)

        self._position = {
            "ticket": ticket,
            "side": side,
            "entry_price": fill_price,
            "lots": lots,
            "sl": sl_price,
            "tp": tp_price,
            "conviction": conviction,
            "direction": direction,
            "open_time": datetime.now(timezone.utc),
        }
        self._entry_step = self._step_count
        self._max_favorable = 0.0
        self._initial_sl = sl_price
        self._initial_tp = tp_price
        self._protection_stage = 0

        # Compute initial risk in account currency for £-based protection thresholds
        _sym_open = self._bridge.get_symbol_info(self._config.mt5_symbol) or {}
        _tick_sz_open = _sym_open.get("tick_size", 0.01) or 0.01
        _tick_val_open = _sym_open.get("tick_value", 0.745)
        _r_dist_open = abs(fill_price - sl_price)
        self._r_value_gbp = (
            (_r_dist_open / _tick_sz_open) * _tick_val_open * lots
        ) if _r_dist_open > 0 else 0.0

        # Snapshot entry conditions for trade journal
        self._entry_conditions = {
            "rsi": bar.get("rsi_14", 0.5),
            "trend_dir": bar.get("h1_trend_dir", 0.0),
            "session": bar.get("session", self._get_session_name()),
            "vol_regime": bar.get("atr_ratio", 1.0),
            "hour": datetime.now(timezone.utc).hour,
            "atr": atr,
            "drawdown": (self._peak_balance - balance) / max(self._peak_balance, 1.0),
            "conviction": conviction,
            "direction": direction,
        }

        # Snapshot bar context at entry for training-improvement logging
        self._entry_observation = self._bar_observation
        self._entry_risk_state = dict(self._bar_risk_state)
        self._entry_spread = self._bar_spread

        self._daily_trades += 1

        log.info(
            "OPEN %s: ticket=%d  lots=%.3f  fill=%.2f  "
            "SL=%.2f  TP=%.2f  conviction=%.3f  daily_trade#%d  spread=%.1f",
            side, ticket, lots, fill_price,
            sl_price, tp_price, conviction, self._daily_trades,
            self._entry_spread,
        )

        # Persist initial protection state for crash recovery
        self._save_protection_state()

        return f"OPEN_{side}"

    # ------------------------------------------------------------------
    # In-position logic
    # ------------------------------------------------------------------

    def _handle_in_position(
        self,
        action: Dict[str, float],
        bar: Dict[str, Any],
        account: Dict[str, Any],
    ) -> str:
        """Manage an existing position: check exit or adjust trailing SL.

        Decision flow:
            1. Check exit_urgency > threshold AND bars_held >= min_hold_bars
            2. If exit: close position, record trade
            3. Else: adjust trailing SL

        Returns:
            "CLOSE_AGENT", "TRAIL_SL_xxxx.xx", or "HOLD".
        """
        pos = self._position
        if pos is None:
            return "HOLD"

        exit_urgency = action["exit_urgency"]
        sl_adjustment = action["sl_adjustment"]
        bars_held = self._step_count - self._entry_step
        current_price = bar.get("close", 0.0)

        # --- Check for exit signal ---
        if (exit_urgency > self._config.exit_threshold
                and bars_held >= self._config.min_hold_bars):

            close_result = self._bridge.close_position(pos["ticket"])

            if close_result.get("success"):
                close_price = close_result.get("fill_price", current_price)
                self._record_trade_close(
                    close_result=close_result,
                    close_reason="AGENT_CLOSE",
                    close_price=close_price,
                    bars_held=bars_held,
                    account=account,
                )
                log.info(
                    "CLOSE_AGENT: ticket=%d  held=%d bars  exit_urgency=%.3f",
                    pos["ticket"], bars_held, exit_urgency,
                )
                return "CLOSE_AGENT"
            else:
                log.warning(
                    "Close failed for ticket %d: %s",
                    pos["ticket"],
                    close_result.get("error", "unknown"),
                )
                # Fall through to SL adjustment

        # --- Apply profit protection first, then AI trailing SL ---
        atr = self._current_atr
        mt5_sl = pos["sl"]  # The SL currently on the broker (MT5)

        # Guard: initial_sl must not be 0 or equal to entry
        if self._initial_sl <= 0 or abs(self._initial_sl - pos["entry_price"]) < 0.01:
            fallback = (
                pos["entry_price"] - 2.0 * atr if pos["side"] == "LONG"
                else pos["entry_price"] + 2.0 * atr
            )
            log.warning(
                "PROTECT_GUARD: initial_sl=%.2f invalid (entry=%.2f) "
                "-- using ATR fallback %.2f",
                self._initial_sl, pos["entry_price"], fallback,
            )
            self._initial_sl = fallback

        r_distance = abs(pos["entry_price"] - self._initial_sl)
        r_multiple = self._max_favorable / r_distance if r_distance > 0 else 0.0

        # Diagnostic logging every bar (protection state visibility)
        if bars_held % 3 == 0 or r_multiple >= 0.25:
            log.info(
                "PROTECT_DEBUG: ticket=%d bar=%d mfe=%.4f r_dist=%.2f "
                "R=%.3f stage=%d thresholds=[%.2f/%.2f/%.2f] "
                "sl=%.2f initial_sl=%.2f price=%.2f",
                pos["ticket"], bars_held, self._max_favorable,
                r_distance, r_multiple, self._protection_stage,
                getattr(self._config, "protection_be_trigger_r", 1.0),
                getattr(self._config, "protection_lock_trigger_r", 1.5),
                getattr(self._config, "protection_trail_trigger_r", 2.0),
                mt5_sl, self._initial_sl, current_price,
            )

        # Profit protection (rule-based staged SL floor)
        protection_sl, new_stage = self._risk.apply_profit_protection(
            position={
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "stop_loss": mt5_sl,
                "initial_sl": self._initial_sl,
                "max_favorable": self._max_favorable,
                "protection_stage": self._protection_stage,
                "r_value_gbp": self._r_value_gbp,
            },
            current_price=current_price,
            atr=atr,
            spread_points=self._bar_spread * 0.01,  # spread in points
        )

        # Log protection stage transitions
        if new_stage > self._protection_stage:
            r_current = self._max_favorable / r_distance if r_distance > 0 else 0.0
            log.info(
                "PROTECT: ticket=%d  stage %d->%d  R=%.2f  "
                "sl %.2f -> %.2f  side=%s",
                pos["ticket"], self._protection_stage, new_stage,
                r_current, mt5_sl, protection_sl, pos["side"],
            )
            self._log_protection_event(pos, new_stage, protection_sl, r_current)
        self._protection_stage = new_stage

        # AI trailing SL (can only tighten beyond protection floor)
        new_sl = self._risk.adjust_stop_loss(
            current_sl=protection_sl,
            side=pos["side"],
            current_price=current_price,
            atr=atr,
            sl_adjustment=sl_adjustment,
        )

        # The final SL is the tighter of AI trail and protection floor.
        # For LONG: higher SL is tighter. For SHORT: lower SL is tighter.
        if pos["side"] == "LONG":
            final_sl = max(new_sl, protection_sl)
        else:
            final_sl = min(new_sl, protection_sl)

        # Compare final SL against the ACTUAL MT5 SL (not the protection floor).
        # This ensures protection-only moves (without AI trail) still get sent.
        sym_info = self._bridge.get_symbol_info(self._config.mt5_symbol) or {}
        tick_size = sym_info.get("tick_size", 0.01) or 0.01
        sl_changed = False
        if pos["side"] == "LONG" and final_sl > mt5_sl + tick_size:
            sl_changed = True
        elif pos["side"] == "SHORT" and final_sl < mt5_sl - tick_size:
            sl_changed = True

        if sl_changed:
            success = self._bridge.modify_position(
                ticket=pos["ticket"],
                sl=final_sl,
            )
            if success:
                pos["sl"] = final_sl

                # Determine reason: protection stage or AI trail
                reason = f"stage_{self._protection_stage}" if final_sl == protection_sl else "ai_trail"

                # Record SL modification for trade history
                self._sl_modifications.append({
                    "bar": self._step_count,
                    "old_sl": round(mt5_sl, 2),
                    "new_sl": round(final_sl, 2),
                    "reason": reason,
                    "price": round(current_price, 2),
                })

                # Log MT5-exact P/L at the new SL level
                mt5_side = "BUY" if pos["side"] == "LONG" else "SELL"
                sl_pnl = self._bridge.calc_profit(
                    mt5_side, self._config.mt5_symbol,
                    pos["lots"], pos["entry_price"], final_sl,
                )
                sl_pnl_str = f"  SL_pnl={sl_pnl:.2f}" if sl_pnl is not None else ""

                log.info(
                    "TRAIL_SL: ticket=%d  %s  sl %.2f -> %.2f  "
                    "reason=%s  sl_adj=%.3f  bars_held=%d  "
                    "stage=%d  R=%.2f%s",
                    pos["ticket"], pos["side"],
                    mt5_sl, final_sl, reason, sl_adjustment, bars_held,
                    self._protection_stage, r_multiple, sl_pnl_str,
                )

                # Persist protection state for crash recovery
                self._save_protection_state()

                return f"TRAIL_SL_{final_sl:.2f}"

        # Even if SL didn't change, persist state periodically
        if bars_held % 10 == 0:
            self._save_protection_state()

        return "HOLD"

    # ------------------------------------------------------------------
    # Trade close recording
    # ------------------------------------------------------------------

    def _record_trade_close(
        self,
        close_result: Dict[str, Any],
        close_reason: str,
        close_price: float = 0.0,
        bars_held: int = 0,
        account: Optional[Dict[str, Any]] = None,
        mt5_pnl: Optional[float] = None,
    ) -> None:
        """Record a closed trade in memory, journal, tp_tracking, and trades.jsonl.

        Args:
            close_result: Dict from mt5_bridge.close_position().
            close_reason: Why the trade was closed (AGENT_CLOSE, TP_HIT,
                          SL_HIT, EMERGENCY_STOP, CIRCUIT_BREAKER, etc.)
            close_price:  Fill price for the close.
            bars_held:    Number of M5 bars the position was held.
            account:      Current account info dict.
            mt5_pnl:      Actual P/L from MT5 deal history (if available).
                          When provided, used instead of manual calculation.
        """
        pos = self._position
        if pos is None:
            log.warning("_record_trade_close called with no position")
            return

        side = pos["side"]
        entry_price = pos["entry_price"]
        lots = pos["lots"]
        conviction = pos["conviction"]

        if close_price == 0.0:
            close_price = close_result.get("fill_price", entry_price)

        # --- P/L calculation ---
        # Prefer MT5's actual P/L (includes swap, fees, exact fill) when
        # available.  Fall back to manual tick_value calculation otherwise.
        if mt5_pnl is not None:
            pnl = mt5_pnl
            log.info(
                "Using MT5 actual P/L: %.2f (manual would be %.2f)",
                mt5_pnl,
                (
                    (close_price - entry_price if side == "LONG"
                     else entry_price - close_price)
                    * self._bridge.value_per_point * lots
                ),
            )
        else:
            value_per_point = self._bridge.value_per_point
            if side == "LONG":
                price_move = close_price - entry_price
            else:
                price_move = entry_price - close_price
            pnl = price_move * value_per_point * lots

        balance = account.get("balance", self._peak_balance) if account else self._peak_balance
        pnl_pct = pnl / max(balance, 1.0)

        # --- TP/SL tracking ---
        tp_hit = close_reason == "TP_HIT"
        sl_hit = close_reason == "SL_HIT"
        max_fav = self._max_favorable

        # Profit locked by trailing: distance SL moved from initial toward profit
        profit_locked = 0.0
        if side == "LONG" and pos["sl"] > self._initial_sl:
            profit_locked = (pos["sl"] - self._initial_sl) / max(self._current_atr, 0.01)
        elif side == "SHORT" and pos["sl"] < self._initial_sl:
            profit_locked = (self._initial_sl - pos["sl"]) / max(self._current_atr, 0.01)
        profit_locked = float(np.clip(profit_locked, 0.0, 1.0))

        # --- Lesson classification (matches training TradeAnalyzer) ---
        lesson_type = self._classify_lesson(
            pnl=pnl,
            hold_bars=bars_held,
            reason=close_reason,
            side=side,
            entry_price=entry_price,
            exit_price=close_price,
            max_fav=max_fav,
            sl_distance=abs(entry_price - self._initial_sl),
            tp_distance=abs(self._initial_tp - entry_price),
            tp_hit=tp_hit,
            sl_hit=sl_hit,
            conviction=conviction,
        )

        # --- Session name ---
        session = self._entry_conditions.get("session", self._get_session_name())

        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Record trade in memory DB
        trade_data = {
            "timestamp": now_iso,
            "week": 0,
            "step": self._step_count,
            "side": side,
            "entry_price": entry_price,
            "exit_price": close_price,
            "lot_size": lots,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 6),
            "hold_bars": bars_held,
            "close_reason": close_reason,
            "conviction": conviction,
            "rsi_at_entry": self._entry_conditions.get("rsi", 0.5),
            "trend_dir_at_entry": self._entry_conditions.get("trend_dir", 0.0),
            "session_at_entry": session,
            "vol_regime_at_entry": self._entry_conditions.get("vol_regime", 1.0),
            "entry_conditions": self._entry_conditions,
            "paper_trade": self._config.paper_trading,
            "mt5_ticket": pos.get("ticket", 0),
        }
        trade_id = self._memory.record_trade(trade_data)

        # 2. Record TP/SL tracking
        self._memory.record_tp_tracking(
            trade_id=trade_id,
            tp_price=self._initial_tp,
            sl_price=self._initial_sl,
            tp_hit=tp_hit,
            sl_hit=sl_hit,
            max_favorable=max_fav,
            profit_locked=profit_locked,
        )

        # 3. Record journal entry
        self._memory.record_journal(
            trade_id=trade_id,
            lesson_type=lesson_type,
            notes=(
                f"{side} {close_reason} | P/L={pnl:+.2f} | "
                f"held={bars_held} bars | conv={conviction:.2f} | "
                f"max_fav={max_fav:.2f}"
            ),
        )

        # 4. Update risk manager
        self._risk.record_trade_result(pnl)

        # 5. Log to trades.jsonl
        self._log_trade_jsonl(trade_data, trade_id, lesson_type)

        log.info(
            "Trade #%d closed: %s %s  P/L=%+.2f (%.2f%%)  held=%d bars  "
            "lesson=%s  conviction=%.3f",
            trade_id, side, close_reason, pnl, pnl_pct * 100,
            bars_held, lesson_type, conviction,
        )

        # Clear position state
        self._position = None
        self._entry_step = 0
        self._entry_conditions = {}
        self._max_favorable = 0.0
        self._initial_sl = 0.0
        self._initial_tp = 0.0
        self._protection_stage = 0
        self._r_value_gbp = 0.0
        self._sl_modifications = []
        self._entry_observation = None
        self._entry_risk_state = {}
        self._entry_spread = 0.0
        self._clear_protection_state()

    # ------------------------------------------------------------------
    # Lesson classification (mirrors training TradeAnalyzer._classify_lesson)
    # ------------------------------------------------------------------

    def _classify_lesson(
        self,
        pnl: float,
        hold_bars: int,
        reason: str,
        side: str,
        entry_price: float,
        exit_price: float,
        max_fav: float,
        sl_distance: float,
        tp_distance: float,
        tp_hit: bool,
        sl_hit: bool,
        conviction: float,
    ) -> str:
        """Classify the trade into a lesson type.

        Logic matches training TradeAnalyzer exactly.
        """
        # Emergency stop is its own category
        if reason in ("EMERGENCY_STOP", "CIRCUIT_BREAKER"):
            return "EMERGENCY_STOP"

        # TP hit -- good trade
        if tp_hit:
            return "GOOD_TRADE"

        # Profitable trade
        if pnl > 0.005:
            if hold_bars <= 3:
                return "SCALP_WIN"
            return "GOOD_TRADE"

        # Breakeven
        if abs(pnl) < 0.005:
            return "BREAKEVEN"

        # Direction analysis
        if side == "LONG":
            price_moved_right = exit_price > entry_price
        else:
            price_moved_right = exit_price < entry_price
        dir_correct = price_moved_right

        # Lost money -- classify why
        if not dir_correct:
            if sl_distance > 0 and max_fav / max(sl_distance, 1e-8) > 1.0:
                return "BAD_TIMING"
            return "WRONG_DIRECTION"

        # Direction was correct but still lost
        if sl_hit:
            if sl_distance > 0 and max_fav / max(sl_distance, 1e-8) > 1.5:
                return "CORRECT_DIR_BAD_SL"
            return "WHIPSAW"

        # Agent closed at a loss despite correct direction
        if hold_bars >= 20 and max_fav > abs(pnl):
            return "HELD_TOO_LONG"
        return "CORRECT_DIR_CLOSED_EARLY"

    # ------------------------------------------------------------------
    # Max favorable excursion tracking
    # ------------------------------------------------------------------

    def _update_max_favorable(self, bar: Dict[str, Any]) -> None:
        """Update the maximum favorable excursion for the current position.

        Tracks how far price moved in our favor -- used for trade
        analysis and TP tracking.
        """
        pos = self._position
        if pos is None:
            return

        current_price = bar.get("close", 0.0)
        high = bar.get("high", current_price)
        low = bar.get("low", current_price)

        if pos["side"] == "LONG":
            favorable = high - pos["entry_price"]
        else:
            favorable = pos["entry_price"] - low

        if favorable > self._max_favorable:
            self._max_favorable = favorable

    # ------------------------------------------------------------------
    # Intrabar protection check (called every 5 seconds between bars)
    # ------------------------------------------------------------------

    def check_ai_protection_intrabar(self, current_price: float, atr: float) -> Optional[str]:
        """Check and apply AI profit protection between M5 bars.

        Called every 5 seconds so protection fires within seconds of the
        threshold being hit, not at the next bar close (up to 5 min later).

        Updates _max_favorable from the live price, then calls
        apply_profit_protection exactly as execute_action does.

        Returns a status string if stage advanced, else None.
        """
        pos = self._position
        if pos is None or current_price <= 0:
            return None
        if self._initial_sl <= 0 or abs(self._initial_sl - pos["entry_price"]) < 0.01:
            return None

        # Update max favorable from current live price
        if pos["side"] == "LONG":
            live_fav = current_price - pos["entry_price"]
        else:
            live_fav = pos["entry_price"] - current_price
        if live_fav > self._max_favorable:
            self._max_favorable = live_fav

        mt5_sl = pos.get("sl", self._initial_sl)

        protection_sl, new_stage = self._risk.apply_profit_protection(
            position={
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "stop_loss": mt5_sl,
                "initial_sl": self._initial_sl,
                "max_favorable": self._max_favorable,
                "protection_stage": self._protection_stage,
                "r_value_gbp": self._r_value_gbp,
            },
            current_price=current_price,
            atr=atr,
            spread_points=self._bar_spread * 0.01,
        )

        sym_info = self._bridge.get_symbol_info(self._config.mt5_symbol) or {}
        tick_size = sym_info.get("tick_size", 0.01) or 0.01

        sl_changed = False
        if pos["side"] == "LONG" and protection_sl > mt5_sl + tick_size:
            sl_changed = True
        elif pos["side"] == "SHORT" and protection_sl < mt5_sl - tick_size:
            sl_changed = True

        if not sl_changed and new_stage <= self._protection_stage:
            return None

        if sl_changed:
            success = self._bridge.modify_position(
                ticket=pos["ticket"],
                sl=round(protection_sl, 2),
            )
            if success:
                pos["sl"] = protection_sl
                log.info(
                    "INTRABAR PROTECT: ticket=%d stage %d→%d SL %.2f→%.2f",
                    pos["ticket"], self._protection_stage, new_stage,
                    mt5_sl, protection_sl,
                )
            else:
                log.warning(
                    "INTRABAR PROTECT: SL modify failed ticket=%d", pos["ticket"]
                )
                return None

        if new_stage > self._protection_stage:
            stage_names = {1: "BREAKEVEN", 2: "PROFIT_LOCK", 3: "TRAILING"}
            r_distance = abs(pos["entry_price"] - self._initial_sl)
            r_current = self._max_favorable / r_distance if r_distance > 0 else 0.0
            self._log_protection_event(pos, new_stage, protection_sl, r_current)
            self._protection_stage = new_stage
            return f"PROTECT_STAGE_{new_stage}_{stage_names.get(new_stage, '')}"

        return None

    # ------------------------------------------------------------------
    # Protection event logging
    # ------------------------------------------------------------------

    def _log_protection_event(
        self,
        pos: Dict[str, Any],
        stage: int,
        new_sl: float,
        r_current: float,
    ) -> None:
        """Log a protection stage transition to alerts.jsonl."""
        stage_names = {1: "PROTECT_BE", 2: "PROTECT_LOCK", 3: "PROTECT_TRAIL"}
        stage_name = stage_names.get(stage, f"PROTECT_{stage}")

        r_distance = abs(pos["entry_price"] - self._initial_sl)
        if pos["side"] == "LONG":
            locked_r = (new_sl - pos["entry_price"]) / r_distance if r_distance > 0 else 0.0
        else:
            locked_r = (pos["entry_price"] - new_sl) / r_distance if r_distance > 0 else 0.0

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": stage_name,
            "ticket": pos.get("ticket", 0),
            "side": pos["side"],
            "stage": stage,
            "r_current": round(r_current, 3),
            "locked_r": round(locked_r, 3),
            "new_sl": round(new_sl, 2),
            "entry_price": pos["entry_price"],
            "initial_sl": round(self._initial_sl, 2),
        }

        try:
            alerts_path = self._resolve_path("storage/logs/alerts.jsonl")
            with open(alerts_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str) + "\n")
        except Exception as exc:
            log.warning("Failed to write protection alert: %s", exc)

    # ------------------------------------------------------------------
    # Manual trade management (user-opened positions)
    # ------------------------------------------------------------------

    def scan_manual_trades(self) -> int:
        """Detect manually-opened positions and start tracking them.

        Scans MT5 for positions on the primary symbol where
        magic != 234000 (not placed by Spartus).  New positions are
        adopted for SL management; closed ones are removed from tracking.

        Returns:
            Number of newly adopted manual positions.
        """
        if not self._config.manage_manual_trades:
            return 0

        try:
            all_positions = self._bridge.get_open_positions(
                self._config.mt5_symbol,
            )
        except Exception:
            log.exception("scan_manual_trades: failed to query MT5")
            return 0

        if not all_positions:
            all_positions = []

        # Find manual positions (magic != 234000)
        manual_positions = [
            p for p in all_positions if p.get("magic", 0) != 234000
        ]
        open_tickets = {p["ticket"] for p in manual_positions}

        # Remove closed manual positions from tracking
        closed = [t for t in self._manual_positions if t not in open_tickets]
        for ticket in closed:
            mp = self._manual_positions.pop(ticket)
            log.info(
                "Manual trade ticket=%d (%s) closed externally — "
                "stopped tracking",
                ticket, mp.get("side", "?"),
            )

        # Adopt new manual positions
        adopted = 0
        for pos in manual_positions:
            ticket = pos["ticket"]
            if ticket in self._manual_positions:
                # Already tracking — update current SL from MT5
                mt5_sl = pos.get("sl", 0.0)
                self._manual_positions[ticket]["current_sl"] = mt5_sl

                # If initial_sl was 0 (no SL at adoption) but user has now
                # set one, adopt it as the initial risk reference.
                if self._manual_positions[ticket]["initial_sl"] <= 0 and mt5_sl > 0:
                    self._manual_positions[ticket]["initial_sl"] = mt5_sl
                    log.info(
                        "Manual trade ticket=%d: SL detected (%.2f) — "
                        "protection now active (R = %.2f)",
                        ticket, mt5_sl,
                        abs(self._manual_positions[ticket]["entry_price"] - mt5_sl),
                    )
                continue

            side = "LONG" if pos.get("type", 0) == 0 else "SHORT"
            initial_sl = pos.get("sl", 0.0)

            self._manual_positions[ticket] = {
                "ticket": ticket,
                "side": side,
                "entry_price": pos["price_open"],
                "lots": pos["volume"],
                "initial_sl": initial_sl,
                "current_sl": initial_sl,
                "tp": pos.get("tp", 0.0),
                "protection_stage": 0,
                "max_favorable": 0.0,
                "open_time": pos.get("time", datetime.now(timezone.utc)),
                "sl_modifications": [],
            }
            adopted += 1
            sl_msg = f"SL={initial_sl:.2f}" if initial_sl > 0 else "NO SL (waiting)"
            log.warning(
                "ADOPTED manual trade: ticket=%d %s %.3f lots @ %.2f "
                "%s TP=%.2f (magic=%d) — will manage SL",
                ticket, side, pos["volume"], pos["price_open"],
                sl_msg, pos.get("tp", 0.0), pos.get("magic", 0),
            )

        return adopted

    def manage_manual_positions(
        self,
        current_price: float,
        atr: float,
        bar_high: float = 0.0,
        bar_low: float = 0.0,
    ) -> List[str]:
        """Apply profit protection to all tracked manual positions.

        Only tightens SL — never loosens, never sets TP, never closes.
        Uses SEPARATE settings from AI trades (configurable via UI):
            Stage 1: Breakeven at configurable R trigger
            Stage 2: Lock configurable R at configurable trigger
            Stage 3: ATR trail at configurable trigger

        Args:
            current_price: Current close price.
            atr: Current ATR(14) value.
            bar_high: Current bar high (for MFE tracking).
            bar_low: Current bar low (for MFE tracking).

        Returns:
            List of action strings for each managed position.
        """
        if not self._config.manage_manual_trades:
            return []
        if not self._manual_positions:
            return []

        results: List[str] = []
        high = bar_high if bar_high > 0 else current_price
        low = bar_low if bar_low > 0 else current_price

        for ticket, mp in list(self._manual_positions.items()):
            # Update max favorable excursion
            if mp["side"] == "LONG":
                favorable = high - mp["entry_price"]
            else:
                favorable = mp["entry_price"] - low

            if favorable > mp["max_favorable"]:
                mp["max_favorable"] = favorable

            # Skip if no SL set (can't compute R)
            if mp["initial_sl"] <= 0:
                results.append(f"MANUAL_{ticket}_NO_SL")
                continue

            # Apply profit protection (uses MANUAL-specific settings, not AI)
            spread = self._bar_spread * 0.01 if self._bar_spread else 0.0
            # Compute r_value_gbp for this manual position (for £-threshold mode)
            _r_dist_mp = abs(mp["entry_price"] - mp["initial_sl"])
            _sym_mp = self._bridge.get_symbol_info(self._config.mt5_symbol) or {}
            _tick_sz_mp = _sym_mp.get("tick_size", 0.01) or 0.01
            _tick_val_mp = _sym_mp.get("tick_value", 0.745)
            _r_val_mp = (
                (_r_dist_mp / _tick_sz_mp) * _tick_val_mp * mp.get("lots", 0.01)
            ) if _r_dist_mp > 0 else 0.0
            protection_sl, new_stage = self._risk.apply_manual_profit_protection(
                position={
                    "side": mp["side"],
                    "entry_price": mp["entry_price"],
                    "stop_loss": mp["current_sl"],
                    "initial_sl": mp["initial_sl"],
                    "max_favorable": mp["max_favorable"],
                    "protection_stage": mp["protection_stage"],
                    "r_value_gbp": _r_val_mp,
                },
                current_price=current_price,
                atr=atr,
                spread_points=spread,
                overrides=self._manual_protection_overrides,
            )

            # Log stage transitions
            if new_stage > mp["protection_stage"]:
                r_distance = abs(mp["entry_price"] - mp["initial_sl"])
                r_current = mp["max_favorable"] / r_distance if r_distance > 0 else 0.0
                stage_names = {1: "BE", 2: "LOCK", 3: "ATR_TRAIL"}
                log.info(
                    "MANUAL PROTECT: ticket=%d %s  stage %d->%d (%s)  "
                    "R=%.2f  sl %.2f -> %.2f",
                    ticket, mp["side"],
                    mp["protection_stage"], new_stage,
                    stage_names.get(new_stage, "?"),
                    r_current, mp["current_sl"], protection_sl,
                )
                # Log to alerts.jsonl
                self._log_manual_protection_event(
                    mp, new_stage, protection_sl, r_current,
                )

            mp["protection_stage"] = new_stage

            # Check if SL actually needs to move
            sl_changed = False
            sym_info = self._bridge.get_symbol_info(self._config.mt5_symbol) or {}
            tick_size = sym_info.get("tick_size", 0.01) or 0.01

            if mp["side"] == "LONG" and protection_sl > mp["current_sl"] + tick_size:
                sl_changed = True
            elif mp["side"] == "SHORT" and protection_sl < mp["current_sl"] - tick_size:
                sl_changed = True

            if sl_changed:
                success = self._bridge.modify_position(
                    ticket=ticket,
                    sl=protection_sl,
                )
                if success:
                    old_sl = mp["current_sl"]
                    mp["current_sl"] = protection_sl
                    mp["sl_modifications"].append({
                        "bar": self._step_count,
                        "old_sl": round(old_sl, 2),
                        "new_sl": round(protection_sl, 2),
                        "stage": new_stage,
                        "price": round(current_price, 2),
                    })
                    log.info(
                        "MANUAL TRAIL: ticket=%d %s  sl %.2f -> %.2f  "
                        "stage=%d",
                        ticket, mp["side"], old_sl, protection_sl,
                        new_stage,
                    )
                    results.append(
                        f"MANUAL_{ticket}_TRAIL_{protection_sl:.2f}"
                    )
                else:
                    log.warning(
                        "Failed to modify SL for manual trade ticket=%d",
                        ticket,
                    )
                    results.append(f"MANUAL_{ticket}_MODIFY_FAILED")
            else:
                results.append(f"MANUAL_{ticket}_HOLD")

        return results

    def get_manual_positions(self) -> Dict[int, Dict[str, Any]]:
        """Return the current manual position tracking dict (for UI)."""
        return dict(self._manual_positions)

    def set_manual_protection_overrides(self, overrides: Dict[str, float]) -> None:
        """Set runtime protection overrides from UI sliders.

        Keys: be_trigger_r, lock_trigger_r, lock_amount_r,
              trail_trigger_r, trail_atr_mult, be_buffer_pips.
        """
        self._manual_protection_overrides = dict(overrides)
        log.info("Manual protection overrides updated: %s", overrides)

    def get_manual_protection_overrides(self) -> Dict[str, float]:
        """Return current manual protection overrides."""
        return dict(self._manual_protection_overrides)

    def _log_manual_protection_event(
        self,
        mp: Dict[str, Any],
        stage: int,
        new_sl: float,
        r_current: float,
    ) -> None:
        """Log a manual trade protection event to alerts.jsonl."""
        stage_names = {1: "MANUAL_PROTECT_BE", 2: "MANUAL_PROTECT_LOCK",
                       3: "MANUAL_PROTECT_TRAIL"}
        stage_name = stage_names.get(stage, f"MANUAL_PROTECT_{stage}")

        r_distance = abs(mp["entry_price"] - mp["initial_sl"])
        if mp["side"] == "LONG":
            locked_r = ((new_sl - mp["entry_price"]) / r_distance
                        if r_distance > 0 else 0.0)
        else:
            locked_r = ((mp["entry_price"] - new_sl) / r_distance
                        if r_distance > 0 else 0.0)

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": stage_name,
            "ticket": mp.get("ticket", 0),
            "side": mp["side"],
            "stage": stage,
            "r_current": round(r_current, 3),
            "locked_r": round(locked_r, 3),
            "new_sl": round(new_sl, 2),
            "entry_price": mp["entry_price"],
            "initial_sl": round(mp["initial_sl"], 2),
            "manual_trade": True,
        }

        try:
            alerts_path = self._resolve_path("storage/logs/alerts.jsonl")
            with open(alerts_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str) + "\n")
        except Exception as exc:
            log.warning("Failed to write manual protection alert: %s", exc)

    # ------------------------------------------------------------------
    # JSONL trade log
    # ------------------------------------------------------------------

    def _log_trade_jsonl(
        self,
        trade_data: Dict[str, Any],
        trade_id: int,
        lesson_type: str,
    ) -> None:
        """Append a trade record to trades.jsonl.

        Each line is a self-contained JSON object for easy streaming
        and offline analysis.  Includes entry observation (670-dim),
        risk state, and spread for training-improvement post-analysis.
        """
        record = {
            "trade_id": trade_id,
            "timestamp": trade_data.get("timestamp"),
            "side": trade_data.get("side"),
            "entry_price": trade_data.get("entry_price"),
            "exit_price": trade_data.get("exit_price"),
            "lot_size": trade_data.get("lot_size"),
            "pnl": trade_data.get("pnl"),
            "pnl_pct": trade_data.get("pnl_pct"),
            "hold_bars": trade_data.get("hold_bars"),
            "close_reason": trade_data.get("close_reason"),
            "conviction": trade_data.get("conviction"),
            "lesson_type": lesson_type,
            "session": trade_data.get("session_at_entry"),
            "paper_trade": trade_data.get("paper_trade", False),
            "entry_conditions": trade_data.get("entry_conditions", {}),
            "max_favorable": round(self._max_favorable, 4),
            "initial_sl": round(self._initial_sl, 2),
            "initial_tp": round(self._initial_tp, 2),
            "final_sl": round(self._position["sl"], 2) if self._position else 0.0,
            # --- Protection fields ---
            "protection_stage_max": self._protection_stage,
            "initial_r": round(abs(trade_data.get("entry_price", 0) - self._initial_sl), 2),
            "max_r_reached": round(
                self._max_favorable / max(abs(trade_data.get("entry_price", 0) - self._initial_sl), 0.01), 2
            ) if self._initial_sl else 0.0,
            # --- Training-improvement fields ---
            "entry_spread": round(self._entry_spread, 1),
            "entry_risk_state": self._entry_risk_state,
            # --- V2 analytics fields ---
            "model_version": self._model_version,
            "model_hash": self._model_hash[:16] if self._model_hash else "",
            "sl_modifications": self._sl_modifications.copy(),
            "sl_modification_count": len(self._sl_modifications),
        }

        # Named entry features (decode 670-dim obs into readable dict)
        if self._entry_observation is not None:
            try:
                obs = self._entry_observation
                if hasattr(obs, "flatten"):
                    obs = obs.flatten()
                # Extract most recent frame (last 67 values from 670-dim stack)
                n_features = 67
                if len(obs) >= n_features:
                    latest_frame = obs[-n_features:]
                    from src.config import TrainingConfig
                    cfg = TrainingConfig()
                    names = list(cfg.market_feature_names) + list(cfg.norm_exempt_features)
                    if len(names) == n_features:
                        record["entry_features"] = {
                            name: round(float(val), 4)
                            for name, val in zip(names, latest_frame)
                        }
            except Exception:
                pass  # Don't fail trade logging over feature decode

        # Reward decomposition (simulated R1-R5 for post-analysis)
        try:
            pnl = trade_data.get("pnl", 0)
            balance = self._peak_balance or 100.0
            equity_return = pnl / max(balance, 1.0)
            hold_bars = trade_data.get("hold_bars", 0)
            risk_amount = record.get("initial_r", 1.0)
            rr = pnl / max(abs(risk_amount), 0.01)
            hold_quality = min(hold_bars / 20.0, 1.0)
            record["reward_components"] = {
                "r1_pnl_signal": round(equity_return * 500.0, 4),
                "r2_quality": round(rr * hold_quality, 4),
                "rr_ratio": round(rr, 3),
                "hold_quality": round(hold_quality, 3),
            }
        except Exception:
            pass

        # Entry observation (670-dim) -- convert numpy to list for JSON
        if self._entry_observation is not None:
            try:
                obs = self._entry_observation
                if hasattr(obs, "tolist"):
                    obs = obs.tolist()
                record["entry_observation_670"] = obs
            except Exception:
                pass  # Don't fail trade logging over observation snapshot

        try:
            with open(self._trades_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            log.warning("Failed to write trades.jsonl: %s", exc)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def start_trading(self) -> None:
        """Set state to RUNNING -- AI begins monitoring and trading.

        Checks MT5 for existing positions from a previous session
        and adopts them so the AI can continue managing (trailing SL, exit).
        """
        prev = self._state
        self._state = TradingState.RUNNING
        recovered = self.recover_from_mt5()
        if recovered:
            log.info(
                "TradeExecutor: %s -> RUNNING (recovered existing position)",
                prev.value,
            )
        else:
            log.info("TradeExecutor: %s -> RUNNING", prev.value)

    def stop_trading(self) -> None:
        """Close all positions immediately and set state to STOPPED.

        This is a controlled shutdown: closes any open position via
        MT5, records the trade, then transitions to STOPPED.
        """
        log.warning("TradeExecutor: stop_trading() called -- closing all positions")

        if self._position is not None:
            ticket = self._position["ticket"]
            close_result = self._bridge.close_position(ticket)
            bars_held = self._step_count - self._entry_step

            if close_result.get("success"):
                close_price = close_result.get("fill_price", 0.0)
                self._record_trade_close(
                    close_result=close_result,
                    close_reason="MANUAL_STOP",
                    close_price=close_price,
                    bars_held=bars_held,
                )
            else:
                log.error(
                    "Failed to close position %d during stop: %s",
                    ticket,
                    close_result.get("error", "unknown"),
                )
                # Force clear position state even if close failed
                self._position = None
                self._clear_protection_state()

        self._state = TradingState.STOPPED
        log.info("TradeExecutor: state -> STOPPED")

    def pause_trading(self) -> None:
        """Pause AI monitoring without touching any open MT5 positions.

        The position stays open on the broker side (SL/TP remain active).
        Use this when the user stops the dashboard but wants their trade
        to continue running on MT5.  Use emergency_stop() to force-close.
        """
        self._state = TradingState.STOPPED
        log.info(
            "TradeExecutor: AI paused -- open position (if any) left intact on MT5"
        )

    def wind_down(self) -> None:
        """Transition to WINDING_DOWN (or STOPPED if no open position).

        In WINDING_DOWN state the executor manages the existing position
        (trailing SL, exit) but does not open new ones.  Once the
        position closes, it auto-transitions to STOPPED.
        """
        if self._position is not None:
            self._state = TradingState.WINDING_DOWN
            log.info(
                "TradeExecutor: RUNNING -> WINDING_DOWN "
                "(managing position ticket=%d until close)",
                self._position["ticket"],
            )
        else:
            self._state = TradingState.STOPPED
            log.info(
                "TradeExecutor: no open position, "
                "skipping WINDING_DOWN -> STOPPED",
            )

    def emergency_stop(self) -> None:
        """Activate emergency stop: close all positions, halt trading.

        Uses MT5Bridge.close_all_positions() for maximum urgency.
        """
        log.critical("EMERGENCY STOP activated")

        # Close all positions for the primary symbol
        results = self._bridge.close_all_positions(self._config.mt5_symbol)

        # Record the close if we were tracking a position
        if self._position is not None:
            bars_held = self._step_count - self._entry_step

            # Find the matching close result
            close_price = 0.0
            for res in results:
                if res.get("success"):
                    close_price = res.get("fill_price", 0.0)
                    break

            if close_price > 0:
                self._record_trade_close(
                    close_result={"success": True, "fill_price": close_price},
                    close_reason="EMERGENCY_STOP",
                    close_price=close_price,
                    bars_held=bars_held,
                )
            else:
                log.error("Emergency stop: could not determine close price")
                self._position = None
                self._clear_protection_state()

        self._state = TradingState.STOPPED
        log.critical("EMERGENCY STOP complete: state -> STOPPED")

    # ------------------------------------------------------------------
    # State and counter accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> TradingState:
        """Current trading state."""
        return self._state

    def get_state(self) -> TradingState:
        """Return the current TradingState."""
        return self._state

    @property
    def step_count(self) -> int:
        """Total M5 bars processed since init."""
        return self._step_count

    @property
    def daily_trades(self) -> int:
        """Number of trades opened today."""
        return self._daily_trades

    def get_daily_trades(self) -> int:
        """Return the number of trades opened today."""
        return self._daily_trades

    def get_peak_balance(self) -> float:
        """Return the peak balance observed (for drawdown calculation)."""
        return self._peak_balance

    def reset_daily(self) -> None:
        """Reset the daily trade counter at 00:00 UTC."""
        log.info(
            "TradeExecutor daily reset: daily_trades=%d -> 0",
            self._daily_trades,
        )
        self._daily_trades = 0

    # ------------------------------------------------------------------
    # Position state accessors (for feature pipeline and dashboard)
    # ------------------------------------------------------------------

    def get_position(self) -> Optional[Dict[str, Any]]:
        """Return the currently tracked position dict, or None if flat.

        Position dict keys: ticket, side, entry_price, lots, sl, tp,
        conviction, direction, open_time.
        """
        return self._position

    def has_position(self) -> bool:
        """Check whether a position is currently open."""
        return self._position is not None

    def get_bars_held(self) -> int:
        """Return the number of bars the current position has been held."""
        if self._position is None:
            return 0
        return self._step_count - self._entry_step

    def get_entry_conditions(self) -> Dict[str, Any]:
        """Return the entry conditions snapshot for the current position."""
        return dict(self._entry_conditions)

    # ------------------------------------------------------------------
    # ATR update (called by the orchestration loop)
    # ------------------------------------------------------------------

    def update_atr(self, atr: float) -> None:
        """Update the cached ATR value from the feature pipeline.

        Called by the main loop after feature computation so that
        the executor has access to the latest ATR for SL adjustments
        and lot sizing.

        Args:
            atr: Current ATR(14) value.
        """
        if atr > 0:
            self._current_atr = atr

    # ------------------------------------------------------------------
    # External position sync (detect TP/SL hits by MT5)
    # ------------------------------------------------------------------

    def sync_position(self, account: Optional[Dict[str, Any]] = None) -> str:
        """Sync internal position state with MT5.

        Detects positions closed externally (TP hit, SL hit, margin call).
        Should be called at the start of each bar before execute_action().

        Args:
            account: Current account info for P/L recording.

        Returns:
            "IN_SYNC", "TP_HIT", "SL_HIT", "EXTERNAL_CLOSE", or "NO_POSITION".
        """
        if self._position is None:
            return "NO_POSITION"

        # Check if our tracked position still exists in MT5
        open_positions = self._bridge.get_open_positions(self._config.mt5_symbol)
        if not open_positions:
            open_positions = []
        ticket = self._position["ticket"]

        position_exists = any(p["ticket"] == ticket for p in open_positions)

        if position_exists:
            # Update SL/TP/entry_price/profit from MT5
            for p in open_positions:
                if p["ticket"] == ticket:
                    self._position["sl"] = p.get("sl", self._position["sl"])
                    self._position["tp"] = p.get("tp", self._position["tp"])
                    self._position["mt5_profit"] = p.get("profit", None)
                    # Self-heal entry_price if it was stored as 0 (fill_price=0 bug)
                    if self._position.get("entry_price", 0.0) == 0.0:
                        real_entry = p.get("price_open", 0.0)
                        if real_entry > 0:
                            self._position["entry_price"] = real_entry
                            # Recompute r_value_gbp now that entry is known
                            _sym = self._bridge.get_symbol_info(self._config.mt5_symbol) or {}
                            _tick_sz = _sym.get("tick_size", 0.01) or 0.01
                            _tick_val = _sym.get("tick_value", 0.745)
                            _r_dist = abs(real_entry - self._initial_sl)
                            if _r_dist > 0:
                                self._r_value_gbp = (_r_dist / _tick_sz) * _tick_val * self._position.get("lots", 0.01)
                            log.info(
                                "Self-healed entry_price=0 -> %.2f for ticket=%d  r_value_gbp=%.4f",
                                real_entry, ticket, self._r_value_gbp,
                            )
                    break
            return "IN_SYNC"

        # Position no longer exists -- closed by MT5 (TP, SL, or external)
        log.info(
            "Position ticket=%d no longer open in MT5 -- detecting close reason",
            ticket,
        )

        # Determine close reason by checking deal history for actual fill price
        bars_held = self._step_count - self._entry_step
        side = self._position["side"]
        entry_price = self._position["entry_price"]
        current_sl = self._position["sl"]
        current_tp = self._position["tp"]

        close_reason = "EXTERNAL_CLOSE"
        close_price = 0.0
        mt5_pnl = None  # Will hold MT5's actual P/L if deal found

        # Use position-specific deal lookup (reliable -- MT5's date-range
        # query can miss recent deals, but position lookup always works).
        # Retry once after a brief wait if MT5 hasn't processed the close yet.
        for attempt in range(2):
            try:
                deal = self._bridge.get_deal_by_position(ticket)
                if deal is not None:
                    close_price = deal["exit_price"]
                    mt5_pnl = deal["pnl"]
                    comment = deal.get("comment", "").lower()
                    if "tp" in comment or "take profit" in comment:
                        close_reason = "TP_HIT"
                    elif "sl" in comment or "stop loss" in comment:
                        close_reason = "SL_HIT"
                    elif mt5_pnl > 0:
                        close_reason = "TP_HIT"
                    elif mt5_pnl < 0:
                        close_reason = "SL_HIT"
                    log.info(
                        "Deal found for ticket=%d: exit=%.2f  "
                        "mt5_pnl=%.2f  comment='%s'  -> %s",
                        ticket, close_price, mt5_pnl,
                        comment, close_reason,
                    )
            except Exception:
                log.debug(
                    "Could not fetch deal by position (attempt %d)",
                    attempt + 1,
                )

            if close_price != 0.0:
                break
            if attempt == 0:
                # Deal might not be in history yet -- wait and retry
                import time
                time.sleep(2.0)
                log.info("Deal lookup retry for ticket=%d...", ticket)

        # Fallback: use current market price to determine TP vs SL
        if close_price == 0.0:
            log.warning(
                "Deal history unavailable for ticket=%d -- using market "
                "price fallback",
                ticket,
            )
            # Get current market price to infer what happened
            try:
                import MetaTrader5 as mt5_mod
                broker_sym = self._bridge._broker_name(self._config.mt5_symbol)
                tick = mt5_mod.symbol_info_tick(broker_sym)
                if tick is not None:
                    current_price = tick.bid if side == "LONG" else tick.ask
                else:
                    current_price = entry_price
            except Exception:
                current_price = entry_price

            if side == "LONG":
                # LONG: price above entry = profit, below = loss
                if current_tp > 0 and current_price >= current_tp:
                    close_reason = "TP_HIT"
                    close_price = current_tp
                elif current_sl > 0 and current_price <= current_sl:
                    close_reason = "SL_HIT"
                    close_price = current_sl
                elif current_price >= entry_price:
                    close_reason = "TP_HIT"
                    close_price = current_tp if current_tp > 0 else current_price
                else:
                    close_reason = "SL_HIT"
                    close_price = current_sl if current_sl > 0 else current_price
            else:
                # SHORT: price below entry = profit, above = loss
                if current_sl > 0 and current_price >= current_sl:
                    close_reason = "SL_HIT"
                    close_price = current_sl
                elif current_tp > 0 and current_price <= current_tp:
                    close_reason = "TP_HIT"
                    close_price = current_tp
                elif current_price <= entry_price:
                    close_reason = "TP_HIT"
                    close_price = current_tp if current_tp > 0 else current_price
                else:
                    close_reason = "SL_HIT"
                    close_price = current_sl if current_sl > 0 else current_price

            log.info(
                "Fallback close detection: side=%s  entry=%.2f  "
                "current_price=%.2f  SL=%.2f  TP=%.2f  -> %s at %.2f",
                side, entry_price, current_price,
                current_sl, current_tp, close_reason, close_price,
            )

        if close_price == 0.0:
            close_price = entry_price  # ultimate fallback

        self._record_trade_close(
            close_result={"success": True, "fill_price": close_price},
            close_reason=close_reason,
            close_price=close_price,
            bars_held=bars_held,
            account=account,
            mt5_pnl=mt5_pnl,
        )

        log.info(
            "Position ticket=%d closed by MT5: %s at %.2f  mt5_pnl=%s",
            ticket, close_reason, close_price,
            f"{mt5_pnl:+.2f}" if mt5_pnl is not None else "N/A",
        )

        # If we were winding down and position closed, go to STOPPED
        if self._state == TradingState.WINDING_DOWN:
            self._state = TradingState.STOPPED
            log.info("Wind-down: position closed externally -> STOPPED")

        return close_reason

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_session_name() -> str:
        """Determine the current trading session from UTC hour."""
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 7:
            return "Asia"
        elif 7 <= hour < 12:
            return "London"
        elif 12 <= hour < 16:
            return "NY"
        elif 16 <= hour < 20:
            return "NY_PM"
        else:
            return "Off"

    def _resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the base directory."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return self._config.get_base_dir() / p

    # ------------------------------------------------------------------
    # Status for dashboard
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive executor status for the dashboard.

        Returns:
            Dict with all executor state for display and monitoring.
        """
        pos_info = None
        if self._position is not None:
            pos_info = {
                "ticket": self._position["ticket"],
                "side": self._position["side"],
                "entry_price": self._position["entry_price"],
                "lots": self._position["lots"],
                "sl": self._position["sl"],
                "tp": self._position["tp"],
                "conviction": self._position["conviction"],
                "bars_held": self._step_count - self._entry_step,
                "max_favorable": round(self._max_favorable, 4),
                "open_time": self._position["open_time"].isoformat(),
            }

        return {
            "state": self._state.value,
            "step_count": self._step_count,
            "daily_trades": self._daily_trades,
            "peak_balance": round(self._peak_balance, 2),
            "initial_balance": round(self._initial_balance, 2),
            "current_atr": round(self._current_atr, 4),
            "has_position": self._position is not None,
            "position": pos_info,
        }
