"""Live risk management with circuit breakers and emergency stops.

Mirrors the training RiskManager lot-sizing and SL/TP logic exactly,
then layers on live-only safety: circuit breakers, daily DD halt,
consecutive-loss tracking, and post-rounding risk caps.

All monetary calculations use the MT5-exact tick_value / tick_size formula
so account-currency conversion is handled automatically.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np

from config.live_config import LiveConfig

logger = logging.getLogger(__name__)


class LiveRiskManager:
    """Live risk management with circuit breakers and emergency stops.

    Includes: lot sizing, SL/TP calculation, trailing SL.
    Adds: circuit breakers, daily DD halt, consecutive loss tracking.
    """

    def __init__(self, config: LiveConfig) -> None:
        self.cfg = config

        # --- Daily / weekly tracking ---
        self._daily_trade_count: int = 0
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._total_trades: int = 0
        self._total_wins: int = 0

        # --- Circuit-breaker pause state ---
        self._pause_until: datetime | None = None

        # --- Timestamps ---
        self._last_daily_reset: datetime = datetime.now(timezone.utc)
        self._last_weekly_reset: datetime = datetime.now(timezone.utc)

        logger.info(
            "LiveRiskManager initialised  max_risk=%.1f%%  max_dd=%.0f%%  "
            "daily_hard_cap=%d",
            self.cfg.max_risk_pct * 100,
            self.cfg.max_dd * 100,
            self.cfg.daily_trade_hard_cap,
        )

    # ------------------------------------------------------------------
    # Position-allowed gate
    # ------------------------------------------------------------------

    def check_position_allowed(
        self,
        balance: float,
        peak_balance: float,
        daily_trade_count: int,
        conviction: float,
    ) -> Tuple[bool, str]:
        """Check if opening a new position is allowed.

        Args:
            balance: Current account balance (account currency).
            peak_balance: Highest balance achieved.
            daily_trade_count: Trades opened today.
            conviction: Agent's conviction [0, 1].

        Returns:
            (allowed, reason) tuple.  ``reason`` is ``"ok"`` when allowed.
        """
        # DD emergency stop -- block at 90 % of max_dd to avoid slippage
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        if dd >= self.cfg.max_dd * 0.90:
            return False, "drawdown_limit"

        # Balance check
        if balance <= 0:
            return False, "bankrupt"

        # Hard daily trade cap -- absolute limit, no exceptions
        if daily_trade_count >= self.cfg.daily_trade_hard_cap:
            return False, "daily_hard_cap"

        # Circuit-breaker pause
        if self._pause_until is not None:
            now = datetime.now(timezone.utc)
            if now < self._pause_until:
                remaining = int((self._pause_until - now).total_seconds())
                return False, f"circuit_breaker_pause ({remaining}s remaining)"
            # Pause expired
            self._pause_until = None

        # Conviction threshold (separate from direction_threshold)
        if conviction < self.cfg.min_conviction:
            return False, "low_conviction"

        return True, "ok"

    # ------------------------------------------------------------------
    # Lot sizing (matches training RiskManager exactly)
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        conviction: float,
        balance: float,
        peak_balance: float,
        atr: float,
        symbol_info: Dict,
        *,
        side: str = "BUY",
        entry_price: float = 0.0,
        mt5_calc_profit=None,
    ) -> float:
        """Calculate position size based on conviction and risk budget.

        When *mt5_calc_profit* is provided, uses MT5's own
        ``order_calc_profit()`` for exact account-currency P/L instead
        of the manual ``tick_value / tick_size`` approximation.

        Lot-sizing formula:
            risk_amount = balance * max_risk_pct * conviction * dd_multiplier
            sl_distance = max(2.5 - conviction, 1.0) * atr
            lots = risk_amount / mt5_calc_profit(side, lots=1, sl_distance)
                   (or fallback: risk_amount / (sl_distance * tick_value/tick_size))

        Drawdown reduction:
            DD >= 10% -> lots = 0 (emergency stop)
            DD >  8%  -> lots * 0.30, conviction capped at 0.30
            DD >  5%  -> lots * 0.60, conviction capped at 0.60

        Args:
            conviction: Agent's conviction [0, 1].
            balance: Current account balance.
            peak_balance: Highest balance achieved.
            atr: Current ATR(14) value.
            symbol_info: Dict with tick_value, tick_size, volume_min/max/step.
            side: "BUY" or "SELL" (for MT5 calc direction).
            entry_price: Current price (for MT5 calc; 0 = skip MT5 calc).
            mt5_calc_profit: Optional callable(side, lots, open, close) -> float.
                If provided, used for exact P/L instead of tick_value formula.

        Returns:
            Position size in lots (0.0 if risk budget exhausted).
        """
        if balance <= 0 or atr <= 0 or conviction <= 0:
            return 0.0

        # Drawdown check
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        if dd >= self.cfg.max_dd:  # 10 %
            logger.warning("Lot sizing blocked: DD %.2f%% >= max %.2f%%", dd * 100, self.cfg.max_dd * 100)
            return 0.0

        # DD reduction: both multiplier AND conviction cap
        if dd > 0.08:
            dd_mult = 0.30
            conviction = min(conviction, 0.30)
        elif dd > 0.05:
            dd_mult = 0.60
            conviction = min(conviction, 0.60)
        else:
            dd_mult = 1.0

        # Risk budget
        risk_amount = balance * self.cfg.max_risk_pct * conviction * dd_mult

        # SL distance in price points
        sl_atr_mult = max(2.5 - conviction, 1.0)  # min_sl_atr = 1.0
        sl_distance = sl_atr_mult * atr

        # --- Compute value_per_point using MT5 API when available ---
        use_mt5 = (
            mt5_calc_profit is not None
            and entry_price > 0
        )

        if use_mt5:
            # Ask MT5: "what is the P/L for 1 lot moving sl_distance points?"
            if side.upper() == "BUY":
                sl_price = entry_price - sl_distance
            else:
                sl_price = entry_price + sl_distance

            loss_1lot = mt5_calc_profit(side, 1.0, entry_price, sl_price)
            if loss_1lot is not None and loss_1lot != 0:
                value_per_point = abs(loss_1lot) / sl_distance
                raw_lots = risk_amount / abs(loss_1lot)
                logger.debug(
                    "MT5-exact lot sizing: 1-lot SL loss=%.4f %s, "
                    "value_per_point=%.4f, raw_lots=%.6f",
                    abs(loss_1lot),
                    self.cfg.__class__.__name__,
                    value_per_point,
                    raw_lots,
                )
            else:
                # MT5 calc failed -- fall back to manual
                logger.warning("MT5 calc_profit returned %s, falling back to manual", loss_1lot)
                use_mt5 = False

        if not use_mt5:
            # Fallback: manual tick_value / tick_size formula
            tick_value = symbol_info.get("trade_tick_value", 1.0)
            tick_size = symbol_info.get("trade_tick_size", 0.01)
            value_per_point = tick_value / tick_size
            raw_lots = risk_amount / (sl_distance * value_per_point)

        # Clamp to broker limits
        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100.0)
        vol_step = symbol_info.get("volume_step", 0.01)
        point = symbol_info.get("point", 0.01)

        # Compute SL price for risk-cap checks
        if side.upper() == "BUY":
            sl_price = entry_price - sl_distance
        else:
            sl_price = entry_price + sl_distance

        # --- Diagnostic logging ---
        logger.info(
            "LOT_SIZING: conv=%.3f  ATR=%.4f  SL_mult=%.2f  "
            "SL_dist=%.2f price (%.0f pts)  entry=%.2f  sl_price=%.2f  "
            "risk_budget=%.2f  raw_lots=%.6f  vol_min=%.4f",
            conviction, atr, sl_atr_mult,
            sl_distance, sl_distance / point, entry_price, sl_price,
            risk_amount, raw_lots, vol_min,
        )

        raw_lots = max(raw_lots, 0.0)

        # --- Safe min-lot promotion ---
        # When raw_lots < vol_min, check if vol_min is still within the
        # risk cap.  If so, promote to vol_min ("small guy can start small").
        # This replaces both the old dangerous override AND the blanket skip.
        if raw_lots < vol_min:
            if raw_lots <= 0:
                logger.debug("Lot size zero -- trade skipped")
                return 0.0

            # Compute worst-case SL loss at vol_min
            if use_mt5 and entry_price > 0:
                loss_minlot_raw = mt5_calc_profit(
                    side, vol_min, entry_price, sl_price,
                )
                loss_minlot = abs(loss_minlot_raw) if loss_minlot_raw else (
                    vol_min * sl_distance * value_per_point
                )
            else:
                loss_minlot = vol_min * sl_distance * value_per_point

            risk_cap_abs = balance * self.cfg.absolute_risk_cap_pct

            logger.info(
                "MIN-LOT CHECK: raw_lots=%.6f < vol_min=%.4f  |  "
                "loss_at_vol_min=%.2f  risk_cap=%.2f (%.1f%%)  "
                "risk_budget=%.2f",
                raw_lots, vol_min, loss_minlot, risk_cap_abs,
                self.cfg.absolute_risk_cap_pct * 100, risk_amount,
            )

            if loss_minlot <= risk_cap_abs:
                # Safe min-lot promotion: vol_min is within the hard cap
                logger.info(
                    "SAFE MIN-LOT PROMOTION: %.6f -> %.4f  "
                    "(loss=%.2f <= cap=%.2f, %.1f%% of balance)",
                    raw_lots, vol_min, loss_minlot,
                    risk_cap_abs, loss_minlot / balance * 100,
                )
                raw_lots = vol_min
            else:
                # Even vol_min exceeds the absolute risk cap -> SKIP
                logger.warning(
                    "SKIP: vol_min=%.4f exceeds risk cap.  "
                    "loss_minlot=%.2f (%.1f%%) > cap=%.2f (%.1f%%).  "
                    "Balance too small for this ATR.",
                    vol_min, loss_minlot, loss_minlot / balance * 100,
                    risk_cap_abs, self.cfg.absolute_risk_cap_pct * 100,
                )
                return 0.0

        raw_lots = min(raw_lots, vol_max)

        # Floor to step (never round UP into higher risk)
        lots = math.floor(raw_lots / vol_step) * vol_step
        lots = max(lots, vol_min)  # Ensure at least vol_min after floor

        # --- Hard post-rounding risk cap (always enforced) ---
        # Compute actual worst-case SL loss at the rounded lot size.
        if use_mt5 and entry_price > 0:
            actual_risk_abs = mt5_calc_profit(side, lots, entry_price, sl_price)
            actual_risk = abs(actual_risk_abs) if actual_risk_abs else lots * sl_distance * value_per_point
        else:
            actual_risk = lots * sl_distance * value_per_point

        risk_cap_abs = balance * self.cfg.absolute_risk_cap_pct
        if actual_risk > risk_cap_abs:
            # Try to reduce lots to fit within the hard cap
            target_lots = risk_cap_abs / (sl_distance * value_per_point)
            target_lots = math.floor(target_lots / vol_step) * vol_step
            if target_lots >= vol_min:
                logger.info(
                    "Post-rounding risk cap: lots %.2f -> %.2f "
                    "(risk %.2f > cap %.2f)",
                    lots, target_lots, actual_risk, risk_cap_abs,
                )
                lots = target_lots
            else:
                # Even vol_min exceeds the absolute risk cap -> SKIP trade
                actual_risk_pct = actual_risk / balance * 100 if balance > 0 else 0
                logger.warning(
                    "SKIP: vol_min=%.4f exceeds risk cap.  "
                    "actual_risk=%.2f (%.1f%%) > cap=%.2f (%.1f%%).  "
                    "Balance too small for this ATR.",
                    vol_min, actual_risk, actual_risk_pct,
                    risk_cap_abs, self.cfg.absolute_risk_cap_pct * 100,
                )
                return 0.0

        logger.info(
            "LOT_SIZING RESULT: lots=%.4f  actual_risk=%.2f (%.1f%%)  "
            "cap=%.2f (%.1f%%)  mt5_exact=%s",
            lots, actual_risk, actual_risk / balance * 100 if balance > 0 else 0,
            risk_cap_abs, self.cfg.absolute_risk_cap_pct * 100, use_mt5,
        )

        return lots

    # ------------------------------------------------------------------
    # SL / TP calculations (match training exactly)
    # ------------------------------------------------------------------

    def calculate_sl(
        self,
        side: str,
        entry_price: float,
        atr: float,
        conviction: float,
    ) -> float:
        """Calculate initial stop-loss price.

        SL distance = max(2.5 - conviction, 1.0) * atr.
        Higher conviction -> tighter SL (agent is more confident).
        """
        sl_atr_mult = max(2.5 - conviction, 1.0)
        sl_distance = sl_atr_mult * atr

        if side == "LONG":
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance

    def calculate_tp(
        self,
        side: str,
        entry_price: float,
        atr: float,
        conviction: float,
    ) -> float:
        """Calculate take-profit price.

        TP distance = (1.5 + 2.5 * conviction) * atr.
        Higher conviction -> wider TP (agent expects bigger move).
        """
        tp_atr_mult = 1.5 + 2.5 * conviction
        tp_distance = tp_atr_mult * atr

        if side == "LONG":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    # ------------------------------------------------------------------
    # Trailing SL (match training exactly)
    # ------------------------------------------------------------------

    def adjust_stop_loss(
        self,
        current_sl: float,
        side: str,
        current_price: float,
        atr: float,
        sl_adjustment: float,
    ) -> float:
        """Adjust trailing stop-loss.  Can only tighten, never loosen.

        Args:
            current_sl: Current stop-loss price.
            side: ``"LONG"`` or ``"SHORT"``.
            current_price: Current market price.
            atr: Current ATR(14).
            sl_adjustment: Agent's SL adjustment action [0, 1].
                           0 = keep current, 1 = trail tight.

        Returns:
            New stop-loss price (only tighter or same as current).
        """
        # Minimum trail distance: 0.5 * atr
        min_distance = 0.5 * atr

        # Target SL based on agent's adjustment
        trail_distance = min_distance + (1.0 - sl_adjustment) * 2.0 * atr

        if side == "LONG":
            proposed_sl = current_price - trail_distance
            return max(proposed_sl, current_sl)
        else:
            proposed_sl = current_price + trail_distance
            return min(proposed_sl, current_sl)

    # ------------------------------------------------------------------
    # Trade result tracking
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float) -> None:
        """Record a closed trade for safety tracking.

        Updates consecutive-loss counter, daily P/L, and win/loss stats.
        Triggers circuit-breaker pauses when consecutive losses exceed
        the configured thresholds.

        Args:
            pnl: Realised P/L of the closed trade (account currency).
        """
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._daily_trade_count += 1
        self._total_trades += 1

        if pnl < 0:
            self._consecutive_losses += 1
            logger.info(
                "Loss recorded: $%.2f  consecutive=%d  daily_pnl=$%.2f",
                pnl, self._consecutive_losses, self._daily_pnl,
            )

            # Circuit-breaker pause escalation
            now = datetime.now(timezone.utc)
            if self._consecutive_losses >= self.cfg.severe_loss_pause:
                pause_minutes = self.cfg.severe_loss_pause_minutes
                self._pause_until = now.replace(
                    minute=now.minute,
                ) + __import__("datetime").timedelta(minutes=pause_minutes)
                logger.warning(
                    "CIRCUIT BREAKER: %d consecutive losses -> %d min pause",
                    self._consecutive_losses, pause_minutes,
                )
            elif self._consecutive_losses >= self.cfg.consecutive_loss_pause:
                pause_minutes = self.cfg.consecutive_loss_pause_minutes
                self._pause_until = now.replace(
                    minute=now.minute,
                ) + __import__("datetime").timedelta(minutes=pause_minutes)
                logger.warning(
                    "CIRCUIT BREAKER: %d consecutive losses -> %d min pause",
                    self._consecutive_losses, pause_minutes,
                )
        else:
            if self._consecutive_losses > 0:
                logger.info(
                    "Win recorded: $%.2f  breaking %d-loss streak",
                    pnl, self._consecutive_losses,
                )
            self._consecutive_losses = 0
            self._total_wins += 1

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily counters at 00:00 UTC."""
        logger.info(
            "Daily reset: trades=%d  pnl=$%.2f  consec_losses=%d",
            self._daily_trade_count, self._daily_pnl, self._consecutive_losses,
        )
        self._daily_trade_count = 0
        self._daily_pnl = 0.0
        self._pause_until = None
        self._last_daily_reset = datetime.now(timezone.utc)

    def reset_weekly(self) -> None:
        """Reset weekly counters (typically at Monday open)."""
        logger.info("Weekly reset: weekly_pnl=$%.2f", self._weekly_pnl)
        self._weekly_pnl = 0.0
        self._last_weekly_reset = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Safety status (for dashboard display)
    # ------------------------------------------------------------------

    def get_safety_status(self) -> Dict:
        """Return all safety metrics for dashboard display.

        Returns:
            Dictionary with current risk/safety state.
        """
        now = datetime.now(timezone.utc)
        pause_remaining = 0
        if self._pause_until is not None and now < self._pause_until:
            pause_remaining = int((self._pause_until - now).total_seconds())

        win_rate = (
            self._total_wins / self._total_trades
            if self._total_trades > 0
            else 0.0
        )

        return {
            "daily_trade_count": self._daily_trade_count,
            "daily_pnl": round(self._daily_pnl, 2),
            "weekly_pnl": round(self._weekly_pnl, 2),
            "consecutive_losses": self._consecutive_losses,
            "total_trades": self._total_trades,
            "total_wins": self._total_wins,
            "win_rate": round(win_rate, 4),
            "is_paused": pause_remaining > 0,
            "pause_remaining_s": pause_remaining,
            "last_daily_reset": self._last_daily_reset.isoformat(),
            "last_weekly_reset": self._last_weekly_reset.isoformat(),
        }

    # ------------------------------------------------------------------
    # Properties for external access
    # ------------------------------------------------------------------

    @property
    def daily_trade_count(self) -> int:
        return self._daily_trade_count

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def weekly_pnl(self) -> float:
        return self._weekly_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses
