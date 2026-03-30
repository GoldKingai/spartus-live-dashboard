"""Risk management: position sizing, SL/TP calculation, and hard safety rules.

All calculations use ATR-based distances for XAUUSD.
Drawdown-based lot reduction ensures the agent can't bet big when losing.
"""

import numpy as np
from typing import Dict, Tuple

from src.config import TrainingConfig


class RiskManager:
    """Enforces risk rules and calculates position parameters.

    - Lot size: conviction-scaled within max_risk_pct, reduced by drawdown
    - SL/TP: ATR-based distances scaled by conviction
    - Trailing SL: only tightens, minimum distance enforced
    """

    def __init__(self, config: TrainingConfig = None):
        self.cfg = config or TrainingConfig()

    def calculate_lot_size(
        self,
        conviction: float,
        balance: float,
        peak_balance: float,
        atr: float,
        symbol_info: Dict,
        sl_conviction: float = None,
    ) -> float:
        """Calculate position size based on conviction and risk budget.

        Lot sizing formula (MT5-exact):
            risk_amount = balance × max_risk_pct × conviction
            sl_distance = SL in price points (ATR-based)
            value_per_point = tick_value / tick_size (account currency)
            lots = risk_amount / (sl_distance × value_per_point)

        Drawdown reduction:
            DD > 10%  → lots = 0 (emergency stop)
            DD > 8%   → lots × 0.30
            DD > 5%   → lots × 0.60

        Args:
            conviction: Agent's conviction [0, 1].
            balance: Current account balance.
            peak_balance: Highest balance achieved.
            atr: Current ATR(14) value.
            symbol_info: Dict with tick_value, tick_size, volume_min/max/step.
            sl_conviction: Direction-scaled conviction for SL distance (FIX-10).
                          If None, uses conviction directly.

        Returns:
            Position size in lots (0.0 if risk budget exhausted).
        """
        if balance <= 0 or atr <= 0 or conviction <= 0:
            return 0.0

        if sl_conviction is None:
            sl_conviction = conviction

        # Drawdown check
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        if dd >= self.cfg.max_dd:  # 10%
            return 0.0

        # DD reduction: both multiplier AND conviction cap
        # At high DD, force conservative sizing regardless of agent's conviction
        if dd > 0.08:
            dd_mult = 0.30
            conviction = min(conviction, 0.30)  # Cap conviction at 30%
            sl_conviction = min(sl_conviction, 0.30)
        elif dd > 0.05:
            dd_mult = 0.60
            conviction = min(conviction, 0.60)  # Cap conviction at 60%
            sl_conviction = min(sl_conviction, 0.60)
        else:
            dd_mult = 1.0

        # Risk budget
        risk_amount = balance * self.cfg.max_risk_pct * conviction * dd_mult

        # SL distance in price points — uses direction-scaled conviction (FIX-10)
        sl_atr_mult = max(2.5 - sl_conviction, self.cfg.min_sl_atr)
        sl_distance = sl_atr_mult * atr

        # Lots — use MT5-exact value_per_point for correct account currency conversion
        tick_value = symbol_info.get("trade_tick_value", self.cfg.trade_tick_value)
        tick_size = symbol_info.get("trade_tick_size", self.cfg.trade_tick_size)
        value_per_point = tick_value / tick_size  # Account currency per price point per lot
        lots = risk_amount / (sl_distance * value_per_point)

        # Clamp to broker limits
        vol_min = symbol_info.get("volume_min", self.cfg.volume_min)
        vol_max = symbol_info.get("volume_max", self.cfg.volume_max)
        vol_step = symbol_info.get("volume_step", self.cfg.volume_step)

        lots = max(lots, 0.0)
        if lots < vol_min * 0.5:
            return 0.0  # Too far below minimum — risk budget exhausted
        lots = max(lots, vol_min)  # Round up to min lot to preserve conviction granularity
        lots = min(lots, vol_max)

        # Round to step
        lots = round(lots / vol_step) * vol_step

        return lots

    def calculate_sl(
        self,
        side: str,
        entry_price: float,
        atr: float,
        conviction: float,
    ) -> float:
        """Calculate initial stop-loss price.

        SL distance = (2.5 - conviction) × ATR, minimum 1.0 ATR.
        Higher conviction → tighter SL (agent is more confident).
        """
        sl_atr_mult = max(2.5 - conviction, self.cfg.min_sl_atr)
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

        TP distance = (1.5 + 2.5 × conviction) × ATR.
        Higher conviction → wider TP (agent expects bigger move).
        """
        tp_atr_mult = 1.5 + 2.5 * conviction
        tp_distance = tp_atr_mult * atr

        if side == "LONG":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def apply_profit_protection(
        self,
        position: dict,
        current_price: float,
        atr: float,
        spread_points: float = 0.0,
    ) -> tuple:
        """Apply rule-based staged SL protection. Called BEFORE adjust_stop_loss.

        Stages:
            0 — Initial (no protection yet)
            1 — Breakeven (+1.0R): SL = entry + buffer
            2 — Profit Lock (+1.5R): SL = entry + 0.5R
            3 — ATR Trail (+2.0R): SL = price - 1.0*ATR (tighten only)

        Args:
            position: Dict with entry_price, stop_loss, side, max_favorable,
                      and protection_stage (added by trade_env).
            current_price: Current market price.
            atr: Current ATR(14).
            spread_points: Current spread in price points (for BE buffer).

        Returns:
            (new_sl, new_stage) — protection floor SL and updated stage.
        """
        side = position["side"]
        entry = position["entry_price"]
        initial_sl = position.get("initial_sl", position["stop_loss"])
        current_sl = position["stop_loss"]
        mfe = position.get("max_favorable", 0.0)
        stage = position.get("protection_stage", 0)

        # R = initial risk distance
        r_distance = abs(entry - initial_sl)
        if r_distance <= 0:
            return current_sl, stage

        # Current R multiple
        r_multiple = mfe / r_distance

        # Buffer for breakeven: max(spread, configured buffer) in price points
        be_buffer = max(spread_points, self.cfg.protection_be_buffer_pips * self.cfg.point)

        # Determine highest eligible stage (skip stages if price gaps through)
        if r_multiple >= self.cfg.protection_trail_trigger_r:
            target_stage = 3
        elif r_multiple >= self.cfg.protection_lock_trigger_r:
            target_stage = 2
        elif r_multiple >= self.cfg.protection_be_trigger_r:
            target_stage = 1
        else:
            target_stage = 0

        # Only advance stage, never regress
        new_stage = max(stage, target_stage)

        # Calculate protection floor based on stage
        if new_stage >= 3:
            # ATR trail — tighten only
            trail = max(self.cfg.protection_trail_atr_mult * atr, self.cfg.min_sl_trail_atr * atr)
            if side == "LONG":
                floor_sl = current_price - trail
            else:
                floor_sl = current_price + trail
        elif new_stage == 2:
            # Lock +0.5R
            lock_amount = self.cfg.protection_lock_amount_r * r_distance
            if side == "LONG":
                floor_sl = entry + lock_amount
            else:
                floor_sl = entry - lock_amount
        elif new_stage == 1:
            # Breakeven + buffer
            if side == "LONG":
                floor_sl = entry + be_buffer
            else:
                floor_sl = entry - be_buffer
        else:
            return current_sl, new_stage

        # Protection floor: only tighten from current SL
        if side == "LONG":
            new_sl = max(floor_sl, current_sl)
        else:
            new_sl = min(floor_sl, current_sl)

        return new_sl, new_stage

    def adjust_stop_loss(
        self,
        current_sl: float,
        side: str,
        current_price: float,
        atr: float,
        sl_adj: float,
    ) -> float:
        """Adjust trailing stop-loss. Can only tighten, never loosen.

        Args:
            current_sl: Current stop-loss price.
            side: 'LONG' or 'SHORT'.
            current_price: Current market price.
            atr: Current ATR(14).
            sl_adj: Agent's SL adjustment action [0, 1].
                    0 = keep current, 1 = trail tight.

        Returns:
            New stop-loss price (only tighter or same as current).
        """
        # Minimum trail distance
        min_distance = self.cfg.min_sl_trail_atr * atr

        # Target SL based on agent's adjustment
        # sl_adj=0 → keep current, sl_adj=1 → trail to min distance
        trail_distance = min_distance + (1.0 - sl_adj) * 2.0 * atr

        if side == "LONG":
            proposed_sl = current_price - trail_distance
            # Only tighten (move SL up for longs)
            return max(proposed_sl, current_sl)
        else:
            proposed_sl = current_price + trail_distance
            # Only tighten (move SL down for shorts)
            return min(proposed_sl, current_sl)

    def check_position_allowed(
        self,
        balance: float,
        peak_balance: float,
        daily_trade_count: int,
        conviction: float,
    ) -> Tuple[bool, str]:
        """Check if opening a new position is allowed.

        Returns:
            (allowed, reason) tuple.
        """
        # DD emergency stop — block new entries near the limit
        # At 9%+ DD, any spread/slippage can trigger the 10% emergency stop
        # on the same bar, creating pointless 0-bar trades.
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        if dd >= self.cfg.max_dd * 0.90:  # Block at 9% when limit is 10%
            return False, "drawdown_limit"

        # Balance check
        if balance <= 0:
            return False, "bankrupt"

        # Hard daily trade cap — absolute limit, no exceptions
        if daily_trade_count >= self.cfg.daily_trade_hard_cap:
            return False, "daily_hard_cap"

        # Daily trade cap with conviction escalation
        if daily_trade_count >= self.cfg.daily_trade_soft_cap:
            if conviction < self.cfg.elevated_conviction_threshold:
                return False, "daily_cap"

        # Conviction threshold
        if conviction < self.cfg.normal_conviction_threshold:
            return False, "low_conviction"

        return True, "ok"
