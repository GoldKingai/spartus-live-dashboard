"""Account state features (Group G): 8 features.

IDENTICAL to training trade_env._build_account_features logic.
All features are EXEMPT from z-score normalization (already bounded).

Features:
    - has_position: 1.0 if position open, else 0.0
    - position_side: 1.0 (LONG), -1.0 (SHORT), 0.0 (flat)
    - unrealized_pnl: unrealized P/L / balance, clipped [-1, 1]
    - position_duration: bars_held / 100, clipped [0, 1]
    - current_drawdown: (peak_balance - equity) / peak_balance, clipped [0, 1]
    - equity_ratio: equity / initial_balance, clipped [0, 5]
    - sl_distance_ratio: |price - sl| / atr, clipped [0, 10] (0 if no position)
    - profit_locked_pct: profit locked by trail / atr, clipped [0, 5] (0 if no trail)
"""

import numpy as np
from typing import Dict, Optional


def compute_account_features(
    has_position: bool = False,
    position_side: Optional[str] = None,
    entry_price: float = 0.0,
    current_price: float = 0.0,
    lots: float = 0.0,
    stop_loss: float = 0.0,
    entry_step: int = 0,
    current_step: int = 0,
    balance: float = 100.0,
    equity: float = 100.0,
    peak_balance: float = 100.0,
    initial_balance: float = 100.0,
    atr: float = 1.0,
    value_per_point_per_lot: float = 100.0,
) -> Dict[str, float]:
    """Compute all 8 account features from live position state and account info.

    IDENTICAL to training trade_env._build_account_features.

    Args:
        has_position: Whether a position is currently open.
        position_side: "LONG" or "SHORT" or None.
        entry_price: Position entry price (0 if flat).
        current_price: Current market price.
        lots: Position size in lots (0 if flat).
        stop_loss: Current stop loss price (0 if flat).
        entry_step: Step number when position was opened.
        current_step: Current step number (bars since session start).
        balance: Current account balance (closed P/L).
        equity: Current account equity (balance + unrealized P/L).
        peak_balance: Peak balance for drawdown calculation.
        initial_balance: Starting balance.
        atr: Current ATR(14) value.
        value_per_point_per_lot: Account currency per price point per 1.0 lot
            (default 100.0 for USD on XAUUSD: tick_value/tick_size = 1.0/0.01).

    Returns:
        Dict mapping feature_name -> float value (8 features).
    """
    # Feature #30: has_position
    f_has_pos = 1.0 if has_position else 0.0

    # Feature #31: position_side
    f_pos_side = 0.0
    if has_position and position_side:
        f_pos_side = 1.0 if position_side == "LONG" else -1.0

    # Feature #32: unrealized_pnl (normalized by balance)
    f_unrealized = 0.0
    if has_position and position_side:
        # P/L calculation using MT5-exact formula: ticks * tick_value * lots
        if position_side == "LONG":
            price_move = current_price - entry_price
        else:
            price_move = entry_price - current_price
        unrealized_pnl = price_move * value_per_point_per_lot * lots
        f_unrealized = float(np.clip(unrealized_pnl / max(balance, 1.0), -1.0, 1.0))

    # Feature #33: position_duration (bars held / 100)
    f_duration = 0.0
    if has_position:
        f_duration = min((current_step - entry_step) / 100.0, 1.0)

    # Feature #34: current_drawdown
    if peak_balance > 0:
        dd = (peak_balance - equity) / peak_balance
    else:
        dd = 0.0
    f_drawdown = float(np.clip(dd, 0.0, 1.0))

    # Feature #35: equity_ratio
    if initial_balance > 0:
        eq_ratio = equity / initial_balance
    else:
        eq_ratio = 1.0
    f_eq_ratio = float(np.clip(eq_ratio, 0.0, 5.0))

    # Feature #36: sl_distance_ratio
    f_sl_dist = 0.0
    if has_position and stop_loss > 0 and atr > 0:
        if position_side == "LONG":
            sl_dist = current_price - stop_loss
        else:
            sl_dist = stop_loss - current_price
        f_sl_dist = float(np.clip(sl_dist / (atr + 1e-8), 0.0, 10.0))

    # Feature #37: profit_locked_pct (profit locked by SL trailing)
    f_profit_locked = 0.0
    if has_position and stop_loss > 0 and atr > 0:
        if position_side == "LONG":
            locked = stop_loss - entry_price
        else:
            locked = entry_price - stop_loss
        if locked > 0:
            f_profit_locked = float(np.clip(locked / (atr + 1e-8), 0.0, 5.0))

    return {
        "has_position": f_has_pos,
        "position_side": f_pos_side,
        "unrealized_pnl": f_unrealized,
        "position_duration": f_duration,
        "current_drawdown": f_drawdown,
        "equity_ratio": f_eq_ratio,
        "sl_distance_ratio": f_sl_dist,
        "profit_locked_pct": f_profit_locked,
    }
