"""SpartusTradeEnv — Gymnasium environment for XAUUSD RL training.

Central integration point: connects features, memory, market sim, risk, and reward.

Observation: 670 dims (67 features × 10 frame stack)
Action: 4 continuous [-1,1] → [direction, conviction, exit_urgency, sl_management]

Features #1-54 are pre-computed in the DataFrame.
Features #55-62 (account) and #63-67 (memory) are computed live each step.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque
from typing import Any, Dict, Optional, Tuple

from src.config import TrainingConfig
from src.environment.market_simulator import MarketSimulator
from src.environment.reward import RewardCalculator
from src.risk.risk_manager import RiskManager
from src.memory.trading_memory import TradingMemory
from src.memory.trend_tracker import TrendTracker
from src.memory.trade_analyzer import TradeAnalyzer


# Pre-computed feature columns (Groups A-F + Upgrades 1-5, 54 features)
PRECOMPUTED_FEATURES = [
    # Group A: Price & Returns (7)
    "close_frac_diff", "returns_1bar", "returns_5bar", "returns_20bar",
    "bar_range", "close_position", "body_ratio",
    # Group B: Volatility (4)
    "atr_14_norm", "atr_ratio", "bb_width", "bb_position",
    # Group C: Momentum & Trend (6)
    "rsi_14", "macd_signal", "adx_14", "ema_cross", "price_vs_ema200", "stoch_k",
    # Group D: Volume (2)
    "volume_ratio", "obv_slope",
    # Group E: Multi-Timeframe (6)
    "h1_trend_dir", "h4_trend_dir", "d1_trend_dir", "h1_rsi", "mtf_alignment", "htf_momentum",
    # Group F: Time & Session (4)
    "hour_sin", "hour_cos", "day_of_week", "session_quality",
    # Upgrade 1: Correlated Instruments (11)
    "eurusd_returns_20", "eurusd_rsi_14", "eurusd_trend",
    "xagusd_returns_20", "xagusd_rsi_14",
    "usdjpy_returns_20", "usdjpy_trend",
    "us500_returns_20", "us500_rsi_14",
    "usoil_returns_20",
    "gold_silver_ratio_z",
    # Upgrade 2: Economic Calendar & Events (6)
    "hours_to_next_high_impact", "hours_to_next_nfp_fomc",
    "in_event_window", "daily_event_density",
    "london_fix_proximity", "comex_session_active",
    # Upgrade 3: Spread & Liquidity (2)
    "spread_estimate_norm", "volume_spike",
    # Upgrade 4: Regime Detection (2)
    "corr_gold_usd_100", "corr_gold_spx_100",
    # Upgrade 5: Session Microstructure (4)
    "asian_range_norm", "asian_range_position",
    "session_momentum", "london_ny_overlap",
]

assert len(PRECOMPUTED_FEATURES) == 54


class SpartusTradeEnv(gym.Env):
    """Gymnasium environment for XAUUSD trading with SAC.

    Takes pre-computed feature DataFrame (from FeatureBuilder + Normalizer).
    Adds live account and memory features at each step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features_df: pd.DataFrame,
        config: TrainingConfig = None,
        memory: Optional[TradingMemory] = None,
        initial_balance: Optional[float] = None,
        week: int = 0,
        seed: int = None,
        is_validation: bool = False,
    ):
        super().__init__()

        self.cfg = config or TrainingConfig()
        self.week = week
        self.is_validation = is_validation
        self.features_df = features_df.reset_index(drop=True)

        # Validate required columns
        missing = [c for c in PRECOMPUTED_FEATURES if c not in self.features_df.columns]
        assert not missing, f"Missing feature columns: {missing}"
        assert "atr_14_raw" in self.features_df.columns, "Missing atr_14_raw column"

        # Pre-extract numpy arrays for fast per-step access (avoid pandas iloc overhead)
        self._precomp_array = np.nan_to_num(
            self.features_df[PRECOMPUTED_FEATURES].to_numpy(dtype=np.float32),
            nan=0.0,
        )
        self._atr_array = np.nan_to_num(
            self.features_df["atr_14_raw"].to_numpy(dtype=np.float64),
            nan=1.0,
        )

        # Components
        self.sim = MarketSimulator(self.cfg, seed=seed)
        self.risk = RiskManager(self.cfg)
        self.reward_calc = RewardCalculator(self.cfg)
        self.memory = memory or TradingMemory(config=self.cfg)
        self.trend_tracker = TrendTracker(self.memory, self.cfg)
        self.trend_tracker.set_week(week)
        self.trade_analyzer = TradeAnalyzer()

        # Balance (carries forward across episodes)
        self._initial_balance = initial_balance or self.cfg.initial_balance
        self.balance = self._initial_balance
        self.peak_balance = self.balance

        # Spaces — bounded to prevent NaN propagation in neural networks
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.cfg.obs_dim,),  # 670
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Episode state
        self.position = None
        self.step_count = 0
        self.episode_trades = 0
        self.daily_trades = 0
        self.current_date = None
        self.daily_start_equity = self.balance
        self._frame_buffer = deque(maxlen=self.cfg.frame_stack)
        self._trade_result_buffer = None  # Set when a trade closes

        # Re-entry penalty tracking
        self._last_sl_side = None    # "LONG" or "SHORT" — side of last SL hit
        self._last_sl_step = 0       # Step when last SL hit occurred

        # Domain randomization offset (set properly in reset())
        self._start_offset = 0
        self._max_steps = len(self.features_df) - self.cfg.lookback

        # Metrics for callback/dashboard
        self.info_history = []

    # === Gym API ============================================================

    def reset(
        self, *, seed: int = None, options: dict = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Per-episode balance reset — every episode starts fresh at £10K
        # Ensures scale-invariant learning (no compounding lot size explosion)
        self.balance = self._initial_balance
        self.peak_balance = self.balance

        # Domain randomization
        self.sim.reset_episode()
        self._start_offset = np.random.randint(0, self.cfg.start_offset_max + 1)
        # Recalculate max steps accounting for offset
        self._max_steps = len(self.features_df) - self.cfg.lookback - self._start_offset
        self._max_steps = max(self._max_steps, 1)  # Guard against empty data

        # Reset episode state
        self.position = None
        self.step_count = 0
        self.episode_trades = 0
        self.daily_trades = 0
        self._trade_result_buffer = None
        self.info_history = []
        self._last_sl_side = None
        self._last_sl_step = 0

        # Initialize date tracking
        first_bar = self._get_bar(0)
        self.current_date = first_bar["time"].date() if hasattr(first_bar["time"], "date") else None
        self.daily_start_equity = self.balance

        # Reset reward calculator
        self.reward_calc.reset(self.balance)

        # Memory episode reset
        self.memory.reset_for_new_episode()

        # Fill frame buffer with copies of the FIRST frame (no look-ahead)
        # Standard frame-stacking init: repeat frame_0 so the agent sees
        # "no temporal history yet" rather than future bars 1-9.
        self._frame_buffer.clear()
        first_frame = self._build_single_frame(0)
        for _ in range(self.cfg.frame_stack):
            self._frame_buffer.append(first_frame.copy())

        obs = self._get_stacked_obs()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Action space (4 continuous):
            [0] direction:      [-1, 1]  negative=short, positive=long
            [1] conviction:     [-1, 1]  mapped to [0, 1]
            [2] exit_urgency:   [-1, 1]  mapped to [0, 1]
            [3] sl_management:  [-1, 1]  mapped to [0, 1]
        """
        action = np.clip(action, -1.0, 1.0)
        direction = float(action[0])
        conviction = float((action[1] + 1.0) / 2.0)   # [0, 1]
        exit_urgency = float((action[2] + 1.0) / 2.0)  # [0, 1]
        sl_adj = float((action[3] + 1.0) / 2.0)        # [0, 1]

        current_bar = self._get_bar(self.step_count)
        done = False
        truncated = False
        self._trade_result_buffer = None

        # Track date changes for daily DD
        bar_date = current_bar["time"].date() if hasattr(current_bar["time"], "date") else None
        if bar_date and bar_date != self.current_date:
            self.current_date = bar_date
            self.daily_start_equity = self._get_equity()
            self.daily_trades = 0

        # 1. Check SL/TP hit on current bar
        self._check_sl_tp_realistic(current_bar)

        # 2. Update position: profit protection first, then AI trailing SL
        if self.position:
            atr = self._get_atr(self.step_count)

            # 2a. Apply rule-based profit protection (sets floor)
            protection_sl, new_stage = self.risk.apply_profit_protection(
                self.position,
                current_bar["close"],
                atr,
                spread_points=self.sim.get_spread(
                    current_bar["time"].hour if hasattr(current_bar["time"], "hour") else 12
                ),
            )
            self.position["stop_loss"] = protection_sl
            self.position["protection_stage"] = new_stage

            # 2b. AI trailing SL (can only tighten beyond protection floor)
            new_sl = self.risk.adjust_stop_loss(
                self.position["stop_loss"],
                self.position["side"],
                current_bar["close"],
                atr,
                sl_adj,
            )
            self.position["stop_loss"] = new_sl

        # 3. Execute agent actions
        if self.position:
            # Has position — check for exit
            if exit_urgency > self.cfg.exit_threshold:
                hold_bars = self.step_count - self.position["entry_step"]
                if hold_bars >= self.cfg.min_hold_bars:
                    self._close_position(current_bar, reason="AGENT_CLOSE")
        else:
            # No position — check for entry
            if abs(direction) > self.cfg.direction_threshold:
                # FIX-3: Combined signal gate — weak direction + weak conviction = no trade
                combined_signal = abs(direction) * conviction
                if combined_signal >= self.cfg.min_combined_signal:
                    side = "LONG" if direction > 0 else "SHORT"
                    allowed, reason = self.risk.check_position_allowed(
                        self.balance, self.peak_balance,
                        self.daily_trades, conviction,
                    )
                    if allowed:
                        self._open_position(side, conviction, current_bar, abs(direction))

        # 4. Trend tracking
        self.trend_tracker.on_step(
            self.step_count, direction, current_bar["close"]
        )

        # 5. Emergency stops (SET semantics — checked BEFORE reward calc
        #    to avoid corrupting normalizer state with discarded values)
        current_equity = self._get_equity()
        current_dd = (self.peak_balance - current_equity) / self.peak_balance if self.peak_balance > 0 else 0.0

        emergency_stop_reward = None
        if self._check_account_blown():
            emergency_stop_reward = -5.0
            done = True
        elif current_dd >= self.cfg.max_dd:
            if self.position:
                self._force_close_position(current_bar)
                emergency_stop_reward = -4.0
                done = True
            else:
                # FIX: No position + DD exceeded = dead zone.
                # Truncate immediately instead of stepping through
                # hundreds of bars where risk manager blocks all trades.
                truncated = True
        elif self._daily_dd_exceeded():
            if self.position:
                self._force_close_position(current_bar)
            emergency_stop_reward = -3.0
            done = True  # End episode — prevents -3.0 cascade on every remaining step

        if emergency_stop_reward is not None:
            # Skip normal reward calc — don't feed garbage into normalizer
            reward = emergency_stop_reward
            reward_info = {
                "reward": reward, "raw_reward": reward,
                "r1_position_pnl": 0, "r2_trade_quality": 0,
                "r3_drawdown": 0, "r4_sharpe": 0, "r5_risk_bonus": 0,
            }
        else:
            reward_info = self.reward_calc.calculate(
                equity=current_equity,
                peak_equity=self.peak_balance,
                position=self.position,
                trade_result=self._trade_result_buffer,
                max_dd=self.cfg.max_dd,
            )
            reward = reward_info["reward"]

        # 6. Update peak balance (use fresh equity after any force-closes)
        self.peak_balance = max(self.peak_balance, self._get_equity())

        # 7. Advance step
        self.step_count += 1
        if self.step_count >= self._max_steps:
            truncated = True
            if self.position:
                self._force_close_position(self._get_bar(self.step_count - 1))

        # 8. Build observation
        if not done and not truncated:
            frame = self._build_single_frame(self.step_count)
            self._frame_buffer.append(frame)

        obs = self._get_stacked_obs()
        info = self._build_info(reward_info)

        return obs, reward, done, truncated, info

    # === Position Management ================================================

    def _open_position(self, side: str, conviction: float, bar: pd.Series,
                       direction_strength: float = 1.0):
        """Open a new position."""
        atr = self._get_atr(self.step_count)
        hour = bar["time"].hour if hasattr(bar["time"], "hour") else 12

        # FIX-10: Effective conviction for SL uses direction strength
        # Strong direction + high conviction = tight SL; weak = wide SL
        sl_conviction = direction_strength * conviction

        # Calculate lot size (uses sl_conviction for SL distance)
        lots = self.risk.calculate_lot_size(
            conviction, self.balance, self.peak_balance, atr, self.cfg.symbol_info,
            sl_conviction=sl_conviction,
        )
        if lots <= 0:
            return

        # Execution price with spread + slippage
        fill_price, spread, slippage = self.sim.get_execution_price(
            side, bar["close"], hour
        )

        # SL and TP — SL uses direction-scaled conviction, TP uses raw conviction
        sl = self.risk.calculate_sl(side, fill_price, atr, sl_conviction)
        tp = self.risk.calculate_tp(side, fill_price, atr, conviction)

        # Risk amount for R2 calculation (account currency, MT5-exact)
        vpp = self.cfg.value_per_point_per_lot  # tick_value / tick_size
        if side == "LONG":
            risk_amount = (fill_price - sl) * lots * vpp
        else:
            risk_amount = (sl - fill_price) * lots * vpp

        # Capture RAW market conditions at entry for memory/pattern binning
        # Use _raw columns (pre-normalization) to get proper 0-1 ranges
        session = self.sim.get_session(hour)
        rsi = bar.get("rsi_14_raw", bar.get("rsi_14", 0.5)) if isinstance(bar, dict) else getattr(bar, "rsi_14_raw", getattr(bar, "rsi_14", 0.5))
        trend = bar.get("h1_trend_dir_raw", bar.get("h1_trend_dir", 0.0)) if isinstance(bar, dict) else getattr(bar, "h1_trend_dir_raw", getattr(bar, "h1_trend_dir", 0.0))
        vol_regime = bar.get("atr_ratio_raw", bar.get("atr_ratio", 1.0)) if isinstance(bar, dict) else getattr(bar, "atr_ratio_raw", getattr(bar, "atr_ratio", 1.0))

        # Check re-entry penalty at open time
        is_reentry = False
        if (self._last_sl_side is not None
                and self._last_sl_side == side
                and (self.step_count - self._last_sl_step) < self.cfg.reentry_penalty_bars):
            is_reentry = True
            self._last_sl_side = None  # Only penalize once

        self.position = {
            "side": side,
            "entry_price": fill_price,
            "entry_step": self.step_count,
            "lots": lots,
            "stop_loss": sl,
            "initial_sl": sl,
            "take_profit": tp,
            "conviction": conviction,
            "risk_amount": risk_amount,
            "max_favorable": 0.0,
            "protection_stage": 0,
            "is_reentry": is_reentry,
            "spread_cost": spread,
            "slippage_cost": slippage,
            "entry_conditions": {
                "hour": hour,
                "session": session,
                "rsi": float(rsi),
                "trend_dir": float(trend),
                "vol_regime": float(vol_regime),
                "atr": float(atr),
                "spread": float(spread),
                "balance": float(self.balance),
                "drawdown": float((self.peak_balance - self.balance) / self.peak_balance)
                    if self.peak_balance > 0 else 0.0,
            },
        }

        self.episode_trades += 1
        self.daily_trades += 1

    def _close_position(self, bar: pd.Series, reason: str = "AGENT_CLOSE"):
        """Close the current position at market price."""
        if not self.position:
            return

        hour = bar["time"].hour if hasattr(bar["time"], "hour") else 12
        fill_price, spread, slippage = self.sim.get_execution_price(
            "SHORT" if self.position["side"] == "LONG" else "LONG",
            bar["close"], hour,
        )
        self._finalize_close(fill_price, reason)

    def _close_at_price(self, price: float, reason: str):
        """Close position at a specific price (SL/TP hit)."""
        if not self.position:
            return
        self._finalize_close(price, reason)

    def _force_close_position(self, bar: pd.Series):
        """Force close for emergency stop (max drawdown hit)."""
        self._close_position(bar, reason="EMERGENCY_STOP")

    def _finalize_close(self, exit_price: float, reason: str):
        """Calculate P/L and record the trade."""
        pos = self.position
        pnl = self.sim.calculate_pnl(
            pos["side"], pos["entry_price"], exit_price, pos["lots"]
        )
        hold_bars = self.step_count - pos["entry_step"]

        pre_trade_balance = self.balance
        self.balance += pnl

        # Profit locked by trailing SL
        if pos["side"] == "LONG":
            locked = max(0, pos["stop_loss"] - pos["entry_price"])
        else:
            locked = max(0, pos["entry_price"] - pos["stop_loss"])
        atr = self._get_atr(self.step_count)
        profit_locked_pct = locked / (atr + 1e-8)

        # Use entry-time conditions (captured at open, not close)
        ec = pos.get("entry_conditions", {})

        # Record to memory (skip during validation to prevent pollution)
        trade_id = None
        if not self.is_validation:
            trade_id = self.memory.record_trade(
                week=self.week, step=self.step_count,
                side=pos["side"], entry_price=pos["entry_price"],
                exit_price=exit_price, lot_size=pos["lots"],
                pnl=pnl, pnl_pct=pnl / max(pre_trade_balance, 1.0),
                hold_bars=hold_bars, close_reason=reason,
                conviction=pos["conviction"],
                rsi_at_entry=ec.get("rsi", 0.5),
                trend_dir_at_entry=ec.get("trend_dir", 0.0),
                session_at_entry=ec.get("session", "unknown"),
                vol_regime_at_entry=ec.get("vol_regime", 1.0),
                entry_conditions=ec,
            )

            # TP tracking
            self.memory.record_tp_tracking(
                trade_id=trade_id,
                tp_price=pos["take_profit"], sl_price=pos["stop_loss"],
                tp_hit=(reason == "TP_HIT"), sl_hit=(reason == "SL_HIT"),
                max_favorable=pos["max_favorable"],
                profit_locked_by_trail=profit_locked_pct,
            )

            # Post-trade analysis — generate journal entry
            trade_dict = {
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "hold_bars": hold_bars,
                "reason": reason,
                "conviction": pos["conviction"],
            }
            tp_data = {
                "tp_hit": reason == "TP_HIT",
                "sl_hit": reason == "SL_HIT",
                "max_favorable": pos["max_favorable"],
                "tp_price": pos["take_profit"],
                "sl_price": pos["stop_loss"],
            }
            journal = self.trade_analyzer.analyze(trade_dict, tp_data, ec)
            self.memory.record_journal_entry(
                trade_id=trade_id,
                week=self.week,
                lesson_type=journal["lesson_type"],
                entry_reasoning=journal["entry_reasoning"],
                exit_analysis=journal["exit_analysis"],
                summary=journal["summary"],
                direction_correct=journal["direction_correct"],
                sl_quality=journal["sl_quality"],
            )

        # Track SL hits for re-entry penalty
        if reason == "SL_HIT":
            self._last_sl_side = pos["side"]
            self._last_sl_step = self.step_count

        # Commission (calculated by simulator, included in pnl already)
        commission = self.sim.get_commission(pos["lots"])

        # Buffer trade result for reward calculation + dashboard
        self._trade_result_buffer = {
            "pnl": pnl,
            "risk_amount": pos["risk_amount"],
            "hold_bars": hold_bars,
            "reason": reason,
            "side": pos["side"],
            "conviction": pos["conviction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "lots": pos["lots"],
            "commission": commission,
            "is_reentry": pos.get("is_reentry", False),
        }

        self.position = None

    # === SL/TP Checking =====================================================

    def _check_sl_tp_realistic(self, bar: pd.Series):
        """Check if SL or TP was hit during this bar using high/low.

        If both could have been hit, conservatively assume SL hit first.
        """
        if not self.position:
            return

        sl = self.position["stop_loss"]
        tp = self.position["take_profit"]

        if self.position["side"] == "LONG":
            sl_hit = bar["low"] <= sl
            tp_hit = bar["high"] >= tp
            # Track max favorable excursion
            mfe = bar["high"] - self.position["entry_price"]
            self.position["max_favorable"] = max(self.position["max_favorable"], mfe)
        else:
            sl_hit = bar["high"] >= sl
            tp_hit = bar["low"] <= tp
            mfe = self.position["entry_price"] - bar["low"]
            self.position["max_favorable"] = max(self.position["max_favorable"], mfe)

        if sl_hit and tp_hit:
            self._close_at_price(sl, reason="SL_HIT")  # Conservative: SL first
        elif sl_hit:
            self._close_at_price(sl, reason="SL_HIT")
        elif tp_hit:
            self._close_at_price(tp, reason="TP_HIT")

    # === Emergency Stops =====================================================

    def _check_account_blown(self) -> bool:
        return self.balance <= 0

    def _daily_dd_exceeded(self) -> bool:
        """Check if daily drawdown exceeds 3%."""
        if self.daily_start_equity <= 0:
            return False
        daily_dd = (self.daily_start_equity - self._get_equity()) / self.daily_start_equity
        return daily_dd > self.cfg.max_daily_dd

    def _get_equity(self) -> float:
        """Current equity including unrealized P/L."""
        if not self.position:
            return self.balance

        bar = self._get_bar(self.step_count)
        price = bar["close"]
        unrealized = self.sim.calculate_pnl(
            self.position["side"],
            self.position["entry_price"],
            price,
            self.position["lots"],
        )
        return self.balance + unrealized

    # === Observation Building ===============================================

    def _build_single_frame(self, step_idx: int) -> np.ndarray:
        """Build a single 42-feature frame for step_idx."""
        bar_idx = self.cfg.lookback + self._start_offset + step_idx
        bar_idx = min(bar_idx, len(self.features_df) - 1)

        # Features #1-29: pre-computed (fast numpy access)
        precomputed = self._precomp_array[bar_idx].copy()

        # ATR from pre-extracted array
        atr = float(self._atr_array[bar_idx])
        if atr <= 0:
            atr = 1.0

        bar = self.features_df.iloc[bar_idx]
        account = self._build_account_features(bar, atr)

        # Features #38-42: memory features (use RAW values for pattern matching)
        market_state = {
            "rsi": float(bar.get("rsi_14_raw", bar.get("rsi_14", 0.5))),
            "trend_dir": float(bar.get("h1_trend_dir_raw", bar.get("h1_trend_dir", 0.0))),
            "session": self.sim.get_session(
                bar["time"].hour if hasattr(bar["time"], "hour") else 12
            ),
            "vol_regime": float(bar.get("atr_ratio_raw", bar.get("atr_ratio", 1.0))),
        }
        memory_feats = self.memory.get_memory_features(market_state, current_step=self.step_count)

        # Combine: 29 + 8 + 5 = 42
        frame = np.concatenate([precomputed, account, memory_feats]).astype(np.float32)

        # Add observation noise (domain randomization)
        if self.cfg.observation_noise_std > 0:
            noise = np.random.normal(0, self.cfg.observation_noise_std, size=frame.shape)
            frame += noise.astype(np.float32)

        return frame

    def _build_account_features(self, bar: pd.Series, atr: float) -> np.ndarray:
        """Build 8 account features (#30-37)."""
        has_pos = 1.0 if self.position else 0.0
        pos_side = 0.0
        unrealized = 0.0
        pos_duration = 0.0
        sl_distance_ratio = 0.0
        profit_locked = 0.0

        if self.position:
            pos_side = 1.0 if self.position["side"] == "LONG" else -1.0
            price = bar["close"]
            unrealized_pnl = self.sim.calculate_pnl(
                self.position["side"],
                self.position["entry_price"],
                price,
                self.position["lots"],
            )
            unrealized = unrealized_pnl / max(self.balance, 1.0)
            pos_duration = min(
                (self.step_count - self.position["entry_step"]) / 100.0, 1.0
            )

            # SL distance ratio
            if self.position["side"] == "LONG":
                sl_dist = price - self.position["stop_loss"]
            else:
                sl_dist = self.position["stop_loss"] - price
            sl_distance_ratio = sl_dist / (atr + 1e-8)

            # Profit locked by SL
            if self.position["side"] == "LONG":
                locked = self.position["stop_loss"] - self.position["entry_price"]
            else:
                locked = self.position["entry_price"] - self.position["stop_loss"]
            if locked > 0:
                profit_locked = locked / (atr + 1e-8)

        equity = self._get_equity()
        dd = (self.peak_balance - equity) / self.peak_balance if self.peak_balance > 0 else 0.0
        eq_ratio = equity / self._initial_balance if self._initial_balance > 0 else 1.0

        return np.array([
            has_pos,            # #30
            pos_side,           # #31
            unrealized,         # #32
            pos_duration,       # #33
            dd,                 # #34
            eq_ratio,           # #35
            sl_distance_ratio,  # #36
            profit_locked,      # #37
        ], dtype=np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack frame buffer into flat observation vector."""
        while len(self._frame_buffer) < self.cfg.frame_stack:
            self._frame_buffer.appendleft(np.zeros(self.cfg.num_features, dtype=np.float32))

        stacked = np.concatenate(list(self._frame_buffer))
        # Replace any NaN/Inf with 0, then hard clip to observation space bounds
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=10.0, neginf=-10.0)
        stacked = np.clip(stacked, -10.0, 10.0)
        return stacked.astype(np.float32)

    # === Helper Methods =====================================================

    def _get_bar(self, step_idx: int) -> pd.Series:
        """Get the OHLCV + features bar for a given step index."""
        bar_idx = self.cfg.lookback + self._start_offset + step_idx
        bar_idx = min(bar_idx, len(self.features_df) - 1)
        return self.features_df.iloc[bar_idx]

    def _get_atr(self, step_idx: int) -> float:
        """Get ATR(14) for a given step (fast numpy lookup)."""
        bar_idx = self.cfg.lookback + self._start_offset + step_idx
        bar_idx = min(bar_idx, len(self._atr_array) - 1)
        atr = float(self._atr_array[bar_idx])
        return atr if atr > 0 else 1.0

    def _build_info(self, reward_info: Optional[Dict] = None) -> Dict:
        """Build step info dict for logging/callback."""
        info = {
            "step": self.step_count,
            "balance": self.balance,
            "equity": self._get_equity(),
            "peak_balance": self.peak_balance,
            "has_position": self.position is not None,
            "episode_trades": self.episode_trades,
            "daily_trades": self.daily_trades,
            "week": self.week,
        }
        if reward_info:
            info.update(reward_info)
        if self._trade_result_buffer:
            info["last_trade"] = self._trade_result_buffer
        return info
