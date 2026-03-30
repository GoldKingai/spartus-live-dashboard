"""5-component reward system with EMA normalization.

Components (from spec v3.3.2):
    R1 (0.40): Position P/L — rewards profitable trades
    R2 (0.20): Trade completion quality — R:R ratio, hold time
    R3 (0.15): Drawdown penalty — penalizes equity drawdown
    R4 (0.15): Differential Sharpe — rewards consistent risk-adjusted returns
    R5 (0.10): Risk-adjusted bonus — linear decay with drawdown

Normalization: EMA with tau=0.001, clip to [-5, +5].
"""

import numpy as np
from typing import Dict, Optional

from src.config import TrainingConfig


class DifferentialSharpe:
    """Running differential Sharpe ratio using EMA statistics.

    Tracks A (mean return) and B (mean squared return) with exponential
    moving average, then computes incremental Sharpe contribution.
    """

    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        self._initialized = False

    def update(self, ret: float) -> float:
        """Update with a new return and get the differential Sharpe value.

        Uses Moody & Saffell (2001) formula with old A/B for both
        numerator and denominator to avoid mixing stale/fresh values.
        """
        if not self._initialized:
            self.A = ret
            self.B = ret ** 2
            self._initialized = True
            return 0.0

        old_A = self.A
        old_B = self.B

        delta_A = ret - old_A
        delta_B = ret ** 2 - old_B

        # Differential Sharpe: use OLD values for consistent computation
        # Guard: EMA variance estimate can go slightly negative — clamp to zero
        variance = max(old_B - old_A ** 2, 0.0)
        denom = variance ** 1.5
        if denom < 1e-10:
            # Update EMA and return 0 (not enough variance yet)
            self.A = old_A + self.eta * delta_A
            self.B = old_B + self.eta * delta_B
            return 0.0

        numerator = old_B * delta_A - 0.5 * old_A * delta_B

        # Update EMA after computing the differential
        self.A = old_A + self.eta * delta_A
        self.B = old_B + self.eta * delta_B

        return numerator / (denom + 1e-10)

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self._initialized = False

    def get_state(self) -> dict:
        return {"A": self.A, "B": self.B, "_initialized": self._initialized}

    def set_state(self, state: dict):
        self.A = state.get("A", 0.0)
        self.B = state.get("B", 0.0)
        self._initialized = state.get("_initialized", False)


class RewardNormalizer:
    """EMA-based reward normalization.

    Tracks running mean and variance with exponential moving average,
    then normalizes rewards to approximately zero mean, unit variance.
    """

    def __init__(self, tau: float = 0.001, clip: float = 5.0):
        self.tau = tau
        self.clip = clip
        self.mean = 0.0
        self.var = 1.0
        self._count = 0

    def normalize(self, raw_reward: float) -> float:
        """Normalize a reward value using running statistics."""
        self._count += 1

        # Use OLD mean for variance update (Welford-style)
        old_mean = self.mean
        self.mean = (1 - self.tau) * self.mean + self.tau * raw_reward
        self.var = (1 - self.tau) * self.var + self.tau * (raw_reward - old_mean) ** 2

        std = max(np.sqrt(self.var), 1e-8)
        normalized = (raw_reward - self.mean) / std
        return float(np.clip(normalized, -self.clip, self.clip))

    def reset(self):
        self.mean = 0.0
        self.var = 1.0
        self._count = 0

    def get_state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "_count": self._count}

    def set_state(self, state: dict):
        self.mean = state.get("mean", 0.0)
        self.var = state.get("var", 1.0)
        self._count = state.get("_count", 0)


class RewardCalculator:
    """Computes the 5-component composite reward.

    All components are weighted and summed, then normalized.
    Terminal penalties (bankruptcy, DD, daily DD) bypass this
    and use SET semantics (reward = -X.0).
    """

    def __init__(self, config: TrainingConfig = None):
        self.cfg = config or TrainingConfig()
        self.diff_sharpe = DifferentialSharpe(eta=self.cfg.sharpe_eta)
        self.normalizer = RewardNormalizer(
            tau=self.cfg.reward_normalizer_tau,
            clip=self.cfg.reward_clip,
        )
        self._last_equity = None

    def get_state(self) -> dict:
        return {
            "normalizer": self.normalizer.get_state(),
            "diff_sharpe": self.diff_sharpe.get_state(),
        }

    def set_state(self, state: dict):
        if "normalizer" in state:
            self.normalizer.set_state(state["normalizer"])
        if "diff_sharpe" in state:
            self.diff_sharpe.set_state(state["diff_sharpe"])

    def reset(self, initial_equity: float):
        """Reset for a new episode."""
        self.diff_sharpe.reset()
        # Don't reset normalizer — it carries across episodes for stability
        self._last_equity = initial_equity

    def calculate(
        self,
        equity: float,
        peak_equity: float,
        position: Optional[Dict] = None,
        trade_result: Optional[Dict] = None,
        max_dd: float = 0.10,
    ) -> Dict[str, float]:
        """Calculate the 5-component reward.

        Args:
            equity: Current account equity.
            peak_equity: Highest equity achieved.
            position: Current open position dict or None.
            trade_result: Dict with trade outcome if a trade just closed.
            max_dd: Maximum allowed drawdown (for R5 scaling).

        Returns:
            Dict with 'reward' (normalized), 'raw_reward', and all components.
        """
        # Current drawdown
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # Equity return since last step
        equity_return = 0.0
        if self._last_equity and self._last_equity > 0:
            equity_return = (equity - self._last_equity) / self._last_equity
        self._last_equity = equity

        # R1: Position P/L (step-by-step unrealized change + realized on close)
        r1 = self._calc_r1(equity_return, position, trade_result)

        # R2: Trade completion quality (only on trade close)
        r2 = self._calc_r2(trade_result)

        # R3: Drawdown penalty
        r3 = self._calc_r3(dd)

        # R4: Differential Sharpe
        r4 = self._calc_r4(equity_return)

        # R5: Risk-adjusted bonus — scaled by entry conviction (FIX-1)
        entry_conviction = 1.0
        if position and self.cfg.r5_conviction_scaling:
            entry_conviction = position.get("conviction", 1.0)
        r5 = self._calc_r5(dd, max_dd, position is not None, entry_conviction)

        # Weighted sum
        raw = (self.cfg.r1_weight * r1
               + self.cfg.r2_weight * r2
               + self.cfg.r3_weight * r3
               + self.cfg.r4_weight * r4
               + self.cfg.r5_weight * r5)

        normalized = self.normalizer.normalize(raw)

        return {
            "reward": normalized,
            "raw_reward": raw,
            "r1_position_pnl": r1,
            "r2_trade_quality": r2,
            "r3_drawdown": r3,
            "r4_sharpe": r4,
            "r5_risk_bonus": r5,
        }

    def _calc_r1(self, equity_return: float, position: Optional[Dict],
                 trade_result: Optional[Dict] = None) -> float:
        """R1: Position P/L — rewards profitable equity changes.

        FIX-16: Conviction-normalized R1. The equity_return is divided by
        the position's conviction so that low-conviction trades produce
        conviction-AWARE (not invariant). High conviction trades get slightly
        more reward when right, and slightly more punishment when wrong.
        This teaches the model to use conviction meaningfully.

        Fires when: in position (unrealized P/L changed) OR
        a trade just closed (realized P/L on this step).
        """
        if position is None and trade_result is None:
            return 0.0  # Truly flat — no position and no trade closure

        # Base R1 from raw equity return (no conviction normalization)
        r1 = equity_return * 500.0

        # Conviction-aware scaling: high conviction amplifies both wins and losses
        # This gives the model incentive to be HIGH conviction when right
        # and LOW conviction when uncertain (conviction discrimination)
        conviction = 0.5
        if position:
            conviction = max(position.get("conviction", 0.5), 0.1)
        elif trade_result:
            conviction = max(trade_result.get("conviction", 0.5), 0.1)

        # Scale: conv=0.1 → 0.8x, conv=0.5 → 1.0x, conv=1.0 → 1.2x
        conv_scale = 0.8 + 0.4 * conviction
        r1 *= conv_scale

        # Asymmetric loss penalty: punish losses harder than rewarding wins
        if r1 < 0:
            r1 *= self.cfg.loss_penalty_mult

        # Re-entry penalty: apply to ALL re-entries (FIX-11)
        if trade_result and trade_result.get("is_reentry", False):
            if r1 < 0:
                r1 *= self.cfg.reentry_penalty_mult  # 1.5x amplified loss
            else:
                r1 *= self.cfg.reentry_win_discount   # 0.7x reduced win

        return r1

    def _calc_r2(self, trade_result: Optional[Dict]) -> float:
        """R2: Trade completion quality — only on trade close.

        Rewards good R:R ratio, appropriate hold duration, and
        directional correctness (FIX-6).
        """
        if not trade_result:
            return 0.0

        pnl = trade_result.get("pnl", 0.0)
        risk = trade_result.get("risk_amount", 1.0)
        hold_bars = trade_result.get("hold_bars", 0)

        # R:R ratio
        rr = pnl / (abs(risk) + 1e-8)

        # Hold quality: ramps slowly to reward patience
        # Full credit at 20 bars (100 min), 30% at 6 bars (min hold)
        hold_quality = min(hold_bars / 20.0, 1.0)

        # Direction correctness bonus (FIX-6): reward trades where
        # direction was right AND trade was profitable
        direction_bonus = 1.0
        if pnl > 0:
            entry = trade_result.get("entry_price", 0)
            exit_p = trade_result.get("exit_price", 0)
            side = trade_result.get("side", "")
            price_moved_right = (
                (side == "LONG" and exit_p > entry) or
                (side == "SHORT" and exit_p < entry)
            )
            if price_moved_right:
                direction_bonus = 1.5  # 50% bonus for directionally correct wins

        # Combine: good R:R with reasonable hold time and direction quality
        return rr * hold_quality * direction_bonus

    def _calc_r3(self, dd: float) -> float:
        """R3: Drawdown penalty — progressive penalty as DD increases."""
        if dd <= 0.02:
            return 0.0
        # Quadratic penalty past 2% DD
        return -(dd - 0.02) ** 2 * 100.0

    def _calc_r4(self, equity_return: float) -> float:
        """R4: Differential Sharpe — rewards consistent risk-adjusted returns."""
        return self.diff_sharpe.update(equity_return)

    def _calc_r5(self, dd: float, max_dd: float, in_position: bool,
                 entry_conviction: float = 1.0) -> float:
        """R5: Risk-adjusted bonus — mild incentive to trade.

        When in position: max(0, 0.1 * (1 - dd/max_dd)) * entry_conviction
            Capped at 0.1 (FIX-2), scaled by conviction (FIX-1).
        When flat: small negative (opportunity cost).
        """
        if not in_position:
            return self.cfg.r5_flat_penalty
        raw = max(0.0, 0.1 * (1.0 - dd / max_dd))
        return raw * entry_conviction
