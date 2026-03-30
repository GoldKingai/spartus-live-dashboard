"""Market simulation: spreads, slippage, and commissions.

Session-based spread model for XAUUSD with domain randomization.
All values in price points (not pips). 1 pip = 0.10 for XAUUSD.
"""

import numpy as np
from typing import Dict, Tuple

from src.config import TrainingConfig


class MarketSimulator:
    """Simulates realistic market execution costs.

    - Session-based spreads (London, NY, Asia, Off-hours)
    - Random slippage (Gaussian)
    - Round-trip commissions
    - Domain randomization (jitter per episode)
    """

    def __init__(self, config: TrainingConfig = None, seed: int = None):
        self.cfg = config or TrainingConfig()
        self.rng = np.random.RandomState(seed)
        self.pip = self.cfg.pip_price  # 0.10

        # Base spreads in pips
        self._base_spreads = {
            "london":    self.cfg.spread_london_pips,     # 1.5
            "ny":        self.cfg.spread_ny_pips,          # 2.0
            "asia":      self.cfg.spread_asia_pips,        # 3.0
            "off_hours": self.cfg.spread_off_hours_pips,   # 5.0
        }

        # Episode-level jitter (applied on reset)
        self._spread_mult = 1.0
        self._slippage_mult = 1.0
        self._commission_mult = 1.0

    def reset_episode(self):
        """Randomize execution costs for a new episode (domain randomization)."""
        self._spread_mult = 1.0 + self.rng.uniform(
            -self.cfg.spread_jitter, self.cfg.spread_jitter)
        self._slippage_mult = 1.0 + self.rng.uniform(
            -self.cfg.slippage_jitter, self.cfg.slippage_jitter)
        self._commission_mult = 1.0 + self.rng.uniform(
            -self.cfg.commission_jitter, self.cfg.commission_jitter)

    def get_session(self, hour: int) -> str:
        """Map UTC hour to trading session."""
        if 8 <= hour < 12:
            return "london"
        elif 12 <= hour < 20:
            return "ny"
        elif 0 <= hour < 8:
            return "asia"
        else:
            return "off_hours"

    def get_spread(self, hour: int, is_news: bool = False) -> float:
        """Get current spread in price points.

        Args:
            hour: UTC hour of the bar.
            is_news: Whether a high-impact news event is active.

        Returns:
            Spread in price points (multiply by pip for $ value).
        """
        session = self.get_session(hour)
        spread_pips = self._base_spreads[session] * self._spread_mult
        if is_news:
            spread_pips *= self.cfg.spread_news_multiplier
        return spread_pips * self.pip

    def get_slippage(self) -> float:
        """Get random slippage in price points (always adverse)."""
        slip_pips = abs(self.rng.normal(
            self.cfg.slippage_mean_pips * self._slippage_mult,
            self.cfg.slippage_std_pips * self._slippage_mult,
        ))
        return slip_pips * self.pip

    def get_commission(self, lots: float) -> float:
        """Get commission cost in account currency.

        Args:
            lots: Trade size in standard lots.

        Returns:
            Round-trip commission in account currency.
        """
        return lots * self.cfg.commission_per_lot * self._commission_mult

    def get_execution_price(
        self, side: str, market_price: float, hour: int, is_news: bool = False,
    ) -> Tuple[float, float, float]:
        """Calculate fill price including spread and slippage.

        Args:
            side: 'LONG' or 'SHORT'.
            market_price: Current mid-price.
            hour: UTC hour.
            is_news: Whether news event is active.

        Returns:
            (fill_price, spread_cost, slippage_cost) all in price points.
        """
        half_spread = self.get_spread(hour, is_news) / 2.0
        slippage = self.get_slippage()

        if side == "LONG":
            # Buy at ask (mid + half_spread) + slippage
            fill = market_price + half_spread + slippage
        else:
            # Sell at bid (mid - half_spread) - slippage
            fill = market_price - half_spread - slippage

        return fill, half_spread * 2.0, slippage

    def calculate_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        lots: float,
    ) -> float:
        """Calculate P/L in account currency using MT5-exact tick formula.

        Formula: ticks × tick_value × lots
        Where ticks = price_move / tick_size

        This automatically handles any account currency (USD, GBP, EUR)
        because tick_value is denominated in account currency.
          USD account (tick_value=1.00): 1 tick on 1 lot = $1.00
          GBP account (tick_value=0.7412): 1 tick on 1 lot = £0.7412

        Args:
            side: 'LONG' or 'SHORT'.
            entry_price: Entry fill price.
            exit_price: Exit fill price.
            lots: Position size in standard lots.

        Returns:
            Net P/L in account currency (after commission).
        """
        if side == "LONG":
            price_move = exit_price - entry_price
        else:
            price_move = entry_price - exit_price

        # MT5-exact: (price_move / tick_size) × tick_value × lots
        ticks = price_move / self.cfg.trade_tick_size
        gross = ticks * self.cfg.trade_tick_value * lots

        commission = self.get_commission(lots)
        return gross - commission
