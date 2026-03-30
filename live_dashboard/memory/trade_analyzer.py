"""Classify trades into lesson types for the journal."""
import logging

log = logging.getLogger(__name__)


class TradeAnalyzer:
    """Classify trades into lesson types."""

    def __init__(self, memory=None):
        self.memory = memory

    def classify_trade(self, trade_data: dict) -> str:
        """Return lesson_type classification for a completed trade.

        Categories:
        - GOOD_TRADE: Profitable, well-managed
        - WRONG_DIRECTION: Lost money, price moved against immediately
        - BAD_TIMING: Had the right idea but entered too early/late
        - CORRECT_DIR_EARLY: SL hit, but price later moved in predicted direction
        - WHIPSAW: Very short hold, quick loss
        - HELD_TOO_LONG: Had significant unrealized profit but gave it back
        - SMALL_WIN: Small profit
        - NEUTRAL: Doesn't fit other categories
        """
        pnl = trade_data.get('pnl', 0)
        hold_bars = trade_data.get('hold_bars', 0)
        close_reason = trade_data.get('close_reason', '')
        max_favorable = trade_data.get('max_favorable', 0)

        # Whipsaw: very quick loss
        if pnl < 0 and hold_bars < 3:
            return "WHIPSAW"

        # Held too long: had big profit but gave it back
        if pnl < 0 and max_favorable > 0 and max_favorable > 2 * abs(pnl):
            return "HELD_TOO_LONG"

        # Bad timing: was right direction but entered wrong
        if pnl < 0 and max_favorable > abs(pnl):
            return "BAD_TIMING"

        # Correct direction early exit: SL hit but price eventually moved right
        if pnl < 0 and close_reason == "SL_HIT" and max_favorable > 0:
            return "CORRECT_DIR_EARLY"

        # Wrong direction: lost money, never profitable
        if pnl < 0 and (max_favorable <= 0 or close_reason == "SL_HIT"):
            return "WRONG_DIRECTION"

        # Good trade: profitable with decent management
        if pnl > 0 and close_reason in ("TP_HIT", "AGENT") and hold_bars >= 3:
            return "GOOD_TRADE"

        # Small win
        if pnl > 0 and pnl < 1.0:
            return "SMALL_WIN"

        # Default profitable
        if pnl > 0:
            return "GOOD_TRADE"

        return "NEUTRAL"
