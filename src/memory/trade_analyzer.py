"""Post-trade analysis and reasoning engine.

Analyzes each closed trade and generates:
1. Human-readable entry reasoning (why the AI took the trade)
2. Exit analysis (what happened and why)
3. Lesson classification (what can be learned)
4. Self-reflection metrics (fed back into observation vector)
"""

import json
from typing import Dict, Optional, Tuple


# Lesson type classifications
LESSON_TYPES = {
    "GOOD_TRADE":              "Correct direction, profitable exit",
    "CORRECT_DIR_CLOSED_EARLY": "Right direction but closed too early — price continued",
    "CORRECT_DIR_BAD_SL":      "Right direction but SL was too tight",
    "WRONG_DIRECTION":         "Price moved against the position",
    "BAD_TIMING":              "Direction eventually right, but entry timing was poor",
    "WHIPSAW":                 "Caught in choppy/ranging market",
    "EMERGENCY_STOP":          "Account-level forced close (drawdown limit)",
    "BREAKEVEN":               "Closed near entry — no significant P/L",
    "SCALP_WIN":               "Quick small profit — in and out fast",
    "HELD_TOO_LONG":           "Profit eroded by holding past the turning point",
}


class TradeAnalyzer:
    """Generates structured reasoning and lessons for each completed trade.

    This serves two purposes:
    1. Human-readable journal entries (displayed in Trade Journal tab)
    2. Classification metrics fed back into the AI's observation vector
    """

    def analyze(
        self,
        trade: Dict,
        tp_data: Dict,
        entry_conditions: Dict,
        price_after_close: Optional[float] = None,
    ) -> Dict:
        """Analyze a completed trade and return journal entry data.

        Args:
            trade: Trade data (side, entry/exit price, pnl, hold_bars, etc.)
            tp_data: TP tracking data (tp_hit, sl_hit, max_favorable, tp_price, sl_price)
            entry_conditions: Market conditions at entry time
            price_after_close: Price N bars after close (for hindsight analysis).
                               None if not yet available.

        Returns:
            Dict with: lesson_type, entry_reasoning, exit_analysis,
                       summary, direction_correct, sl_quality
        """
        side = trade["side"]
        pnl = trade["pnl"]
        hold_bars = trade["hold_bars"]
        reason = trade["reason"]
        conviction = trade["conviction"]
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]

        max_fav = tp_data.get("max_favorable", 0.0)
        tp_hit = tp_data.get("tp_hit", False)
        sl_hit = tp_data.get("sl_hit", False)
        tp_price = tp_data.get("tp_price", 0.0)
        sl_price = tp_data.get("sl_price", 0.0)

        ec = entry_conditions or {}

        # === Direction Analysis ===
        if side == "LONG":
            price_moved_right = exit_price > entry_price
            sl_distance = entry_price - sl_price if sl_price else 0
            tp_distance = tp_price - entry_price if tp_price else 0
        else:
            price_moved_right = exit_price < entry_price
            sl_distance = sl_price - entry_price if sl_price else 0
            tp_distance = entry_price - tp_price if tp_price else 0

        direction_correct = 1 if price_moved_right else 0

        # === SL Quality ===
        sl_quality = self._assess_sl_quality(max_fav, sl_distance, tp_distance, sl_hit, pnl)

        # === Lesson Classification ===
        lesson_type = self._classify_lesson(
            pnl, hold_bars, reason, direction_correct, sl_hit, tp_hit,
            max_fav, sl_distance, tp_distance, conviction
        )

        # === Entry Reasoning (human-readable) ===
        entry_reasoning = self._build_entry_reasoning(side, conviction, ec)

        # === Exit Analysis (human-readable) ===
        exit_analysis = self._build_exit_analysis(
            side, pnl, hold_bars, reason, max_fav, sl_distance,
            tp_distance, direction_correct, sl_quality
        )

        # === Summary ===
        summary = self._build_summary(
            side, pnl, lesson_type, hold_bars, reason, ec
        )

        return {
            "lesson_type": lesson_type,
            "entry_reasoning": entry_reasoning,
            "exit_analysis": exit_analysis,
            "summary": summary,
            "direction_correct": direction_correct,
            "sl_quality": sl_quality,
        }

    def _assess_sl_quality(
        self, max_fav: float, sl_dist: float, tp_dist: float,
        sl_hit: bool, pnl: float
    ) -> str:
        """Assess whether the SL placement was appropriate.

        sl_dist > 0: SL is on the correct side of entry (normal risk)
        sl_dist <= 0: Trailing SL has moved past entry — profit locked in
        """
        if sl_dist < 0:
            # Trailing SL moved past entry price — profit was locked
            return "GOOD" if pnl > 0 else "TRAILED"

        if sl_dist == 0:
            # SL exactly at entry (breakeven stop)
            return "BREAKEVEN_STOP"

        fav_ratio = max_fav / sl_dist

        if pnl > 0:
            return "GOOD"
        if sl_hit and fav_ratio > 1.5:
            # Price went well in our favor then reversed to hit SL
            return "TOO_TIGHT"
        if sl_hit and fav_ratio < 0.3:
            # Price never went in our favor — wrong direction
            return "OK"
        if not sl_hit and pnl < 0:
            # Lost money without SL hit — emergency stop or agent close
            return "IRRELEVANT"
        return "OK"

    def _classify_lesson(
        self, pnl, hold_bars, reason, dir_correct, sl_hit, tp_hit,
        max_fav, sl_dist, tp_dist, conviction
    ) -> str:
        """Classify the trade into a lesson type."""
        # Emergency stop is its own category
        if reason in ("EMERGENCY_STOP", "CIRCUIT_BREAKER"):
            return "EMERGENCY_STOP"

        # TP hit — good trade
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

        # Lost money — classify why
        if not dir_correct:
            # Wrong direction
            if sl_dist > 0 and max_fav / sl_dist > 1.0:
                return "BAD_TIMING"  # Eventually went right but entered wrong
            return "WRONG_DIRECTION"

        # Direction was correct but still lost
        if sl_hit:
            if sl_dist > 0 and max_fav / sl_dist > 1.5:
                return "CORRECT_DIR_BAD_SL"  # SL too tight
            return "WHIPSAW"  # Market chopped around

        # Agent closed at a loss despite correct direction
        if hold_bars >= 20 and max_fav > abs(pnl):
            return "HELD_TOO_LONG"
        return "CORRECT_DIR_CLOSED_EARLY"

    def _build_entry_reasoning(self, side: str, conviction: float, ec: Dict) -> str:
        """Build human-readable entry reasoning."""
        parts = []

        # Direction reasoning
        trend = ec.get("trend_dir", 0)
        rsi = ec.get("rsi", 0.5)
        session = ec.get("session", "unknown")
        hour = ec.get("hour", 0)
        atr = ec.get("atr", 0)
        dd = ec.get("drawdown", 0)

        # Trend context
        if abs(trend) > 0.5:
            trend_word = "bullish" if trend > 0 else "bearish"
            parts.append(f"H1 trend {trend_word} ({trend:+.2f})")
        else:
            parts.append(f"H1 trend neutral ({trend:+.2f})")

        # RSI context
        if rsi > 0.7:
            parts.append(f"RSI overbought ({rsi:.0%})")
        elif rsi < 0.3:
            parts.append(f"RSI oversold ({rsi:.0%})")
        else:
            parts.append(f"RSI neutral ({rsi:.0%})")

        # Session
        parts.append(f"{session} session (UTC {hour:02d}:00)")

        # Conviction
        conv_word = "high" if conviction > 0.7 else ("moderate" if conviction > 0.4 else "low")
        parts.append(f"{conv_word} conviction ({conviction:.0%})")

        # Drawdown context
        if dd > 0.05:
            parts.append(f"under drawdown pressure ({dd:.1%})")

        action = "BUY" if side == "LONG" else "SELL"
        return f"{action} — {'; '.join(parts)}"

    def _build_exit_analysis(
        self, side, pnl, hold_bars, reason, max_fav, sl_dist, tp_dist,
        dir_correct, sl_quality
    ) -> str:
        """Build human-readable exit analysis."""
        parts = []

        # Exit reason
        reason_map = {
            "TP_HIT": "Take profit reached",
            "SL_HIT": "Stop loss triggered",
            "AGENT_CLOSE": "AI decided to close",
            "EMERGENCY_STOP": "Emergency stop (max drawdown)",
            "CIRCUIT_BREAKER": "Emergency stop (max drawdown)",
            "TRUNCATED": "Episode ended (week boundary)",
        }
        parts.append(reason_map.get(reason, reason))

        # P/L
        if pnl > 0:
            parts.append(f"profit +\u00a3{pnl:.2f}")
        elif pnl < 0:
            parts.append(f"loss -\u00a3{abs(pnl):.2f}")
        else:
            parts.append("breakeven")

        # Hold duration
        parts.append(f"held {hold_bars} bars")

        # Max favorable excursion
        if max_fav > 0 and sl_dist > 0:
            fav_ratio = max_fav / sl_dist
            if fav_ratio > 2.0:
                parts.append(f"price went {fav_ratio:.1f}x SL in favor then reversed")
            elif fav_ratio > 1.0:
                parts.append(f"price went {fav_ratio:.1f}x SL distance in favor")
            else:
                parts.append(f"max favorable was only {fav_ratio:.1f}x SL distance")

        # SL quality note
        if sl_quality == "TOO_TIGHT":
            parts.append("SL was too tight for the move")
        elif sl_quality == "GOOD" and pnl > 0:
            parts.append("good SL placement")

        return "; ".join(parts)

    def _build_summary(self, side, pnl, lesson_type, hold_bars, reason, ec) -> str:
        """Build a concise one-line summary."""
        session = ec.get("session", "?")
        trend = ec.get("trend_dir", 0)
        action = "LONG" if side == "LONG" else "SHORT"

        trend_word = "with trend" if (
            (side == "LONG" and trend > 0.3) or (side == "SHORT" and trend < -0.3)
        ) else "against trend" if (
            (side == "LONG" and trend < -0.3) or (side == "SHORT" and trend > 0.3)
        ) else "sideways"

        pnl_str = f"+\u00a3{pnl:.2f}" if pnl > 0 else f"-\u00a3{abs(pnl):.2f}"

        lesson_short = {
            "GOOD_TRADE": "Good trade",
            "CORRECT_DIR_CLOSED_EARLY": "Right call, exited too early",
            "CORRECT_DIR_BAD_SL": "Right call, SL too tight",
            "WRONG_DIRECTION": "Wrong direction",
            "BAD_TIMING": "Bad entry timing",
            "WHIPSAW": "Whipsawed",
            "EMERGENCY_STOP": "Emergency stop hit",
            "CIRCUIT_BREAKER": "Emergency stop hit",
            "BREAKEVEN": "Breakeven",
            "SCALP_WIN": "Quick scalp profit",
            "HELD_TOO_LONG": "Held past the turn",
        }.get(lesson_type, lesson_type)

        return (
            f"{action} {trend_word} in {session} | "
            f"{pnl_str} over {hold_bars} bars | "
            f"{lesson_short}"
        )
