"""Circuit breaker -- pause or halt trading after consecutive losses or drawdown.

Escalation levels:
    1. 3 consecutive losses  -> pause 30 min
    2. 5 consecutive losses  -> pause 2 hours
    3. Daily DD > 2%         -> halt rest of day (close all)
    4. Daily DD > 3%         -> close all + halt rest of day
    5. Weekly DD > 5%        -> halt until manual reset

The ``should_trade`` method is the single gate that the execution loop
checks before every new entry.  It returns ``(allowed, reason)`` so the
dashboard can show exactly *why* trading is blocked.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple

from config.live_config import LiveConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Pause trading after consecutive losses or daily DD."""

    def __init__(self, config: LiveConfig) -> None:
        self.cfg = config

        # --- Consecutive-loss tracking ---
        self._consecutive_losses: int = 0
        self._pause_until: datetime | None = None

        # --- Drawdown tracking ---
        self._daily_dd: float = 0.0
        self._weekly_dd: float = 0.0

        # --- Halt flags ---
        self._daily_halted: bool = False
        self._weekly_halted: bool = False
        self._close_all_triggered: bool = False

        logger.info(
            "CircuitBreaker initialised  "
            "pause_at=%d (%d min)  severe_at=%d (%d min)  "
            "daily_dd_halt=%.1f%%  daily_dd_close=%.1f%%  weekly_dd_halt=%.1f%%",
            self.cfg.consecutive_loss_pause,
            self.cfg.consecutive_loss_pause_minutes,
            self.cfg.severe_loss_pause,
            self.cfg.severe_loss_pause_minutes,
            self.cfg.daily_dd_halt_pct * 100,
            self.cfg.daily_dd_close_all_pct * 100,
            self.cfg.weekly_dd_halt_pct * 100,
        )

    # ------------------------------------------------------------------
    # Primary gate
    # ------------------------------------------------------------------

    def should_trade(self) -> Tuple[bool, str]:
        """Check whether the circuit breaker allows a new trade.

        Returns:
            (allowed, reason) tuple.
        """
        # Weekly halt takes highest priority -- requires manual reset
        if self._weekly_halted:
            return False, "weekly_dd_halt (manual reset required)"

        # Daily halt -- clears at 00:00 UTC
        if self._daily_halted:
            return False, "daily_dd_halt"

        # Time-based pause from consecutive losses
        if self._pause_until is not None:
            now = datetime.now(timezone.utc)
            if now < self._pause_until:
                remaining = int((self._pause_until - now).total_seconds())
                return False, f"consecutive_loss_pause ({remaining}s remaining)"
            # Pause expired -- clear it
            self._pause_until = None
            logger.info("Consecutive-loss pause expired, trading resumed")

        return True, "ok"

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_loss(self) -> None:
        """Record a losing trade and escalate if thresholds breached."""
        self._consecutive_losses += 1
        now = datetime.now(timezone.utc)

        if self._consecutive_losses >= self.cfg.severe_loss_pause:
            # Level 2: severe pause (e.g. 5 losses -> 2 hours)
            self._pause_until = now + timedelta(
                minutes=self.cfg.severe_loss_pause_minutes,
            )
            logger.warning(
                "CIRCUIT BREAKER L2: %d consecutive losses -> %d min pause",
                self._consecutive_losses,
                self.cfg.severe_loss_pause_minutes,
            )
        elif self._consecutive_losses >= self.cfg.consecutive_loss_pause:
            # Level 1: moderate pause (e.g. 3 losses -> 30 min)
            self._pause_until = now + timedelta(
                minutes=self.cfg.consecutive_loss_pause_minutes,
            )
            logger.warning(
                "CIRCUIT BREAKER L1: %d consecutive losses -> %d min pause",
                self._consecutive_losses,
                self.cfg.consecutive_loss_pause_minutes,
            )

    def record_win(self) -> None:
        """Record a winning trade -- resets the consecutive-loss counter."""
        if self._consecutive_losses > 0:
            logger.info(
                "Win breaks %d-loss streak, resetting consecutive counter",
                self._consecutive_losses,
            )
        self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # Drawdown updates
    # ------------------------------------------------------------------

    def update_dd(self, daily_dd: float, weekly_dd: float) -> None:
        """Update drawdown levels and trigger halts if breached.

        Args:
            daily_dd: Today's drawdown as a fraction (e.g. 0.025 = 2.5%).
            weekly_dd: This week's drawdown as a fraction.
        """
        self._daily_dd = daily_dd
        self._weekly_dd = weekly_dd

        # Weekly DD > 5% -> halt until manual reset
        if weekly_dd >= self.cfg.weekly_dd_halt_pct and not self._weekly_halted:
            self._weekly_halted = True
            self._close_all_triggered = True
            logger.critical(
                "CIRCUIT BREAKER: Weekly DD %.2f%% >= %.2f%% -- "
                "HALT until manual reset.  Close all positions!",
                weekly_dd * 100, self.cfg.weekly_dd_halt_pct * 100,
            )

        # Daily DD > 3% -> close all + halt rest of day
        if daily_dd >= self.cfg.daily_dd_close_all_pct and not self._daily_halted:
            self._daily_halted = True
            self._close_all_triggered = True
            logger.critical(
                "CIRCUIT BREAKER: Daily DD %.2f%% >= %.2f%% -- "
                "CLOSE ALL + halt rest of day",
                daily_dd * 100, self.cfg.daily_dd_close_all_pct * 100,
            )

        # Daily DD > 2% -> halt rest of day (no forced close)
        elif daily_dd >= self.cfg.daily_dd_halt_pct and not self._daily_halted:
            self._daily_halted = True
            logger.warning(
                "CIRCUIT BREAKER: Daily DD %.2f%% >= %.2f%% -- "
                "halt rest of day",
                daily_dd * 100, self.cfg.daily_dd_halt_pct * 100,
            )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_paused(self) -> bool:
        """True if any circuit-breaker condition is active."""
        allowed, _ = self.should_trade()
        return not allowed

    def pause_remaining_seconds(self) -> int:
        """Seconds remaining on a time-based pause.  0 if no pause."""
        if self._pause_until is None:
            return 0
        now = datetime.now(timezone.utc)
        if now >= self._pause_until:
            return 0
        return int((self._pause_until - now).total_seconds())

    @property
    def close_all_triggered(self) -> bool:
        """True if a drawdown breach requires all positions to be closed.

        The execution loop should check this, close all positions, then
        call ``acknowledge_close_all()`` to clear the flag.
        """
        return self._close_all_triggered

    def acknowledge_close_all(self) -> None:
        """Clear the close-all flag after positions have been closed."""
        self._close_all_triggered = False
        logger.info("Close-all flag acknowledged and cleared")

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Manual reset -- required to resume after weekly halt."""
        logger.warning("MANUAL RESET: clearing all circuit-breaker states")
        self._consecutive_losses = 0
        self._pause_until = None
        self._daily_halted = False
        self._weekly_halted = False
        self._close_all_triggered = False
        self._daily_dd = 0.0
        self._weekly_dd = 0.0

    def reset_daily(self) -> None:
        """Daily reset at 00:00 UTC -- clears daily halt and pause."""
        logger.info(
            "Daily circuit-breaker reset: "
            "halted=%s  consec_losses=%d  daily_dd=%.2f%%",
            self._daily_halted, self._consecutive_losses, self._daily_dd * 100,
        )
        self._daily_halted = False
        self._daily_dd = 0.0
        # Do NOT reset consecutive losses or weekly state at daily boundary
        # Only clear the time-based pause if it existed
        self._pause_until = None

    # ------------------------------------------------------------------
    # Status for dashboard
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return full circuit-breaker state for dashboard display."""
        return {
            "consecutive_losses": self._consecutive_losses,
            "is_paused": self.is_paused(),
            "pause_remaining_s": self.pause_remaining_seconds(),
            "daily_halted": self._daily_halted,
            "weekly_halted": self._weekly_halted,
            "close_all_triggered": self._close_all_triggered,
            "daily_dd_pct": round(self._daily_dd * 100, 2),
            "weekly_dd_pct": round(self._weekly_dd * 100, 2),
        }
