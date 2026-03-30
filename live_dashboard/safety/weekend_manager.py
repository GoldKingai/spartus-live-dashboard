"""Weekend manager -- handle Friday close and weekend gap risk.

Schedule (all times UTC):
    Friday 19:00  -- Block new entries (positions can still be managed)
    Friday 20:00  -- Close all positions
    Saturday      -- No trading
    Sunday        -- No trading
    Monday 00:30  -- Resume trading (or Sunday 00:30 if broker opens Sunday)

The execution loop calls ``should_trade()`` before every new entry and
``should_close_all()`` each tick to trigger the Friday forced close.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple

from config.live_config import LiveConfig

logger = logging.getLogger(__name__)


class WeekendManager:
    """Handle Friday close and weekend gap risk."""

    # Weekday constants (datetime.weekday())
    MONDAY = 0
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    def __init__(self, config: LiveConfig) -> None:
        self.cfg = config
        logger.info(
            "WeekendManager initialised  "
            "block_new=%02d:00  close_all=%02d:00  "
            "resume=Mon %02d:%02d UTC",
            self.cfg.friday_block_new_utc_hour,
            self.cfg.friday_close_utc_hour,
            self.cfg.monday_resume_utc_hour,
            self.cfg.monday_resume_utc_minute,
        )

    # ------------------------------------------------------------------
    # Primary gate
    # ------------------------------------------------------------------

    def should_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on day/time.

        Returns:
            (allowed, reason) tuple.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute

        # Saturday -- always closed
        if weekday == self.SATURDAY:
            return False, "weekend (Saturday)"

        # Sunday -- closed until resume time (default Mon 00:30)
        if weekday == self.SUNDAY:
            return False, "weekend (Sunday)"

        # Monday -- blocked until resume time
        if weekday == self.MONDAY:
            resume_minutes = (
                self.cfg.monday_resume_utc_hour * 60
                + self.cfg.monday_resume_utc_minute
            )
            current_minutes = hour * 60 + minute
            if current_minutes < resume_minutes:
                return False, f"pre-market (resume Mon {self.cfg.monday_resume_utc_hour:02d}:{self.cfg.monday_resume_utc_minute:02d} UTC)"

        # Friday -- block new entries after the block hour
        if weekday == self.FRIDAY:
            if hour >= self.cfg.friday_block_new_utc_hour:
                return False, f"Friday wind-down (no new entries after {self.cfg.friday_block_new_utc_hour:02d}:00 UTC)"

        return True, "ok"

    # ------------------------------------------------------------------
    # Friday forced close
    # ------------------------------------------------------------------

    def should_close_all(self) -> bool:
        """True if Friday >= close hour (default 20:00 UTC).

        The execution loop must close all open positions when this
        returns True.
        """
        now = datetime.now(timezone.utc)
        if now.weekday() == self.FRIDAY and now.hour >= self.cfg.friday_close_utc_hour:
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_weekend(self) -> bool:
        """True if current UTC time is outside trading hours."""
        allowed, _ = self.should_trade()
        return not allowed

    def time_to_friday_close(self) -> float:
        """Hours until the Friday forced-close deadline.

        Returns:
            Positive hours until Friday close.  If already past, returns
            hours until *next* Friday's close.  During the weekend the
            value counts forward to next Friday.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()

        # Days until Friday (weekday 4)
        if weekday <= self.FRIDAY:
            days_ahead = self.FRIDAY - weekday
        else:
            # Saturday (5) or Sunday (6) -- next Friday
            days_ahead = self.FRIDAY + 7 - weekday

        # Build next Friday close datetime
        friday_close = now.replace(
            hour=self.cfg.friday_close_utc_hour,
            minute=0,
            second=0,
            microsecond=0,
        ) + timedelta(days=days_ahead)

        diff = (friday_close - now).total_seconds()
        if diff < 0:
            # Already past this Friday's close -- next week
            diff += 7 * 24 * 3600

        return diff / 3600.0

    # ------------------------------------------------------------------
    # Status for dashboard
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return weekend-manager state for dashboard display."""
        allowed, reason = self.should_trade()
        return {
            "trading_allowed": allowed,
            "reason": reason,
            "is_weekend": self.is_weekend(),
            "should_close_all": self.should_close_all(),
            "hours_to_friday_close": round(self.time_to_friday_close(), 2),
        }
