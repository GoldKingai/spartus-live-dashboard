"""Auto-generate high-impact economic calendar events for gold trading.

Produces event dates programmatically so the live system never needs
a manual CSV update, an MQL5 bridge, or any external API.

Rules:
    - NFP (Non-Farm Payrolls): 1st Friday of month, 08:30 ET
    - CPI (Consumer Price Index): ~10th-13th of month, 08:30 ET
      (uses BLS schedule for known years, estimates for unknown)
    - FOMC Rate Decision: 8 scheduled meetings/year (known dates)
    - ISM Manufacturing PMI: 1st business day of month, 10:00 ET
    - ECB Rate Decision: ~6 per year (known dates)

All times are output in UTC (auto-adjusted for EST/EDT).
"""

from datetime import date, datetime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

_TZ_ET = ZoneInfo("America/New_York")
_TZ_UTC = ZoneInfo("UTC")


# ──────────────────────────────────────────────────────────────────────
# FOMC meeting dates (published by the Federal Reserve)
# Add new years as they're announced at:
#   https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# ──────────────────────────────────────────────────────────────────────
_FOMC_DATES: Dict[int, List[str]] = {
    2025: [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    ],
    2026: [
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ],
}

# FOMC announcements are at 14:00 ET (2 PM)
_FOMC_HOUR_ET = 14
_FOMC_MINUTE_ET = 0


# ──────────────────────────────────────────────────────────────────────
# CPI release dates (published by BLS)
# Add new years as they're announced at:
#   https://www.bls.gov/schedule/news_release/cpi.htm
# Falls back to "2nd Tuesday or Wednesday ~10th-13th" heuristic.
# ──────────────────────────────────────────────────────────────────────
_CPI_DATES: Dict[int, List[str]] = {
    2025: [
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
        "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
        "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    ],
    2026: [
        "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14",
        "2026-05-13", "2026-06-10", "2026-07-14", "2026-08-12",
        "2026-09-15", "2026-10-13", "2026-11-10", "2026-12-10",
    ],
}

# CPI released at 08:30 ET
_CPI_HOUR_ET = 8
_CPI_MINUTE_ET = 30


# ──────────────────────────────────────────────────────────────────────
# ECB Rate Decision dates
# Add new years from: https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html
# ──────────────────────────────────────────────────────────────────────
_ECB_DATES: Dict[int, List[str]] = {
    2025: [
        "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05",
        "2025-07-17", "2025-09-11", "2025-10-30", "2025-12-18",
    ],
    2026: [
        "2026-01-22", "2026-03-05", "2026-04-16", "2026-06-04",
        "2026-07-16", "2026-09-10", "2026-10-29", "2026-12-17",
    ],
}

# ECB announcements at 13:15 CET / 12:15 UTC (fixed, not ET)
_ECB_HOUR_UTC = 12
_ECB_MINUTE_UTC = 45


# ──────────────────────────────────────────────────────────────────────
# Helper: first weekday of month
# ──────────────────────────────────────────────────────────────────────

def _first_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the first occurrence of `weekday` (0=Mon, 4=Fri) in month."""
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def _first_business_day_of_month(year: int, month: int) -> date:
    """Return the first business day (Mon-Fri) of the month."""
    d = date(year, month, 1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d += timedelta(days=1)
    return d


def _et_to_utc(d: date, hour_et: int, minute_et: int) -> datetime:
    """Convert a date + ET time to a UTC datetime (handles EST/EDT)."""
    local = datetime(d.year, d.month, d.day, hour_et, minute_et, tzinfo=_TZ_ET)
    return local.astimezone(_TZ_UTC)


def _estimate_cpi_date(year: int, month: int) -> date:
    """Estimate CPI release date when BLS schedule is unknown.

    Heuristic: 2nd Tuesday of the month, or the closest business day
    around the 10th-13th.
    """
    # Find 2nd Tuesday (weekday=1)
    first_tue = _first_weekday_of_month(year, month, 1)  # Tuesday
    second_tue = first_tue + timedelta(days=7)
    return second_tue


# ──────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────

def generate_calendar_events(
    year: int,
    months: List[int] | None = None,
) -> List[Dict]:
    """Generate high-impact economic calendar events for a given year.

    Args:
        year: Calendar year (e.g. 2026).
        months: Optional list of months to generate (1-12).
                 Defaults to all 12 months.

    Returns:
        List of event dicts with keys:
            'datetime_utc' (datetime), 'event_name' (str), 'impact' (str).
    """
    if months is None:
        months = list(range(1, 13))

    events: List[Dict] = []

    for month in months:
        # --- NFP: 1st Friday at 08:30 ET ---
        nfp_date = _first_weekday_of_month(year, month, 4)  # Friday
        nfp_utc = _et_to_utc(nfp_date, 8, 30)
        events.append({
            "datetime_utc": nfp_utc,
            "event_name": "Non-Farm Payrolls",
            "impact": "HIGH",
        })

        # --- ISM Manufacturing PMI: 1st business day at 10:00 ET ---
        ism_date = _first_business_day_of_month(year, month)
        ism_utc = _et_to_utc(ism_date, 10, 0)
        events.append({
            "datetime_utc": ism_utc,
            "event_name": "ISM Manufacturing PMI",
            "impact": "HIGH",
        })

        # --- CPI: from BLS schedule or estimated ---
        if year in _CPI_DATES:
            cpi_dates = _CPI_DATES[year]
            # Find CPI date for this month
            for cpi_str in cpi_dates:
                cpi_d = date.fromisoformat(cpi_str)
                if cpi_d.month == month:
                    cpi_utc = _et_to_utc(cpi_d, _CPI_HOUR_ET, _CPI_MINUTE_ET)
                    events.append({
                        "datetime_utc": cpi_utc,
                        "event_name": "CPI",
                        "impact": "HIGH",
                    })
                    break
        else:
            # Estimate
            cpi_d = _estimate_cpi_date(year, month)
            cpi_utc = _et_to_utc(cpi_d, _CPI_HOUR_ET, _CPI_MINUTE_ET)
            events.append({
                "datetime_utc": cpi_utc,
                "event_name": "CPI",
                "impact": "HIGH",
            })

    # --- FOMC: from published schedule ---
    if year in _FOMC_DATES:
        for fomc_str in _FOMC_DATES[year]:
            fomc_d = date.fromisoformat(fomc_str)
            if fomc_d.month in months:
                fomc_utc = _et_to_utc(fomc_d, _FOMC_HOUR_ET, _FOMC_MINUTE_ET)
                events.append({
                    "datetime_utc": fomc_utc,
                    "event_name": "FOMC Rate Decision",
                    "impact": "HIGH",
                })

    # --- ECB: from published schedule ---
    if year in _ECB_DATES:
        for ecb_str in _ECB_DATES[year]:
            ecb_d = date.fromisoformat(ecb_str)
            if ecb_d.month in months:
                ecb_utc = datetime(
                    ecb_d.year, ecb_d.month, ecb_d.day,
                    _ECB_HOUR_UTC, _ECB_MINUTE_UTC, tzinfo=_TZ_UTC,
                )
                events.append({
                    "datetime_utc": ecb_utc,
                    "event_name": "ECB Interest Rate Decision",
                    "impact": "HIGH",
                })

    events.sort(key=lambda e: e["datetime_utc"])
    return events


def generate_upcoming_events(months_ahead: int = 3) -> List[Dict]:
    """Generate events for the current month + N months ahead.

    Handles year boundaries (e.g., Nov → Jan next year).

    Args:
        months_ahead: How many months into the future to generate.

    Returns:
        Sorted list of event dicts.
    """
    today = date.today()
    events: List[Dict] = []
    seen: set = set()

    for offset in range(months_ahead + 1):
        # Calculate target year/month
        m = today.month + offset
        y = today.year
        while m > 12:
            m -= 12
            y += 1

        year_events = generate_calendar_events(y, months=[m])
        for e in year_events:
            key = (e["datetime_utc"].date(), e["event_name"])
            if key not in seen:
                seen.add(key)
                events.append(e)

    events.sort(key=lambda e: e["datetime_utc"])
    return events
