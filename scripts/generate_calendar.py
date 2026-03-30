"""Generate deterministic economic calendar CSV for training.

Creates a HIGH-impact event schedule from 2015-2026 based on standard
recurring patterns. The AI learns "volatility around this time" not
exact event timestamps, so deterministic schedules are sufficient.

Events:
- NFP: First Friday of every month
- FOMC: 8 meetings per year (standard schedule)
- CPI: 13th of each month (or next business day)
- PPI: 14th of each month (or next business day)
- GDP: 28th of month following quarter end (Jan, Apr, Jul, Oct)
- Retail Sales: 15th of each month (or next business day)
- ISM Manufacturing: 1st business day of each month
- ECB: Second Thursday of Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec

Usage:
    python scripts/generate_calendar.py
"""

import csv
from datetime import date, timedelta
from pathlib import Path


# --- DST helpers ---

def us_dst_start(year: int) -> date:
    """Second Sunday of March."""
    d = date(year, 3, 8)  # Earliest possible second Sunday
    while d.weekday() != 6:  # Sunday
        d += timedelta(days=1)
    return d


def us_dst_end(year: int) -> date:
    """First Sunday of November."""
    d = date(year, 11, 1)
    while d.weekday() != 6:
        d += timedelta(days=1)
    return d


def eu_dst_start(year: int) -> date:
    """Last Sunday of March."""
    d = date(year, 3, 31)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def eu_dst_end(year: int) -> date:
    """Last Sunday of October."""
    d = date(year, 10, 31)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def is_us_dst(d: date) -> bool:
    return us_dst_start(d.year) <= d < us_dst_end(d.year)


def is_eu_dst(d: date) -> bool:
    return eu_dst_start(d.year) <= d < eu_dst_end(d.year)


def next_business_day(d: date) -> date:
    """If d is weekend, advance to Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# --- Event generators ---

def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:  # Friday
        d += timedelta(days=1)
    return d


def second_thursday(year: int, month: int) -> date:
    d = date(year, month, 8)  # Earliest second Thursday
    while d.weekday() != 3:  # Thursday
        d += timedelta(days=1)
    return d


def generate_nfp(start_year: int, end_year: int, end_month: int) -> list:
    """Non-Farm Payrolls: first Friday, 13:30 UTC (12:30 during EDT)."""
    events = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_month:
                break
            d = first_friday(y, m)
            time = "12:30" if is_us_dst(d) else "13:30"
            events.append((d.isoformat(), time, "Non-Farm Payrolls", "USD", "HIGH"))
    return events


def generate_fomc(start_year: int, end_year: int, end_month: int) -> list:
    """FOMC rate decisions: ~8 per year, Wednesdays.
    Standard months: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec (3rd Wednesday approx).
    """
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    events = []
    for y in range(start_year, end_year + 1):
        for m in fomc_months:
            if y == end_year and m > end_month:
                break
            # Third Wednesday of the month
            d = date(y, m, 15)
            while d.weekday() != 2:  # Wednesday
                d += timedelta(days=1)
            time = "18:00" if is_us_dst(d) else "19:00"
            events.append((d.isoformat(), time, "Fed Interest Rate Decision", "USD", "HIGH"))
    return events


def generate_cpi(start_year: int, end_year: int, end_month: int) -> list:
    """CPI: ~13th of each month, 13:30 UTC (12:30 EDT)."""
    events = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_month:
                break
            d = next_business_day(date(y, m, 13))
            time = "12:30" if is_us_dst(d) else "13:30"
            events.append((d.isoformat(), time, "CPI", "USD", "HIGH"))
    return events


def generate_ppi(start_year: int, end_year: int, end_month: int) -> list:
    """PPI: ~14th of each month, 13:30 UTC (12:30 EDT)."""
    events = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_month:
                break
            d = next_business_day(date(y, m, 14))
            time = "12:30" if is_us_dst(d) else "13:30"
            events.append((d.isoformat(), time, "PPI", "USD", "HIGH"))
    return events


def generate_gdp(start_year: int, end_year: int, end_month: int) -> list:
    """GDP advance: 28th of Jan, Apr, Jul, Oct, 13:30 UTC."""
    gdp_months = [1, 4, 7, 10]
    events = []
    for y in range(start_year, end_year + 1):
        for m in gdp_months:
            if y == end_year and m > end_month:
                break
            d = next_business_day(date(y, m, 28))
            time = "12:30" if is_us_dst(d) else "13:30"
            events.append((d.isoformat(), time, "GDP", "USD", "HIGH"))
    return events


def generate_retail_sales(start_year: int, end_year: int, end_month: int) -> list:
    """Retail Sales: ~15th of each month, 13:30 UTC."""
    events = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_month:
                break
            d = next_business_day(date(y, m, 15))
            time = "12:30" if is_us_dst(d) else "13:30"
            events.append((d.isoformat(), time, "Retail Sales", "USD", "HIGH"))
    return events


def generate_ism(start_year: int, end_year: int, end_month: int) -> list:
    """ISM Manufacturing PMI: 1st business day, 15:00 UTC (14:00 EDT)."""
    events = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_month:
                break
            d = next_business_day(date(y, m, 1))
            time = "14:00" if is_us_dst(d) else "15:00"
            events.append((d.isoformat(), time, "ISM Manufacturing PMI", "USD", "HIGH"))
    return events


def generate_ecb(start_year: int, end_year: int, end_month: int) -> list:
    """ECB rate decisions: second Thursday of Jan,Mar,Apr,Jun,Jul,Sep,Oct,Dec."""
    ecb_months = [1, 3, 4, 6, 7, 9, 10, 12]
    events = []
    for y in range(start_year, end_year + 1):
        for m in ecb_months:
            if y == end_year and m > end_month:
                break
            d = second_thursday(y, m)
            time = "11:45" if is_eu_dst(d) else "12:45"
            events.append((d.isoformat(), time, "ECB Interest Rate Decision", "EUR", "HIGH"))
    return events


def main():
    start_year = 2015
    end_year = 2026
    end_month = 2  # Up to Feb 2026

    all_events = []
    all_events.extend(generate_nfp(start_year, end_year, end_month))
    all_events.extend(generate_fomc(start_year, end_year, end_month))
    all_events.extend(generate_cpi(start_year, end_year, end_month))
    all_events.extend(generate_ppi(start_year, end_year, end_month))
    all_events.extend(generate_gdp(start_year, end_year, end_month))
    all_events.extend(generate_retail_sales(start_year, end_year, end_month))
    all_events.extend(generate_ism(start_year, end_year, end_month))
    all_events.extend(generate_ecb(start_year, end_year, end_month))

    # Sort by date, then time
    all_events.sort(key=lambda x: (x[0], x[1]))

    # Write CSV
    out_path = Path("data/calendar/economic_calendar.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "time_utc", "event_name", "currency", "impact"])
        writer.writerows(all_events)

    print(f"Generated {len(all_events)} events")
    print(f"Saved to: {out_path.resolve()}")

    # Summary by event type
    from collections import Counter
    counts = Counter(e[2] for e in all_events)
    for event, count in sorted(counts.items()):
        print(f"  {event}: {count}")

    # Date range
    print(f"  Date range: {all_events[0][0]} to {all_events[-1][0]}")


if __name__ == "__main__":
    main()
