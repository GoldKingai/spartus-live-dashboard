"""Economic calendar and market event features (Upgrade 2): 6 features.

IDENTICAL to training calendar_features.py logic.
All features are EXEMPT from z-score normalization (bounded/binary).

3-tier calendar source: MQL5 bridge JSON > User CSV > Static known events > neutral defaults.

Features:
    - hours_to_next_high_impact: hours until next high-impact event / 48, clipped [0, 1]
    - hours_to_next_nfp_fomc: hours until NFP or FOMC / 168, clipped [0, 1]
    - in_event_window: 1.0 if within +/-30 min of high-impact event, else 0.0
    - daily_event_density: count of events today / 10, clipped [0, 1]
    - london_fix_proximity: proximity to London Fix (10:30 or 15:00 London time), scaled [0, 1]
    - comex_session_active: 1.0 if COMEX gold futures open (8:20-13:30 NY time weekday), else 0.0
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


# Timezone references
_TZ_LONDON = ZoneInfo("Europe/London")
_TZ_NY = ZoneInfo("America/New_York")
_TZ_UTC = ZoneInfo("UTC")

# London Fix times (local London time)
_FIX_AM_HOUR, _FIX_AM_MIN = 10, 30
_FIX_PM_HOUR, _FIX_PM_MIN = 15, 0

# COMEX gold futures session (local New York time)
_COMEX_OPEN_HOUR, _COMEX_OPEN_MIN = 8, 20
_COMEX_CLOSE_HOUR, _COMEX_CLOSE_MIN = 13, 30

# Major events that move gold the most (matches training exactly)
_MAJOR_EVENTS = {"NFP", "Non-Farm Payrolls", "Nonfarm Payrolls",
                 "FOMC", "Fed Interest Rate Decision",
                 "Federal Funds Rate"}

# ---------------------------------------------------------------------------
# MQL5 bridge event cache (avoids file I/O every bar)
# ---------------------------------------------------------------------------
_mql5_cache: List[Dict] = []
_mql5_cache_time: float = 0.0          # monotonic timestamp
_mql5_cache_mtime: float = 0.0         # file mtime when last read
_MQL5_CACHE_TTL_S: float = 60.0        # re-check file at most every 60s
_MQL5_MAX_AGE_S: float = 7200.0        # reject JSON older than 2 hours


def compute_calendar_features(
    timestamp: datetime,
    calendar_events: Optional[List[Dict]] = None,
    mql5_bridge_path: Optional[Path] = None,
    calendar_csv_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute all 6 calendar/event features for a given timestamp.

    IDENTICAL to training calendar_features.calc_calendar_features logic.

    3-tier calendar source priority:
        1. MQL5 bridge JSON (live events from MT5)
        2. User-provided CSV file
        3. Directly passed calendar_events list
        4. Neutral defaults (no events)

    Args:
        timestamp: UTC datetime for the current bar.
        calendar_events: Pre-loaded list of event dicts with keys:
            'datetime_utc' (datetime), 'event_name' (str), 'impact' (str).
        mql5_bridge_path: Path to MQL5 bridge JSON file with live calendar events.
        calendar_csv_path: Path to economic calendar CSV.

    Returns:
        Dict mapping feature_name -> float value (6 features).
    """
    # Load events from best available source
    events = _load_events(calendar_events, mql5_bridge_path, calendar_csv_path)

    # --- Event-based features (1-4) ---
    if events:
        event_feats = _calc_event_features(timestamp, events)
    else:
        event_feats = {
            "hours_to_next_high_impact": 1.0,   # "no event soon"
            "hours_to_next_nfp_fomc": 1.0,
            "in_event_window": 0.0,
            "daily_event_density": 0.0,
        }

    # --- London Fix proximity (5) ---
    london_fix = _calc_london_fix_proximity(timestamp)

    # --- COMEX session active (6) ---
    comex = _calc_comex_active(timestamp)

    return {
        "hours_to_next_high_impact": event_feats["hours_to_next_high_impact"],
        "hours_to_next_nfp_fomc": event_feats["hours_to_next_nfp_fomc"],
        "in_event_window": event_feats["in_event_window"],
        "daily_event_density": event_feats["daily_event_density"],
        "london_fix_proximity": london_fix,
        "comex_session_active": comex,
    }


# ---------------------------------------------------------------------------
# Event loading (3-tier)
# ---------------------------------------------------------------------------

def _load_events(
    calendar_events: Optional[List[Dict]],
    mql5_bridge_path: Optional[Path],
    calendar_csv_path: Optional[Path],
) -> List[Dict]:
    """Load calendar events from best available source.

    Returns list of dicts with 'datetime_utc', 'event_name', 'impact' keys.

    Performance: MQL5 JSON is cached for 60s to avoid file I/O every bar.
    Staleness: MQL5 JSON older than 2 hours is rejected (falls back to CSV).
    """
    global _mql5_cache, _mql5_cache_time, _mql5_cache_mtime

    import time as _time

    # Tier 1: MQL5 bridge JSON (live from MT5) -- with caching
    if mql5_bridge_path and mql5_bridge_path.exists():
        now_mono = _time.monotonic()
        # Only re-read file if cache expired AND file has changed
        if now_mono - _mql5_cache_time > _MQL5_CACHE_TTL_S:
            try:
                file_mtime = os.path.getmtime(mql5_bridge_path)
                if file_mtime != _mql5_cache_mtime or not _mql5_cache:
                    with open(mql5_bridge_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Staleness check: reject if updated_at > 2 hours ago
                    updated_str = data.get("updated_at", "")
                    if updated_str:
                        try:
                            updated_dt = pd.to_datetime(updated_str, utc=True)
                            age_s = (
                                datetime.now(ZoneInfo("UTC"))
                                - updated_dt.to_pydatetime()
                            ).total_seconds()
                            if age_s > _MQL5_MAX_AGE_S:
                                _mql5_cache = []  # stale, fall through
                                _mql5_cache_time = now_mono
                                _mql5_cache_mtime = file_mtime
                            else:
                                # Parse events -- MQL5 uses "time", "name", "importance"
                                events = []
                                for ev in data.get("events", []):
                                    raw_time = ev.get("time", "")
                                    if not raw_time:
                                        continue
                                    dt = pd.to_datetime(raw_time, utc=True)
                                    events.append({
                                        "datetime_utc": dt.to_pydatetime(),
                                        "event_name": ev.get("name", ""),
                                        "impact": ev.get("importance", "LOW").upper(),
                                    })
                                _mql5_cache = [e for e in events if e["impact"] == "HIGH"]
                                _mql5_cache_time = now_mono
                                _mql5_cache_mtime = file_mtime
                        except (ValueError, TypeError):
                            _mql5_cache = []
                            _mql5_cache_time = now_mono
                            _mql5_cache_mtime = file_mtime
                    else:
                        _mql5_cache = []
                        _mql5_cache_time = now_mono
                        _mql5_cache_mtime = file_mtime
            except (json.JSONDecodeError, OSError):
                _mql5_cache_time = now_mono  # don't retry for 60s

        if _mql5_cache:
            return _mql5_cache

    # Tier 2: User CSV
    if calendar_csv_path and calendar_csv_path.exists():
        try:
            df = load_calendar_csv(calendar_csv_path)
            events = []
            for row in df.itertuples(index=False):
                dt_val = row.datetime_utc
                if hasattr(dt_val, "to_pydatetime"):
                    dt_val = dt_val.to_pydatetime()
                events.append({
                    "datetime_utc": dt_val,
                    "event_name": getattr(row, "event_name", ""),
                    "impact": "HIGH",
                })
            return events
        except Exception:
            pass

    # Tier 3: Pre-loaded events
    if calendar_events:
        return [e for e in calendar_events if e.get("impact", "").upper() == "HIGH"]

    # Tier 4: No events
    return []


def load_calendar_csv(csv_path: Path) -> pd.DataFrame:
    """Load and parse economic calendar CSV.

    Matches training calendar_features.load_calendar exactly.

    Expected columns: date, time_utc, event_name, currency, impact.
    Returns DataFrame with 'datetime_utc' column (timezone-aware).
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[df["impact"].str.upper() == "HIGH"].copy()
    df["datetime_utc"] = pd.to_datetime(
        df["date"] + " " + df["time_utc"], format="mixed", utc=True
    )
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Event features (1-4)
# ---------------------------------------------------------------------------

def _calc_event_features(
    timestamp: datetime,
    events: List[Dict],
) -> Dict[str, float]:
    """Compute 4 event features for a single timestamp.

    Matches training _calc_event_features logic.
    """
    # Ensure timestamp is tz-aware UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=_TZ_UTC)

    # Separate all events and major events
    all_event_times = sorted([e["datetime_utc"] for e in events])
    major_event_times = sorted([
        e["datetime_utc"] for e in events
        if any(kw in str(e.get("event_name", "")) for kw in _MAJOR_EVENTS)
    ])

    # hours_to_next_high_impact
    hours_to_high = _hours_to_next(timestamp, all_event_times, cap_hours=48.0)
    hours_to_high_norm = hours_to_high / 48.0

    # hours_to_next_nfp_fomc
    hours_to_major = _hours_to_next(timestamp, major_event_times, cap_hours=168.0)
    hours_to_major_norm = hours_to_major / 168.0

    # in_event_window: within 30 min of any high-impact event
    in_window = _in_event_window(timestamp, all_event_times, window_minutes=30)

    # daily_event_density: events today / 10, capped at 1.0
    today = timestamp.date()
    today_count = sum(
        1 for t in all_event_times
        if hasattr(t, 'date') and t.date() == today
    )
    daily_density = min(today_count / 10.0, 1.0)

    return {
        "hours_to_next_high_impact": float(np.clip(hours_to_high_norm, 0.0, 1.0)),
        "hours_to_next_nfp_fomc": float(np.clip(hours_to_major_norm, 0.0, 1.0)),
        "in_event_window": in_window,
        "daily_event_density": daily_density,
    }


def _hours_to_next(
    timestamp: datetime,
    event_times: List[datetime],
    cap_hours: float,
) -> float:
    """Hours until next event, capped."""
    if not event_times:
        return cap_hours

    for et in event_times:
        # Ensure tz-aware comparison
        if et.tzinfo is None:
            et = et.replace(tzinfo=_TZ_UTC)
        diff = (et - timestamp).total_seconds() / 3600.0
        if diff >= 0:
            return min(diff, cap_hours)

    return cap_hours


def _in_event_window(
    timestamp: datetime,
    event_times: List[datetime],
    window_minutes: int = 30,
) -> float:
    """Binary: 1.0 if within window_minutes before/after any event."""
    if not event_times:
        return 0.0

    window = timedelta(minutes=window_minutes)

    for et in event_times:
        if et.tzinfo is None:
            et = et.replace(tzinfo=_TZ_UTC)
        if abs(et - timestamp) <= window:
            return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# London Fix proximity (5)
# ---------------------------------------------------------------------------

def _calc_london_fix_proximity(timestamp: datetime) -> float:
    """Normalized distance to nearest London Fix.

    1.0 = during fix, 0.0 = >2 hours away.
    London Fix AM: 10:30 London time, PM: 15:00 London time.
    Handles DST automatically via zoneinfo.

    Matches training _calc_london_fix_proximity exactly.
    """
    # Convert to London local time
    if timestamp.tzinfo is None:
        ts_utc = timestamp.replace(tzinfo=_TZ_UTC)
    else:
        ts_utc = timestamp
    london_time = ts_utc.astimezone(_TZ_LONDON)

    london_hour = london_time.hour
    london_minute = london_time.minute
    minutes_of_day = london_hour * 60 + london_minute

    # Fix times in minutes since midnight
    fix_am = _FIX_AM_HOUR * 60 + _FIX_AM_MIN   # 10:30 = 630
    fix_pm = _FIX_PM_HOUR * 60 + _FIX_PM_MIN    # 15:00 = 900

    # Distance to nearest fix (in minutes)
    dist_am = abs(minutes_of_day - fix_am)
    dist_pm = abs(minutes_of_day - fix_pm)
    min_dist = min(dist_am, dist_pm)

    # Normalize: 0 minutes = 1.0, 120 minutes (2 hours) = 0.0
    return float(np.clip(1.0 - min_dist / 120.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# COMEX session active (6)
# ---------------------------------------------------------------------------

def _calc_comex_active(timestamp: datetime) -> float:
    """Binary: 1.0 if COMEX gold futures session is open.

    COMEX gold: 8:20 AM - 1:30 PM New York time, weekdays only.
    Handles DST automatically via zoneinfo.

    Matches training _calc_comex_active exactly.
    """
    # Convert to New York local time
    if timestamp.tzinfo is None:
        ts_utc = timestamp.replace(tzinfo=_TZ_UTC)
    else:
        ts_utc = timestamp
    ny_time = ts_utc.astimezone(_TZ_NY)

    # Weekend check
    if ny_time.weekday() >= 5:
        return 0.0

    ny_hour = ny_time.hour
    ny_minute = ny_time.minute
    minutes_of_day = ny_hour * 60 + ny_minute

    comex_open = _COMEX_OPEN_HOUR * 60 + _COMEX_OPEN_MIN     # 8:20 = 500
    comex_close = _COMEX_CLOSE_HOUR * 60 + _COMEX_CLOSE_MIN  # 13:30 = 810

    if comex_open <= minutes_of_day < comex_close:
        return 1.0

    return 0.0
