"""Economic calendar & market event features (Upgrade 2).

6 features:
- hours_to_next_high_impact: proximity to next high-impact USD event [0, 1]
- hours_to_next_nfp_fomc: countdown to NFP/FOMC [0, 1]
- in_event_window: binary, within 30 min of high-impact event
- daily_event_density: count of high-impact events today / 10, capped at 1.0
- london_fix_proximity: distance to nearest London Fix [0, 1]
- comex_session_active: binary, COMEX gold futures open

All features are EXEMPT from z-score normalization (bounded/binary).
London Fix and COMEX times use zoneinfo for DST-aware computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import TrainingConfig

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

# Major events that move gold the most
_MAJOR_EVENTS = {"NFP", "Non-Farm Payrolls", "Nonfarm Payrolls",
                 "FOMC", "Fed Interest Rate Decision",
                 "Federal Funds Rate"}


def calc_calendar_features(
    m5: pd.DataFrame,
    calendar_df: pd.DataFrame = None,
    config: TrainingConfig = None,
) -> pd.DataFrame:
    """Compute all 6 calendar/event features from M5 timestamps.

    Args:
        m5: M5 OHLCV with 'time' column (timezone-aware UTC).
        calendar_df: Economic calendar with columns:
            date (str YYYY-MM-DD), time_utc (str HH:MM), event_name (str),
            currency (str), impact (str HIGH/MEDIUM/LOW).
            If None, event features default to neutral values.
        config: TrainingConfig.

    Returns:
        DataFrame with 6 feature columns, indexed like m5.
    """
    cfg = config or TrainingConfig()
    times = m5["time"]
    n = len(times)

    # --- Event-based features (1-4) ---
    if calendar_df is not None and not calendar_df.empty:
        event_feats = _calc_event_features(times, calendar_df)
    else:
        # No calendar data — neutral defaults
        event_feats = pd.DataFrame({
            "hours_to_next_high_impact": np.full(n, 1.0),  # "no event soon"
            "hours_to_next_nfp_fomc": np.full(n, 1.0),
            "in_event_window": np.zeros(n),
            "daily_event_density": np.zeros(n),
        })

    # --- London Fix proximity (5) ---
    london_fix = _calc_london_fix_proximity(times)

    # --- COMEX session active (6) ---
    comex = _calc_comex_active(times)

    return pd.DataFrame({
        "hours_to_next_high_impact": event_feats["hours_to_next_high_impact"].values,
        "hours_to_next_nfp_fomc": event_feats["hours_to_next_nfp_fomc"].values,
        "in_event_window": event_feats["in_event_window"].values,
        "daily_event_density": event_feats["daily_event_density"].values,
        "london_fix_proximity": london_fix,
        "comex_session_active": comex,
    })


def load_calendar(csv_path: Path) -> pd.DataFrame:
    """Load and parse economic calendar CSV.

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


def _calc_event_features(
    times: pd.Series,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 4 event features from calendar data.

    Uses vectorized searchsorted for efficient nearest-event lookup.
    """
    n = len(times)

    # Parse calendar event times
    if "datetime_utc" not in calendar_df.columns:
        calendar_df = calendar_df.copy()
        calendar_df["datetime_utc"] = pd.to_datetime(
            calendar_df["date"] + " " + calendar_df["time_utc"],
            format="mixed", utc=True,
        )

    # All high-impact events
    all_events = calendar_df["datetime_utc"].sort_values().values
    # Major events only (NFP, FOMC)
    is_major = calendar_df["event_name"].apply(
        lambda x: any(kw in str(x) for kw in _MAJOR_EVENTS)
    )
    major_events = calendar_df.loc[is_major, "datetime_utc"].sort_values().values

    times_np = times.values.astype("datetime64[ns]")

    # --- hours_to_next_high_impact ---
    hours_to_high = _hours_to_next_event(times_np, all_events, cap_hours=48.0)
    hours_to_high_norm = hours_to_high / 48.0  # Normalize to [0, 1]

    # --- hours_to_next_nfp_fomc ---
    hours_to_major = _hours_to_next_event(times_np, major_events, cap_hours=168.0)
    hours_to_major_norm = hours_to_major / 168.0  # Normalize to [0, 1]

    # --- in_event_window ---
    in_window = _in_event_window(times_np, all_events, window_minutes=30)

    # --- daily_event_density ---
    daily_density = _daily_event_density(times, calendar_df)

    return pd.DataFrame({
        "hours_to_next_high_impact": hours_to_high_norm,
        "hours_to_next_nfp_fomc": hours_to_major_norm,
        "in_event_window": in_window,
        "daily_event_density": daily_density,
    })


def _hours_to_next_event(
    times_np: np.ndarray, events: np.ndarray, cap_hours: float,
) -> np.ndarray:
    """Vectorized hours until next event using searchsorted."""
    if len(events) == 0:
        return np.full(len(times_np), cap_hours)

    indices = np.searchsorted(events, times_np, side="left")
    result = np.full(len(times_np), cap_hours)

    valid = indices < len(events)
    if valid.any():
        diffs = (events[indices[valid]] - times_np[valid]).astype("timedelta64[s]").astype(float)
        hours = diffs / 3600.0
        result[valid] = np.clip(hours, 0.0, cap_hours)

    return result


def _in_event_window(
    times_np: np.ndarray, events: np.ndarray, window_minutes: int = 30,
) -> np.ndarray:
    """Binary: 1 if within window_minutes before/after any event."""
    if len(events) == 0:
        return np.zeros(len(times_np), dtype=np.float32)

    window_ns = np.timedelta64(window_minutes, "m")
    result = np.zeros(len(times_np), dtype=np.float32)

    indices = np.searchsorted(events, times_np, side="left")

    for offset in [0, -1]:
        check_idx = indices + offset
        valid = (check_idx >= 0) & (check_idx < len(events))
        if valid.any():
            diffs = np.abs(events[check_idx[valid]] - times_np[valid])
            result[valid] = np.maximum(result[valid], (diffs <= window_ns).astype(np.float32))

    return result


def _daily_event_density(times: pd.Series, calendar_df: pd.DataFrame) -> np.ndarray:
    """Count of high-impact events per day / 10, capped at 1.0."""
    dates = times.dt.date
    cal_dates = pd.to_datetime(calendar_df["date"]).dt.date
    counts = cal_dates.value_counts()
    density = dates.map(counts).fillna(0).values / 10.0
    return np.clip(density, 0.0, 1.0)


def _calc_london_fix_proximity(times: pd.Series) -> np.ndarray:
    """Normalized distance to nearest London Fix.

    1.0 = during fix, 0.0 = >2 hours away.
    London Fix AM: 10:30 London time, PM: 15:00 London time.
    Handles DST automatically via zoneinfo.
    """
    n = len(times)
    result = np.zeros(n, dtype=np.float32)

    # Convert to London local time for fix proximity
    # Handle tz-naive timestamps by localizing to UTC first
    if times.dt.tz is None:
        london_times = times.dt.tz_localize("UTC").dt.tz_convert(_TZ_LONDON)
    else:
        london_times = times.dt.tz_convert(_TZ_LONDON)
    london_hour = london_times.dt.hour.values
    london_minute = london_times.dt.minute.values

    # Minutes since midnight (London time)
    minutes_of_day = london_hour * 60 + london_minute

    # Fix times in minutes since midnight
    fix_am = _FIX_AM_HOUR * 60 + _FIX_AM_MIN  # 10:30 = 630
    fix_pm = _FIX_PM_HOUR * 60 + _FIX_PM_MIN   # 15:00 = 900

    # Distance to nearest fix (in minutes)
    dist_am = np.abs(minutes_of_day - fix_am)
    dist_pm = np.abs(minutes_of_day - fix_pm)
    min_dist = np.minimum(dist_am, dist_pm)

    # Normalize: 0 minutes = 1.0, 120 minutes (2 hours) = 0.0
    result = np.clip(1.0 - min_dist / 120.0, 0.0, 1.0).astype(np.float32)

    return result


def _calc_comex_active(times: pd.Series) -> np.ndarray:
    """Binary: 1 if COMEX gold futures session is open.

    COMEX gold: 8:20 AM - 1:30 PM New York time.
    Handles DST automatically via zoneinfo.
    """
    # Handle tz-naive timestamps by localizing to UTC first
    if times.dt.tz is None:
        ny_times = times.dt.tz_localize("UTC").dt.tz_convert(_TZ_NY)
    else:
        ny_times = times.dt.tz_convert(_TZ_NY)
    ny_hour = ny_times.dt.hour.values
    ny_minute = ny_times.dt.minute.values

    minutes_of_day = ny_hour * 60 + ny_minute
    comex_open = _COMEX_OPEN_HOUR * 60 + _COMEX_OPEN_MIN    # 8:20 = 500
    comex_close = _COMEX_CLOSE_HOUR * 60 + _COMEX_CLOSE_MIN  # 13:30 = 810

    active = ((minutes_of_day >= comex_open) & (minutes_of_day < comex_close)).astype(np.float32)

    # Only on weekdays
    weekday = times.dt.dayofweek.values
    active[weekday >= 5] = 0.0

    return active
