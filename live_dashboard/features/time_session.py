"""Time and session features (Group F): 4 features.

IDENTICAL to training feature_builder._calc_time_features logic.
All features are EXEMPT from z-score normalization (already bounded).

Features:
    - hour_sin: sin(2*pi*hour/24)
    - hour_cos: cos(2*pi*hour/24)
    - day_of_week: dayofweek / 4.0 (Mon=0.0, Fri=1.0)
    - session_quality: trading session quality score [0, 1]
"""

import numpy as np
from datetime import datetime
from typing import Dict


# ---------------------------------------------------------------------------
# Session quality mapping (EXACT match to training feature_builder.py)
# ---------------------------------------------------------------------------

def _session_quality(hour: int) -> float:
    """Map UTC hour to trading session quality score [0, 1].

    Matches training feature_builder._session_quality exactly.
    """
    if 8 <= hour < 12:
        return 1.0    # London AM (best liquidity)
    elif 13 <= hour < 17:
        return 0.95   # NY overlap
    elif 12 <= hour < 13:
        return 0.9    # London PM
    elif 17 <= hour < 20:
        return 0.7    # NY PM
    elif 0 <= hour < 8:
        return 0.4    # Asia
    else:
        return 0.2    # Off hours


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_time_session_features(timestamp: datetime) -> Dict[str, float]:
    """Compute all 4 time/session features for a given timestamp.

    IDENTICAL to training feature_builder._calc_time_features.

    Args:
        timestamp: UTC datetime for the current bar.

    Returns:
        Dict mapping feature_name -> float value (4 features).
    """
    hour = timestamp.hour
    day = timestamp.weekday()  # Mon=0, Sun=6

    return {
        # Cyclical hour encoding (sin)
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        # Cyclical hour encoding (cos)
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        # Day of week [0, 1] (Mon=0.0, Fri=1.0)
        "day_of_week": day / 4.0,
        # Session quality [0, 1]
        "session_quality": _session_quality(hour),
    }
