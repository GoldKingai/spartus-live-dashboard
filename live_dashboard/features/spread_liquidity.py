"""Spread and liquidity features (Upgrade 3): 2 features.

IDENTICAL to training feature_builder._calc_spread_liquidity logic.
Both features are EXEMPT from z-score normalization (already bounded/clipped).

Features:
    - spread_estimate_norm: For LIVE, uses actual broker spread / ATR(14), clipped [0, 5].
      Falls back to session-based estimate if live spread unavailable.
    - volume_spike: volume_ratio / 5.0, clipped [0, 1]
      (volume_ratio = tick_volume / SMA20 of tick_volume)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Session spread mapping (matches training feature_builder._session_spread)
# ---------------------------------------------------------------------------

def _session_spread(hour: int) -> float:
    """Map UTC hour to estimated spread in pips.

    Matches training feature_builder._session_spread exactly.
    Used as fallback when live spread is not available.
    """
    if 8 <= hour < 12:
        return 1.5    # London AM
    elif 13 <= hour < 17:
        return 2.0    # NY overlap
    elif 12 <= hour < 13:
        return 1.8    # London PM
    elif 17 <= hour < 20:
        return 2.5    # NY PM
    elif 0 <= hour < 8:
        return 3.0    # Asia
    else:
        return 5.0    # Off hours


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_spread_liquidity_features(
    m5: pd.DataFrame,
    atr_14: float,
    live_spread: Optional[float] = None,
    pip_price: float = 0.10,
) -> Dict[str, float]:
    """Compute all 2 spread/liquidity features for the latest M5 bar.

    IDENTICAL to training feature_builder._calc_spread_liquidity for the
    session-based estimate. For LIVE, uses actual broker spread when available.

    Args:
        m5: M5 OHLCV DataFrame with 'time' and 'volume' columns.
            At least 20 bars for volume SMA.
        atr_14: Current ATR(14) value in price units.
        live_spread: Actual broker spread in price units (e.g., 0.21 for 21 points).
            If None, falls back to session-based estimate.
        pip_price: Price value of 1 pip (default 0.10 for XAUUSD).

    Returns:
        Dict mapping feature_name -> float value (2 features).
    """
    volume = m5["volume"]
    hour = m5["time"].iloc[-1].hour if hasattr(m5["time"].iloc[-1], "hour") else 12

    # --- spread_estimate_norm ---
    if live_spread is not None and live_spread > 0:
        # LIVE mode: use actual broker spread
        spread_norm = live_spread / (atr_14 + 1e-8)
    else:
        # Fallback: session-based estimate (same as training)
        spread_pips = _session_spread(hour)
        spread_price = spread_pips * pip_price  # Convert pips to price
        spread_norm = spread_price / (atr_14 + 1e-8)

    spread_norm = float(np.clip(spread_norm, 0.0, 5.0))

    # --- volume_spike ---
    vol_sma_20 = volume.rolling(20, min_periods=1).mean()
    vol_ratio = volume.iloc[-1] / (vol_sma_20.iloc[-1] + 1e-8)
    volume_spike = float(np.clip(vol_ratio / 5.0, 0.0, 1.0))

    return {
        "spread_estimate_norm": spread_norm,
        "volume_spike": volume_spike,
    }
