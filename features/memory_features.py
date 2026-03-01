"""Memory features (Group H): 5 features.

IDENTICAL to training trade_env memory feature computation.
Wrapper that gets memory features from TradingMemory with safe defaults.

Features (all values [0, 1]):
    - recent_win_rate: blended win rate + journal direction accuracy
    - similar_pattern_winrate: Bayesian win rate for similar market conditions
    - trend_prediction_accuracy: trend prediction accuracy
    - tp_hit_rate: blended TP hit rate + good trade rate
    - avg_sl_trail_profit: blended SL trail profit + journal SL quality

When TradingMemory is unavailable or has insufficient data, returns
neutral defaults (0.5 for all features).
"""

import numpy as np
from typing import Dict, Optional


# Default neutral values for all 5 memory features
_NEUTRAL_DEFAULTS = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

# Feature names in order (matches training observation vector indices)
MEMORY_FEATURE_NAMES = (
    "recent_win_rate",
    "similar_pattern_winrate",
    "trend_prediction_accuracy",
    "tp_hit_rate",
    "avg_sl_trail_profit",
)


def compute_memory_features(
    memory=None,
    market_state: Optional[Dict] = None,
    current_step: int = 0,
) -> Dict[str, float]:
    """Compute all 5 memory features with safe defaults.

    IDENTICAL to training trade_env memory feature extraction.
    Calls memory.get_memory_features(market_state, step) when available.

    Args:
        memory: TradingMemory instance (or None for neutral defaults).
            Must implement get_memory_features(market_state, current_step)
            returning a 5-element numpy array.
        market_state: Dict with market context for pattern matching:
            {
                "rsi": float (0-1, RSI/100),
                "trend_dir": float (-3 to 3, H1 trend direction),
                "session": str ("london", "ny", "asia", etc.),
                "vol_regime": float (ATR ratio),
            }
            If None, pattern matching uses priors only.
        current_step: Current step number (for cache optimization).

    Returns:
        Dict mapping feature_name -> float value (5 features, all [0, 1]).
    """
    if memory is None:
        return _get_neutral_features()

    try:
        feats = memory.get_memory_features(market_state, current_step=current_step)

        # Validate shape and values
        if feats is None or len(feats) != 5:
            return _get_neutral_features()

        # Clip to [0, 1] range and replace NaN
        feats = np.clip(np.nan_to_num(feats, nan=0.5), 0.0, 1.0)

        return {
            MEMORY_FEATURE_NAMES[i]: float(feats[i])
            for i in range(5)
        }

    except Exception:
        return _get_neutral_features()


def _get_neutral_features() -> Dict[str, float]:
    """Return neutral default memory features (all 0.5).

    0.5 is the uninformative prior for all memory features:
    - 50% win rate
    - 50% pattern match rate
    - 50% trend accuracy
    - 50% TP hit rate
    - 50% SL quality
    """
    return {name: 0.5 for name in MEMORY_FEATURE_NAMES}


def get_memory_features_array(
    memory=None,
    market_state: Optional[Dict] = None,
    current_step: int = 0,
) -> np.ndarray:
    """Return memory features as a 5-element numpy array.

    Convenience function matching the training observation vector format.

    Args:
        memory: TradingMemory instance (or None).
        market_state: Market context dict.
        current_step: Current step number.

    Returns:
        5-element float32 numpy array with values in [0, 1].
    """
    feats = compute_memory_features(memory, market_state, current_step)
    return np.array([feats[name] for name in MEMORY_FEATURE_NAMES], dtype=np.float32)
