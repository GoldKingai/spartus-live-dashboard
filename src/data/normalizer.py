"""Expanding-window normalization for market features.

Applies rolling z-score to market features (#1-25) only.
Time, account, and memory features (#26-42) are EXEMPT.
Uses only past data at each point — no look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Set

from src.config import TrainingConfig


class ExpandingWindowNormalizer:
    """Rolling z-score normalizer with exemption support.

    - Market features (Groups A-E): rolling z-score, 200-bar window, clip ±5
    - Exempt features (Groups F-H): passed through unchanged
    """

    def __init__(self, config: TrainingConfig = None):
        cfg = config or TrainingConfig()
        self.window = cfg.norm_window       # 200
        self.clip_val = cfg.norm_clip        # 5.0
        self.min_periods = 50               # Minimum bars for meaningful stats
        self.exempt: Set[str] = set(cfg.norm_exempt_features)
        self.market_features: tuple = cfg.market_feature_names

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a DataFrame of features in-place (vectorized).

        Market feature columns get rolling z-score normalization.
        All other columns are passed through unchanged.

        Args:
            df: DataFrame with feature columns (from FeatureBuilder).

        Returns:
            New DataFrame with normalized market features.
        """
        result = df.copy()

        for col in self.market_features:
            if col not in result.columns:
                continue
            series = result[col]
            rolling_mean = series.rolling(
                window=self.window, min_periods=self.min_periods
            ).mean()
            rolling_std = series.rolling(
                window=self.window, min_periods=self.min_periods
            ).std()
            normalized = (series - rolling_mean) / (rolling_std + 1e-8)
            result[col] = normalized.clip(-self.clip_val, self.clip_val)

        return result

    def normalize_single(
        self, series: pd.Series, idx: int, feature_name: str
    ) -> float:
        """Normalize a single value using only past data.

        Used for real-time / step-by-step normalization in the environment.

        Args:
            series: Full feature series up to and including idx.
            idx: Index to normalize.
            feature_name: Name of the feature (for exemption check).

        Returns:
            Normalized value, or raw value if exempt.
        """
        if feature_name in self.exempt:
            return series.iloc[idx]

        start = max(0, idx - self.window + 1)
        window_data = series.iloc[start: idx + 1]

        if len(window_data) < self.min_periods:
            return 0.0

        mean = window_data.mean()
        std = window_data.std()
        val = (series.iloc[idx] - mean) / (std + 1e-8)
        return float(np.clip(val, -self.clip_val, self.clip_val))
