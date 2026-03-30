"""Data Cleaner - 8-step cleaning pipeline for OHLCV data."""

import numpy as np
import pandas as pd


FREQ_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


class DataCleaner:
    """Clean OHLCV data through an 8-step pipeline.

    Steps:
    1. Remove duplicate timestamps
    2. Sort by time
    3. Fix OHLC integrity
    4. Remove zero/negative prices
    5. Fill small gaps (1-2 bars via interpolation)
    6. Smooth price spikes > max_pct_change
    7. Fill NaN (forward fill, then backward fill)
    8. Fix zero volume (rolling median)
    """

    def __init__(
        self,
        max_pct_change: float = 0.03,
        volume_window: int = 20,
        max_gap_bars: int = 2,
    ):
        self.max_pct_change = max_pct_change
        self.volume_window = volume_window
        self.max_gap_bars = max_gap_bars

    def clean(self, df: pd.DataFrame, timeframe: str = "M5") -> tuple[pd.DataFrame, dict]:
        """Run the full cleaning pipeline.

        Args:
            df: DataFrame with columns [time, open, high, low, close, volume].
            timeframe: Timeframe for gap detection frequency.

        Returns:
            Tuple of (cleaned DataFrame, cleaning report dict).
        """
        if df.empty:
            return df.copy(), {"empty": True}

        report = {}
        data = df.copy()

        # Step 1: Remove duplicates
        before = len(data)
        data = data.drop_duplicates(subset=["time"], keep="first")
        report["duplicates_removed"] = before - len(data)

        # Step 2: Sort by time
        data = data.sort_values("time").reset_index(drop=True)

        # Step 3: Fix OHLC integrity
        ohlc_fixed = 0
        bad_high = data["high"] < data[["open", "close"]].max(axis=1)
        bad_low = data["low"] > data[["open", "close"]].min(axis=1)
        ohlc_fixed = (bad_high | bad_low).sum()
        data["high"] = data[["open", "high", "low", "close"]].max(axis=1)
        data["low"] = data[["open", "high", "low", "close"]].min(axis=1)
        report["ohlc_violations_fixed"] = int(ohlc_fixed)

        # Step 4: Remove zero/negative prices
        price_cols = ["open", "high", "low", "close"]
        bad_prices = (data[price_cols] <= 0).any(axis=1)
        report["rows_removed"] = int(bad_prices.sum())
        data = data[~bad_prices].reset_index(drop=True)

        # Step 5: Fill small gaps
        gaps_filled = self._fill_gaps(data, timeframe)
        data = gaps_filled[0]
        report["gaps_filled"] = gaps_filled[1]

        # Step 6: Smooth price spikes
        spikes = self._smooth_spikes(data)
        data = spikes[0]
        report["spikes_smoothed"] = spikes[1]

        # Step 6b: Re-fix OHLC integrity (spike smoothing can break it)
        data["high"] = data[["open", "high", "low", "close"]].max(axis=1)
        data["low"] = data[["open", "high", "low", "close"]].min(axis=1)

        # Step 7: Fill NaN
        for col in price_cols:
            data[col] = data[col].ffill().bfill()

        # Step 8: Fix zero volume
        zero_vol = self._fix_zero_volume(data)
        data = zero_vol[0]
        report["zero_volume_fixed"] = zero_vol[1]

        data = data.reset_index(drop=True)
        return data, report

    def _fill_gaps(self, df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, int]:
        """Fill gaps of 1-2 missing bars via interpolation."""
        freq = FREQ_MAP.get(timeframe)
        if freq is None or len(df) < 3:
            return df, 0

        # Create expected time index
        full_idx = pd.date_range(
            start=df["time"].min(),
            end=df["time"].max(),
            freq=freq,
        )

        # Only fill weekday gaps (forex closed on weekends)
        full_idx = full_idx[full_idx.weekday < 5]

        before_len = len(df)
        df = df.set_index("time")
        df = df.reindex(full_idx)

        # Only interpolate small gaps (up to max_gap_bars consecutive NaNs)
        # Identify gap sizes
        is_nan = df["close"].isna()
        if is_nan.any():
            # Group consecutive NaNs
            groups = (~is_nan).cumsum()
            gap_sizes = is_nan.groupby(groups).transform("sum")
            # Only fill gaps <= max_gap_bars
            fillable = is_nan & (gap_sizes <= self.max_gap_bars)
            # Interpolate price columns where fillable
            for col in ["open", "high", "low", "close"]:
                interpolated = df[col].interpolate(method="linear")
                df.loc[fillable, col] = interpolated.loc[fillable]
            # Interpolate volume for same bars
            vol_interp = df["volume"].interpolate(method="linear")
            df.loc[fillable, "volume"] = vol_interp.loc[fillable]

        # Drop remaining NaN rows (larger gaps = legitimate market closures)
        df = df.dropna(subset=["close"])
        df = df.reset_index()
        df = df.rename(columns={"index": "time"})

        gaps_filled = len(df) - before_len
        return df, max(0, gaps_filled)

    def _smooth_spikes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Replace bars with >max_pct_change price moves with interpolated values."""
        if len(df) < 3:
            return df, 0

        smoothed = 0
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            pct = df[col].pct_change().abs()
            spike_mask = pct > self.max_pct_change
            # Don't flag the first bar
            spike_mask.iloc[0] = False

            if spike_mask.any():
                smoothed += spike_mask.sum()
                # Replace spikes with interpolated values
                df.loc[spike_mask, col] = np.nan
                df[col] = df[col].interpolate(method="linear")

        # Deduplicate spike count (counted per column, but report per bar)
        # This is approximate — just report total column-level fixes
        return df, int(smoothed)

    def _fix_zero_volume(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Replace zero/NaN volume with rolling median."""
        zero_mask = (df["volume"] <= 0) | df["volume"].isna()
        count = zero_mask.sum()

        if count > 0:
            rolling_med = df["volume"].rolling(
                window=self.volume_window, min_periods=5, center=True
            ).median()
            df.loc[zero_mask, "volume"] = rolling_med.loc[zero_mask]
            # Fallback: any remaining NaN/zero gets volume=1
            still_bad = (df["volume"] <= 0) | df["volume"].isna()
            df.loc[still_bad, "volume"] = 1.0

        return df, int(count)

    @staticmethod
    def print_report(report: dict, label: str = ""):
        """Print a formatted cleaning report."""
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Cleaning report:")
        for key, val in report.items():
            print(f"  {key}: {val}")
