"""Storage Manager - Organize validated OHLCV data into weekly Parquet files."""

from pathlib import Path

import pandas as pd


class StorageManager:
    """Split continuous OHLCV data into weekly Parquet files.

    Output structure:
        storage/data/processed/{year}/week_{NN}_{TF}.parquet

    Each file has columns: time, open, high, low, close, volume
    """

    def __init__(self, base_dir: str = "storage/data/processed"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_weekly(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Split DataFrame into weekly Parquet files.

        Args:
            df: DataFrame with columns [time, open, high, low, close, volume].
            timeframe: Timeframe label (M5, H1, H4, D1).

        Returns:
            Dict with metadata: total_weeks, total_bars, year_range, files_written.
        """
        if df.empty:
            return {"total_weeks": 0, "total_bars": 0, "files_written": []}

        df = df.copy()
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], utc=True)

        # Extract ISO year and week
        df["_iso_year"] = df["time"].dt.isocalendar().year.astype(int)
        df["_iso_week"] = df["time"].dt.isocalendar().week.astype(int)

        files_written = []
        grouped = df.groupby(["_iso_year", "_iso_week"])

        for (year, week), group in grouped:
            year_dir = self.base_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            filename = f"week_{week:02d}_{timeframe}.parquet"
            filepath = year_dir / filename

            out = group[["time", "open", "high", "low", "close", "volume"]].copy()
            out = out.sort_values("time").reset_index(drop=True)
            out.to_parquet(filepath, engine="pyarrow", index=False)
            files_written.append(str(filepath))

        years = sorted(df["_iso_year"].unique())
        return {
            "timeframe": timeframe,
            "total_weeks": len(files_written),
            "total_bars": len(df),
            "year_range": f"{years[0]}-{years[-1]}",
            "files_written": files_written,
        }

    def save_raw(self, df: pd.DataFrame, source: str, timeframe: str) -> str:
        """Save raw (pre-cleaning) data as a single Parquet file.

        Args:
            df: Raw OHLCV DataFrame.
            source: Source label (mt5, dukascopy).
            timeframe: Timeframe label.

        Returns:
            Path to saved file.
        """
        raw_dir = self.base_dir.parent / "raw" / source
        raw_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_dir / f"{source}_XAUUSD_{timeframe}.parquet"
        df.to_parquet(filepath, engine="pyarrow", index=False)
        return str(filepath)

    def load_weekly(self, year: int, week: int, timeframe: str) -> pd.DataFrame:
        """Load a specific weekly Parquet file.

        Returns:
            DataFrame or empty DataFrame if file doesn't exist.
        """
        filepath = self.base_dir / str(year) / f"week_{week:02d}_{timeframe}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath, engine="pyarrow")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    def load_range(
        self, start_year: int, end_year: int, timeframe: str
    ) -> pd.DataFrame:
        """Load all weekly files for a year range and concatenate.

        Returns:
            Single concatenated DataFrame sorted by time.
        """
        frames = []
        for year in range(start_year, end_year + 1):
            year_dir = self.base_dir / str(year)
            if not year_dir.exists():
                continue
            pattern = f"week_*_{timeframe}.parquet"
            for f in sorted(year_dir.glob(pattern)):
                frames.append(pd.read_parquet(f, engine="pyarrow"))

        if not frames:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def get_summary(self) -> dict:
        """Scan processed directory and return summary stats."""
        summary = {}
        for year_dir in sorted(self.base_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            year = year_dir.name
            files = list(year_dir.glob("*.parquet"))
            by_tf = {}
            for f in files:
                # Parse timeframe from filename: week_01_M5.parquet
                parts = f.stem.split("_")
                if len(parts) >= 3:
                    tf = parts[-1]
                    by_tf.setdefault(tf, 0)
                    by_tf[tf] += 1
            summary[year] = by_tf
        return summary

    def print_summary(self):
        """Print a formatted summary of stored data."""
        summary = self.get_summary()
        if not summary:
            print("No processed data found.")
            return

        print("\nProcessed Data Summary:")
        print(f"{'Year':<8}", end="")
        all_tfs = sorted({tf for by_tf in summary.values() for tf in by_tf})
        for tf in all_tfs:
            print(f"{tf:>8}", end="")
        print()
        print("-" * (8 + 8 * len(all_tfs)))

        for year in sorted(summary):
            print(f"{year:<8}", end="")
            for tf in all_tfs:
                count = summary[year].get(tf, 0)
                print(f"{count:>8}", end="")
            print()
