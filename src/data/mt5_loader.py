"""MT5 Data Loader - Pull historical OHLCV data from MetaTrader 5."""

import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5DataLoader:
    """Load historical OHLCV data from MetaTrader 5."""

    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self._ensure_connected()

    def _ensure_connected(self):
        """Initialize MT5 connection if not already connected."""
        if not mt5.initialize():
            raise ConnectionError(
                f"MT5 initialize() failed: {mt5.last_error()}. "
                "Ensure MetaTrader 5 is running and logged in."
            )
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise ValueError(
                f"Symbol '{self.symbol}' not found. "
                "Check your broker supports this symbol."
            )
        if not info.visible:
            mt5.symbol_select(self.symbol, True)

    def pull_range(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Pull OHLCV data for a date range.

        Args:
            timeframe: One of M5, M15, H1, H4, D1.
            start_date: Start datetime (UTC).
            end_date: End datetime (UTC).
            max_retries: Number of retries on failure.

        Returns:
            DataFrame with columns [time, open, high, low, close, volume].
        """
        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Use: {list(TIMEFRAMES)}")

        start_utc = start_date.replace(tzinfo=timezone.utc)
        end_utc = end_date.replace(tzinfo=timezone.utc)

        rates = None
        for attempt in range(max_retries):
            self._ensure_connected()
            # Try copy_rates_range first; fall back to copy_rates_from
            rates = mt5.copy_rates_range(self.symbol, tf, start_utc, end_utc)
            if rates is not None and len(rates) > 0:
                break
            # Fallback: copy_rates_from with a large count
            rates = mt5.copy_rates_from(self.symbol, tf, start_utc, 999_999)
            if rates is not None and len(rates) > 0:
                break
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        if rates is None or len(rates) == 0:
            raise ValueError(
                f"No data for {self.symbol} {timeframe} "
                f"from {start_date} to {end_date}. Error: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def pull_week(self, year: int, week_number: int) -> dict:
        """Pull all timeframes for a specific ISO week.

        Returns:
            Dict keyed by timeframe name, values are DataFrames.
        """
        monday = datetime.fromisocalendar(year, week_number, 1)
        saturday = datetime.fromisocalendar(year, week_number, 6)

        result = {}
        for tf_name in TIMEFRAMES:
            try:
                df = self.pull_range(tf_name, monday, saturday)
                result[tf_name] = df
            except ValueError:
                result[tf_name] = pd.DataFrame(
                    columns=["time", "open", "high", "low", "close", "volume"]
                )
        return result

    def pull_from_pos(self, timeframe: str, count: int = 500_000) -> pd.DataFrame:
        """Pull bars counting backward from the most recent bar.

        Some brokers reject copy_rates_range for low timeframes but
        accept copy_rates_from_pos. This method works around that.
        """
        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'.")

        self._ensure_connected()
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data via copy_rates_from_pos for {self.symbol} {timeframe}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        return df.sort_values("time").reset_index(drop=True)

    def pull_all_available(self) -> dict:
        """Pull maximum available history for all timeframes.

        Uses copy_rates_range for H1+ and copy_rates_from_pos for M5/M15
        (many brokers reject range queries for low timeframes).

        Returns:
            Dict keyed by timeframe name, values are DataFrames.
        """
        now = datetime.now(timezone.utc)
        # Start dates for range-based pull (H1+)
        start_dates = {
            "H1": datetime(2015, 1, 1),
            "H4": datetime(2005, 1, 1),
            "D1": datetime(2000, 1, 1),
        }
        # For M5/M15, use position-based pull (backward from newest)
        # Broker-specific limits — most cap at ~50K-80K bars
        pos_counts = {
            "M5": 50_000,
            "M15": 80_000,
        }

        result = {}
        for tf_name in TIMEFRAMES:
            try:
                if tf_name in pos_counts:
                    df = self.pull_from_pos(tf_name, pos_counts[tf_name])
                else:
                    start = start_dates.get(tf_name, datetime(2015, 1, 1))
                    df = self.pull_range(tf_name, start, now)

                result[tf_name] = df
                print(f"  MT5 {tf_name}: {len(df):,} bars "
                      f"({df['time'].min().date()} to {df['time'].max().date()})")
            except ValueError as e:
                print(f"  MT5 {tf_name}: No data ({e})")
                result[tf_name] = pd.DataFrame(
                    columns=["time", "open", "high", "low", "close", "volume"]
                )
        return result

    def shutdown(self):
        """Shutdown MT5 connection."""
        mt5.shutdown()
