"""Dukascopy Data Loader - Download free historical OHLCV from Dukascopy.

Dukascopy stores tick data in binary format at known URLs. We download
hourly bi5 files (LZMA-compressed), decompress, parse ticks, then
aggregate into OHLCV bars at the desired timeframe.

URL pattern:
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1}/{DD}/{HH}h_ticks.bi5
  Note: Month is 0-indexed (January = 00).

Binary tick format (20 bytes per tick):
  - 4 bytes: milliseconds since hour start (uint32, big-endian)
  - 4 bytes: ask price (uint32, big-endian) * point_value
  - 4 bytes: bid price (uint32, big-endian) * point_value
  - 4 bytes: ask volume (float32, big-endian)
  - 4 bytes: bid volume (float32, big-endian)

For XAUUSD, point_value = 1e-3 (prices stored as integers / 1000).
"""

import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
TICK_SIZE = 20  # bytes per tick record

POINT_VALUES = {
    "XAUUSD": 1e-3,
    "EURUSD": 1e-5,
    "GBPUSD": 1e-5,
    "XAGUSD": 1e-3,            # Precious metals — same scale as XAUUSD
    "USDJPY": 1e-3,
    "USA500IDXUSD": 1e-3,      # S&P 500 index — verified: 3250 not 325000
    "LIGHTCMDUSD": 1e-3,       # WTI Crude Oil (Dukascopy symbol: LIGHT.CMD/USD)
}

# Map our internal symbol names to Dukascopy API symbols
DUKASCOPY_SYMBOL_MAP = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "XAGUSD": "XAGUSD",
    "USDJPY": "USDJPY",
    "US500": "USA500IDXUSD",
    "USOIL": "LIGHTCMDUSD",
}

RESAMPLE_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


class DukascopyLoader:
    """Download and parse Dukascopy tick data into OHLCV bars.

    Uses concurrent HTTP requests (ThreadPoolExecutor) to download
    multiple hours in parallel, significantly speeding up acquisition.
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        raw_dir: str = "storage/data/raw/dukascopy",
        max_workers: int = 10,
    ):
        self.symbol = symbol
        # Resolve Dukascopy API symbol name
        self._dukascopy_symbol = DUKASCOPY_SYMBOL_MAP.get(symbol, symbol)
        self.point_value = POINT_VALUES.get(self._dukascopy_symbol, 1e-3)
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def _new_session(self) -> requests.Session:
        """Create a new requests session (thread-safe)."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        return s

    def _build_url(self, dt: datetime) -> str:
        """Build Dukascopy data URL for a specific hour."""
        month_0idx = dt.month - 1  # Dukascopy uses 0-indexed months
        return (
            f"{BASE_URL}/{self._dukascopy_symbol}/"
            f"{dt.year}/{month_0idx:02d}/{dt.day:02d}/"
            f"{dt.hour:02d}h_ticks.bi5"
        )

    def _download_and_parse_hour(self, hour_dt: datetime) -> list:
        """Download, decompress, and parse one hour of tick data.

        Returns list of (time, price, volume) tuples.
        """
        url = self._build_url(hour_dt)
        session = self._new_session()

        for attempt in range(3):
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code == 404 or len(resp.content) == 0:
                    return []
                if resp.status_code == 200:
                    raw = self._decompress_bi5(resp.content)
                    return self._parse_ticks_fast(raw, hour_dt)
            except requests.RequestException:
                pass
            if attempt < 2:
                time.sleep(0.5 + attempt)
        return []

    def _decompress_bi5(self, data: bytes) -> bytes:
        """Decompress LZMA/bi5 data."""
        try:
            return lzma.decompress(data)
        except lzma.LZMAError:
            return b""

    def _parse_ticks_fast(self, raw: bytes, hour_dt: datetime) -> list:
        """Parse binary tick data using struct.iter_unpack (fast)."""
        if not raw or len(raw) < TICK_SIZE:
            return []

        # Trim to exact multiple of TICK_SIZE
        n_ticks = len(raw) // TICK_SIZE
        raw = raw[: n_ticks * TICK_SIZE]

        ticks = []
        base_ts = hour_dt.timestamp()
        for ms, ask_int, bid_int, ask_vol, bid_vol in struct.iter_unpack(">IIIff", raw):
            price = (ask_int + bid_int) * self.point_value / 2  # mid-price
            volume = ask_vol + bid_vol
            tick_ts = base_ts + ms / 1000.0
            ticks.append((tick_ts, price, volume))
        return ticks

    def _ticks_to_ohlcv(self, tick_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample tick data to OHLCV bars."""
        if tick_df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        freq = RESAMPLE_MAP.get(timeframe)
        if freq is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Use: {list(RESAMPLE_MAP)}")

        df = tick_df.set_index("time")
        ohlcv = df["price"].resample(freq).agg(
            open="first", high="max", low="min", close="last"
        )
        ohlcv["volume"] = df["volume"].resample(freq).sum()
        ohlcv = ohlcv.dropna(subset=["open"])
        ohlcv = ohlcv.reset_index()
        if ohlcv["time"].dt.tz is None:
            ohlcv["time"] = ohlcv["time"].dt.tz_localize("UTC")
        return ohlcv[["time", "open", "high", "low", "close", "volume"]]

    def download_day(self, date: datetime) -> pd.DataFrame:
        """Download all ticks for a single day using concurrent requests."""
        hours = [
            datetime(date.year, date.month, date.day, h, tzinfo=timezone.utc)
            for h in range(24)
        ]

        all_ticks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._download_and_parse_hour, h): h for h in hours}
            for future in as_completed(futures):
                ticks = future.result()
                if ticks:
                    all_ticks.extend(ticks)

        if not all_ticks:
            return pd.DataFrame(columns=["time", "price", "volume"])

        # Build DataFrame from (timestamp, price, volume) tuples
        arr = np.array(all_ticks)
        df = pd.DataFrame({
            "time": pd.to_datetime(arr[:, 0], unit="s", utc=True),
            "price": arr[:, 1],
            "volume": arr[:, 2],
        })
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def download_all(
        self,
        start_year: int = 2015,
        end_year: int = 2026,
        timeframes: list[str] | None = None,
        show_progress: bool = True,
    ) -> dict:
        """Download all requested timeframes for the full date range.

        Downloads ticks month-by-month, resamples to each timeframe,
        and saves monthly checkpoints.

        Args:
            start_year: First year to download.
            end_year: Last year to download (inclusive).
            timeframes: List of timeframes. Defaults to [M5, H1, H4, D1].
            show_progress: Show progress bars.

        Returns:
            Dict keyed by timeframe name, values are DataFrames.
        """
        if timeframes is None:
            timeframes = ["M5", "H1", "H4", "D1"]

        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        end = min(
            datetime(end_year, 12, 31, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
        )

        print(f"\nDownloading Dukascopy ticks: {start.date()} to {end.date()}")
        print(f"Using {self.max_workers} concurrent connections...")

        result = {tf: [] for tf in timeframes}
        current_month_start = start

        while current_month_start <= end:
            if current_month_start.month == 12:
                next_month = datetime(current_month_start.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                next_month = datetime(
                    current_month_start.year, current_month_start.month + 1, 1,
                    tzinfo=timezone.utc,
                )

            month_label = current_month_start.strftime("%Y-%m")

            # Check if checkpoint exists
            checkpoint_path = self.raw_dir / f"{self.symbol}_{month_label}_ticks.parquet"
            if checkpoint_path.exists():
                if show_progress:
                    print(f"  {month_label}: Loading from checkpoint...")
                combined = pd.read_parquet(checkpoint_path)
                if "time" in combined.columns and not combined.empty:
                    for tf in timeframes:
                        ohlcv = self._ticks_to_ohlcv(combined, tf)
                        if not ohlcv.empty:
                            result[tf].append(ohlcv)
                    current_month_start = next_month
                    continue

            # Build list of weekdays in this month
            days_in_month = []
            d = current_month_start
            while d < next_month and d <= end:
                if d.weekday() < 5:
                    days_in_month.append(d)
                d += timedelta(days=1)

            if not days_in_month:
                current_month_start = next_month
                continue

            month_ticks = []
            iterator = tqdm(
                days_in_month,
                desc=f"  {month_label}",
                disable=not show_progress,
            )

            for day in iterator:
                tick_df = self.download_day(day)
                if not tick_df.empty:
                    month_ticks.append(tick_df)

            if month_ticks:
                combined = pd.concat(month_ticks, ignore_index=True)
                combined = combined.sort_values("time").reset_index(drop=True)

                # Resample to each timeframe
                for tf in timeframes:
                    ohlcv = self._ticks_to_ohlcv(combined, tf)
                    if not ohlcv.empty:
                        result[tf].append(ohlcv)

                # Save checkpoint as Parquet (much faster than CSV)
                combined.to_parquet(checkpoint_path, engine="pyarrow", index=False)
                if show_progress:
                    print(f"    Saved checkpoint: {len(combined):,} ticks")

            current_month_start = next_month

        # Concatenate all months for each timeframe
        final = {}
        for tf in timeframes:
            if result[tf]:
                df = pd.concat(result[tf], ignore_index=True)
                df = df.sort_values("time").reset_index(drop=True)
                df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
                final[tf] = df
                print(f"  Dukascopy {tf}: {len(df):,} bars "
                      f"({df['time'].min().date()} to {df['time'].max().date()})")
            else:
                final[tf] = pd.DataFrame(
                    columns=["time", "open", "high", "low", "close", "volume"]
                )
                print(f"  Dukascopy {tf}: No data")

        return final
