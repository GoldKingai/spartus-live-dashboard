"""Download M5 data for correlated instruments from Dukascopy.

Uses the same proven DukascopyLoader that successfully downloaded XAUUSD
with 92% coverage. Downloads day-by-day with 10 concurrent workers per day,
which avoids Dukascopy rate limiting.

Stores weekly Parquet files at:
    storage/data/correlated/{SYMBOL}/{year}/week_{NN}_M5.parquet

Usage (run ONE symbol at a time):
    python scripts/download_correlated.py --symbol EURUSD
    python scripts/download_correlated.py --symbol XAGUSD
    python scripts/download_correlated.py --symbol USDJPY
    python scripts/download_correlated.py --symbol US500
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.data.dukascopy_loader import DukascopyLoader, RESAMPLE_MAP
from src.data.storage_manager import StorageManager


def download_symbol(symbol: str, start_year: int, end_year: int, config: TrainingConfig):
    """Download M5 data for one correlated instrument.

    Uses the standard DukascopyLoader (10 workers, day-by-day) which is
    proven to get 90%+ coverage without triggering rate limits.
    """
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} M5 data ({start_year}-{end_year})")
    print(f"Using 10 concurrent workers (same as XAUUSD)")
    print(f"{'='*60}")

    loader = DukascopyLoader(
        symbol=symbol,
        raw_dir=str(config.correlated_data_dir / symbol / "raw"),
        max_workers=10,
    )

    # Use the proven download_all method (day-by-day, 10 workers)
    # This is what successfully downloaded XAUUSD with 92% coverage
    result = loader.download_all(
        start_year=start_year,
        end_year=end_year,
        timeframes=["M5"],
        show_progress=True,
    )

    m5 = result.get("M5")
    if m5 is None or m5.empty:
        print(f"  WARNING: No M5 data downloaded for {symbol}")
        return

    print(f"\n  {symbol}: {len(m5):,} M5 bars total")
    print(f"  Range: {m5['time'].min().date()} to {m5['time'].max().date()}")
    bars_per_week = len(m5) / max(1, (m5['time'].max() - m5['time'].min()).days / 7)
    print(f"  Avg bars/week: {bars_per_week:.0f}")

    # Save as weekly Parquet files
    storage = StorageManager(
        base_dir=str(config.correlated_data_dir / symbol)
    )
    result = storage.save_weekly(m5, "M5")
    print(f"  Saved {result['total_weeks']} weekly files to {config.correlated_data_dir / symbol}")


def main():
    parser = argparse.ArgumentParser(
        description="Download correlated instrument data from Dukascopy"
    )
    parser.add_argument("--start-year", type=int, default=2015,
                        help="First year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2026,
                        help="Last year (default: 2026)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Download a single symbol (REQUIRED)")
    args = parser.parse_args()

    if not args.symbol:
        print("ERROR: You must specify --symbol to download one at a time.")
        print("Example: python scripts/download_correlated.py --symbol EURUSD")
        print("Available: EURUSD, XAGUSD, USDJPY, US500")
        sys.exit(1)

    config = TrainingConfig()
    symbols = config.correlated_symbols

    if args.symbol not in symbols:
        print(f"Unknown symbol: {args.symbol}. Available: {symbols}")
        sys.exit(1)

    print(f"Downloading M5 data for: {args.symbol}")
    print(f"Date range: {args.start_year} to {args.end_year}")
    print(f"Storage: {config.correlated_data_dir / args.symbol}")
    print(f"Workers: 10 (rate-limit safe)")

    download_symbol(args.symbol, args.start_year, args.end_year, config)

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
