"""
Spartus Data Acquisition Pipeline
===================================
Master script to download, validate, clean, and store XAUUSD historical data.

Usage:
    python scripts/download_data.py [--skip-mt5] [--skip-dukascopy] [--start-year YYYY]

Pipeline:
    1. Connect to MT5 → pull all available history → save raw
    2. Download Dukascopy tick data → resample to OHLCV → save raw
    3. Merge sources (Dukascopy primary, MT5 fills recent gaps)
    4. Validate all data
    5. Clean any issues
    6. Re-validate (must pass)
    7. Split into weekly Parquet files
    8. Print summary
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_cleaner import DataCleaner
from src.data.data_validator import DataValidator
from src.data.storage_manager import StorageManager

TIMEFRAMES = ["M5", "H1", "H4", "D1"]
SYMBOL = "XAUUSD"


def pull_mt5_data() -> dict:
    """Step 1: Pull all available data from MT5."""
    print("\n" + "=" * 60)
    print("STEP 1: Pulling data from MetaTrader 5")
    print("=" * 60)

    try:
        from src.data.mt5_loader import MT5DataLoader
        loader = MT5DataLoader(symbol=SYMBOL)
        data = loader.pull_all_available()
        loader.shutdown()
        return data
    except Exception as e:
        print(f"\n  MT5 ERROR: {e}")
        print("  Continuing without MT5 data...")
        return {}


def pull_dukascopy_data(start_year: int) -> dict:
    """Step 2: Download deep history from Dukascopy."""
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Dukascopy historical data")
    print("=" * 60)

    from src.data.dukascopy_loader import DukascopyLoader

    loader = DukascopyLoader(
        symbol=SYMBOL,
        raw_dir=str(PROJECT_ROOT / "storage" / "data" / "raw" / "dukascopy"),
    )

    current_year = datetime.now(timezone.utc).year
    data = loader.download_all(
        start_year=start_year,
        end_year=current_year,
        timeframes=TIMEFRAMES,
    )
    return data


def merge_sources(mt5_data: dict, duka_data: dict) -> dict:
    """Step 3: Merge data sources. Dukascopy is primary, MT5 fills recent gaps."""
    print("\n" + "=" * 60)
    print("STEP 3: Merging data sources")
    print("=" * 60)

    merged = {}

    for tf in TIMEFRAMES:
        duka_df = duka_data.get(tf, pd.DataFrame())
        mt5_df = mt5_data.get(tf, pd.DataFrame())

        if duka_df.empty and mt5_df.empty:
            print(f"  {tf}: No data from either source!")
            merged[tf] = pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "volume"]
            )
            continue

        if duka_df.empty:
            print(f"  {tf}: MT5 only ({len(mt5_df):,} bars)")
            merged[tf] = mt5_df
            continue

        if mt5_df.empty:
            print(f"  {tf}: Dukascopy only ({len(duka_df):,} bars)")
            merged[tf] = duka_df
            continue

        # Dukascopy is primary. Append MT5 bars that are newer than Dukascopy's latest.
        duka_max_time = duka_df["time"].max()
        mt5_newer = mt5_df[mt5_df["time"] > duka_max_time]

        if len(mt5_newer) > 0:
            combined = pd.concat([duka_df, mt5_newer], ignore_index=True)
            combined = combined.sort_values("time").reset_index(drop=True)
            combined = combined.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
            print(
                f"  {tf}: Dukascopy ({len(duka_df):,}) + MT5 new ({len(mt5_newer):,}) "
                f"= {len(combined):,} bars"
            )
            merged[tf] = combined
        else:
            print(f"  {tf}: Dukascopy ({len(duka_df):,} bars), MT5 had no newer data")
            merged[tf] = duka_df

        # Cross-source validation on overlapping period
        overlap_start = max(duka_df["time"].min(), mt5_df["time"].min())
        overlap_end = min(duka_df["time"].max(), mt5_df["time"].max())
        duka_overlap = duka_df[(duka_df["time"] >= overlap_start) & (duka_df["time"] <= overlap_end)]
        mt5_overlap = mt5_df[(mt5_df["time"] >= overlap_start) & (mt5_df["time"] <= overlap_end)]

        if len(duka_overlap) > 0 and len(mt5_overlap) > 0:
            # Merge on time for comparison
            comp = pd.merge(
                duka_overlap[["time", "close"]],
                mt5_overlap[["time", "close"]],
                on="time",
                suffixes=("_duka", "_mt5"),
            )
            if len(comp) > 0:
                pct_diff = ((comp["close_duka"] - comp["close_mt5"]).abs() / comp["close_mt5"] * 100)
                mismatch = (pct_diff > 0.1).sum()
                print(
                    f"    Cross-validation: {len(comp)} overlapping bars, "
                    f"mean diff={pct_diff.mean():.4f}%, "
                    f"mismatches (>0.1%)={mismatch} ({mismatch/len(comp)*100:.1f}%)"
                )

    return merged


def validate_and_clean(data: dict) -> dict:
    """Steps 4-6: Validate → Clean → Re-validate."""
    print("\n" + "=" * 60)
    print("STEP 4-6: Validate, Clean, Re-validate")
    print("=" * 60)

    # Spike thresholds per timeframe (gold can move more on higher TFs)
    spike_thresholds = {"M5": 0.03, "H1": 0.05, "H4": 0.08, "D1": 0.10}
    cleaned = {}

    for tf in TIMEFRAMES:
        df = data.get(tf, pd.DataFrame())
        if df.empty:
            print(f"\n  {tf}: SKIP (no data)")
            cleaned[tf] = df
            continue

        spike_pct = spike_thresholds.get(tf, 0.03)
        validator = DataValidator(max_spike_pct=spike_pct)
        cleaner = DataCleaner(max_pct_change=spike_pct)

        # Step 4: Initial validation
        print(f"\n  {tf}: Validating {len(df):,} bars (spike threshold: {spike_pct*100:.0f}%)...")
        report = validator.validate(df, timeframe=tf)
        if not report["passed"]:
            failed = [c for c in report["checks"] if not c["passed"]]
            for c in failed:
                print(f"    [{c['name']}] {c['detail']}")

        # Step 5: Clean
        print(f"  {tf}: Cleaning...")
        df_clean, clean_report = cleaner.clean(df, timeframe=tf)
        for key, val in clean_report.items():
            if val and val != 0:
                print(f"    {key}: {val}")

        # Step 6: Re-validate
        report2 = validator.validate(df_clean, timeframe=tf)
        status = "PASS" if report2["passed"] else "FAIL"
        print(f"  {tf}: Re-validation: {status}")
        if not report2["passed"]:
            failed = [c for c in report2["checks"] if not c["passed"]]
            for c in failed:
                print(f"    [{c['name']}] {c['detail']}")

        cleaned[tf] = df_clean

    return cleaned


def save_to_parquet(data: dict, raw_data: dict = None, source: str = "merged"):
    """Step 7: Split into weekly Parquet files."""
    print("\n" + "=" * 60)
    print("STEP 7: Saving to weekly Parquet files")
    print("=" * 60)

    storage = StorageManager(base_dir=str(PROJECT_ROOT / "storage" / "data" / "processed"))

    # Save raw data if provided
    if raw_data:
        for src_name, src_data in raw_data.items():
            for tf, df in src_data.items():
                if not df.empty:
                    path = storage.save_raw(df, src_name, tf)
                    print(f"  Raw saved: {path}")

    # Save cleaned data as weekly Parquets
    for tf in TIMEFRAMES:
        df = data.get(tf, pd.DataFrame())
        if df.empty:
            print(f"  {tf}: No data to save")
            continue

        meta = storage.save_weekly(df, tf)
        print(
            f"  {tf}: {meta['total_weeks']} weeks, {meta['total_bars']:,} bars "
            f"({meta['year_range']})"
        )

    return storage


def print_final_summary(storage: StorageManager, data: dict):
    """Step 8: Print summary."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    storage.print_summary()

    # Target check
    print("\nData Targets:")
    targets = {"M5": 300_000, "H1": 40_000, "H4": 10_000, "D1": 5_000}
    for tf, target in targets.items():
        df = data.get(tf, pd.DataFrame())
        actual = len(df)
        status = "OK" if actual >= target else "BELOW TARGET"
        print(f"  {tf}: {actual:>10,} bars (target: {target:>10,}) [{status}]")
        if not df.empty:
            print(f"       Range: {df['time'].min().date()} to {df['time'].max().date()}")


def main():
    parser = argparse.ArgumentParser(description="Spartus Data Acquisition Pipeline")
    parser.add_argument("--skip-mt5", action="store_true", help="Skip MT5 data pull")
    parser.add_argument("--skip-dukascopy", action="store_true", help="Skip Dukascopy download")
    parser.add_argument("--start-year", type=int, default=2015, help="Dukascopy start year (default: 2015)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SPARTUS DATA ACQUISITION PIPELINE")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    start_time = time.time()
    raw_sources = {}

    # Step 1: MT5
    if args.skip_mt5:
        print("\nSkipping MT5 (--skip-mt5 flag)")
        mt5_data = {}
    else:
        mt5_data = pull_mt5_data()
        if mt5_data:
            raw_sources["mt5"] = mt5_data

    # Step 2: Dukascopy
    if args.skip_dukascopy:
        print("\nSkipping Dukascopy (--skip-dukascopy flag)")
        duka_data = {}
    else:
        duka_data = pull_dukascopy_data(start_year=args.start_year)
        if duka_data:
            raw_sources["dukascopy"] = duka_data

    # Step 3: Merge
    merged = merge_sources(mt5_data, duka_data)

    # Steps 4-6: Validate & Clean
    cleaned = validate_and_clean(merged)

    # Step 7: Save
    storage = save_to_parquet(cleaned, raw_data=raw_sources)

    # Step 8: Summary
    print_final_summary(storage, cleaned)

    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed/60:.1f} minutes.")
    print("Data saved to: storage/data/processed/")


if __name__ == "__main__":
    main()
