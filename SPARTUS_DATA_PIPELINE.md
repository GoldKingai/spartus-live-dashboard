# Spartus Trading AI - Data Pipeline Specification

**Companion document to [SPARTUS_TRADING_AI.md](SPARTUS_TRADING_AI.md)**

> **DATA ACQUISITION IS PRIORITY #1.**
> Before writing ANY training code, secure years of clean XAUUSD data.
> Without data, there is nothing to train on.

---

## 1. Data Flow Overview

```
MT5 Terminal (Broker Data)
    │
    ▼
mt5_loader.py ── Pull OHLCV for M5/M15/H1/H4/D1
    │
    ▼
data_validator.py ── Quality checks, gap detection, integrity
    │
    ▼
feature_builder.py ── Calculate 42 features per bar
    │
    ▼
normalizer.py ── Expanding-window normalization (NO LOOK-AHEAD)
    │
    ▼
Parquet files ── Stored per-week in storage/data/
    │
    ▼
SpartusTradeEnv ── Loads weekly data for training episodes
```

---

## 2. Data Extraction from MT5

### Pull Historical Data

```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

class MT5DataLoader:
    """Load historical OHLCV data from MetaTrader 5."""

    TIMEFRAMES = {
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    def __init__(self, symbol="XAUUSD"):
        if not mt5.initialize():
            raise RuntimeError("MT5 not running")
        self.symbol = symbol

    def pull_range(self, timeframe, start_date, end_date):
        """Pull OHLCV data for a date range."""
        tf = self.TIMEFRAMES[timeframe]
        rates = mt5.copy_rates_range(self.symbol, tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No data for {self.symbol} {timeframe} {start_date}-{end_date}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]

    def pull_week(self, year, week_number):
        """Pull all timeframes for a specific ISO week."""
        # Calculate week start (Monday) and end (Saturday)
        jan1 = datetime(year, 1, 1, tzinfo=timezone.utc)
        # ISO week 1 contains the first Thursday of the year
        week_start = jan1 + timedelta(weeks=week_number - 1)
        # Adjust to Monday
        week_start -= timedelta(days=week_start.weekday())
        week_end = week_start + timedelta(days=6)  # Through Sunday

        data = {}
        for tf_name in self.TIMEFRAMES:
            try:
                df = self.pull_range(tf_name, week_start, week_end)
                data[tf_name] = df
            except ValueError:
                # No data for this timeframe (might be holiday week)
                data[tf_name] = pd.DataFrame()

        return data

    def pull_years(self, start_year, end_year):
        """Pull all weeks for a range of years."""
        all_weeks = []
        for year in range(start_year, end_year + 1):
            for week in range(1, 53):
                try:
                    data = self.pull_week(year, week)
                    # Only include weeks with enough M5 data
                    if len(data.get('M5', [])) >= 1000:
                        all_weeks.append({
                            'year': year,
                            'week': week,
                            'data': data
                        })
                except Exception as e:
                    print(f"Skipping {year} week {week}: {e}")
        return all_weeks
```

### Data Acquisition Plan (DO THIS FIRST)

```
PRIORITY ORDER:
═══════════════
1. MT5 Broker Data    → Pull immediately (you already have MT5)
2. Dukascopy          → Download same day (best free source for deep M5 history)
3. Kaggle / HistData  → Supplement for 20+ year daily/H4 context
4. Cross-validate     → Compare overlapping periods between sources

MINIMUM REQUIREMENT TO START TRAINING:
  → 5 years of clean M5 data (260 weeks × ~1300 bars = ~338,000 bars)
  → 10 years of H1/H4/D1 data for multi-timeframe context
```

---

## 2.5 Data Sources (Comprehensive)

### Source 1: MetaTrader 5 Broker — FASTEST TO GET

**Pros:** Already installed, exact broker pricing, includes your account's actual spreads
**Cons:** Limited M5 history (1-3 years typically), broker-dependent quality

**What you get:**
| Timeframe | Typical Depth | Bars |
|-----------|--------------|------|
| D1 | 10-20 years | 2,500-5,000 |
| H4 | 5-10 years | 7,500-15,000 |
| H1 | 3-5 years | 15,000-25,000 |
| M15 | 2-3 years | 35,000-70,000 |
| M5 | 1-3 years | 75,000-225,000 |

**Download code (already in MT5DataLoader above):**
```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

def download_all_mt5_data(symbol="XAUUSD"):
    """Download maximum available data from your MT5 broker."""
    if not mt5.initialize():
        raise RuntimeError("MT5 not running — launch MetaTrader 5 first")

    timeframes = {
        'M5':  mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1':  mt5.TIMEFRAME_H1,
        'H4':  mt5.TIMEFRAME_H4,
        'D1':  mt5.TIMEFRAME_D1,
    }

    results = {}
    for tf_name, tf_enum in timeframes.items():
        # Pull maximum available bars (MT5 limit is typically 999999)
        rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, 999999)

        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

            # Save to parquet
            output_path = f"storage/data/raw/mt5_{symbol}_{tf_name}.parquet"
            df.to_parquet(output_path, index=False)

            results[tf_name] = {
                'bars': len(df),
                'start': df['time'].iloc[0],
                'end': df['time'].iloc[-1],
                'path': output_path
            }
            print(f"  {tf_name}: {len(df):,} bars ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")

    mt5.shutdown()
    return results
```

### Source 2: Dukascopy — BEST FREE SOURCE FOR DEEP HISTORY

**Pros:** Swiss bank quality, tick-level data, 10+ years of M5, completely free
**Cons:** Requires download scripts, data in their own format

**What you get:**
| Data Type | Depth | Quality |
|-----------|-------|---------|
| Tick data | 2010-present | Institutional grade |
| M1 OHLCV | 2010-present | Excellent |
| M5 OHLCV | 2010-present | Excellent |
| All timeframes | Aggregated from ticks | Perfect |

**Download methods:**

```python
# ═══════════════════════════════════════════
# METHOD A: duka library (automated)
# ═══════════════════════════════════════════
# pip install duka

from duka.app import app as duka_download
import os

def download_dukascopy_data(symbol="XAUUSD", start_year=2015, end_year=2025):
    """Download XAUUSD data from Dukascopy."""
    output_dir = "storage/data/raw/dukascopy"
    os.makedirs(output_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        print(f"Downloading {symbol} {year}...")

        try:
            # Download M1 data (we'll resample to M5 ourselves)
            duka_download(
                symbol,
                start,
                end,
                1,  # 1 = M1 timeframe
                folder=output_dir
            )
            print(f"  {year}: Downloaded successfully")
        except Exception as e:
            print(f"  {year}: Failed - {e}")

def resample_m1_to_m5(m1_path, output_path):
    """Resample M1 data to M5 bars."""
    df = pd.read_csv(m1_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    m5 = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    m5.reset_index(inplace=True)
    m5.to_parquet(output_path, index=False)
    return m5


# ═══════════════════════════════════════════
# METHOD B: Manual download from website
# ═══════════════════════════════════════════
# 1. Go to: https://www.dukascopy.com/swiss/english/marketwatch/historical/
# 2. Select instrument: XAUUSD (Gold)
# 3. Select period: Custom range
# 4. Select timeframe: 5 Min (or 1 Min if you want to resample)
# 5. Click Download
# 6. Save CSV files to storage/data/raw/dukascopy/


# ═══════════════════════════════════════════
# METHOD C: tick_vault library
# ═══════════════════════════════════════════
# pip install tick-vault

# from tick_vault import TickVault
# vault = TickVault()
# vault.download("XAUUSD", "2015-01-01", "2025-12-31", timeframe="M5")
```

### Source 3: Kaggle — DEEPEST HISTORY (20+ years)

**Pros:** 20+ years of data, easy to download, community-validated
**Cons:** Quality varies by dataset, may need cleaning

**Best Kaggle datasets for XAUUSD:**

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials: place kaggle.json in ~/.kaggle/
# Get your API key from: https://www.kaggle.com/settings

# Download XAUUSD historical data
kaggle datasets download -d lucastrenzado/xauusd-gold-price-historical-data-2004-2024
# OR
kaggle datasets download -d mattiuzc/gold-price-historical-data

# Unzip
unzip xauusd-gold-price-historical-data-2004-2024.zip -d storage/data/raw/kaggle/
```

```python
def load_kaggle_data(kaggle_dir="storage/data/raw/kaggle"):
    """Load and standardize Kaggle XAUUSD data."""
    import glob

    all_files = glob.glob(f"{kaggle_dir}/*.csv")
    frames = []

    for f in all_files:
        df = pd.read_csv(f)

        # Standardize column names (varies by dataset)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'date' in col_lower or 'time' in col_lower:
                col_map[col] = 'time'
            elif 'open' in col_lower:
                col_map[col] = 'open'
            elif 'high' in col_lower:
                col_map[col] = 'high'
            elif 'low' in col_lower:
                col_map[col] = 'low'
            elif 'close' in col_lower:
                col_map[col] = 'close'
            elif 'vol' in col_lower:
                col_map[col] = 'volume'

        df.rename(columns=col_map, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        frames.append(df[['time', 'open', 'high', 'low', 'close', 'volume']])

    combined = pd.concat(frames).sort_values('time').drop_duplicates('time')
    return combined
```

### Source 4: HistData.com — RELIABLE M1 DATA (2000+)

**Pros:** Very deep history, M1 data, reliable institutional source
**Cons:** Manual download (monthly ZIP files), no API

```
HOW TO DOWNLOAD:
1. Go to: https://www.histdata.com/download-free-forex-historical-data/
2. Select: ASCII format
3. Select: XAUUSD (under Metals)
4. Select year/month → Download ZIP
5. Each ZIP contains a CSV with: DateTime, Open, High, Low, Close, Volume

FILE FORMAT:
20200101 000000;1517.030;1517.190;1517.030;1517.180;0
20200101 000100;1517.180;1517.200;1517.050;1517.090;0
```

```python
def load_histdata(histdata_dir="storage/data/raw/histdata"):
    """Load HistData CSV files."""
    import glob

    files = sorted(glob.glob(f"{histdata_dir}/*.csv"))
    frames = []

    for f in files:
        df = pd.read_csv(f, sep=';', header=None,
                         names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S', utc=True)
        frames.append(df[['time', 'open', 'high', 'low', 'close', 'volume']])

    return pd.concat(frames).sort_values('time')
```

### Source 5: yfinance — QUICK DAILY DATA (for validation)

```python
# pip install yfinance
import yfinance as yf

# Gold futures (GC=F) — D1 data only
gold = yf.download("GC=F", start="2010-01-01", end="2025-12-31", interval="1d")

# NOTE: yfinance does NOT have intraday gold data.
# Use ONLY for D1 cross-validation with other sources.
```

### Cross-Source Validation

```python
def validate_across_sources(mt5_df, dukascopy_df, timeframe='D1'):
    """
    Compare data from different sources to catch errors.
    Prices should match within 0.1% for the same bar.
    """
    # Merge on timestamp
    merged = pd.merge(
        mt5_df[['time', 'close']].rename(columns={'close': 'close_mt5'}),
        dukascopy_df[['time', 'close']].rename(columns={'close': 'close_duka'}),
        on='time',
        how='inner'
    )

    # Check price difference
    merged['diff_pct'] = abs(merged['close_mt5'] - merged['close_duka']) / merged['close_mt5'] * 100

    mismatches = merged[merged['diff_pct'] > 0.1]  # More than 0.1% difference

    print(f"Overlapping bars: {len(merged)}")
    print(f"Mismatches (>0.1%): {len(mismatches)}")
    print(f"Max difference: {merged['diff_pct'].max():.2f}%")
    print(f"Mean difference: {merged['diff_pct'].mean():.4f}%")

    if len(mismatches) > len(merged) * 0.01:
        print("WARNING: More than 1% of bars mismatch — investigate!")
    else:
        print("PASSED: Data sources are consistent")

    return merged
```

### Complete Data Download Script

```python
# scripts/download_data.py
"""
STEP 1 OF SPARTUS: Download all historical XAUUSD data.
Run this BEFORE anything else.

Usage: python scripts/download_data.py
"""

import os
import sys

def main():
    os.makedirs("storage/data/raw", exist_ok=True)
    os.makedirs("storage/data/processed", exist_ok=True)

    print("=" * 60)
    print("SPARTUS DATA ACQUISITION")
    print("=" * 60)

    # Step 1: MT5
    print("\n[1/4] Downloading from MT5 broker...")
    try:
        results = download_all_mt5_data()
        for tf, info in results.items():
            print(f"  ✓ {tf}: {info['bars']:,} bars")
    except Exception as e:
        print(f"  ✗ MT5 failed: {e}")
        print("  → Make sure MetaTrader 5 is running with XAUUSD available")

    # Step 2: Dukascopy
    print("\n[2/4] Downloading from Dukascopy...")
    try:
        download_dukascopy_data(start_year=2015, end_year=2025)
        print("  ✓ Dukascopy data downloaded")
    except Exception as e:
        print(f"  ✗ Dukascopy failed: {e}")
        print("  → Try manual download from https://www.dukascopy.com")

    # Step 3: Validate
    print("\n[3/4] Validating data quality...")
    # Run DataValidator on all downloaded files
    validator = DataValidator()
    # ... validation logic

    # Step 4: Process into weekly parquet files
    print("\n[4/4] Processing into weekly files...")
    # ... organize into storage/data/processed/YYYY/week_NN_TF.parquet

    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 3. Data Validation

### Quality Checks

```python
class DataValidator:
    """Validate data quality before it enters the pipeline."""

    def validate_week(self, week_data):
        """Run all validation checks on a week of data."""
        errors = []

        for tf_name, df in week_data.items():
            if df.empty:
                errors.append(f"{tf_name}: No data")
                continue

            # 1. OHLC Integrity
            bad_bars = df[
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ]
            if len(bad_bars) > 0:
                errors.append(f"{tf_name}: {len(bad_bars)} bars with invalid OHLC")

            # 2. No negative prices
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                errors.append(f"{tf_name}: Negative or zero prices found")

            # 3. No NaN values
            if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
                errors.append(f"{tf_name}: NaN values found")

            # 4. Time continuity (no duplicate timestamps)
            if df['time'].duplicated().any():
                errors.append(f"{tf_name}: Duplicate timestamps")

            # 5. Sorted by time
            if not df['time'].is_monotonic_increasing:
                errors.append(f"{tf_name}: Not sorted by time")

            # 6. No price spikes > 3% in one bar
            returns = df['close'].pct_change().abs()
            spikes = returns[returns > 0.03]
            if len(spikes) > 0:
                errors.append(f"{tf_name}: {len(spikes)} bars with >3% price spike")

            # 7. Volume > 0
            zero_vol = df[df['volume'] <= 0]
            if len(zero_vol) > len(df) * 0.1:  # Allow up to 10% zero volume
                errors.append(f"{tf_name}: >10% bars with zero volume")

        # 8. M5 has minimum bar count
        if 'M5' in week_data and len(week_data['M5']) < 1000:
            errors.append(f"M5: Only {len(week_data['M5'])} bars (need 1000+)")

        return len(errors) == 0, errors
```

### Multi-Timeframe Alignment Check

```python
def validate_timeframe_alignment(week_data):
    """Ensure higher timeframe data covers the M5 trading period."""
    if 'M5' not in week_data or week_data['M5'].empty:
        return False, "No M5 data"

    m5_start = week_data['M5']['time'].iloc[0]
    m5_end = week_data['M5']['time'].iloc[-1]

    issues = []
    for tf in ['H1', 'H4', 'D1']:
        if tf not in week_data or week_data[tf].empty:
            issues.append(f"{tf}: Missing data")
            continue

        tf_start = week_data[tf]['time'].iloc[0]
        tf_end = week_data[tf]['time'].iloc[-1]

        # HTF data should cover at least the M5 trading period
        if tf_start > m5_start:
            issues.append(f"{tf}: Starts after M5 ({tf_start} > {m5_start})")
        if tf_end < m5_end:
            issues.append(f"{tf}: Ends before M5 ({tf_end} < {m5_end})")

    return len(issues) == 0, issues
```

---

## 3.5 Data Cleaning (Before Feature Calculation)

> **The training engine is only as good as the data it learns from.**
> Dirty data = bad learning = bad model. Clean data is non-negotiable.
> Validation (Section 3) detects problems. This section FIXES them.

### Cleaning Pipeline

```python
class DataCleaner:
    """Clean and repair raw data before it enters the feature pipeline.
    Run AFTER validation, BEFORE feature calculation."""

    def clean_week(self, week_data):
        """Apply all cleaning steps to a week of data."""
        for tf_name, df in week_data.items():
            if df.empty:
                continue

            # 1. Remove duplicate timestamps (keep first occurrence)
            df = df.drop_duplicates(subset='time', keep='first')

            # 2. Sort by time (must be ascending)
            df = df.sort_values('time').reset_index(drop=True)

            # 3. Fix OHLC integrity violations
            df = self._fix_ohlc_integrity(df)

            # 4. Remove zero/negative prices
            df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]

            # 5. Fill small gaps (interpolate missing bars)
            df = self._fill_gaps(df, tf_name)

            # 6. Handle price spikes (smooth outliers)
            df = self._smooth_spikes(df, max_pct_change=0.03)

            # 7. Fill NaN values (forward fill, then backward fill)
            df[['open', 'high', 'low', 'close']] = (
                df[['open', 'high', 'low', 'close']].ffill().bfill()
            )

            # 8. Replace zero volume with rolling median
            df = self._fix_zero_volume(df)

            week_data[tf_name] = df

        return week_data

    def _fix_ohlc_integrity(self, df):
        """Ensure High >= max(Open, Close) and Low <= min(Open, Close)."""
        df = df.copy()
        # High must be the highest value
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        # Low must be the lowest value
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        return df

    def _fill_gaps(self, df, timeframe):
        """Fill gaps of 1-2 missing bars with interpolation.
        Larger gaps are left as-is (likely weekend/holiday)."""
        expected_freq = {
            'M5': '5min', 'M15': '15min', 'H1': '1h', 'H4': '4h', 'D1': '1D'
        }
        freq = expected_freq.get(timeframe, '5min')

        # Create complete time index
        full_index = pd.date_range(df['time'].iloc[0], df['time'].iloc[-1], freq=freq)

        # Only fill gaps of 1-2 bars (larger gaps = legitimate market closure)
        df = df.set_index('time').reindex(full_index)
        # Interpolate only where gaps are small
        max_fill = 2  # Fill at most 2 consecutive missing bars
        df[['open', 'high', 'low', 'close']] = (
            df[['open', 'high', 'low', 'close']].interpolate(limit=max_fill)
        )
        df['volume'] = df['volume'].fillna(0)
        df = df.dropna(subset=['close']).reset_index()
        df.rename(columns={'index': 'time'}, inplace=True)
        return df

    def _smooth_spikes(self, df, max_pct_change=0.03):
        """Replace price spikes > max_pct_change with interpolated values."""
        df = df.copy()
        for col in ['open', 'high', 'low', 'close']:
            pct_change = df[col].pct_change().abs()
            spike_mask = pct_change > max_pct_change
            if spike_mask.any():
                # Replace spikes with average of surrounding bars
                df.loc[spike_mask, col] = None
                df[col] = df[col].interpolate(method='linear')
        return df

    def _fix_zero_volume(self, df, window=20):
        """Replace zero volume with rolling median (not zero — zero distorts indicators)."""
        df = df.copy()
        rolling_median = df['volume'].rolling(window, min_periods=5).median()
        zero_mask = df['volume'] <= 0
        df.loc[zero_mask, 'volume'] = rolling_median[zero_mask]
        # If rolling median is also zero, use 1 as minimum
        df['volume'] = df['volume'].fillna(1).clip(lower=1)
        return df
```

### Cleaning Report

```python
def generate_cleaning_report(raw_week, cleaned_week):
    """Log what was cleaned so we can audit data quality."""
    report = {
        'duplicates_removed': 0,
        'ohlc_violations_fixed': 0,
        'gaps_filled': 0,
        'spikes_smoothed': 0,
        'zero_volume_fixed': 0,
        'rows_removed': 0,
    }
    for tf in raw_week:
        if tf in cleaned_week:
            report['rows_removed'] += len(raw_week[tf]) - len(cleaned_week[tf])
    return report

# The cleaning report is logged to storage/logs/data_cleaning.jsonl
# so we know EXACTLY what was changed in the data
```

### Data Quality Pipeline Order

```
RAW DATA (from MT5 / Dukascopy / Kaggle)
    │
    ▼
DataValidator.validate_week()     ← Detect problems (Section 3)
    │
    ▼
DataCleaner.clean_week()          ← Fix problems (THIS section)
    │
    ▼
DataValidator.validate_week()     ← Re-validate (must pass clean)
    │
    ▼
FeatureBuilder.build_all_features()  ← Calculate indicators (Section 4)
```

> **The data that enters the feature builder must be CLEAN.**
> Every bar must have valid OHLCV, no gaps, no spikes, no NaN values.
> This is the foundation the entire training engine builds on.

---

## 4. Feature Calculation

### Complete Feature Builder

```python
import talib as ta
import pandas_ta as pta
import numpy as np
from fracdiff import fdiff

class FeatureBuilder:
    """Calculate all 42 features from raw OHLCV data."""

    def __init__(self, config):
        self.config = config
        self.normalizer = ExpandingWindowNormalizer(
            method=config.normalization,
            window=config.normalization_window  # 200
        )

    def build_all_features(self, m5_df, htf_data):
        """
        Build all features for an entire M5 dataframe.
        Returns a DataFrame with normalized features ready for training.
        """
        df = m5_df.copy()

        # ═══════════════════════════════════════════
        # A. PRICE & RETURNS (7 features)
        # ═══════════════════════════════════════════

        # 1. Fractionally differentiated close
        optimal_d = self._find_optimal_d(df['close'])
        df['close_frac_diff'] = fdiff(df['close'].values, d=optimal_d)

        # 2-4. Log returns at different horizons
        df['returns_1bar'] = np.log(df['close'] / df['close'].shift(1))
        df['returns_5bar'] = np.log(df['close'] / df['close'].shift(5))
        df['returns_20bar'] = np.log(df['close'] / df['close'].shift(20))

        # 5. Bar range
        df['bar_range'] = (df['high'] - df['low']) / df['close']

        # 6. Close position within bar
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # 7. Body ratio
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)

        # ═══════════════════════════════════════════
        # B. VOLATILITY (4 features)
        # ═══════════════════════════════════════════

        # 8. Normalized ATR
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_14_norm'] = df['atr_14'] / df['close'] * 100

        # 9. ATR ratio (expansion/contraction)
        atr_7 = ta.ATR(df['high'], df['low'], df['close'], timeperiod=7)
        atr_21 = ta.ATR(df['high'], df['low'], df['close'], timeperiod=21)
        df['atr_ratio'] = atr_7 / (atr_21 + 1e-8)

        # 10-11. Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # ═══════════════════════════════════════════
        # C. MOMENTUM & TREND (6 features)
        # ═══════════════════════════════════════════

        # 12. RSI
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14) / 100

        # 13. MACD histogram normalized by ATR
        macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_signal'] = macd_hist / (df['atr_14'] + 1e-8)

        # 14. ADX (trend strength)
        df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14) / 100

        # 15. EMA cross signal
        ema_20 = ta.EMA(df['close'], timeperiod=20)
        ema_50 = ta.EMA(df['close'], timeperiod=50)
        df['ema_cross'] = (ema_20 - ema_50) / (df['atr_14'] + 1e-8)

        # 16. Price vs EMA 200
        ema_200 = ta.EMA(df['close'], timeperiod=200)
        df['price_vs_ema200'] = (df['close'] - ema_200) / (df['atr_14'] + 1e-8)

        # 17. Stochastic
        slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'],
                                 fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk / 100

        # ═══════════════════════════════════════════
        # D. VOLUME (2 features)
        # ═══════════════════════════════════════════

        # 18. Volume ratio
        vol_sma = ta.SMA(df['volume'].astype(float), timeperiod=20)
        df['volume_ratio'] = df['volume'] / (vol_sma + 1e-8)

        # 19. OBV slope
        obv = ta.OBV(df['close'], df['volume'].astype(float))
        df['obv_slope'] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        # Normalize OBV slope
        df['obv_slope'] = df['obv_slope'] / (df['obv_slope'].rolling(50).std() + 1e-8)

        # ═══════════════════════════════════════════
        # E. MULTI-TIMEFRAME CONTEXT (6 features)
        # ═══════════════════════════════════════════
        df = self._add_htf_features(df, htf_data)

        # ═══════════════════════════════════════════
        # F. TIME & SESSION (4 features)
        # ═══════════════════════════════════════════

        # 26-27. Cyclical hour encoding
        hours = df['time'].dt.hour + df['time'].dt.minute / 60
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # 28. Day of week
        df['day_of_week'] = df['time'].dt.dayofweek / 4  # Mon=0, Fri=1

        # 29. Session quality
        df['session_quality'] = hours.apply(self._get_session_quality)

        # ═══════════════════════════════════════════
        # NORMALIZE market features (expanding window)
        # ═══════════════════════════════════════════
        # NOTE: Only market features (#1-25) are normalized.
        # Time/session features (#26-29) are EXEMPT — they are already
        # cyclical [-1,1] or bounded [0,1]. Normalizing cyclical features
        # (hour_sin, hour_cos) would destroy their encoding.
        # Account (#30-37) and Memory (#38-42) features are also exempt
        # (computed at runtime, already ratios/flags).
        # See SPARTUS_TRAINING_METHODOLOGY.md Section 3 for full rationale.
        feature_cols_to_normalize = [
            'close_frac_diff', 'returns_1bar', 'returns_5bar', 'returns_20bar',
            'bar_range', 'close_position', 'body_ratio',
            'atr_14_norm', 'atr_ratio', 'bb_width', 'bb_position',
            'rsi_14', 'macd_signal', 'adx_14', 'ema_cross', 'price_vs_ema200', 'stoch_k',
            'volume_ratio', 'obv_slope',
            'h1_trend_dir', 'h4_trend_dir', 'd1_trend_dir',
            'h1_rsi', 'mtf_alignment', 'htf_momentum',
        ]
        # EXEMPT from normalization (already bounded/cyclical):
        # 'hour_sin', 'hour_cos', 'day_of_week', 'session_quality'

        for col in feature_cols_to_normalize:
            if col in df.columns:
                df[f'{col}_norm'] = self.normalizer.normalize_series(df[col])

        return df

    def _add_htf_features(self, m5_df, htf_data):
        """Add higher timeframe features aligned to M5 bars."""
        # For each M5 bar, find the most recent H1/H4/D1 bar
        for tf_name, col_prefix in [('H1', 'h1'), ('H4', 'h4'), ('D1', 'd1')]:
            htf_df = htf_data.get(tf_name)
            if htf_df is None or htf_df.empty:
                m5_df[f'{col_prefix}_trend_dir'] = 0.0
                m5_df[f'{col_prefix}_rsi'] = 0.5
                continue

            # Calculate HTF indicators
            htf_ema20 = ta.EMA(htf_df['close'], timeperiod=20)
            htf_ema20_slope = htf_ema20.diff(3) / (htf_df['close'] * 0.01 + 1e-8)

            htf_rsi = ta.RSI(htf_df['close'], timeperiod=14) / 100

            htf_macd = ta.MACD(htf_df['close'])[2]  # Histogram
            htf_atr = ta.ATR(htf_df['high'], htf_df['low'], htf_df['close'], timeperiod=14)

            # Map to M5 timeframe (use most recent HTF bar at or before M5 time)
            htf_df = htf_df.copy()
            htf_df['trend_dir'] = htf_ema20_slope.clip(-3, 3)
            htf_df['rsi'] = htf_rsi.fillna(0.5)
            htf_df['momentum'] = (htf_macd / (htf_atr + 1e-8)).clip(-3, 3)

            # Merge with M5 using as-of join (no future data)
            m5_df = pd.merge_asof(
                m5_df.sort_values('time'),
                htf_df[['time', 'trend_dir', 'rsi', 'momentum']].sort_values('time'),
                on='time',
                direction='backward',  # CRITICAL: only use past HTF bars
                suffixes=('', f'_{col_prefix}')
            )
            m5_df.rename(columns={
                'trend_dir': f'{col_prefix}_trend_dir',
                'rsi': f'{col_prefix}_rsi',
                'momentum': f'{col_prefix}_momentum' if col_prefix == 'h4' else f'drop_{col_prefix}'
            }, inplace=True)

        # MTF alignment (average of all trend directions)
        trend_cols = ['h1_trend_dir', 'h4_trend_dir', 'd1_trend_dir']
        m5_df['mtf_alignment'] = m5_df[trend_cols].mean(axis=1).clip(-1, 1)

        # HTF momentum (use H4)
        m5_df['htf_momentum'] = m5_df.get('h4_momentum', 0.0)

        return m5_df

    def _get_session_quality(self, hour):
        """Trading session quality score."""
        if 8 <= hour < 12:     return 1.0   # London AM (best)
        elif 13 <= hour < 17:  return 0.95  # NY overlap
        elif 12 <= hour < 13:  return 0.9   # London PM
        elif 17 <= hour < 20:  return 0.7   # NY PM
        elif 0 <= hour < 8:    return 0.4   # Asia
        else:                  return 0.2   # Off hours

    def _find_optimal_d(self, price_series, max_d=1.0, significance=0.05):
        """Find minimum fractional diff order for stationarity."""
        from statsmodels.tsa.stattools import adfuller
        for d in np.arange(0.1, max_d, 0.05):
            try:
                diff = fdiff(price_series.values, d=d)
                clean = diff[~np.isnan(diff)]
                if len(clean) > 100:
                    pval = adfuller(clean)[1]
                    if pval < significance:
                        return round(d, 2)
            except:
                continue
        return 0.5  # Default
```

---

## 5. Normalization (CRITICAL)

### The Rule

> **At bar T, normalization can ONLY use data from bars 0 to T.**
> Using bars T+1 and beyond is LOOK-AHEAD BIAS and invalidates all results.

### Implementation

```python
class ExpandingWindowNormalizer:
    """
    Normalize features using only past data.

    Two methods available:
    1. Rolling z-score: (x - rolling_mean) / rolling_std
    2. Expanding min-max: (x - expanding_min) / (expanding_max - expanding_min)

    Both guarantee no future data is used.
    """

    def __init__(self, method="rolling_zscore", window=200):
        self.method = method
        self.window = window

    def normalize_series(self, series):
        """Normalize an entire series using only past data at each point."""
        if self.method == "rolling_zscore":
            rolling_mean = series.rolling(window=self.window, min_periods=50).mean()
            rolling_std = series.rolling(window=self.window, min_periods=50).std()
            normalized = (series - rolling_mean) / (rolling_std + 1e-8)
            return normalized.clip(-5, 5)

        elif self.method == "expanding_minmax":
            expanding_min = series.expanding(min_periods=50).min()
            expanding_max = series.expanding(min_periods=50).max()
            normalized = (series - expanding_min) / (expanding_max - expanding_min + 1e-8)
            return normalized.clip(0, 1)

    def normalize_single(self, series, idx):
        """Normalize a single value at index idx using only past data."""
        if self.method == "rolling_zscore":
            start = max(0, idx - self.window)
            window_data = series.iloc[start:idx + 1]
            if len(window_data) < 10:
                return 0.0
            mean = window_data.mean()
            std = window_data.std()
            return np.clip((series.iloc[idx] - mean) / (std + 1e-8), -5, 5)
```

### Anti-Leakage Verification

```python
# This test runs on EVERY build
def test_normalization_no_leakage():
    """Verify that changing future data doesn't affect current normalization."""
    np.random.seed(42)
    series = pd.Series(np.random.randn(500).cumsum() + 2000)  # Simulated prices
    normalizer = ExpandingWindowNormalizer(method="rolling_zscore", window=200)

    for test_idx in [200, 300, 400]:
        # Normalize normally
        val_normal = normalizer.normalize_single(series, test_idx)

        # Corrupt ALL future data
        corrupted = series.copy()
        corrupted.iloc[test_idx + 1:] = 9999999

        # Normalize with corrupted future
        val_corrupted = normalizer.normalize_single(corrupted, test_idx)

        # Must be IDENTICAL
        assert val_normal == val_corrupted, \
            f"LEAKAGE at idx {test_idx}: {val_normal} != {val_corrupted}"
```

---

## 6. Data Storage Format

### Weekly Parquet Files

```
storage/data/
├── 2020/
│   ├── week_01_M5.parquet
│   ├── week_01_H1.parquet
│   ├── week_01_H4.parquet
│   ├── week_01_D1.parquet
│   ├── week_02_M5.parquet
│   └── ...
├── 2021/
│   └── ...
├── 2022/
│   └── ...
├── 2023/
│   └── ...
├── 2024/
│   └── ...
└── 2025/
    └── ...
```

**Why Parquet:**
- 5-10x smaller than CSV
- Preserves dtypes (no string-to-float conversion errors)
- Fast read/write
- Columnar format (efficient for feature selection)

### Pre-Computed Feature Files

```
storage/features/
├── 2020/
│   ├── week_01_features.parquet    # All 42 features, pre-normalized
│   └── ...
└── ...
```

Pre-computing features saves time during training (features calculated once, used many times).

---

## 7. Multi-Timeframe Alignment

### The `merge_asof` Approach

```python
# CRITICAL: Use 'backward' direction to avoid look-ahead bias
# This finds the most recent HTF bar AT OR BEFORE the M5 timestamp

m5_aligned = pd.merge_asof(
    m5_df.sort_values('time'),
    h1_df[['time', 'trend', 'rsi']].sort_values('time'),
    on='time',
    direction='backward'  # ← ONLY looks backward in time
)
```

**What this means:**
- At M5 bar 12:35, the H1 feature comes from the H1 bar that closed at 12:00 (or earlier)
- The H1 bar closing at 13:00 is NOT used (that's future data)
- This creates a natural lag in HTF features, which is realistic (same as live trading)

---

## 8. Data Split Strategy

```
ALL AVAILABLE DATA (e.g., 2020-2025 = ~260 weeks)
│
├── TRAINING SET (70%): Weeks 1-182
│   Used for: SAC training, memory building
│   AI trades through these weeks sequentially
│
├── VALIDATION SET (15%): Weeks 183-221
│   Used for: Walk-forward validation, hyperparameter tuning
│   AI is tested here (no learning)
│   Purge: 2-week gap between train and validation
│
└── TEST SET (15%): Weeks 222-260
    Used for: FINAL evaluation only
    NEVER used for any tuning or model selection
    Touched ONCE at the very end
    Purge: 2-week gap between validation and test
```

### Purging Between Splits

```
... Week 180 | Week 181 | [PURGE: Week 182-183] | Week 184 | Week 185 ...
     TRAIN       TRAIN     EMBARGO (unused)        VAL         VAL
```

The 2-week purge ensures no autocorrelation leaks between splits.

---

## 8.5 Data Augmentation for Training

> **Problem:** ~200 weeks of training data = ~200 episodes. This is a small dataset for deep RL.
> Data augmentation creates realistic variations that teach the agent to generalize beyond the
> exact historical sequences.

### Augmentation Methods (Applied at Episode Load Time)

```python
class TrainingDataAugmenter:
    """
    Time-series-aware data augmentation.
    Applied when loading a week for training — NOT applied to validation/test data.
    After augmentation, ALL features are recalculated from augmented OHLCV.
    """

    def augment_episode(self, week_data, rng):
        """Apply random augmentations to a week of data."""
        augmented = week_data.copy()

        # 1. MAGNITUDE SCALING
        #    Scale OHLCV by random factor near 1.0
        #    Simulates slightly different price levels (gold at $1900 vs $2100)
        #    The agent should trade the same regardless of absolute price level
        scale = rng.uniform(0.95, 1.05)
        for col in ['open', 'high', 'low', 'close']:
            augmented['M5'][col] *= scale
            augmented['H1'][col] *= scale
            augmented['H4'][col] *= scale
            augmented['D1'][col] *= scale
        # Volume scaled proportionally (higher prices → proportionally different volume)
        augmented['M5']['tick_volume'] *= rng.uniform(0.90, 1.10)

        # 2. WINDOW SLICING
        #    Randomly trim 0-100 bars from start and/or end
        #    Creates different episode lengths (prevents learning "bar 500 = exit time")
        trim_start = rng.randint(0, 50)
        trim_end = rng.randint(0, 50)
        if trim_start + trim_end < len(augmented['M5']) - 200:  # Keep at least 200 bars
            augmented['M5'] = augmented['M5'].iloc[trim_start:len(augmented['M5']) - trim_end]

        # 3. RECALCULATE FEATURES from augmented OHLCV
        #    CRITICAL: Never augment features directly — always recalculate from price data
        #    This ensures indicator consistency (e.g., RSI from augmented close is correct)
        augmented['features'] = feature_builder.build_all(augmented)

        return augmented
```

### What NOT to Augment

| Don't Augment | Why |
|---------------|-----|
| Time features (hour, day) | These are real calendar features, not price-derived |
| Session quality | Determined by time, not price |
| Feature values directly | Would break indicator consistency — always recalculate from OHLCV |
| Validation/test data | Augmentation is training-only to prevent evaluation bias |

### Regime Mixing (Curriculum Stage 2 Only)

During curriculum Stage 2 (weeks 31-80), randomly shuffle week order within the stage. This prevents the agent from memorizing specific temporal sequences (e.g., "after this week's pattern, next week always trends up"). Stage 3 uses original chronological order for realism.

---

## 9. Data Acquisition Checklist

Before proceeding to training engine development, verify:

```
□ MT5 data downloaded (all available timeframes)
□ Dukascopy M5 data downloaded (2015-2025 minimum)
□ At least 5 years of M5 data available (260+ weeks)
□ At least 10 years of D1/H4/H1 data for context
□ Cross-source validation passed (< 0.1% price difference)
□ Data organized into storage/data/processed/ by year/week
□ DataValidator passes on all week files
□ No gaps > 2 hours during London/NY sessions
□ Holiday weeks identified and excluded

TOTAL DATA TARGET:
  M5:  300,000+ bars (5+ years)
  H1:  40,000+ bars (5+ years)
  H4:  10,000+ bars (5+ years)
  D1:  5,000+ bars (20 years ideal)
```

---

**Document Version:** 3.3
**Updated:** 2026-02-23
**Status:** Complete - Companion to SPARTUS_TRADING_AI.md
**Changes in v3.0:** Added comprehensive data source details with download code, cross-validation, complete download script, data acquisition checklist
**Changes in v3.1:** Added Section 3.5 Data Cleaning with full DataCleaner class (OHLC repair, gap filling, spike smoothing, zero volume fix), cleaning reports, and data quality pipeline order
**Changes in v3.2:** Updated feature count references from 38→42 to match expanded feature set (added sl_distance_ratio, profit_locked_pct, tp_hit_rate, avg_sl_trail_profit)
**Changes in v3.3:** Added Section 8.5 Data Augmentation for Training (magnitude scaling, window slicing, regime mixing). Added TrainingDataAugmenter class with augment_episode() method. Documented what NOT to augment (time features, features directly, validation data).
**Changes in v3.3.1:** Cross-reference alignment audit. Fixed normalization: removed time/session features (hour_sin, hour_cos, day_of_week, session_quality) from normalization list — these are cyclical/bounded and must NOT be z-scored (aligned with SPARTUS_TRAINING_METHODOLOGY.md Section 3 exemptions).
**Changes in v3.3.2:** profit_locked_pct feature (#37) now uses ATR-based denominator instead of entry_price. For XAUUSD at ~$2650, entry_price denominator gave ~0.0004 (invisible to network); ATR gives ~0.5-2.5 (meaningful signal).
