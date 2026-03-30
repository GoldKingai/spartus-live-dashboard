# Spartus AI — Feature Upgrades & Training Plan (ARCHIVED)

> **Last Updated:** 2026-03-06
> **Status:** COMPLETED — All Part A upgrades implemented and deployed in W170
> **Superseded by:** `Training_Upgrade_Plan.md` (V2 upgrade planning based on live results)

This document is kept as a historical reference. Part A (5 feature upgrades) was fully implemented.
Part B (progressive training roadmap) is OUTDATED — superseded by Training_Upgrade_Plan.md.

---

Original document follows for reference:

---

This document contains TWO separate plans:

- **Part A: Standalone Feature Upgrades** — ALL 5 IMPLEMENTED (42 → 67 features, 670 obs dims)
- **Part B: Progressive Training Roadmap** — OUTDATED (replaced by Training_Upgrade_Plan.md)

---
---

# PART A: Standalone Feature Upgrades

> **Status:** READY FOR IMPLEMENTATION
> **Context:** The AI plateaued at Week 118 with 49.9% direction accuracy (coin flip). Root cause analysis revealed all 25 market features are derived from XAUUSD price alone — gold price is driven by external factors (USD strength, risk sentiment, safe-haven flows, commodities) that are completely absent from the feature set. The AI mastered basic mechanics (risk management, trade execution, session awareness) but cannot predict direction without external market context.
> **Decision:** Implement ALL 5 upgrades before resuming training. The AI starts fresh with an enriched 70-feature observation space. The current 42-feature checkpoint cannot be reused (observation dimensionality changes).
> **Impact:** 42 → 67 features (420 → 670 observation dimensions with frame stack x10).

---

## Upgrade 1: Correlated Instruments (Multi-Market Context)

**Expected improvement:** 10-20% (direction accuracy — the critical missing piece)
**Priority:** CRITICAL — Root cause of the plateau
**Complexity:** Medium
**New features:** 11

### What

Add price action from 5 correlated instruments to give the AI awareness of the forces that actually drive gold prices:

- **EURUSD** — Primary USD strength proxy. Gold has -0.97 correlation with DXY; EURUSD is the best single proxy for dollar strength available on every MT5 broker.
- **XAGUSD** — Precious metals co-movement. Silver has 0.92 historical correlation with gold and often leads breakouts by 1-3 bars.
- **USDJPY** — Safe-haven flow indicator. Both gold and JPY strengthen during risk-off events (-0.94 weekly correlation). When USDJPY drops sharply, gold typically rises.
- **US500** (S&P 500) — Risk sentiment. Gold's correlation with equities is regime-dependent: negative during crises (risk-off → gold up, stocks down), positive during reflation. The AI needs this to detect regime shifts.
- **USOIL** (WTI Crude) — Inflation expectations and commodity cycle. Oil-gold correlation is moderate (0.3-0.5) but spikes during inflation episodes.

### Why These 5 and Not Others

| Instrument | Gold Correlation | Signal Type | MT5 Available | Dukascopy M5 From |
|-----------|-----------------|-------------|---------------|-------------------|
| EURUSD | -0.97 (via DXY) | USD strength | Every broker | 2003 |
| XAGUSD | 0.92 | Precious metals | Most brokers | 2011 |
| USDJPY | -0.94 weekly | Safe haven flows | Every broker | 2003 |
| US500 | Regime-dependent | Risk sentiment | Most brokers | 2010 |
| USOIL | 0.3-0.5 | Inflation/commodity | Most brokers | 2012 |

**Rejected:** VIX (not on most MT5 brokers), US 10Y yields (not standard MT5), Copper (limited MT5 availability), DXY directly (not tradeable on MT5 — EURUSD is a superior proxy).

### Features to Add (11 new features)

| # | Feature | Calculation | Normalization | Why |
|---|---------|-------------|---------------|-----|
| 1 | `eurusd_returns_20` | 20-bar log return of EURUSD M5 | Z-score | USD strength direction |
| 2 | `eurusd_rsi_14` | RSI(14) of EURUSD / 100 | Z-score | USD momentum (overbought/oversold) |
| 3 | `eurusd_trend` | EMA(20) slope of EURUSD, normalized | Z-score | USD trend direction |
| 4 | `xagusd_returns_20` | 20-bar log return of XAGUSD M5 | Z-score | Silver leading gold moves |
| 5 | `xagusd_rsi_14` | RSI(14) of XAGUSD / 100 | Z-score | Silver momentum |
| 6 | `usdjpy_returns_20` | 20-bar log return of USDJPY M5 | Z-score | Safe-haven flow direction |
| 7 | `usdjpy_trend` | EMA(20) slope of USDJPY, normalized | Z-score | Yen trend (risk proxy) |
| 8 | `us500_returns_20` | 20-bar log return of US500 M5 | Z-score | Equity risk sentiment |
| 9 | `us500_rsi_14` | RSI(14) of US500 / 100 | Z-score | Equity momentum |
| 10 | `usoil_returns_20` | 20-bar log return of USOIL M5 | Z-score | Commodity/inflation direction |
| 11 | `gold_silver_ratio_z` | Z-score of XAU/XAG ratio vs 200-bar rolling mean | Z-score | Precious metals relative value |

**Removed:** `gold_oil_ratio_z` — the gold-oil ratio is a macro indicator that operates on daily/weekly timeframes. At M5 frequency it's noise. `usoil_returns_20` already captures the directional oil signal.

**Feature design principles:**
- **Returns (20-bar log):** Captures direction and momentum in a stationary, normalized form
- **RSI:** Captures overbought/oversold extremes — valuable for mean-reversion signals
- **Trend (EMA slope):** Captures medium-term direction, smoothed to reduce noise
- **Cross-ratio (Z-score):** Captures relative value between instruments — mean-reverting by construction

### Implementation Plan

**Step 1: Data Acquisition**
```
scripts/download_correlated.py (new file, ~120 lines)
- Download M5 data for EURUSD, XAGUSD, USDJPY, US500, USOIL from Dukascopy
- Same weekly Parquet format as XAUUSD: data/raw/{SYMBOL}/{SYMBOL}_M5_{YEAR}W{WEEK}.parquet
- Store in data/raw/EURUSD/, data/raw/XAGUSD/, data/raw/USDJPY/, data/raw/US500/, data/raw/USOIL/
- Dukascopy symbol mapping:
  - EURUSD → "EURUSD"
  - XAGUSD → "XAGUSD"
  - USDJPY → "USDJPY"
  - US500  → "USA500IDXUSD" (verify exact Dukascopy symbol)
  - USOIL  → "USOUSD" (WTI Crude Oil)
- Date range: 2015-01-01 to present (aligned with XAUUSD M5 availability)
- Use existing DukascopyLoader infrastructure from data_pipeline.py
```

**Step 2: Feature Engineering**
```
src/data/correlation_features.py (new file, ~150 lines)
- load_correlated_data(week_start, week_end, symbols) -> dict of DataFrames
- calc_instrument_features(df) -> returns_20, rsi_14, trend for one instrument
- calc_cross_ratios(xau_df, xag_df) -> gold_silver_ratio_z
- calc_correlation_features(xau_m5, corr_data: dict) -> DataFrame with 11 columns
- Use pd.merge_asof(direction='backward') to align timestamps
- Handle missing data: forward-fill up to 5 bars, then fill with 0.0
- No look-ahead bias: all features use only past/current bars
```

**Step 3: Integrate into Feature Builder**
```
src/data/feature_builder.py
- Import correlation_features
- In _build_features(), load correlated instrument data and compute 11 features
- Append 11 new columns to the feature DataFrame
- Update FEATURE_NAMES list
```

**Step 4: Config Update**
```
src/config.py
- correlated_symbols: list = ["EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"]
- correlated_data_dir: str = "data/raw"
- Update observation space: +110 dims (11 features x 10 frame stack)
```

### Data Sourcing

| Instrument | Historical Source | Format | Available From | Live Source (MT5) |
|-----------|-----------------|--------|---------------|-------------------|
| EURUSD | Dukascopy (free) | M5 Parquet | 2003 | `mt5.copy_rates_from_pos("EURUSD", M5, 0, 200)` |
| XAGUSD | Dukascopy (free) | M5 Parquet | 2011 | `mt5.copy_rates_from_pos("XAGUSD", M5, 0, 200)` |
| USDJPY | Dukascopy (free) | M5 Parquet | 2003 | `mt5.copy_rates_from_pos("USDJPY", M5, 0, 200)` |
| US500 | Dukascopy (free) | M5 Parquet | 2010 | `mt5.copy_rates_from_pos("US500", M5, 0, 200)` |
| USOIL | Dukascopy (free) | M5 Parquet | 2012 | `mt5.copy_rates_from_pos("USOIL", M5, 0, 200)` |

**Note:** MT5 symbol names vary by broker (e.g., "US500" vs "US500.cash" vs "SPX500"). The live implementation must include a symbol mapping config or auto-detection via `mt5.symbols_get()`.

### Training-Live Parity

| Aspect | Training | Live | Match? |
|--------|----------|------|--------|
| Data source | Historical M5 Parquet (Dukascopy) | `mt5.copy_rates_from_pos()` per symbol | YES |
| Timestamp alignment | `merge_asof(direction='backward')` | Same bar timestamp across symbols | YES |
| Feature calculation | Identical code path | Identical code path | YES |
| Missing data handling | Forward-fill then 0.0 | Same (+ `data_available` binary if symbol unavailable) | YES |

### Risks

- **Broker symbol availability:** Not all brokers offer US500 or USOIL. Mitigation: add per-symbol `data_available` binary feature; fill missing instruments with 0.0. The AI learns to ignore zeroed-out instruments.
- **Timestamp misalignment:** Different instruments may have slightly different bar timestamps (especially during low-liquidity hours). Mitigation: `merge_asof` with 5-minute tolerance handles this robustly.
- **Dukascopy vs MT5 data quality:** Historical and live data come from different providers. Feature distributions should be similar (same underlying market), but slight OHLCV differences are expected. Domain randomization in the training loop helps bridge this gap.
- **Correlation regime changes:** Gold correlations are NOT static — they shifted significantly in 2022-2025 (gold decorrelated from traditional drivers during central bank buying). This is a feature, not a bug — Upgrade 4 (Regime Detection) explicitly captures these shifts.

---

## Upgrade 2: Economic Calendar & Key Market Events

**Expected improvement:** 3-7% (volatility prediction and trade timing)
**Priority:** HIGH
**Complexity:** Low-Medium
**New features:** 6

### What

Encode upcoming high-impact economic events AND key recurring gold market events as deterministic schedule features. The AI learns:
1. **WHEN** volatility spikes are likely (economic data releases)
2. **WHEN** gold-specific price-setting events occur (London Fix, COMEX)

**Key insight:** Economic calendars are published weeks in advance. London Fix and COMEX hours are fixed by regulation. These are perfectly deterministic — no prediction needed, just encoding when events happen.

### Why London Fix and COMEX Matter

- **London Fix (LBMA Gold Price):** Set twice daily at 10:30 AM and 3:00 PM London time. The world's benchmark gold price. Major institutional flows converge on these times — gold volatility and volume spike reliably around fixes. The AM fix correlates with European trading open; the PM fix with the NYSE open.
- **COMEX Gold Futures:** The primary exchange for gold futures (8:20 AM - 1:30 PM ET). When COMEX is open, gold spot prices are strongly influenced by futures trading. Higher liquidity, tighter spreads, and stronger trends during COMEX hours.

### Features to Add (6 new features)

| # | Feature | Calculation | Normalization | Why |
|---|---------|-------------|---------------|-----|
| 1 | `hours_to_next_high_impact` | Hours until next high-impact USD event, capped at 48, normalized to [0,1] | Exempt | General event awareness |
| 2 | `hours_to_next_nfp_fomc` | Hours until next NFP or FOMC (biggest gold movers), capped at 168, normalized to [0,1] | Exempt | Major event countdown |
| 3 | `in_event_window` | Binary: 1 if within 30 min before/after high-impact event | Exempt | Active event volatility |
| 4 | `daily_event_density` | Count of high-impact events today / 10, capped at 1.0 | Exempt | Busy news day awareness |
| 5 | `london_fix_proximity` | Normalized distance to nearest London Fix (10:30 AM or 3:00 PM London). 1.0 = during fix, 0.0 = >2 hours away | Exempt | Gold price-setting event |
| 6 | `comex_session_active` | Binary: 1 if COMEX gold futures session is open (8:20 AM - 1:30 PM ET) | Exempt | Gold futures market activity |

### Implementation Plan

**Step 1: Historical Calendar Data**
```
data/calendar/economic_calendar.csv
Columns: date, time_utc, event_name, currency, impact (HIGH/MEDIUM/LOW)

Sources for historical data (free):
- ForexFactory historical calendar (scrape or manual download)
- Myfxbook economic calendar API
- investing.com economic calendar

Filter to HIGH impact events affecting gold:
- USD events: NFP, FOMC, CPI, PPI, GDP, Retail Sales, ISM Manufacturing
- EUR events: ECB rate decisions (affects EURUSD → gold)
- Safe-haven triggers: Geopolitical events tagged as HIGH impact
```

**Step 2: London Fix & COMEX Schedule**
```
No external data needed — these are deterministic:
- London Fix AM: 10:30 London time (UTC+0 winter, UTC+1 BST summer) every business day
- London Fix PM: 15:00 London time every business day
- COMEX gold: 08:20-13:30 ET (UTC-5 EST winter, UTC-4 EDT summer) every business day

Compute from UTC timestamp + DST calendar.
Store DST transition dates in a small lookup table (or use pytz/zoneinfo).
```

**Step 3: Feature Engineering**
```
src/data/calendar_features.py (new file, ~150 lines)
- load_calendar(csv_path) -> DataFrame
- calc_event_features(m5_timestamps, calendar_df) -> 4 event features
- calc_london_fix_proximity(m5_timestamps) -> 1 feature
- calc_comex_active(m5_timestamps) -> 1 feature
- Vectorized computation, no look-ahead
- Cache results per week to Parquet (events are static once past)
```

**Step 4: Integrate into Feature Builder**
```
src/data/feature_builder.py
- Import calendar_features
- In _build_features(), compute 6 calendar features
- Append to feature DataFrame
```

**Step 5: Config Update**
```
src/config.py
- calendar_csv_path: str = "data/calendar/economic_calendar.csv"
- Update observation space: +60 dims (6 features x 10 frame stack)
```

**Step 6: Live Implementation**
```
- Download next 2 weeks of calendar from ForexFactory/Myfxbook API at startup
- Refresh daily
- London Fix and COMEX times computed from system clock (same code as training)
- Same feature calculation code as training
```

### Data Sourcing

| Feature | Historical Source | Live Source | Deterministic? |
|---------|-----------------|------------|----------------|
| Event proximity (1-4) | economic_calendar.csv | ForexFactory/Myfxbook API (refresh daily) | YES (published weeks ahead) |
| London Fix (5) | Computed from UTC + DST rules | Same computation from system clock | YES (regulatory schedule) |
| COMEX session (6) | Computed from UTC + DST rules | Same computation from system clock | YES (exchange hours) |

### Training-Live Parity

| Aspect | Training | Live | Match? |
|--------|----------|------|--------|
| Event schedule | Historical CSV (ForexFactory) | Live API (same events, known in advance) | YES |
| London Fix times | Computed from UTC + DST rules | Same computation | EXACT |
| COMEX hours | Computed from UTC + DST rules | Same computation | EXACT |
| Feature calculation | Identical code | Identical code | YES |

### Risks

- **Event rescheduling:** Some events get rescheduled (e.g., FOMC emergency meetings). Marginal impact — the model learns "volatility likely around this time" not "exact event at this second."
- **Historical calendar data quality:** Different sources may disagree on exact event times or impact ratings. Mitigation: use ForexFactory as primary source (most widely used), cross-reference with Myfxbook for validation.
- **DST transitions:** London and New York switch DST on different dates. The DST lookup table must handle both correctly. Edge case: the week of DST change has shifted Fix/COMEX times in UTC.

---

## Upgrade 3: Spread & Liquidity Awareness

**Expected improvement:** 1-3% (trade timing and cost avoidance)
**Priority:** MEDIUM
**Complexity:** Low
**New features:** 2

### What

Give the AI awareness of current trading costs and unusual market activity:
1. **Spread estimate:** "How expensive is it to trade right now?" — avoids entering during wide-spread periods
2. **Volume spike:** "Is something unusual happening?" — detects institutional activity, news reactions, or liquidity crunches

### Features to Add (2 new features)

| # | Feature | Calculation | Normalization | Why |
|---|---------|-------------|---------------|-----|
| 1 | `spread_estimate_norm` | Session-based spread estimate / ATR(14) | Exempt | Trading cost relative to volatility |
| 2 | `volume_spike` | Current bar volume / SMA(20) of volume, capped at 5.0, normalized to [0,1] | Exempt | Unusual activity detection |

### Implementation Plan

**Step 1: Spread Feature (Training)**
```
Already have session-based spread data in MarketSimulator:
- London AM: 1.5 pips      → Low cost
- NY overlap: 2.0 pips     → Medium cost
- Asia: 3.0 pips            → Higher cost
- Off hours: 5.0 pips       → Expensive

Map hour → spread estimate → normalize by ATR(14)
Add to feature computation in feature_builder.py
```

**Step 2: Spread Feature (Live)**
```
actual_spread = mt5.symbol_info("XAUUSD").spread * mt5.symbol_info("XAUUSD").point
spread_feature = actual_spread / current_atr_14
```

**Step 3: Volume Spike Feature**
```
volume_ratio = current_bar_volume / sma_20_volume
volume_spike = min(volume_ratio / 5.0, 1.0)  # Cap at 5x average, normalize to [0,1]

Training: Use historical volume from OHLCV data (already available)
Live: Use mt5.copy_rates_from_pos() volume field
```

**Step 4: Integrate into Feature Builder**
```
src/data/feature_builder.py
- Add spread_estimate_norm to time-based feature section
- Add volume_spike to volume feature section (extend existing volume features)
```

### Training-Live Parity

| Aspect | Training | Live | Match? |
|--------|----------|------|--------|
| Spread source | Session-based estimate | `mt5.symbol_info().spread` | APPROXIMATE |
| Spread normalization | Same (/ ATR) | Same (/ ATR) | YES |
| Volume source | Historical OHLCV volume | `mt5.copy_rates_from_pos()` volume | YES |
| Volume normalization | Same (/ SMA20, cap 5) | Same | YES |

**Note:** Training spread is estimated, live is actual. Distributions will be similar but not identical. Domain randomization (+/-30% jitter, already implemented) bridges this gap. The AI learns "high spread = expensive" not "spread = exactly 2.3 pips."

### Risks

- **Volume data quality:** MT5 volume is tick volume (number of price changes), not actual traded volume. Historical Dukascopy volume may differ from live broker volume. Mitigation: normalize by rolling average — the model learns relative spikes, not absolute values.
- **Spread estimate accuracy:** Session-based estimates are averages; real spreads vary within sessions. Acceptable — the model needs directional awareness ("expensive" vs "cheap") not exact values.

---

## Upgrade 4: Intermarket Regime Detection

**Expected improvement:** 3-8% (regime awareness — knowing WHEN correlations hold)
**Priority:** HIGH
**Complexity:** Medium
**New features:** 2
**Prerequisite:** Upgrade 1 must be implemented (uses correlated instrument data)

### What

Gold's correlations with other markets are NOT constant — they shift dramatically during regime changes:
- **Normal regime:** Gold ↔ EURUSD correlation ~ -0.85 (inverse via USD)
- **Risk-off crisis:** Gold ↔ US500 correlation flips negative (gold up, stocks down)
- **Inflation regime:** Gold ↔ USOIL correlation strengthens to 0.7+ (both hedge inflation)
- **2022-2025 structural shift:** Gold decorrelated from traditional drivers during central bank buying spree

Rolling correlations capture these regime changes in real time. The AI learns "the normal USD relationship is breaking down" or "risk-off mode is activating."

### Why This Matters

Even with Upgrade 1's correlated instruments, the AI would see EURUSD going down and gold going up and treat these as independent signals. Rolling correlations explicitly encode WHETHER instruments are currently moving together, which tells the AI what regime the market is in. This is the difference between "seeing the data" and "understanding the context."

### Features to Add (2 new features)

| # | Feature | Calculation | Normalization | Why |
|---|---------|-------------|---------------|-----|
| 1 | `corr_gold_usd_100` | 100-bar rolling Pearson correlation of gold vs EURUSD 5-bar returns | Z-score | USD correlation regime |
| 2 | `corr_gold_spx_100` | 100-bar rolling Pearson correlation of gold vs US500 5-bar returns | Z-score | Risk sentiment regime |

**Removed:**
- `corr_regime_change` (derivative of rolling correlation) — Redundant with frame stacking. The AI sees 10 consecutive values of `corr_gold_usd_100` via frame stack x10. If correlation is shifting, the AI detects the transition from the trajectory directly. The derivative also amplifies the estimation noise inherent in 100-bar rolling correlations on overlapping returns.
- `risk_regime_score` (sign(SPX) * sign(-gold) composite) — Hand-engineers an interaction the neural network can learn from raw inputs (`us500_returns_20` + `returns_20bar`). The sign() function also throws away magnitude information, making "S&P fell 0.01%" identical to "S&P crashed 3%."

**Design notes:**
- 100-bar window (~8.3 hours on M5) balances responsiveness vs stability
- Uses 5-bar returns (not 1-bar) to reduce noise in correlation estimates
- Frame stacking x10 provides regime change detection naturally — the AI sees the correlation trajectory over 50 bars

### Implementation Plan

**Step 1: Feature Engineering**
```
src/data/regime_features.py (new file, ~60 lines)
- calc_rolling_correlation(series_a, series_b, window=100) -> Series
- calc_regime_features(gold_m5, eurusd_m5, us500_m5) -> DataFrame with 2 columns
- All vectorized with pandas rolling operations
- No look-ahead bias: rolling window uses only past data
```

**Step 2: Integrate into Feature Builder**
```
src/data/feature_builder.py
- Import regime_features
- In _build_features(), after computing correlated instrument features, compute 2 regime features
- Requires aligned EURUSD and US500 DataFrames (from Upgrade 1 data loading)
```

**Step 3: Config Update**
```
src/config.py
- regime_corr_window: int = 100  # Rolling correlation window (bars)
- Update observation space: +20 dims (2 features x 10 frame stack)
```

### Training-Live Parity

| Aspect | Training | Live | Match? |
|--------|----------|------|--------|
| Correlation computation | `pd.Series.rolling(100).corr()` | Identical computation | EXACT |
| Input data | Historical M5 returns | Live M5 returns | YES |

### Risks

- **Correlation noise at low sample sizes:** 100-bar rolling correlations can be noisy when markets are quiet (low variance → unstable correlation). Mitigation: clip output to [-1,1] and let the normalizer handle scaling.
- **Regime detection lag:** Rolling correlations detect regime changes AFTER they happen (lagging indicator). This is acceptable — even detecting a regime change 1-2 hours late is valuable vs not detecting it at all.
- **Spurious correlations:** Short-window correlations can show false signals. The frame stack (10 bars of history) helps the AI distinguish sustained regime shifts from noise.

---

## Upgrade 5: Session Microstructure

**Expected improvement:** 2-5% (intraday timing and pattern awareness)
**Priority:** MEDIUM
**Complexity:** Low
**New features:** 4

### What

Gold has distinct intraday patterns that repeat daily:
- **Asian session (00:00-07:00 UTC):** Low volatility, range-bound. Establishes a range that often becomes the reference for London breakouts.
- **London open (07:00-08:00 UTC):** First major liquidity injection. Breakouts from the Asian range are high-probability setups.
- **London-NY overlap (12:00-16:00 UTC):** Highest liquidity and volume of the day. Trends established here are the most reliable.
- **NY close (20:00-21:00 UTC):** Declining liquidity, wider spreads. Positions should be managed, not initiated.

The AI currently has `session_quality` (a single 0-1 value) and hour sin/cos. These capture WHEN it is but not WHAT HAS HAPPENED in the session so far. Session microstructure features tell the AI: "The Asian range was 15 pips and price just broke above it during London open" — a specific, actionable signal.

### Features to Add (4 new features)

| # | Feature | Calculation | Normalization | Why |
|---|---------|-------------|---------------|-----|
| 1 | `asian_range_norm` | (Asian high - Asian low) / ATR(14) | Exempt | How wide was the overnight range? |
| 2 | `asian_range_position` | (Current price - Asian midpoint) / (Asian range / 2), clipped [-2, 2] | Exempt | Price position relative to Asian range |
| 3 | `session_momentum` | (Current price - current session open) / ATR(14), clipped [-3, 3] | Exempt | How far has price moved this session? |
| 4 | `london_ny_overlap` | Binary: 1 if current time is 12:00-16:00 UTC | Exempt | Highest liquidity window |

**Design notes:**
- Asian range is computed fresh each day at 07:00 UTC (end of Asian session)
- `asian_range_position` uses midpoint for symmetric ±1 encoding; values beyond range (breakouts) allowed up to ±2
- `session_momentum` resets at each session boundary (00:00, 07:00, 12:00 UTC)
- `london_ny_overlap` is the single most important liquidity window for gold

### Implementation Plan

**Step 1: Feature Engineering**
```
src/data/session_features.py (new file, ~80 lines)
- calc_asian_range(ohlcv_m5) -> high, low, midpoint for each day
  - Asian session: bars where hour_utc in [0, 7)
  - Forward-fill range values for the rest of the day
- calc_asian_range_position(price, asian_mid, asian_range) -> [-2, 2]
- calc_session_momentum(ohlcv_m5) -> per-bar momentum since session open
  - Session boundaries: 00:00, 07:00, 12:00 UTC
- calc_london_ny_overlap(timestamps) -> binary
- calc_session_features(ohlcv_m5) -> DataFrame with 4 columns
- All vectorized, no look-ahead (Asian range only available after 07:00 UTC)
```

**Step 2: Handle Asian Range Look-Ahead**
```
CRITICAL: The Asian range (00:00-07:00 UTC) is only known AFTER 07:00 UTC.
- Before 07:00: use PREVIOUS day's Asian range (stale but valid, no look-ahead)
- After 07:00: use today's completed Asian range
- This is naturally handled by computing the range at 07:00 and forward-filling
```

**Step 3: Integrate into Feature Builder**
```
src/data/feature_builder.py
- Import session_features
- In _build_features(), compute 4 session features from M5 OHLCV
- Append to feature DataFrame
```

**Step 4: Config Update**
```
src/config.py
- asian_session_end_utc: int = 7  # UTC hour when Asian range completes
- session_boundaries_utc: list = [0, 7, 12]  # Session open times
- Update observation space: +40 dims (4 features x 10 frame stack)
```

### Training-Live Parity

| Aspect | Training | Live | Match? |
|--------|----------|------|--------|
| Asian range computation | From historical M5 bars (00:00-07:00 UTC) | From live M5 bars (same window) | EXACT |
| Session momentum | Price delta from historical session open | Price delta from live session open | EXACT |
| London-NY overlap | UTC timestamp check | UTC timestamp check | EXACT |
| All features | Deterministic time-based computation | Same | EXACT |

### Risks

- **Weekend/holiday gaps:** Asian range may be distorted on Sunday open (gap from Friday close) or holidays. Mitigation: clip `asian_range_norm` extremes; ATR normalization handles most distortion.
- **DST impact:** Session boundaries in UTC don't change with DST, but market participant behavior shifts slightly. Minor impact — the AI learns from the aggregate pattern.
- **Thin Asian sessions:** Some days have extremely tight Asian ranges (3-5 pips on gold). These produce large `asian_range_position` values on small moves. Mitigation: already clipped to [-2, 2].

---

## Upgrade Summary

| Upgrade | New Features | Impact Est. | Effort | Training-Live Parity | Priority | Status |
|---------|-------------|-------------|--------|---------------------|----------|--------|
| 1: Correlated Instruments | +11 | 10-20% | Medium | Full | CRITICAL | READY |
| 2: Economic Calendar & Events | +6 | 3-7% | Low-Med | Full/Exact | HIGH | READY |
| 3: Spread & Liquidity | +2 | 1-3% | Low | Approximate | MEDIUM | READY |
| 4: Intermarket Regime Detection | +2 | 3-8% | Medium | Exact | HIGH | READY |
| 5: Session Microstructure | +4 | 2-5% | Low | Exact | MEDIUM | READY |

**Combined total: +25 new features (42 → 67 features, 420 → 670 observation dimensions)**

**Combined improvement estimate:** 15-30% across all metrics, with the largest gain in direction accuracy (currently at coin-flip 49.9%). Diminishing returns when stacked, but each upgrade addresses a different blind spot.

**Implementation order:** All 5 upgrades implemented together before resuming training. The AI starts fresh with 67 features. The current 42-feature checkpoint cannot be reused (observation dimensionality changes).

### Normalization Classification (CRITICAL)

The existing normalizer (`src/data/normalizer.py`) applies rolling z-score to features in `market_feature_names` and passes through features in `norm_exempt_features` unchanged. New features MUST be classified correctly — binary and time-based features through z-score normalization produce garbage (std → 0 when values are constant, z-score → infinity).

**Add to `market_feature_names` (z-score normalized, 13 features):**
```
eurusd_returns_20, eurusd_rsi_14, eurusd_trend,
xagusd_returns_20, xagusd_rsi_14,
usdjpy_returns_20, usdjpy_trend,
us500_returns_20, us500_rsi_14,
usoil_returns_20,
gold_silver_ratio_z,
corr_gold_usd_100, corr_gold_spx_100
```

**Add to `norm_exempt_features` (already bounded, 12 features):**
```
hours_to_next_high_impact,    # [0, 1] continuous
hours_to_next_nfp_fomc,       # [0, 1] continuous
in_event_window,               # binary 0/1
daily_event_density,           # [0, 1] continuous
london_fix_proximity,          # [0, 1] continuous (time-based, like session_quality)
comex_session_active,          # binary 0/1
spread_estimate_norm,          # [0, ~3] already ATR-normalized
volume_spike,                  # [0, 1] already capped and normalized
asian_range_norm,              # [0, ~3] already ATR-normalized
asian_range_position,          # [-2, 2] already clipped
session_momentum,              # [-3, 3] already ATR-normalized and clipped
london_ny_overlap              # binary 0/1
```

**Why this matters:** Binary features (0/1) through z-score: when the rolling window is all-1s (e.g., middle of COMEX session), std = 0 and z-score = infinity → clipped to ±5. The feature becomes meaningless noise that could confuse training.

### Network Size Consideration

Current architecture: `net_arch_pi = [256, 256]`, `net_arch_qf = [256, 256]`.

- **Before upgrade:** 420 inputs → 256 units = 1.64:1 compression
- **After upgrade:** 670 inputs → 256 units = 2.62:1 compression

The first hidden layer must compress 670 dimensions through a 256-unit bottleneck. This is more aggressive but not necessarily a problem — the network only needs to extract the ~20-30 most important signals from 670 noisy inputs.

**Recommendation:** Start with 256x256. If the AI shows signs of underfitting (loss plateaus very early, features don't affect behavior, accuracy stuck at random despite clear signals in the data), bump to `[384, 384]`. This is a config-only change (`net_arch_pi` and `net_arch_qf` in `src/config.py`) — no code changes needed.

### Implementation Dependencies

```
Upgrade 1 (Correlated Instruments) ← No dependencies, but required by Upgrade 4
Upgrade 2 (Calendar & Events)      ← No dependencies
Upgrade 3 (Spread & Liquidity)     ← No dependencies
Upgrade 4 (Regime Detection)       ← Requires Upgrade 1 data (EURUSD, US500)
Upgrade 5 (Session Microstructure) ← No dependencies
```

### New Files to Create

| File | Purpose | ~Lines |
|------|---------|--------|
| `scripts/download_correlated.py` | Download M5 data for 5 correlated instruments from Dukascopy | ~120 |
| `src/data/correlation_features.py` | Per-instrument features + cross-ratio (11 features) | ~140 |
| `src/data/calendar_features.py` | Economic calendar + London Fix + COMEX (6 features) | ~150 |
| `src/data/regime_features.py` | Rolling correlations (2 features) | ~60 |
| `src/data/session_features.py` | Asian range + session momentum + overlap (4 features) | ~80 |
| `data/calendar/economic_calendar.csv` | Historical high-impact event schedule (2015-2026) | ~5000 rows |

### Existing Files to Modify

| File | Changes |
|------|---------|
| `src/data/feature_builder.py` | Import and call 4 new feature modules, update FEATURE_NAMES, update feature count |
| `src/config.py` | Add new config fields (symbols, paths, windows), update num_features to 67, update market_feature_names and norm_exempt_features |
| `src/environment/trade_env.py` | Update PRECOMPUTED_FEATURES list and observation space bounds for 67 features (670 dims) |
| `src/data/data_pipeline.py` | Add Dukascopy symbol mappings for 5 new instruments |

### Impact on Part B (Progressive Training Roadmap)

Implementing all 5 Part A upgrades NOW means the AI starts Phase 1 with 67 features instead of 42. Key implications:
- **Phase 1** trains on 67 features from the start (not 42) — graduation criteria unchanged
- **Phase 2** adds pattern features on top of 67 (total becomes 75 instead of 50)
- **Phase 3** no longer needs to introduce Upgrade 1 separately (already present)
- **Phase 4** no longer needs to introduce Upgrades 2-5 separately (already present)
- Phase graduation criteria and learning objectives remain valid — the AI still masters risk management before patterns, patterns before strategies, etc.
- Part B feature counts will be updated when implementation begins

---

## Deferred Features (Cannot Train On)

These features are available live via the MT5 API but **have no historical equivalent**, making them unusable for training. An RL model trained without these features would not know how to interpret them live.

### Tick-Level Microstructure

**MT5 API:** `mt5.copy_ticks_from()` — individual bid/ask ticks with millisecond timestamps.

**Why deferred:**
- Historical tick data goes back ~2 years on most brokers (vs 18 years of M5 bars)
- Tick patterns are highly broker-specific (ECN vs market maker)
- Storage would be massive (millions of ticks per day)
- Would require a completely different model architecture (CNN/transformer on tick sequences)

**Potential features if ever implemented:**
- Tick arrival rate (trades/second)
- Bid-ask bounce frequency
- Order flow imbalance
- Tick volatility vs bar volatility ratio

### Broker Sentiment / Retail Positioning

**MT5 API:** Some brokers expose % long vs % short via custom feeds.

**Why deferred:**
- Not available historically
- Broker-specific (different client bases = different sentiment)
- Would create broker lock-in
- Contrarian signal value is debatable

### Depth of Market (DOM)

**MT5 API:** `mt5.market_book_get()` — current order book.

**Why deferred:**
- No historical order book data available
- Gold is primarily OTC, so broker DOM is synthetic (aggregated from LPs)
- DOM changes millisecond-by-millisecond, impossible to reconstruct historically
- Would add enormous observation space for minimal proven value in gold

---
---

# PART B: Progressive Training Roadmap

> **Status:** Phase 1 (Foundation) in progress
> **Principle:** The AI learns like a child — master the basics before introducing advanced subjects. Each phase builds on top of the previous knowledge without retraining from scratch.
> **Note:** With all Part A upgrades now implemented in Phase 1, the AI starts with 67 features (670 dims) instead of the original 42 (420 dims). Phase graduation criteria remain unchanged. Feature counts in phase descriptions below reflect the ORIGINAL plan and will be updated when Part A implementation is complete.

---

## How Progressive Training Works

The AI's knowledge lives in its neural network weights. When we load a trained checkpoint and continue training with new features or harder conditions, the existing knowledge is preserved. The AI doesn't start over — it builds on top of what it already knows.

**Technical approach:**
- Phase 1 trains with the current 42-feature observation space
- When Phase 2 begins, we expand the observation space (e.g., to 50+ features) by adding new input slots
- We load the Phase 1 checkpoint, which preserves all learned weights for the original 42 features
- New feature weights initialize randomly, but the AI's foundation (risk management, basic direction, session awareness) remains intact
- Training continues with the enriched data — the AI learns to incorporate the new information on top of its existing skills

This is analogous to how LLMs work: a base model is trained on fundamentals, then fine-tuned on specific tasks without losing its general knowledge.

---

## Phase 1: Foundation — Basic Market Mechanics

**Status:** IN PROGRESS
**Goal:** The AI learns to not blow up, trade in the right direction, and manage risk.

### What the AI is learning:
- When to enter and exit positions
- Which direction to trade (LONG vs SHORT)
- Basic risk management (position sizing, stop losses)
- Session awareness (Asia, London, New York have different characteristics)
- Drawdown control (don't lose everything)
- Trade frequency management (don't overtrade)

### Current feature set (42 features):

| Group | Features | Count |
|-------|----------|-------|
| A: Price & Returns | Fractional diff close, log returns (1/5/20 bar), bar range, close position, body ratio | 7 |
| B: Volatility | ATR(14) norm, ATR ratio (7/21), BB width, BB position | 4 |
| C: Momentum & Trend | RSI(14), MACD signal, ADX(14), EMA cross (20/50), price vs EMA200, Stoch %K | 6 |
| D: Volume | Volume ratio (vs SMA20), OBV slope | 2 |
| E: Multi-Timeframe | H1/H4/D1 trend direction, H1 RSI, MTF alignment, HTF momentum | 6 |
| F: Time & Session | Hour sin/cos, day of week, session quality | 4 |
| G: Account State | Unrealized PnL, position duration, drawdown, exposure ratio, bars since trade, daily trade count, win streak, loss streak | 8 |
| H: Memory | Recent win rate, similar pattern win rate, trend prediction accuracy, TP hit rate, avg SL trail profit | 5 |

**Total observation:** 42 features x 10 frame stack = 420 input dimensions.

### Phase 1 Graduation Criteria

The AI must **consistently** meet ALL of these over at least 10 consecutive weeks (not just touch them once):

| Criteria | Threshold | Why |
|----------|-----------|-----|
| Win rate | >= 25% (10-week rolling average) | Must win at least 1 in 4 trades consistently |
| Emergency stop rate | < 15% (10-week rolling average) | Must not be hitting hard drawdown limits regularly |
| Direction accuracy | >= 35% (10-week rolling average) | Must be better than random at predicting direction |
| Profit factor | >= 0.30 (10-week rolling average) | Losses are shrinking relative to wins |
| Average hold bars | >= 2.0 (10-week rolling average) | Not just randomly entering and immediately exiting |
| Action std | > 0.2 | Policy hasn't collapsed — still exploring |
| Convergence state | Not COLLAPSED or WARMING_UP | Training is stable |

**What "consistently" means:** The 10-week rolling average must stay above the threshold for at least 10 consecutive evaluation points. A single bad week doesn't disqualify — the rolling average smooths out noise. But the AI can't just spike above the threshold once and drop back down.

### Expected training time: 50-150 weeks
This is the hardest phase — the AI starts from zero knowledge.

---

## Phase 2: Pattern Recognition

**Status:** PLANNED (waiting for Phase 1 graduation)
**Goal:** The AI learns to recognize and trade based on price patterns and market structure.
**Prerequisite:** Phase 1 graduation criteria met.

### What the AI will learn:
- Candlestick patterns (engulfing, pin bars, doji, hammers)
- Support and resistance levels (recent swing highs/lows)
- Price action structure (higher highs, lower lows, consolidation)
- Trend strength and quality

### New features to add (8 features, total becomes 50):

| # | Feature | Calculation | Why |
|---|---------|-------------|-----|
| 43 | `engulfing_signal` | +1 bullish engulfing, -1 bearish engulfing, 0 none | Classic reversal patterns |
| 44 | `pin_bar_signal` | +1 bullish pin bar, -1 bearish, 0 none | Rejection at levels |
| 45 | `dist_to_resistance` | Distance to nearest resistance / ATR | Where is the ceiling? |
| 46 | `dist_to_support` | Distance to nearest support / ATR | Where is the floor? |
| 47 | `swing_high_low_ratio` | Recent swing range / ATR | Market structure width |
| 48 | `consecutive_higher_highs` | Count of consecutive HH (or LL if negative) / 10 | Trend structure |
| 49 | `consolidation_bars` | Bars in current consolidation range / 100 | Ranging detection |
| 50 | `pattern_confluence` | Count of agreeing pattern signals / max possible | How many signals agree? |

### Implementation plan:

**Step 1: Feature engineering** (~100 lines)
```
src/data/pattern_features.py (new file)
- detect_candlestick_patterns(ohlcv) -> engulfing, pin_bar signals
- calc_support_resistance(ohlcv, lookback=100) -> nearest levels
- calc_market_structure(ohlcv) -> swing counts, consolidation
- All vectorized with pandas/numpy, no look-ahead bias
```

**Step 2: Integrate into feature builder**
```
src/data/feature_builder.py
- Import and call pattern_features in _build_features()
- Append 8 new columns to the feature DataFrame
```

**Step 3: Config update**
```
src/config.py
- num_features: 42 -> 50
- Observation space becomes 500 dims (50 x 10 frame stack)
```

**Step 4: Load Phase 1 checkpoint and continue training**
```
- Load spartus_latest.zip from Phase 1
- Create new model with 500-dim observation space
- Copy Phase 1 weights for first 420 input dims
- New weights for features 43-50 initialize randomly
- Continue training — Phase 1 knowledge preserved
```

### Phase 2 Graduation Criteria

Must consistently meet ALL over 10+ consecutive weeks:

| Criteria | Threshold | Why |
|----------|-----------|-----|
| Win rate | >= 35% (10-week rolling) | Pattern recognition should boost accuracy |
| Emergency stop rate | < 10% (10-week rolling) | Risk management must stay strong |
| Direction accuracy | >= 45% (10-week rolling) | Patterns should help direction calls |
| Profit factor | >= 0.50 (10-week rolling) | Approaching breakeven |
| Prediction accuracy | >= 55% (10-week rolling) | Better than random at predicting moves |
| TP hit rate | >= 5% (10-week rolling) | Starting to reach take profit targets |

### Expected additional training time: 30-80 weeks
Faster than Phase 1 because the foundation is already solid.

---

## Phase 3: Strategy & Regime Awareness

**Status:** PLANNED (waiting for Phase 2 graduation)
**Goal:** The AI learns to adapt its behavior to different market regimes and develops actual trading strategies.
**Prerequisite:** Phase 2 graduation criteria met.
**Uses:** Upgrade 1 (Correlated Instruments) from Part A — see that section for full data acquisition and parity details.

### What the AI will learn:
- Trend-following vs mean-reversion (different strategies for different markets)
- Regime detection and adaptation (trending / ranging / volatile / quiet)
- Multi-timeframe confluence (when H1, H4, D1 all agree, take bigger positions)
- Session-specific strategies (London breakout, NY reversal, Asia range)
- Correlation-based signals (USD strength via EURUSD, silver leading gold via XAGUSD)

### New features to add (10 features, total becomes 60):

| # | Feature | Calculation | Why |
|---|---------|-------------|-----|
| 51 | `regime_trending` | Probability current regime is trending (ADX + slope) | Regime-aware trading |
| 52 | `regime_ranging` | Probability current regime is ranging (ATR + BB width) | Different strategy needed |
| 53 | `regime_volatile` | Probability current regime is volatile (ATR spike + wide range) | Risk adjustment |
| 54 | `mtf_confluence_score` | Weighted agreement across H1/H4/D1 trends, -1 to +1 | When all timeframes agree |
| 55 | `session_momentum` | Price change since session open / ATR | Session-specific moves |
| 56 | `eurusd_returns_20` | 20-bar log return of EURUSD M5 | USD strength proxy (from Upgrade 1) |
| 57 | `eurusd_rsi_14` | RSI(14) of EURUSD / 100 | USD momentum (from Upgrade 1) |
| 58 | `xagusd_returns_20` | 20-bar log return of XAGUSD M5 | Silver correlation (from Upgrade 1) |
| 59 | `gold_silver_ratio_z` | Z-score of XAU/XAG ratio vs 200-bar rolling mean | Relative value (from Upgrade 1) |
| 60 | `strategy_signal` | Composite: mean-reversion if ranging, trend-follow if trending | Suggested strategy direction |

**Note:** Features 56-59 come from Upgrade 1 (Correlated Instruments). See Part A for full data download instructions, MT5 API calls, and Training-Live Parity details. The remaining 2 features from Upgrade 1 (`eurusd_trend`, `xagusd_rsi_14`) can be added here too if testing shows value — bringing total to 62.

### Implementation plan:

**Step 1: Correlated instrument data** (see Upgrade 1 in Part A for full details)
```
scripts/download_correlated.py (new file)
- Download EURUSD and XAGUSD M5 data (2007-2026)
- Same weekly Parquet format as XAUUSD
- Store in data/raw/EURUSD/ and data/raw/XAGUSD/
```

**Step 2: Regime detection engine**
```
src/data/regime_features.py (new file, ~150 lines)
- calc_regime_probabilities(ohlcv) -> trending/ranging/volatile scores
- calc_mtf_confluence(h1_trend, h4_trend, d1_trend) -> score
- calc_session_momentum(ohlcv, session_open_bar)
- calc_strategy_signal(regime, patterns, trend) -> composite
```

**Step 3: Correlated instrument features** (see Upgrade 1 in Part A for full details)
```
src/data/correlation_features.py (new file, ~80 lines)
- calc_correlation_features(xau_m5, eur_m5, xag_m5) -> 4-6 features
- Use pd.merge_asof(direction='backward') for alignment
- No look-ahead bias
```

**Step 4: Config + observation space update**
```
src/config.py
- num_features: 50 -> 60 (or 62 with all Upgrade 1 features)
- Observation space: 600 dims (60 x 10 frame stack)
- Need EURUSD and XAGUSD data paths
```

**Step 5: Load Phase 2 checkpoint and continue**

### Phase 3 Graduation Criteria

Must consistently meet ALL over 10+ consecutive weeks:

| Criteria | Threshold | Why |
|----------|-----------|-----|
| Win rate | >= 45% (10-week rolling) | Should be winning nearly half |
| Emergency stop rate | < 5% (10-week rolling) | Rarely hitting hard limits |
| Direction accuracy | >= 55% (10-week rolling) | Clearly better than random |
| Profit factor | >= 0.80 (10-week rolling) | Close to breakeven |
| Prediction accuracy | >= 58% (10-week rolling) | Consistent edge |
| Positive P/L weeks | >= 30% of weeks | Starting to have green weeks |

### Expected additional training time: 30-60 weeks

### Data requirement:
- EURUSD M5 historical data (2007-2026)
- XAGUSD M5 historical data (2007-2026)
- Both available from same MT5 broker

---

## Phase 4: Advanced — Profitability & Edge Refinement

**Status:** PLANNED (waiting for Phase 3 graduation)
**Goal:** The AI becomes consistently profitable and develops a genuine trading edge.
**Prerequisite:** Phase 3 graduation criteria met.
**Uses:** Upgrade 2 (Economic Calendar) and Upgrade 3 (Spread Estimate) from Part A — see those sections for full implementation and parity details.

### What the AI will learn:
- Economic calendar awareness (when to avoid / seek volatility)
- Spread cost optimization (trade when spreads are tight)
- Position management refinement (trailing stops, partial exits)
- Risk-adjusted returns (maximize Sharpe, not just win rate)
- Drawdown recovery strategies

### New features to add (5 features, total becomes 65):

| # | Feature | Calculation | Why |
|---|---------|-------------|-----|
| 61 | `hours_to_high_impact_event` | Hours to next NFP/FOMC/CPI, normalized (from Upgrade 2) | Avoid/exploit news events |
| 62 | `in_event_window` | Binary: within 30 min of high-impact event (from Upgrade 2) | Event-aware trading |
| 63 | `daily_event_density` | High-impact events today / 10 (from Upgrade 2) | Busy news day awareness |
| 64 | `spread_estimate_norm` | Current session spread / ATR (from Upgrade 3) | Trading cost awareness |
| 65 | `drawdown_recovery_phase` | Bars since last peak / 500, capped at 1.0 | Recovery patience |

**Note:** Features 61-63 come from Upgrade 2 and feature 64 comes from Upgrade 3. See Part A for full implementation plans, data sources, MT5 API calls, and Training-Live Parity details. The 4th feature from Upgrade 2 (`hours_to_next_nfp`) can be added here too if testing shows value — bringing total to 66.

### Implementation plan:

**Step 1: Economic calendar data** (see Upgrade 2 in Part A for full details)
```
data/calendar/economic_calendar.csv
- Historical high-impact events (NFP, FOMC, CPI, PPI, GDP, etc.)
- Sources: ForexFactory, investing.com, Myfxbook
- Filter to USD-affecting events only
```

**Step 2: Calendar feature engine** (see Upgrade 2 in Part A for full details)
```
src/data/calendar_features.py (new file, ~100 lines)
- Load calendar CSV
- For each M5 bar, compute hours to next event
- Vectorized calculation, no look-ahead
- Cache per-week results
```

**Step 3: Spread feature** (see Upgrade 3 in Part A for full details)
```
src/data/feature_builder.py
- Add spread_estimate_norm to _calc_time_features()
- Training: session-based estimate / ATR
- Live: mt5.symbol_info().spread / ATR
```

**Step 4: Config + observation space update**
```
src/config.py
- num_features: 60 -> 65 (or 66 with hours_to_next_nfp)
- Observation space: 650 dims (65 x 10 frame stack)
```

### Phase 4 Graduation Criteria (Ready for Live Trading Evaluation)

Must consistently meet ALL over 20+ consecutive weeks:

| Criteria | Threshold | Why |
|----------|-----------|-----|
| Win rate | >= 50% (20-week rolling) | Winning more than losing |
| Emergency stop rate | < 3% (20-week rolling) | Almost never hitting hard limits |
| Profit factor | >= 1.2 (20-week rolling) | Consistently profitable |
| Sharpe ratio | >= 0.5 (20-week rolling) | Risk-adjusted positive returns |
| Max drawdown | < 15% (over the 20-week window) | Controlled risk |
| Positive P/L weeks | >= 50% of weeks | More green weeks than red |
| Direction accuracy | >= 55% (20-week rolling) | Genuine directional edge |

### Expected additional training time: 20-50 weeks

**When Phase 4 criteria are met:** The AI is a candidate for paper trading on a live MT5 demo account. This is NOT live trading — it's a final validation step to confirm that training performance translates to real market conditions with real spreads, real slippage, and real execution.

---

## Phase 5: Live Validation (Paper Trading)

**Status:** FUTURE
**Goal:** Validate the trained model on a live demo account before risking real money.
**Prerequisite:** Phase 4 graduation criteria met over 20+ consecutive weeks.

### What happens:
- Deploy the Phase 4 model to MT5 demo account
- Run for 4-8 weeks minimum with real market data
- Compare live performance to training performance
- Check for:
  - Spread mismatch (training estimates vs actual broker spreads)
  - Execution slippage impact
  - Performance degradation from training to live
  - Any systematic bias (e.g., always too late on entries)

### Go-Live Criteria:
- Paper trading Sharpe >= 0.3 (some degradation from training is expected)
- Paper trading profit factor >= 1.0
- No catastrophic drawdown events (> 20%)
- Performance within 30% of training metrics

---

## Summary

### Standalone Upgrades (Part A)

| Upgrade | New Features | Impact | Effort | Parity | Priority | Status |
|---------|-------------|--------|--------|--------|----------|--------|
| 1: Correlated Instruments | +11 | 10-20% | Medium | Full | CRITICAL | READY |
| 2: Economic Calendar & Events | +6 | 3-7% | Low-Med | Full/Exact | HIGH | READY |
| 3: Spread & Liquidity | +2 | 1-3% | Low | Approximate | MEDIUM | READY |
| 4: Intermarket Regime Detection | +2 | 3-8% | Medium | Exact | HIGH | READY |
| 5: Session Microstructure | +4 | 2-5% | Low | Exact | MEDIUM | READY |
| **Total** | **+25** | **15-30%** | | | | |

### Progressive Phases (Part B)

| Phase | Focus | Features | Graduation Win Rate | Est. Training Time |
|-------|-------|----------|--------------------|--------------------|
| 1: Foundation | Risk mgmt, direction, sessions | 67 (42 base + 25 from Part A) | >= 25% | 50-150 weeks |
| 2: Patterns | Candlesticks, S/R, structure | +8 = 75 | >= 35% | 30-80 weeks |
| 3: Strategy | Regime, correlation, confluence | +10 = 85 | >= 45% | 30-60 weeks |
| 4: Edge | Advanced refinement | +5 = 90 | >= 50% + PF > 1.2 | 20-50 weeks |
| 5: Validation | Paper trading on live demo | 90 (same) | Real market test | 4-8 weeks |

**Note:** With all Part A upgrades implemented in Phase 1, Phases 3 and 4 will have fewer NEW features to add (some already present). Phase feature counts above may be adjusted during implementation.

### How Upgrades Map to Phases

| Upgrade | Implemented In | Used By |
|---------|---------------|---------|
| Upgrade 1: Correlated Instruments | Phase 1 (all 11 features) | Foundation + all later phases |
| Upgrade 2: Economic Calendar & Events | Phase 1 (all 6 features) | Foundation + all later phases |
| Upgrade 3: Spread & Liquidity | Phase 1 (all 2 features) | Foundation + all later phases |
| Upgrade 4: Intermarket Regime Detection | Phase 1 (all 2 features) | Foundation + all later phases |
| Upgrade 5: Session Microstructure | Phase 1 (all 4 features) | Foundation + all later phases |

**Total estimated time to live-ready:** 134-348 weeks of training (roughly 6-16 months of wall-clock time at current training speed). With 25 additional features providing directional context from day one, the AI should progress through phases significantly faster than the original estimate.

**Key principle:** Never rush to the next phase. The foundation must be solid. A child who can't add numbers won't learn algebra. An AI that can't manage risk won't benefit from pattern recognition — it'll just find new ways to lose money.
