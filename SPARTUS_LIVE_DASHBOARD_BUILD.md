# Spartus Live Trading Dashboard — Build Plan v2.1

> **Purpose:** Standalone GitHub-ready application that loads a trained Spartus model and trades live on MetaTrader 5. Detects account currency automatically, computes all 67 features in real-time, executes trades, and displays a full monitoring dashboard with training-equivalent diagnostics.
>
> **Champion Model:** W170 (`spartus_champion_W170.zip`, 9.4 MB) — PF 2.818 test, MaxDD 13.6%, survives 2x spread stress
>
> **Folder:** `live_dashboard/` (self-contained, uploadable to GitHub separately)
>
> **Input:** Trained model ZIP + config → **Output:** Live trades on MT5 + comprehensive monitoring data

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Folder Structure](#2-folder-structure)
3. [Phase 1: MT5 Bridge & Account Detection](#3-phase-1-mt5-bridge--account-detection)
4. [Phase 2: Live Feature Pipeline](#4-phase-2-live-feature-pipeline)
5. [Phase 3: Model Inference Engine](#5-phase-3-model-inference-engine)
6. [Phase 4: Trade Executor](#6-phase-4-trade-executor)
7. [Phase 5: Risk & Safety Layer](#7-phase-5-risk--safety-layer)
8. [Phase 6: Memory System (Live)](#8-phase-6-memory-system-live)
9. [Phase 7: Dashboard UI (6 Tabs)](#9-phase-7-dashboard-ui)
10. [Phase 8: Model Package Loader](#10-phase-8-model-package-loader)
11. [Phase 9: Paper Trading Mode](#11-phase-9-paper-trading-mode)
12. [Phase 10: Startup Validation & Deployment Checklist](#12-phase-10-startup-validation--deployment-checklist)
13. [CLI Live Monitor](#13-cli-live-monitor)
14. [Enhanced Logging](#14-enhanced-logging)
15. [Data Flow Diagram](#15-data-flow-diagram)
16. [Configuration](#16-configuration)
17. [Implementation Order & Dependencies](#17-implementation-order--dependencies)
18. [Testing Strategy](#18-testing-strategy)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SPARTUS LIVE TRADING DASHBOARD                     │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │  MT5 Bridge  │──▶│ Feature      │──▶│   Model      │             │
│  │  (data +     │   │ Pipeline     │   │   Inference   │             │
│  │  execution)  │◀──│ (67 feats)   │   │   (SAC)      │             │
│  └──────┬───────┘   └──────────────┘   └──────┬───────┘             │
│         │                                       │                    │
│         │           ┌──────────────┐           │                    │
│         │           │   Risk &     │◀──────────┘                    │
│         │◀──────────│   Safety     │    4 actions:                   │
│         │  execute  │   Layer      │    direction, conviction,       │
│         │  order    └──────────────┘    exit, SL management         │
│         │                                                            │
│  ┌──────▼───────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  MetaTrader 5 │   │   Memory     │   │  Dashboard   │            │
│  │  (Broker)     │   │   (SQLite)   │   │  (PyQt6)     │            │
│  └───────────────┘   └──────────────┘   └──────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Principle

The live dashboard replicates the **exact same** observation pipeline as training. The agent sees identical 670-dim observations whether training or live. No approximations, no shortcuts.

### Key Differences from Training

| Aspect | Training | Live |
|--------|----------|------|
| Data source | Historical parquet files | Real-time MT5 API |
| Execution | MarketSimulator (simulated) | MT5 OrderSend (real money) |
| Spreads/slippage | Domain-randomized | Actual broker values |
| Action mode | Stochastic (exploration) | Deterministic (no exploration) |
| Balance | Simulated (starts at 100) | Real broker account balance |
| Account currency | Config setting (USD default) | Auto-detected from MT5 |
| Feature warmup | Full history available | Must accumulate 200+ bars first |

### Deployment Model — Fully Standalone

The live dashboard is a **self-contained application** with zero runtime dependency on the training codebase:

- **GitHub-ready:** The entire `live_dashboard/` folder is its own repository. Users clone it, install dependencies, drop in a model, and run.
- **No training repo needed:** All feature computation, normalizer, memory system, and risk management code lives inside `live_dashboard/`. Nothing is imported from the training repository.
- **Different machine OK:** The dashboard can (and typically will) run on a different computer than training. The only artifact needed from training is the model ZIP file.
- **Model drop-in:** Place the model ZIP (e.g., `spartus_champion_W170.zip`) in the `model/` folder. The system auto-discovers and loads it.
- **Feature parity maintained via tests:** A parity test (`tests/test_parity.py`) validates that standalone feature computation matches reference observations saved during training.

**Quick Start:**
```bash
git clone <repo>/spartus-live-dashboard
cd spartus-live-dashboard
pip install -r requirements.txt
# Copy model ZIP into model/ folder
python main.py
# Dashboard opens → click "Start Trading" when ready
```

---

## 2. Folder Structure

```
live_dashboard/
├── main.py                          # Entry point — launches dashboard
├── requirements.txt                 # Dependencies (PyQt6, MetaTrader5, etc.)
├── README.md                        # Setup instructions
│
├── config/
│   ├── live_config.py               # LiveConfig dataclass
│   └── default_config.yaml          # User-editable config file
│
├── core/
│   ├── mt5_bridge.py                # MT5 connection, data feeds, account info
│   ├── feature_pipeline.py          # Real-time 67-feature computation
│   ├── live_normalizer.py           # Rolling z-score normalizer (200-bar buffer)
│   ├── model_loader.py              # Load model ZIP + reward state + config
│   ├── inference_engine.py          # SAC model inference (670-dim → 4 actions)
│   ├── trade_executor.py            # Translate actions → MT5 orders
│   ├── risk_manager.py              # Live risk: circuit breakers, DD halt, etc.
│   └── position_manager.py          # Track open positions, trailing SL, recovery
│
├── features/
│   ├── technical.py                 # Groups A-E: price, volatility, momentum, vol, MTF
│   ├── time_session.py              # Group F: hour encoding, session quality
│   ├── correlation.py               # Upgrade 1: correlated instruments (11)
│   ├── calendar.py                  # Upgrade 2: economic calendar (6)
│   ├── spread_liquidity.py          # Upgrade 3: spread + volume (2)
│   ├── regime.py                    # Upgrade 4: regime detection (2)
│   ├── session_micro.py             # Upgrade 5: session microstructure (4)
│   ├── account_features.py          # Group G: 8 live account features
│   └── memory_features.py           # Group H: 5 memory features
│
├── memory/
│   ├── trading_memory.py            # SQLite: trades, patterns, predictions, journal
│   ├── trend_tracker.py             # Predict → verify cycle
│   └── trade_analyzer.py            # Lesson classification
│
├── dashboard/
│   ├── main_window.py               # PyQt6 main window + tab layout
│   ├── tab_live_status.py           # Tab 1: Live status + position
│   ├── tab_performance.py           # Tab 2: Performance metrics + charts
│   ├── tab_trade_journal.py         # Tab 3: Trade journal + decisions
│   ├── tab_model_state.py           # Tab 4: Model info + feature health
│   ├── tab_alerts.py                # Tab 5: Alerts + safety status
│   ├── tab_analytics.py             # Tab 6: Analytics & diagnostics
│   ├── widgets.py                   # Shared UI components
│   └── theme.py                     # Dark theme (matching training dashboard)
│
├── safety/
│   ├── circuit_breaker.py           # Consecutive loss / daily DD halt
│   ├── weekend_manager.py           # Friday close + gap protection
│   ├── connection_monitor.py        # MT5 heartbeat + reconnection
│   └── emergency_stop.py            # One-click halt all trading
│
├── scripts/
│   └── live_monitor.py              # CLI monitoring tool (like training's monitor.py)
│
├── utils/
│   ├── logger.py                    # Live trade logging (JSONL)
│   ├── timeframe_aggregator.py      # M5 → H1/H4/D1 aggregation
│   └── symbol_mapper.py             # Broker symbol name mapping
│
├── mt5_scripts/
│   └── CalendarBridge.mq5           # MQL5 service → exports calendar events to JSON
│
├── data/
│   └── calendar/
│       ├── known_events.json        # Static annual calendar (NFP/FOMC/CPI dates)
│       └── economic_calendar.csv    # Optional: user-supplied CSV override
│
├── storage/                         # Created at runtime
│   ├── state/
│   │   └── normalizer_state.json    # Normalizer buffer persistence (auto-saved)
│   ├── memory/
│   │   └── live_trading.db          # Live trades SQLite
│   ├── logs/
│   │   ├── trades.jsonl             # Every trade with full details
│   │   ├── actions.jsonl            # Every action decision
│   │   ├── alerts.jsonl             # Alert history
│   │   ├── observations.jsonl       # Periodic obs + action dumps (configurable)
│   │   ├── feature_stats.jsonl      # Feature distribution snapshots per session
│   │   └── weekly_summary.jsonl     # Weekly performance aggregates
│   └── reports/
│       └── weekly/                  # Auto-generated weekly report JSONs
│
├── model/                           # User places model here
│   └── (empty — user drops model ZIP here)
│
└── tests/
    ├── fixtures/                    # Reference observations + feature snapshots
    │   └── reference_obs_W170.npz  # Saved 670-dim observations for parity test
    ├── test_mt5_bridge.py
    ├── test_feature_pipeline.py
    ├── test_live_normalizer.py
    ├── test_normalizer_persistence.py
    ├── test_trade_executor.py
    ├── test_risk_manager.py
    ├── test_model_loader.py
    ├── test_startup_validator.py
    ├── test_parity.py              # CRITICAL: observation parity vs reference
    ├── test_feature_drift.py
    ├── test_logging.py
    └── test_weekly_report.py
```

---

## 3. Phase 1: MT5 Bridge & Account Detection

### File: `core/mt5_bridge.py`

The MT5 bridge is the foundation — everything depends on it.

### Class: `MT5Bridge`

```python
class MT5Bridge:
    """Manages all communication with MetaTrader 5.

    Responsibilities:
    - Connect/disconnect/reconnect to MT5 terminal
    - Auto-detect account currency and symbol parameters
    - Stream M5 bars for XAUUSD + correlated instruments
    - Provide H1/H4/D1 bars (aggregated from M5 or fetched separately)
    - Report current spread, tick values, account balance/equity
    - Place/modify/close orders
    """
```

### Methods

```python
def connect(self) -> bool:
    """Initialize MT5, validate connection, detect account."""
    # 1. mt5.initialize()
    # 2. mt5.account_info() → account_currency, balance, equity, leverage
    # 3. mt5.symbol_info("XAUUSD") → tick_value, tick_size, contract_size,
    #    volume_min/max/step, spread, point
    # 4. Validate all required symbols available (XAUUSD + correlated)
    # 5. Store symbol_info dict for risk manager

def get_account_info(self) -> dict:
    """Return current account state.
    Returns: {
        'currency': 'USD'|'GBP'|'EUR'|...,
        'balance': float,
        'equity': float,
        'margin': float,
        'free_margin': float,
        'leverage': int,
        'server': str,
        'name': str,
    }
    """

def get_symbol_info(self, symbol: str = "XAUUSD") -> dict:
    """Return broker's symbol parameters.
    Returns: {
        'tick_value': float,      # In account currency
        'tick_size': float,
        'contract_size': float,
        'volume_min': float,
        'volume_max': float,
        'volume_step': float,
        'point': float,
        'spread': int,            # Current spread in points
        'digits': int,
    }
    """

def get_latest_bars(self, symbol: str, timeframe: int,
                    count: int) -> pd.DataFrame:
    """Fetch last N bars. Used for warmup and per-step updates.
    Returns DataFrame with: time, open, high, low, close, volume
    """

def get_current_spread(self, symbol: str = "XAUUSD") -> float:
    """Current spread in price points (not pips)."""

def get_tick_value(self, symbol: str = "XAUUSD") -> float:
    """Current tick_value in account currency. Called per-trade for accuracy."""

def get_open_positions(self, symbol: str = "XAUUSD") -> list:
    """Return list of open positions for recovery on restart."""

def send_market_order(self, symbol: str, side: str, lots: float,
                      sl: float, tp: float, comment: str) -> dict:
    """Place a market order. Returns fill info or error."""

def modify_position(self, ticket: int, sl: float = None,
                    tp: float = None) -> bool:
    """Modify SL/TP of existing position."""

def close_position(self, ticket: int) -> dict:
    """Close a specific position. Returns fill info."""

def close_all_positions(self, symbol: str = "XAUUSD") -> list:
    """Emergency: close all positions on symbol."""

def is_market_open(self) -> bool:
    """Check if market is currently tradeable."""

def disconnect(self):
    """Clean shutdown of MT5 connection."""
```

### Account Currency Auto-Detection

```python
def _detect_account_setup(self):
    """Auto-detect everything from MT5 — zero manual config needed."""
    acc = mt5.account_info()
    sym = mt5.symbol_info("XAUUSD")

    self.account_currency = acc.currency       # "USD", "GBP", "EUR", etc.
    self.tick_value = sym.trade_tick_value      # Already in account currency
    self.tick_size = sym.trade_tick_size        # 0.01
    self.contract_size = sym.trade_contract_size  # 100
    self.value_per_point = self.tick_value / self.tick_size

    # Validate
    assert self.tick_size > 0, "Invalid tick_size"
    assert self.tick_value > 0, "Invalid tick_value"

    log.info(f"Account: {self.account_currency}, "
             f"tick_value={self.tick_value}, "
             f"value_per_point={self.value_per_point}")
```

### Correlated Symbol Mapping

Different brokers use different names for the same instruments:

```python
# Map our standard names → broker-specific names
SYMBOL_MAP_DEFAULT = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD",
    "XAGUSD": "XAGUSD",
    "USDJPY": "USDJPY",
    "US500":  "US500",      # May be: SPX500, USA500, SP500m
    "USOIL":  "USOIL",      # May be: WTI, CL-OIL, USOUSD
}

def _validate_symbols(self):
    """Check all required symbols exist on broker. Suggest alternatives."""
    for our_name, broker_name in self.symbol_map.items():
        info = mt5.symbol_info(broker_name)
        if info is None:
            # Try common alternatives
            alternatives = BROKER_ALTERNATIVES.get(our_name, [])
            found = False
            for alt in alternatives:
                if mt5.symbol_info(alt) is not None:
                    self.symbol_map[our_name] = alt
                    found = True
                    break
            if not found:
                log.warning(f"{our_name}: not found. "
                           f"Features will default to 0.")
```

### Reconnection Logic

```python
def _heartbeat_loop(self):
    """Background thread: check MT5 connection every 5 seconds."""
    while self._running:
        if not mt5.terminal_info():
            self._connected = False
            self._reconnect_attempts += 1
            log.warning(f"MT5 disconnected. Attempt {self._reconnect_attempts}")
            if self._reconnect_attempts > 6:  # 30 seconds
                self._trigger_emergency_stop()
            else:
                mt5.initialize()
        else:
            self._connected = True
            self._reconnect_attempts = 0
        time.sleep(5)
```

---

## 4. Phase 2: Live Feature Pipeline

### File: `core/feature_pipeline.py`

### Class: `LiveFeaturePipeline`

This is the most critical component. It must produce **identical** 670-dim observations as training.

```python
class LiveFeaturePipeline:
    """Computes all 67 features in real-time from MT5 data.

    Maintains rolling buffers for:
    - 200+ M5 bars (for indicators and normalization)
    - H1/H4/D1 bars (for multi-timeframe features)
    - Correlated instrument M5 bars (for cross-instrument features)
    - 10-frame observation buffer (for frame stacking)
    """
```

### Warmup Procedure

On startup, the pipeline must accumulate enough history before trading:

```python
def warmup(self, mt5_bridge: MT5Bridge) -> bool:
    """Load historical data to initialize all indicators.

    Required history:
    - M5: 400+ bars (~33 hours) for EMA200 + 200-bar normalization
    - H1: 250+ bars (~10 days) for H1 RSI, H1 trend
    - H4: 100+ bars (~17 days) for H4 trend, HTF momentum
    - D1: 250+ bars (~1 year) for D1 trend, fracDiff warmup
    - Correlated M5: 200+ bars per instrument

    After warmup:
    - All indicators produce valid values
    - Normalizer has 200+ samples per feature
    - Frame buffer has 10 valid frames
    - Ready to trade
    """
    # 1. Fetch M5 bars (at least 500 to be safe)
    m5 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_M5, 500)

    # 2. Fetch HTF bars
    h1 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_H1, 300)
    h4 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_H4, 120)
    d1 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_D1, 300)

    # 3. Fetch correlated instruments
    correlated = {}
    for symbol in ["EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"]:
        broker_sym = mt5_bridge.symbol_map.get(symbol, symbol)
        bars = mt5_bridge.get_latest_bars(broker_sym, mt5.TIMEFRAME_M5, 300)
        if bars is not None and len(bars) > 50:
            correlated[symbol] = bars

    # 4. Build full feature set using standalone FeatureBuilder
    features_df = self.feature_builder.build_features(
        m5, h1, h4, d1, correlated_m5=correlated
    )

    # 5. Initialize normalizer with historical data
    self.normalizer.fit(features_df)

    # 6. Fill frame buffer with last 10 frames
    for i in range(max(0, len(features_df) - 10), len(features_df)):
        frame = self._build_frame(features_df.iloc[i])
        self.frame_buffer.append(frame)

    self._warmed_up = True
    return True
```

### Per-Bar Update

Every 5 minutes when a new M5 bar closes:

```python
def on_new_bar(self, mt5_bridge: MT5Bridge,
               account_state: dict, memory_features: np.ndarray
               ) -> np.ndarray:
    """Process a new M5 bar and return 670-dim observation.

    Steps:
    1. Fetch latest M5 bar + update rolling buffer
    2. Fetch/update HTF bars if new bar closed (H1 every 12 M5 bars, etc.)
    3. Fetch correlated instrument latest bars
    4. Recompute all 54 precomputed features from rolling buffers
    5. Normalize market features (rolling z-score)
    6. Compute 8 account features from account_state
    7. Get 5 memory features from memory system
    8. Build 67-dim frame → append to frame buffer
    9. Return 670-dim observation (10 stacked frames)
    """
```

### Feature Groups — Implementation

Each feature group has its own module in `features/`:

| Module | Features | Input | Notes |
|--------|----------|-------|-------|
| `technical.py` | A(7) + B(4) + C(6) + D(2) = 19 | M5 OHLCV buffer | Uses `ta` library |
| `time_session.py` | F(4) | Current timestamp | Pure time math |
| `correlation.py` | Upgrade 1 (11) | 5 correlated M5 buffers | merge_asof backward; neutral fills (see below) |
| `calendar.py` | Upgrade 2 (6) | MQL5 calendar bridge / static events / CSV | Timezone-aware |
| `spread_liquidity.py` | Upgrade 3 (2) | ATR + volume + hour | Session-based |
| `regime.py` | Upgrade 4 (2) | Gold + EURUSD + US500 returns | Rolling correlation |
| `session_micro.py` | Upgrade 5 (4) | M5 OHLCV + ATR + hour | Asian range logic |
| `account_features.py` | G(8) | Position state + balance | Live from MT5 |
| `memory_features.py` | H(5) | SQLite memory DB | Cached 50 steps |

### Missing/Stale Feed Handling (CRITICAL — Must Match Training)

When a correlated instrument feed is unavailable or stale, features must be filled with **semantically neutral values**, not zero. Zero creates false signals (e.g., RSI=0 looks extremely oversold).

```python
# Neutral fill values — matches training's correlation_features.py
NEUTRAL_FILLS = {
    # RSI features: 0.5 = neutral (not overbought or oversold)
    "eurusd_rsi_14": 0.5,
    "xagusd_rsi_14": 0.5,
    "us500_rsi_14": 0.5,
    # Returns/trend/ratio: 0.0 = no change (neutral)
    "eurusd_returns_20": 0.0, "eurusd_trend": 0.0,
    "xagusd_returns_20": 0.0,
    "usdjpy_returns_20": 0.0, "usdjpy_trend": 0.0,
    "us500_returns_20": 0.0,
    "usoil_returns_20": 0.0,
    "gold_silver_ratio_z": 0.0,
}
```

### gold_silver_ratio_z — Raw Ratio (No Internal Z-Score)

The `gold_silver_ratio_z` feature is computed as the **raw ratio** `close_xau / close_xag`. It is NOT internally z-scored — the rolling normalizer in the pipeline handles z-scoring once. This prevents double z-scoring that would distort the signal.

### Regime Features — Forward Fill Limit

Regime correlation features (`corr_gold_usd_100`, `corr_gold_spx_100`) use `ffill(limit=60)` — stale data older than 60 bars (~5 hours) is NOT propagated. After 60 bars of missing data, the value becomes NaN and is handled by the normalizer.

**Multi-timeframe features (Group E, 6 features):**

```python
# Option A: Fetch H1/H4/D1 directly from MT5 (simplest)
h1 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_H1, 30)
h4 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_H4, 15)
d1 = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_D1, 10)

# Option B: Aggregate from M5 (if broker limits API calls)
# Use timeframe_aggregator.py to build H1/H4/D1 from M5 buffer
```

### Rolling Normalizer

```python
class LiveNormalizer:
    """Maintains per-feature rolling buffers for z-score normalization.

    Matches the training system's ExpandingWindowNormalizer exactly:
    - Window: 200 bars
    - Min periods: 50
    - Clip: [-5, +5]
    """

    def __init__(self, market_features: list, window: int = 200,
                 clip: float = 5.0):
        self.window = window
        self.clip = clip
        self.buffers = {f: deque(maxlen=window) for f in market_features}

    def normalize(self, feature_name: str, raw_value: float) -> float:
        """Add value to buffer and return z-score normalized value."""
        self.buffers[feature_name].append(raw_value)
        buf = self.buffers[feature_name]
        if len(buf) < 50:
            return 0.0  # Not enough data yet
        mean = np.mean(buf)
        std = np.std(buf)
        z = (raw_value - mean) / (std + 1e-8)
        return float(np.clip(z, -self.clip, self.clip))

    def get_state(self) -> dict:
        """Export buffer state for persistence."""
        return {f: list(buf) for f, buf in self.buffers.items()}

    def set_state(self, state: dict):
        """Restore buffer state from persistence."""
        for f, values in state.items():
            if f in self.buffers:
                self.buffers[f] = deque(values, maxlen=self.window)
```

### Live Feature Health Monitoring

The training system includes `_validate_features()` (in feature_builder.py) that checks for NaN, inf, and constant features. The live dashboard must run the same checks every bar:

```python
def _check_feature_health(self, frame: np.ndarray) -> dict:
    """Per-bar feature health check. Runs every 5 minutes.

    Returns: {
        'nan_count': int,       # Number of NaN values in 67-dim frame
        'inf_count': int,       # Number of +/-inf values
        'constant_count': int,  # Features with zero variance over last 50 bars
        'stale_feeds': list,    # Correlated instruments not updated in >60 bars
        'healthy': bool,        # True if all checks pass
    }
    """
    # If nan_count > 0 or inf_count > 0 → replace with 0.0, log WARNING
    # If constant_count > 5 → log WARNING (possible data feed issue)
    # If stale_feeds → log which instruments, use neutral fills
```

### Duplicate Bar Guard

Prevents processing the same M5 bar twice on network glitches or MT5 API retries:

```python
def on_new_bar(self, mt5_bridge, account_state, memory_features):
    latest_bar = mt5_bridge.get_latest_bars("XAUUSD", mt5.TIMEFRAME_M5, 1)
    bar_time = latest_bar['time'].iloc[0]

    if bar_time <= self._last_bar_time:
        log.debug(f"Skipping duplicate bar: {bar_time}")
        return None  # Caller checks for None and skips cycle

    self._last_bar_time = bar_time
    # ... proceed with feature computation
```

### Normalizer State Persistence

The rolling normalizer maintains 200-bar buffers for 38 features. If the system restarts, these buffers would be lost, requiring a full warmup. Persistence avoids this:

```python
class LiveNormalizer:
    # ... existing methods ...

    def save_state(self, path: str):
        """Save all 38 feature buffers + frame buffer to JSON.
        Called on: clean shutdown, and every hour as backup.
        """
        state = {name: list(buf) for name, buf in self.buffers.items()}
        state["_frame_buffer"] = [frame.tolist() for frame in self.frame_buffer]
        state["_saved_at"] = time.time()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f)

    def load_state(self, path: str) -> bool:
        """Restore buffers from disk. Returns True if loaded successfully.
        Only restores if file is < 1 hour old (3600s). Otherwise, do full warmup.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
            saved_at = state.get("_saved_at", 0)
            if time.time() - saved_at > 3600:
                log.info("Normalizer state too old (>1h), will do full warmup")
                return False
            for name, values in state.items():
                if name.startswith("_"):
                    continue
                if name in self.buffers:
                    self.buffers[name] = deque(values, maxlen=self.window)
            if "_frame_buffer" in state:
                for frame_list in state["_frame_buffer"]:
                    self.frame_buffer.append(np.array(frame_list, dtype=np.float32))
            log.info(f"Normalizer state restored ({len(self.buffers)} features)")
            return True
        except Exception as e:
            log.warning(f"Failed to load normalizer state: {e}")
            return False
```

**Save path:** `storage/state/normalizer_state.json`
**Save triggers:** Clean shutdown (guaranteed), hourly backup (covers crashes)
**Restore logic:** On startup, attempt restore. If file missing or >1 hour old → full warmup from MT5 history.

### Live Spread Feature (spread_estimate_norm)

The `spread_estimate_norm` feature (Upgrade 3) differs between training and live:

- **Training:** `session_spread_estimate / ATR(14)` — uses hardcoded session-based spread values
- **Live:** `actual_broker_spread / ATR(14)` — uses real-time spread from MT5 API

```python
# In features/spread_liquidity.py
def calc_spread_feature_live(mt5_bridge, atr_14):
    """Use actual broker spread instead of session estimate.

    This is semantically equivalent to training (both measure "how expensive
    is it to trade right now relative to volatility") but live uses the real
    value. Training's domain randomization (±30% spread jitter) already
    prepared the model for spread variation.
    """
    spread_price = mt5_bridge.get_current_spread("XAUUSD")  # In price points
    return min(spread_price / (atr_14 + 1e-8), 5.0)  # Same clip as training
```

### Feature Builder Strategy: Standalone

The live dashboard is a **self-contained, deployable package** — it runs on any machine with MT5
and Python, independent of the training repository. All feature computation code lives inside
the `live_dashboard/features/` modules.

- All 54 precomputed feature calculations are implemented directly in `features/*.py`
- `test_parity.py` validates that live feature outputs match training outputs on identical data
- The dashboard ships via GitHub — anyone can clone it, drop in a model ZIP, and run
- Zero dependency on the training codebase at runtime

**Parity safeguard:** The `test_parity.py` test loads a known feature cache from the model
package's training config and verifies that the standalone feature modules produce identical
output (tolerance: 1e-6). This catches any drift if features are updated in future training
rounds.

### Feature Parity Checklist

All 67 features must match training exactly:

```
PRECOMPUTED (54):
  [x] A: close_frac_diff, returns_1/5/20bar, bar_range, close_position, body_ratio
  [x] B: atr_14_norm, atr_ratio, bb_width, bb_position
  [x] C: rsi_14, macd_signal, adx_14, ema_cross, price_vs_ema200, stoch_k
  [x] D: volume_ratio, obv_slope
  [x] E: h1_trend_dir, h4_trend_dir, d1_trend_dir, h1_rsi, mtf_alignment, htf_momentum
  [x] F: hour_sin, hour_cos, day_of_week, session_quality
  [x] U1: eurusd_returns_20, eurusd_rsi_14, eurusd_trend, xagusd_returns_20,
       xagusd_rsi_14, usdjpy_returns_20, usdjpy_trend, us500_returns_20,
       us500_rsi_14, usoil_returns_20, gold_silver_ratio_z
  [x] U2: hours_to_next_high_impact, hours_to_next_nfp_fomc, in_event_window,
       daily_event_density, london_fix_proximity, comex_session_active
  [x] U3: spread_estimate_norm, volume_spike
  [x] U4: corr_gold_usd_100, corr_gold_spx_100
  [x] U5: asian_range_norm, asian_range_position, session_momentum, london_ny_overlap

LIVE (13):
  [x] G: has_position, position_side, unrealized_pnl, position_duration,
       current_drawdown, equity_ratio, sl_distance_ratio, profit_locked_pct
  [x] H: recent_win_rate, similar_pattern_winrate, trend_prediction_accuracy,
       tp_hit_rate, avg_sl_trail_profit
```

### Economic Calendar Data Source (Upgrade 2 Features)

The MT5 Python API does **NOT** include calendar functions — those only exist in MQL5 (the
native scripting language). The live dashboard uses a 3-tier fallback for calendar events:

**Tier 1: MQL5 Calendar Bridge (Primary — real-time)**

A small MQL5 service script (`mt5_scripts/CalendarBridge.mq5`) runs inside the MT5 terminal
and exports upcoming economic events to a shared JSON file every 15 minutes:

```
mt5_scripts/CalendarBridge.mq5  →  storage/state/calendar_events.json
```

The script uses MQL5's native `CalendarValueHistory()` and `CalendarEventByCountry()` to
pull real-time event data including importance level, forecast, previous, and actual values.

Installation: User compiles `CalendarBridge.mq5` in MT5's MetaEditor and attaches it as a
service. The JSON output path is configurable.

```json
// storage/state/calendar_events.json (auto-generated by MQL5 service)
{
  "updated_at": "2026-03-01T14:00:00Z",
  "events": [
    {"time": "2026-03-07T13:30:00Z", "name": "Non-Farm Payrolls", "currency": "USD",
     "importance": "HIGH", "forecast": "180K", "previous": "175K"},
    {"time": "2026-03-19T18:00:00Z", "name": "FOMC Rate Decision", "currency": "USD",
     "importance": "HIGH", "forecast": "5.25%", "previous": "5.25%"}
  ]
}
```

**Tier 2: Static Known Events (Built-in fallback)**

Ships with the dashboard as `data/calendar/known_events.json` — contains annually-published
dates for all recurring high-impact events:

- **NFP:** First Friday of every month, 13:30 UTC
- **FOMC:** 8 scheduled dates per year (published annually by the Fed)
- **CPI:** Monthly, ~13:30 UTC
- **London Fix:** Daily 10:30 UTC and 15:00 UTC (deterministic)
- **COMEX sessions:** Regular hours (deterministic)

These events are sufficient for the 6 calendar features because the model primarily needs
*proximity to events* and *whether an event window is active*, not the actual data values.
The static calendar is updated once per year when the Fed and BLS publish their schedules.

**Tier 3: User-Supplied CSV (Optional override)**

If the user provides `data/calendar/economic_calendar.csv` (e.g., from ForexFactory export),
it takes priority over the static events. Same format as training.

**Priority:** MQL5 Bridge > User CSV > Static Known Events > Neutral defaults

**Safe degradation:** If no calendar data is available (all 3 tiers fail), all 6 calendar
features default to neutral values (matching training's fallback behavior). The model was
trained to handle neutral calendar features gracefully — it simply ignores event proximity
when the data is absent.

### Normalization Split

**Z-Score normalized (38 features):** Groups A-E (25) + Upgrade 1 (11) + Upgrade 4 (2)
**Exempt / pass-through (29 features):** Group F (4) + Groups G-H (13) + Upgrades 2 (6), 3 (2), 5 (4)

This matches `config.market_feature_names` (38) and `config.norm_exempt_features` (29).

**Note:** `gold_silver_ratio_z` is in the Z-score group — the normalizer applies the z-score, not the feature computation itself. The feature computes a raw ratio (`close_xau / close_xag`) and the rolling normalizer handles standardization. This prevents double z-scoring.

---

## 5. Phase 3: Model Inference Engine

### File: `core/inference_engine.py`

### Class: `InferenceEngine`

```python
class InferenceEngine:
    """Loads trained SAC model and runs inference.

    Takes 670-dim observation → produces 4 action values.
    Uses deterministic policy (no exploration noise).
    """

    def __init__(self, model_path: str):
        self.model = SAC.load(model_path)

    def predict(self, observation: np.ndarray) -> dict:
        """Run model inference on observation.

        Args:
            observation: 670-dim numpy array (10 stacked frames)

        Returns: {
            'direction': float,    # [-1, 1]: negative=short, positive=long
            'conviction': float,   # [0, 1]: position sizing confidence
            'exit_urgency': float, # [0, 1]: close signal strength
            'sl_adjustment': float # [0, 1]: trailing SL tightness
        }
        """
        action, _ = self.model.predict(observation, deterministic=True)

        return {
            'direction': float(action[0]),
            'conviction': float((action[1] + 1) / 2),
            'exit_urgency': float((action[2] + 1) / 2),
            'sl_adjustment': float((action[3] + 1) / 2),
        }
```

---

## 6. Phase 4: Trade Executor

### File: `core/trade_executor.py`

### Class: `TradeExecutor`

Translates the AI's 4 action values into MT5 orders.

```python
class TradeExecutor:
    """Converts AI actions into MT5 trade operations.

    Decision flow per bar:
    1. Get action from InferenceEngine
    2. If no position:
       a. direction > 0.3 → check if LONG allowed → open LONG
       b. direction < -0.3 → check if SHORT allowed → open SHORT
       c. else → do nothing (flat)
    3. If in position:
       a. exit_urgency > 0.5 and held >= min_hold_bars → close
       b. else → adjust trailing SL based on sl_adjustment
    """

    def execute_action(self, action: dict, current_bar: dict,
                       account: dict) -> str:
        """Main decision loop — called every M5 bar.

        Returns: action_taken (str) for logging
        """
        if not self.position:
            return self._handle_no_position(action, current_bar, account)
        else:
            return self._handle_in_position(action, current_bar, account)

    def _handle_no_position(self, action, bar, account):
        # 1. Check direction threshold
        direction = action['direction']
        conviction = action['conviction']

        if abs(direction) < self.cfg.direction_threshold:
            return "HOLD_FLAT"

        side = "LONG" if direction > 0 else "SHORT"

        # 2. Check risk manager allows it
        allowed, reason = self.risk_mgr.check_position_allowed(
            balance=account['balance'],
            peak_balance=self.peak_balance,
            daily_trade_count=self.daily_trades,
            conviction=conviction,
        )
        if not allowed:
            return f"BLOCKED_{reason}"

        # 3. Calculate lot size using LIVE tick_value from MT5
        tick_value = self.mt5_bridge.get_tick_value()
        symbol_info = self.mt5_bridge.get_symbol_info()
        lots = self.risk_mgr.calculate_lot_size(
            conviction, account['balance'], self.peak_balance,
            self.current_atr, symbol_info
        )
        if lots <= 0:
            return "LOTS_ZERO"

        # 4. Calculate SL/TP
        entry_price = bar['close']  # Approximate — real fill may differ
        sl = self.risk_mgr.calculate_sl(side, entry_price,
                                         self.current_atr, conviction)
        tp = self.risk_mgr.calculate_tp(side, entry_price,
                                         self.current_atr, conviction)

        # 5. Send order to MT5
        result = self.mt5_bridge.send_market_order(
            symbol="XAUUSD", side=side, lots=lots,
            sl=sl, tp=tp, comment=f"Spartus_{side}_{conviction:.2f}"
        )

        if result['success']:
            self.position = {
                'ticket': result['ticket'],
                'side': side,
                'entry_price': result['fill_price'],
                'lots': lots,
                'stop_loss': sl,
                'take_profit': tp,
                'conviction': conviction,
                'entry_step': self.step_count,
                'entry_time': datetime.utcnow(),
            }
            self.daily_trades += 1
            return f"OPEN_{side}"

        return f"ORDER_FAILED_{result.get('error', 'unknown')}"

    def _handle_in_position(self, action, bar, account):
        # 1. Check exit signal
        if (action['exit_urgency'] > self.cfg.exit_threshold
                and self._bars_held() >= self.cfg.min_hold_bars):
            result = self.mt5_bridge.close_position(self.position['ticket'])
            if result['success']:
                self._record_trade_close(result)
                return "CLOSE_AGENT"

        # 2. Adjust trailing SL
        new_sl = self.risk_mgr.adjust_stop_loss(
            self.position['stop_loss'],
            self.position['side'],
            bar['close'],
            self.current_atr,
            action['sl_adjustment'],
        )
        if new_sl != self.position['stop_loss']:
            self.mt5_bridge.modify_position(
                self.position['ticket'], sl=new_sl
            )
            self.position['stop_loss'] = new_sl
            return f"TRAIL_SL_{new_sl:.2f}"

        return "HOLD"
```

### SL/TP Handling — Server vs Client Side

**In training:** SL/TP is checked client-side using bar high/low.
**In live:** SL/TP is set as a **server-side order** on MT5. The broker monitors it.

```
Training: Agent checks if bar.low <= SL → close at SL price
Live:     MT5 broker monitors SL → auto-closes when price hits SL
          Agent only needs to: (a) set initial SL/TP, (b) trail SL
```

This is **safer** for live — the broker executes SL even if our app crashes.

---

## 7. Phase 5: Risk & Safety Layer

### File: `core/risk_manager.py` (Live version)

Standalone risk manager with all live-specific safety features:

```python
class LiveRiskManager:
    """Live risk management with circuit breakers and emergency stops.

    Includes: lot sizing, SL/TP calculation, trailing SL (ported from training logic)
    Adds: circuit breakers, daily DD halt, consecutive loss tracking
    """
```

### Trading State Machine (CRITICAL — No Auto-Start)

The dashboard **NEVER** starts trading automatically. The user must explicitly click
"Start Trading" after reviewing the startup checks. Three trading states:

```python
class TradingState(Enum):
    STOPPED = "stopped"       # Default on launch — AI is completely idle
    RUNNING = "running"       # AI actively monitoring and trading
    WINDING_DOWN = "winding_down"  # No new trades, managing open position until close
```

**State transitions:**

```
                  ┌──────────┐
    Launch ──────▶│ STOPPED  │◀──── position closed (from WINDING_DOWN)
                  └────┬─────┘
                       │ User clicks [Start Trading]
                       ▼
                  ┌──────────┐
                  │ RUNNING  │◀──── CB pause expires / manual resume
                  └──┬───┬───┘
                     │   │ User clicks [Wind Down]
                     │   ▼
                     │ ┌──────────────┐
                     │ │ WINDING_DOWN │──── position closes naturally ──▶ STOPPED
                     │ └──────────────┘
                     │
                     │ User clicks [Stop Now] or EMERGENCY STOP
                     ▼
                  STOPPED  (closes all positions immediately)
```

**Dashboard controls (Tab 1 header bar + Tab 5 controls):**

| Button | Action | When Visible |
|--------|--------|-------------|
| **Start Trading** | Transition STOPPED → RUNNING | When STOPPED and all startup checks pass |
| **Wind Down** | Transition RUNNING → WINDING_DOWN (no new trades, AI continues managing open position until it closes, then auto-transitions to STOPPED) | When RUNNING |
| **Stop Now** | Close all positions immediately, transition → STOPPED | When RUNNING or WINDING_DOWN |
| **Emergency Stop** | Same as Stop Now + block all orders + require manual reset | Always visible |

**Why no "Pause" button:** A generic pause is dangerous — if the AI is paused with an open
position, the position sits unmanaged (no SL trailing, no exit decisions). Instead:
- **Wind Down** = safe pause (AI finishes managing the current trade, then stops)
- **Stop Now** = immediate halt (closes everything, clean slate)
- **Circuit breaker pauses** = automatic temporary pauses that resume on their own

**WINDING_DOWN behavior:**
- The AI continues to run inference every bar (SL adjustments, exit decisions)
- The AI will NOT open new positions (direction/conviction actions are ignored)
- When the open position closes (TP, SL, or agent exit), state transitions to STOPPED
- If no position is open when Wind Down is clicked, immediately transitions to STOPPED

### Circuit Breakers

```python
class CircuitBreaker:
    """Pause trading after consecutive losses or daily DD.

    Rules:
    - 3 consecutive losses → pause 30 minutes
    - 5 consecutive losses → pause 2 hours
    - Daily DD > 2% → halt for rest of day (matches training done=True)
    - Daily DD > 3% → close all + halt for rest of day
    - Weekly DD > 5% → halt until manual reset

    IMPORTANT: Daily DD halt must END the trading day entirely.
    In training, exceeding daily DD triggers done=True (episode ends).
    Live must match: close all positions and block new entries until
    the next trading day. Do NOT just log a warning and continue.

    DAILY RESET TIMING:
    - Daily DD tracking resets at 00:00 UTC (Monday-Friday)
    - Daily trade count resets at 00:00 UTC
    - Weekend: Friday 20:00 UTC → Monday 00:30 UTC (no trading)
    - The "trading day" runs 00:00 UTC to 23:59 UTC
    - Circuit breaker pauses DO reset at 00:00 UTC (fresh day = fresh start)
    - Weekly DD tracks from Monday 00:00 UTC → Friday 20:00 UTC

    NOTE: Circuit breaker pauses only apply during RUNNING state.
    They temporarily block new entries but the AI continues managing
    open positions (SL trailing, exit decisions). When the pause
    expires, RUNNING state resumes automatically.
    """

    def should_trade(self) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
```

### Weekend Manager

```python
class WeekendManager:
    """Handle Friday close and weekend gap risk.

    Rules:
    - Friday 20:00 UTC: Close all positions (no weekend holding)
    - Friday 19:00 UTC: Block new entries (1 hour buffer)
    - Sunday open: Wait 30 minutes for spreads to normalize
    - Monday 00:30 UTC: Resume trading
    """
```

### Emergency Stop

```python
class EmergencyStop:
    """One-click kill switch accessible from dashboard.

    Actions:
    1. Close all open positions immediately
    2. Cancel all pending orders
    3. Block all new orders
    4. Log emergency event
    5. Send alert (email/webhook if configured)

    Requires manual reset to resume trading.
    """
```

---

## 8. Phase 6: Memory System (Live)

### Standalone Memory System

The memory system (SQLite: trades, patterns, predictions, tp_tracking, journal) uses the **same schema and logic** as the training environment but is fully standalone — no imports from the training codebase:

- `memory/trading_memory.py` — standalone module (same schema as training's `src/memory/trading_memory.py`)
- `memory/trend_tracker.py` — standalone module (same logic as training's `src/memory/trend_tracker.py`)
- `memory/trade_analyzer.py` — standalone module (same logic as training's `src/memory/trade_analyzer.py`)

All three modules are self-contained within the live dashboard repository. They implement the same SQLite tables, queries, and memory feature calculations to ensure the model receives identical memory features (5 dims) as during training.

### Fresh Start Handling

A new live account starts with **zero trade history**. Memory features default to neutral (0.5):

```python
def get_memory_features_safe(self, market_state, step) -> np.ndarray:
    """Return memory features with safe defaults for fresh accounts.

    First 50 trades: Use 0.5 defaults (neutral).
    After 50 trades: Use actual computed features.
    Gradual blend between trades 20-50.
    """
    trade_count = self.get_trade_count()
    if trade_count < 20:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    real_features = self.get_memory_features(market_state, step)
    if trade_count >= 50:
        return real_features

    # Blend: weight increases from 0 to 1 over trades 20-50
    blend = (trade_count - 20) / 30.0
    defaults = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    return defaults * (1 - blend) + real_features * blend
```

---

## 9. Phase 7: Dashboard UI

### Framework: PyQt6

Matches the training dashboard for consistency. Dark theme, bright signal colors, GPU-accelerated charts via pyqtgraph.

### Tab Layout — 6 Tabs

#### Tab 1: LIVE STATUS

```
┌──────────────────────────────────────────────────────────────────┐
│  SPARTUS LIVE TRADING — XAUUSD       ⬤ STOPPED                  │
│  [▶ Start Trading]  [■ Stop Now]  [⏏ Wind Down]  [⚠ EMERGENCY] │
├──────────────────────┬───────────────────────────────────────────┤
│  MT5 CONNECTION       │  ACCOUNT                                │
│  ● Connected          │  Currency: USD                          │
│  Server: Vantage-Live │  Balance:  $1,234.56                    │
│  Latency: 45ms        │  Equity:   $1,245.89                    │
│  Spread: 2.1 pips     │  Margin:   $125.00                     │
│                        │  Free:     $1,120.89                    │
├────────────────────────┼─────────────────────────────────────────┤
│  OPEN POSITION         │  TODAY'S SUMMARY                       │
│  LONG 0.02 lots        │  Trades: 4  (3W / 1L)                 │
│  Entry: $2,651.30      │  P/L: +$12.45                         │
│  Current: $2,653.80    │  Win Rate: 75%                         │
│  P/L: +$5.00           │  Max DD: 1.2%                         │
│  SL: $2,648.50 (▲)     │  Profit Factor: 2.31                  │
│  TP: $2,658.00         │                                        │
│  Duration: 45 min      │                                        │
├────────────────────────┴─────────────────────────────────────────┤
│  AI DECISION LOG (last 8 actions)                               │
│  14:35 TRAIL_SL → $2,649.80 (conviction 0.72)                  │
│  14:30 HOLD (exit_urgency: 0.23)                                │
│  14:25 OPEN_LONG 0.02 lots @ $2,651.30 (conv: 0.72, dir: 0.65)│
│  14:20 HOLD_FLAT (direction: 0.12 < 0.30)                      │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘
```

**Trading state indicator** (top bar, color-coded):
- **⬤ STOPPED** (gray) — Default on launch. AI idle. "Start Trading" enabled.
- **⬤ RUNNING** (green) — AI actively trading. "Wind Down" and "Stop Now" enabled.
- **⬤ WINDING DOWN** (yellow) — Managing open position, no new trades. "Stop Now" enabled.
- **⬤ CB PAUSED** (orange) — Circuit breaker active, will auto-resume. "Stop Now" enabled.

**Button visibility** follows the state machine defined in Phase 5.

#### Tab 2: PERFORMANCE

```
┌──────────────────────────────────────────────────────────────────┐
│  BALANCE CHART (last 500 bars, pyqtgraph)                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │     ╱─╲    ╱───╲                                        │     │
│  │  __╱   ╲__╱     ╲___╱─╲_____╱──────                    │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
├──────────────────────┬───────────────────────────────────────────┤
│  ROLLING METRICS      │  TRADE HISTORY                          │
│  Sharpe (30d): 0.85   │  #  Time   Side  Lots  P/L  Reason     │
│  Win Rate: 54%         │  47 14:25  LONG  0.02  +5.0 TP_HIT    │
│  Profit Factor: 1.62   │  46 13:10  SHORT 0.01  -2.3 SL_HIT   │
│  Avg Trade: +$1.23     │  45 12:40  LONG  0.02  +3.1 AGENT     │
│  Max DD: 4.2%          │  44 11:55  SHORT 0.01  +1.8 TP_HIT   │
│  Total Trades: 47      │  ...                                   │
│  Best: +$8.50          │                                        │
│  Worst: -$4.20         │                                        │
└──────────────────────┴───────────────────────────────────────────┘
```

#### Tab 3: TRADE JOURNAL

```
┌──────────────────────────────────────────────────────────────────┐
│  LESSON SUMMARY          │  TRADE DETAIL                        │
│  GOOD_TRADE: 18          │  Trade #47 — GOOD_TRADE              │
│  WRONG_DIRECTION: 8      │  Entry: LONG @ $2,651.30             │
│  CORRECT_DIR_EARLY: 6    │  Exit: TP_HIT @ $2,658.00            │
│  BAD_TIMING: 4           │  P/L: +$5.00 (0.40%)                │
│  WHIPSAW: 3              │  Hold: 45 min (9 bars)               │
│  HELD_TOO_LONG: 2        │                                      │
│  ...                     │  Entry: RSI=42, trend=+0.8,          │
│                           │         session=London, vol=0.9      │
│                           │  SL Quality: TRAILED (locked +$2.10)│
│                           │  Pattern: rsi=4/trend=5/london/vol=5│
│                           │  Pattern W/R: 62% (13W / 8L)        │
└──────────────────────────┴───────────────────────────────────────┘
```

#### Tab 4: MODEL & FEATURES

```
┌──────────────────────────────────────────────────────────────────┐
│  MODEL INFO               │  FEATURE HEALTH                     │
│  Model: W170 Champion     │  ● 67/67 features active            │
│  Trained: Week 170        │  ● Normalizer: 200/200 bars         │
│  Val Sharpe: 4.091        │  ● Frame buffer: 10/10 frames       │
│  Architecture: [256,256]  │  ● NaN rate: 0.0%                   │
│  Features: 67 (670 obs)   │  ● Inf values: 0                    │
│  Actions: 4 continuous    │  ● Constant feats: 0                │
│                           │                                      │
│  OBSERVATION SAMPLE       │  CORRELATED FEEDS                   │
│  direction: 0.65          │  ● EURUSD: OK (2.1s ago)            │
│  conviction: 0.72         │  ● XAGUSD: OK (2.1s ago)            │
│  exit: 0.23               │  ● USDJPY: OK (2.1s ago)            │
│  sl_adj: 0.41             │  ● US500: OK (4.3s ago)             │
│                           │  ● USOIL: STALE (>60s, neutral fill)│
│  REWARD STATE             │                                      │
│  (Loaded from model pkg)  │  CALENDAR: 2 events today           │
│  Normalizer EMA: OK       │  Next: NFP in 4.2 hours             │
└──────────────────────────┴───────────────────────────────────────┘
```

#### Tab 5: ALERTS & SAFETY

```
┌──────────────────────────────────────────────────────────────────┐
│  SAFETY STATUS             │  ALERT LOG                         │
│  ⬤ RUNNING                 │  14:35 [INFO] Trailing SL to 2649  │
│  ● Circuit Breaker: OFF    │  14:25 [INFO] Opened LONG 0.02     │
│  ● Weekend Close: 5h away  │  13:15 [WARN] Spread spike: 4.2x   │
│  ● Daily DD: 1.2% / 3.0%  │  12:30 [INFO] Closed SHORT +$1.80  │
│  ● Consec. Losses: 1       │  11:55 [WARN] USOIL feed stale     │
│  ● Connection: Stable      │  11:50 [INFO] Opened SHORT 0.01    │
│                            │                                     │
│  CONTROLS                  │  DAILY RISK                         │
│  [▶ Start Trading]         │  Trades: 4 / 10 soft / 20 hard    │
│  [⏏ Wind Down]             │  DD: 1.2% / 3.0% daily halt       │
│  [■ Stop Now]              │  DD: 2.1% / 10% total limit       │
│  [⚠ EMERGENCY STOP]        │  Equity: $1,245.89                 │
│  [Reset Circuit Breaker]   │                                     │
└──────────────────────────┴───────────────────────────────────────┘
```

#### Tab 6: ANALYTICS & DIAGNOSTICS

Mirrors the depth of the training dashboard's AI INTERNALS and METRICS tabs. Provides the data needed to decide if/when to retrain the model.

```
┌──────────────────────────────────────────────────────────────────┐
│  ACTION DISTRIBUTIONS (last 500)  │  TRAINING vs LIVE            │
│  Direction:  ▁▂▃█▅▃▂▁  μ=0.12    │  Metric    Expected  Actual  │
│  Conviction: ▁▁▃█████▅  μ=0.68   │  PF        2.24      1.85    │
│  Exit:       █▅▃▂▁▁▁▁  μ=0.22    │  Win Rate  52%       48%     │
│  SL Adj:     ▂▃▅█▅▃▂▁  μ=0.45   │  MaxDD     20.8%     3.2%    │
│                                    │  T/Day     2.4       1.8     │
│  Flat rate: 62% of bars            │  TIM%      8.9%      6.1%   │
│  Avg trades/day: 1.8               │  [vs stress_matrix_W0170]   │
├────────────────────────────────────┼──────────────────────────────┤
│  SESSION BREAKDOWN                 │  FEATURE DRIFT (54 baselined)│
│  Session   Trades Win%  PF  AvgPL │  Feature        Live   Train │
│  London      12   58%  1.9  +2.1  │  returns_20bar  0.02   0.00  │
│  NY Overlap   8   50%  1.5  +0.8  │  atr_14_norm    1.23   0.95  │
│  Asia          3   33%  0.8  -1.2  │  eurusd_ret20   0.05   0.01  │
│  Off-hours     1    0%  0.0  -3.1  │  ● 52/54 within 2σ          │
│                                    │  ⚠ 2 features drifted        │
│  DAY-OF-WEEK BREAKDOWN             │  (13 live features: no baseline)│
│  Mon: +$4.20  Tue: +$1.80         │                              │
│  Wed: -$2.10  Thu: +$6.50         │  CORRELATION DRIFT (norm.)   │
│  Fri: +$1.20                       │  Score: 0.08  ● OK          │
│                                    │  Persist: 0/3 yel, 0/2 red  │
├────────────────────────────────────┼──────────────────────────────┤
│  FEATURE HEALTH                    │                              │
│  ● Active: 67/67                   │                              │
│  ● Stale feeds: 0                  │                              │
│  ● Dead features: 0                │                              │
├────────────────────────────────────┴──────────────────────────────┤
│  WEEKLY REPORTS (auto-generated every Sunday 23:59 UTC)           │
│  Week 1: PF 1.85, 12 trades, +$14.30, MaxDD 3.2%               │
│  Week 2: PF 2.10, 15 trades, +$22.10, MaxDD 2.8%               │
│  [Export CSV]  [Export Full JSON]                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Action Distributions Panel:**
- Maintain `deque(maxlen=500)` per action dimension (direction, conviction, exit_urgency, sl_adjustment)
- Every bar: append the 4 raw action values (before thresholding)
- Display as histogram (pyqtgraph BarGraphItem, 20 bins per dimension)
- Show: mean, std, min, max per dimension
- Show: flat rate (% of bars where |direction| < 0.3 = no trade signal)
- Show: avg trades/day over rolling 7-day window
- **Red flag:** If flat rate > 90% for 2+ days → model may be refusing to trade
- **Red flag:** If action std < 0.1 for any dimension → policy may have collapsed

**Training vs Live Comparison Panel:**
- Load W170 stress matrix results from `stress_results.json` (bundled in model package)
- Compare to rolling 30-day live metrics
- Columns: Metric | Training (W170 Val) | Training (W170 Test) | Live (30d)
- Rows: PF, Win Rate, MaxDD%, Trades/Day, TIM%, Avg Hold Bars
- Color code: Green if live within 80% of training, Yellow if 60-80%, Red if <60%
- This is the primary decision tool for "do we need to retrain?"

**Session Breakdown Panel:**
- Classify each trade by UTC hour at entry → session:
  - London (07:00-11:59 UTC)
  - NY Overlap (12:00-15:59 UTC)
  - NY PM (16:00-19:59 UTC)
  - Asia (00:00-06:59 UTC)
  - Off-hours (20:00-23:59 UTC)
- Aggregate per session: trade count, win rate, profit factor, avg P/L, net P/L
- Refreshed from SQLite `trades` table
- **Key insight:** If London/NY performance is strong but Asia is losing, the model may need session-specific retraining

**Day-of-Week Breakdown:**
- Net P/L per day of week (Monday-Friday)
- Rolling 4-week average

**Feature Drift Panel (54 precomputed features):**
- Load `feature_baseline.json` from model package (per-feature mean/std from validation data)
- Covers **54 precomputed features** — the 38 z-scored market features + 16 norm-exempt bounded features
- Does **NOT** cover the 13 live features (8 account + 5 memory) — these are inherently per-step and have no meaningful training baseline
- Compare each of the 54 baselined features:
  - Live mean/std (from LiveNormalizer's 200-bar buffers for z-scored; from raw rolling stats for bounded)
  - Training baseline mean/std (from `feature_baseline.json`)
- Flag features where `|live_mean - train_mean| > 2 × train_std`
- Display: **"X/54 baselined features within 2σ"** with list of drifted features
- Below: "(13 live features: account/memory — no baseline, changes per trade)"
- **Key insight:** Significant drift means the market regime has changed since training. If >10 features drift, the model is seeing data it wasn't trained on.

**Correlation Drift Panel (top-20 features):**
- Load `correlation_baseline.json` from model package (20×20 correlation matrix + `frobenius_norm_baseline`)
- Top-20 features selected by variance during packaging (most informative features)
- In live: compute rolling 200-bar correlation matrix for the same 20 features
- **Normalized drift score:** `score = ‖C_live - C_base‖_F / (‖C_base‖_F + ε)` where ε=1e-8
  - Dividing by the baseline norm makes the score dimensionless and stable across different feature sets or baseline magnitudes
  - The baseline Frobenius norm (`frobenius_norm_baseline`) is precomputed during packaging and stored in `correlation_baseline.json`
- **Persistence filter — avoids twitchy alerts from news spikes:**
  - Yellow: score ≥ 0.15 sustained for **3 consecutive windows** (~50 min at 200-bar rolling M5)
  - Red: score ≥ 0.25 sustained for **2 consecutive windows** (~33 min)
  - If the score drops below threshold before the persistence count is met → reset counter, stay green
  - This prevents single news events (NFP, FOMC) from triggering false regime-change alerts
- Display: "Corr Drift: 0.08 (OK)" with color coding per persistence rules above
- **Key insight:** Individual feature means can look normal while correlation structure shifts. This catches "market regime changed" that per-feature drift misses. Example: gold-USD correlation flipping from negative to positive while both features individually look fine.

**Weekly Reports Panel:**
- Auto-generated every Sunday 23:59 UTC, saved to `storage/reports/weekly/`
- Contains: PF, trades, wins, losses, net P/L, max DD, Sharpe, session breakdown, action stats, feature drift count
- Export buttons: CSV (for spreadsheets) and full JSON (for programmatic analysis)
- Scrollable list of all past weekly reports

### Dashboard Update Loop

```python
class LiveDashboard(QMainWindow):
    def __init__(self):
        # ... setup tabs ...
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(1000)  # 1 Hz update for UI

    def _update(self):
        """Pull latest state and refresh all panels."""
        account = self.mt5_bridge.get_account_info()
        position = self.position_manager.get_current_position()
        metrics = self.metrics_tracker.get_current()
        alerts = self.alert_log.get_recent(20)
        # Update all tab widgets...
```

---

## 10. Phase 8: Model Package Loader

### File: `core/model_loader.py`

### Model Distribution Format

A trained model is distributed as a single ZIP containing everything needed:

```
spartus_champion_W170.zip  (9.4 MB)
├── model.zip                    # SB3 SAC model weights
├── config.json                  # Full TrainingConfig as JSON
├── metadata.json                # Training info (week=170, val_sharpe=4.091, etc.)
├── reward_state.json            # RewardNormalizer + DiffSharpe state
├── feature_baseline.json        # Per-feature mean/std for 54 precomputed features (drift detection)
├── correlation_baseline.json    # 20×20 correlation matrix for top features (regime detection)
├── stress_results.json          # Stress matrix results (training-vs-live comparison)
└── README.txt                   # Model info for humans

# The training system packages the champion model via ModelExporter:
#   storage/models/spartus_champion_W170.zip  (primary deployment artifact)
#   storage/models/spartus_secondary_W180.zip (backup)
```

### feature_baseline.json — Feature Drift Detection Baseline

Generated during model packaging from validation set feature distributions:

```json
{
  "close_frac_diff": {"mean": 0.001, "std": 1.02},
  "returns_1bar": {"mean": -0.002, "std": 0.98},
  "returns_5bar": {"mean": -0.008, "std": 1.05},
  "atr_14_norm": {"mean": 0.95, "std": 0.38},
  ...
}
```

- Contains mean, std, min, max, count for all **54 precomputed features** (38 z-scored + 16 norm-exempt)
- Does NOT include the 13 live features (8 account + 5 memory) — these have no training baseline
- Computed from the full validation set (87 weeks, 2022-W45 to 2024-W27, ~325K bars)
- Used by Tab 6 (Analytics) to detect feature distribution drift:
  - If `|live_mean - train_mean| > 2 × train_std` → feature flagged as "drifted"
  - If >10 features drift → market regime may have changed since training

### correlation_baseline.json — Correlation Structure Baseline

Captures the inter-feature correlation structure from validation data. Detects regime shifts that per-feature drift misses (e.g., gold-USD correlation flipping sign while both means look normal):

```json
{
  "features": ["session_momentum", "mtf_alignment", "h1_trend_dir", ...],
  "correlation_matrix": [
    [1.000, 0.123, -0.045, 0.312, ...],
    [0.123, 1.000, 0.067, -0.089, ...],
    ...
  ],
  "frobenius_norm_baseline": 8.028,
  "n_bars": 325000,
  "n_features": 20
}
```

- Contains a 20×20 correlation matrix for the **top-20 most variable features** from validation data
- `frobenius_norm_baseline`: precomputed `‖C_base‖_F` — used as the denominator for normalized drift score
- Top-20 selected by variance (highest variance = most informative for correlation tracking)
- Generated during model packaging alongside feature_baseline.json
- Used by Tab 6 (Analytics) for **normalized** correlation drift detection:
  - Compute rolling 200-bar correlation matrix for same 20 features in live
  - Normalized score: `score = ‖C_live - C_base‖_F / (frobenius_norm_baseline + ε)`
  - **Persistence filter** to avoid news-spike false alarms:
    - Yellow: score ≥ 0.15 for 3 consecutive windows
    - Red: score ≥ 0.25 for 2 consecutive windows
    - Below threshold → reset counter, stay green

### stress_results.json — Training Performance Expectations

Contains the W170 stress matrix results for the training-vs-live comparison panel:

```json
{
  "champion": "W170",
  "val_base": {"pf": 2.238, "sharpe": 4.091, "max_dd": 20.8, "trades": 1044, "win_rate": 52.0, "tim": 8.9, "trades_per_day": 2.4, "avg_hold": 8.7},
  "val_2x_spread": {"pf": 2.013, "max_dd": 35.8, "trades": 948, "win_rate": 49.6},
  "val_3x_spread": {"pf": 1.629, "max_dd": 47.3, "trades": 764, "win_rate": 45.5},
  "val_combined_2x2x": {"pf": 1.801, "max_dd": 32.2, "trades": 870, "win_rate": 47.9},
  "test_base": {"pf": 2.818, "sharpe": 4.215, "max_dd": 13.6, "trades": 481, "win_rate": 52.0},
  "test_2x_spread": {"pf": 2.163, "max_dd": 18.8, "trades": 456, "win_rate": 49.3}
}
```

- Live dashboard's Tab 6 loads these values on startup
- Compares rolling 30-day live metrics against both val and test expectations
- The `val_2x_spread` scenario is the most realistic comparison for live performance (real execution has spread + slippage beyond base conditions)

### Loading

```python
class ModelLoader:
    """Load a model package and prepare for live inference."""

    def load(self, package_path: str) -> dict:
        """Extract and validate model package.

        Returns: {
            'model': SAC model instance,
            'config': TrainingConfig,
            'metadata': dict (week, val_sharpe, etc.),
            'reward_state': dict (normalizer state),
            'feature_baseline': dict (54 feature mean/std),
            'correlation_baseline': dict (20×20 corr matrix),
            'stress_results': dict (stress matrix scenarios),
        }
        """
        # 1. Extract ZIP to temp dir
        # 2. Load model.zip via SAC.load()
        # 3. Load config.json → validate feature count matches
        # 4. Load metadata.json → display in dashboard
        # 5. Load reward_state.json → store for display + future fine-tuning
        #    (live inference does NOT use rewards — model.predict() only)
        # 6. Load feature_baseline.json → Tab 6 drift detection (54 features)
        # 7. Load correlation_baseline.json → Tab 6 correlation drift (20×20 matrix)
        # 8. Load stress_results.json → Tab 6 training-vs-live comparison
        # 9. Validate obs_dim matches (670)
        # 7. Return all components
```

### Package Creator (Training Side)

Add to training system — creates the model package for distribution:

```python
def create_model_package(model_path, config, metadata, reward_state,
                         output_path):
    """Bundle trained model for live deployment.

    Call after training completes:
        python scripts/package_model.py
    """
```

---

## 11. Phase 9: Paper Trading Mode

### Purpose

Run the live system on a **demo account** before risking real money. Everything works the same except no real money is at stake.

```python
class PaperTradingMode:
    """Configuration flag — controls dashboard display only.

    When paper_trading=True:
    - Dashboard shows "PAPER TRADING" banner (bright yellow)
    - All trades logged with paper_trade=True flag
    - Same execution, same risk, same features
    - Can compare paper vs. expected performance

    Duration: minimum 1 week recommended before going live.
    """
```

### Paper → Live Transition Checklist

Displayed in dashboard before switching to live:

```
Before going live, verify:
□ Paper traded for >= 1 week minimum (2 weeks recommended)
□ Paper Sharpe >= 0.3
□ Paper profit factor >= 1.4 (realistic target, not backtest PF)
□ No catastrophic drawdowns (< 10%)
□ All features computing correctly (67/67)
□ Correlated instrument feeds stable
□ Correlation drift score stable (green for 1+ week)
□ Circuit breakers tested
□ Emergency stop tested
□ Weekend close working
□ Account currency correctly detected
□ Risk logging shows actual risk within expected bounds
```

### £100 Demo Deployment — Lot Quantization Reality

The W170 model was trained starting at £100, and the demo should match. However, MT5's minimum lot size (0.01) creates quantization effects that must be understood:

**Empirical analysis (767K M5 bars, 2015-2026):**

```
XAUUSD M5 ATR(14) Distribution:
  Median: 0.94    P25: 0.60    P75: 1.55    P95: 3.65

Trade Viability at £100 (GBP account, tick_value=0.7412):
  Balance   Viable   Rejected(too small)
  £100      76.1%    23.9%
  £150      87.8%    12.2%
  £200      92.7%     7.3%
  £300      96.8%     3.2%
  £500      98.9%     1.1%
```

**Key insights:**
- At £100, ~24% of trade signals are rejected because the computed lot is too far below 0.01 (vol_min × 0.5)
- **This is identical to training** — the same risk_manager code with the same vol_min=0.01 ran during all 170 weeks of training. The model learned to operate within this constraint.
- As the account grows from profitable trades, viability rapidly improves
- At P95 ATR (extreme volatility), ALL trades are rejected at £100 — the system naturally goes flat during danger
- The lot rounding that occurs (e.g., 0.007 → 0.01) is also identical to training behavior

**Overshoot from lot rounding (same as training):**
- Lot rounding causes actual risk to sometimes exceed intended risk
- This is the same rounding behavior the model was trained with
- The risk_analysis block in trades.jsonl logs both intended and actual risk for monitoring

### Initial Observation Period (First 2 Weeks) — OPTIONAL

**This feature changes trading behavior vs training.** The model was trained with dynamic lot sizing from the start. Enabling the observation cap means the live system won't behave identically to training during the cap period. Use it if you want to observe pure behavior before allowing dynamic sizing. Skip it if you want exact training parity.

```python
# In LiveRiskManager — ONLY if observation period is enabled
if self.observation_enabled and datetime.utcnow() < self.observation_period_end:
    lots = min(lots, self.observation_lot_cap)  # Cap during observation
```

**Impact:** At £100 starting balance, lot sizes are already tiny (0.01-0.02). The cap only matters once balance grows enough for larger lots. In practice, the cap has minimal effect during the first 2 weeks unless the account grows significantly.

**Config (disabled by default):**
```python
observation_period_enabled: bool = False  # Set True to enable lot cap
observation_period_days: int = 14         # Cap lots for first N days
observation_lot_cap: float = 0.01         # Max lot during observation
```

### Realistic Performance Expectations

Do NOT compare live performance directly to backtest PF 2.8. The right benchmarks:

| Metric | Backtest (W170 test) | Realistic Live Target | Red Flag |
|--------|---------------------|----------------------|----------|
| PF | 2.818 | ≥ 1.4 | < 1.0 for 2+ weeks |
| Win Rate | 58.2% | ≥ 45% | < 35% |
| Max DD | 13.6% | < 15% | > 25% |
| Trades/Day | 0.34 | 0.2 - 0.5 | 0 for 3+ days |
| TIM% | 0.6% | 0.5 - 5% | > 15% |

**If PF < 1.0 AND drift mostly green:** structural mismatch — investigate execution, spread, data feed
**If PF drops only during yellow/red drift:** model is fine, market regime shifted

---

## 12. Phase 10: Startup Validation & Deployment Checklist

### File: `core/startup_validator.py`

Runs automatically when the dashboard launches. Results are displayed as a green/red
checklist in the dashboard. The "Start Trading" button is **disabled** until all required
checks pass. Trading **never** begins automatically.

```python
class StartupValidator:
    """Pre-flight checks before trading can be enabled.

    Each check returns (pass: bool, message: str).
    ALL required checks must pass before the "Start Trading" button activates.
    Optional checks show warnings but don't block trading.
    """

    required_checks = [
        # MT5 Connection
        ("MT5 Terminal Running", check_mt5_running),
        ("MT5 Account Logged In", check_mt5_account),
        ("XAUUSD Symbol Available", check_symbol_available),
        ("XAUUSD Spread Reasonable", check_spread_reasonable),

        # Account
        ("Account Currency Detected", check_currency_detected),
        ("tick_value Valid", check_tick_value),
        ("Sufficient Balance", check_balance),

        # Model
        ("Model File Exists", check_model_exists),
        ("Model Loads Successfully", check_model_loads),
        ("Observation Dim Correct (670)", check_obs_dim),
        ("Feature Count Matches Config (67)", check_feature_count),

        # Features
        ("M5 History Available (500+ bars)", check_m5_history),
        ("H1/H4/D1 History Available", check_htf_history),
        ("Correlated Instruments Available", check_correlated),
        ("Feature Warmup Complete", check_warmup),
        ("Normalizer Initialized", check_normalizer),

        # Safety
        ("Circuit Breaker Configured", check_circuit_breaker),
        ("Emergency Stop Working", check_emergency_stop),
        ("Weekend Manager Active", check_weekend),
        ("Memory DB Accessible", check_memory_db),
        ("Log Directory Writable", check_logs),
    ]

    optional_checks = [
        # Calendar (features default to neutral if unavailable)
        ("Calendar Data Available", check_calendar_any_source),
        # Stress results for live comparison
        ("Stress Results in Model Package", check_stress_results),
        ("Feature Baseline in Model Package", check_feature_baseline),
    ]
```

**User workflow on first launch:**
1. Dashboard opens in STOPPED state
2. Startup checks run automatically and display results
3. User reviews the checklist, sees any warnings
4. User clicks **"Start Trading"** (only enabled when all required checks pass)
5. Trading begins

---

## 13. CLI Live Monitor

### File: `scripts/live_monitor.py`

Equivalent to training's `scripts/monitor.py --deep`. A read-only CLI tool that inspects the live trading system from any terminal without interfering with trading. Uses SQLite WAL mode for safe concurrent access.

### Usage

```bash
python live_dashboard/scripts/live_monitor.py                 # Quick status
python live_dashboard/scripts/live_monitor.py --deep          # Full diagnostics
python live_dashboard/scripts/live_monitor.py --compare       # Training vs live
python live_dashboard/scripts/live_monitor.py --session       # Per-session breakdown
python live_dashboard/scripts/live_monitor.py --health        # Feature health + drift
python live_dashboard/scripts/live_monitor.py --weekly        # Weekly report summary
```

### Data Sources

```
storage/memory/live_trading.db   — trades, patterns, predictions, journal
storage/logs/trades.jsonl        — full trade details with entry conditions
storage/logs/actions.jsonl       — every action decision (4 values per bar)
storage/logs/feature_stats.jsonl — feature distribution snapshots
storage/logs/weekly_summary.jsonl — weekly aggregated performance
storage/state/normalizer_state.json — normalizer buffer health
```

### Quick Mode (default)

```
======================================================================
SPARTUS LIVE MONITOR — 2026-03-15 14:35:00 UTC
======================================================================
Connection:  ● MT5 Connected (Vantage-Live, 45ms latency)
Account:     GBP | Balance: £1,234.56 | Equity: £1,245.89
Position:    LONG 0.02 lots @ £2,651.30 | P/L: +£5.00 | SL: £2,648.50

TODAY
  Trades: 4  (3W / 1L)  |  P/L: +£12.45  |  Win Rate: 75%
  Max DD: 1.2%  |  PF: 2.31  |  Remaining: 16/20 hard cap

LAST 5 TRADES
  14:25 LONG  0.02  +£5.00  TP_HIT    hold=9 bars  conv=0.72
  13:10 SHORT 0.01  -£2.30  SL_HIT    hold=4 bars  conv=0.45
  12:40 LONG  0.02  +£3.10  AGENT     hold=12 bars conv=0.68
  11:55 SHORT 0.01  +£1.80  TP_HIT    hold=7 bars  conv=0.52
  10:30 LONG  0.01  +£4.85  TP_HIT    hold=15 bars conv=0.61

FEATURES: 67/67 active | Normalizer: 200/200 bars | Stale feeds: 0
ALERTS:   0 ERROR | 1 WARN (spread spike 13:15)
======================================================================
```

### Deep Mode (--deep)

Adds to quick mode:

```
ACTION DISTRIBUTION (last 100 bars)
  Direction:  μ=+0.12  σ=0.48  flat_rate=62%
  Conviction: μ=0.68   σ=0.15  min=0.31  max=0.94
  Exit:       μ=0.22   σ=0.18  close_rate=8%
  SL Adj:     μ=0.45   σ=0.21

MODEL BEHAVIOR
  Direction bias: +0.12 (slight long bias)
  Avg conviction on entry: 0.71
  Avg exit urgency on close: 0.73
  Avg hold bars: 8.2
  TP/SL hit ratio: 1.8:1

TRADE QUALITY (all time)
  Total: 47 trades | Win: 26 (55%) | Loss: 21 (45%)
  PF: 1.85 | Sharpe (30d): 0.92 | Max DD: 4.2%
  Lesson breakdown:
    GOOD_TRADE: 18  WRONG_DIRECTION: 8  BAD_TIMING: 4
    CORRECT_DIR_EARLY: 6  WHIPSAW: 3  HELD_TOO_LONG: 2

SESSION PERFORMANCE
  London:     12 trades  58% win  PF 1.9  avg +£2.10
  NY Overlap:  8 trades  50% win  PF 1.5  avg +£0.80
  NY PM:       3 trades  67% win  PF 2.1  avg +£1.50
  Asia:        3 trades  33% win  PF 0.8  avg -£1.20
  Off-hours:   1 trade    0% win  PF 0.0  avg -£3.10

DAY OF WEEK
  Mon: +£4.20 | Tue: +£1.80 | Wed: -£2.10 | Thu: +£6.50 | Fri: +£1.20

FEATURE DRIFT (54 baselined / 13 live-only excluded)
  52/54 features within 2σ of training baseline
  ⚠ atr_14_norm: live μ=1.23 vs train μ=0.95 (drift: 2.3σ)
  ⚠ eurusd_returns_20: live μ=0.05 vs train μ=0.01 (drift: 2.1σ)

CORRELATION DRIFT (top-20 features, normalized)
  Score: 0.08  ● OK  (consecutive: 0/3 yellow, 0/2 red)

HARDWARE
  CPU: 12% | RAM: 2.1 GB / 16.0 GB | GPU: idle (CPU inference)
```

### Compare Mode (--compare)

```
======================================================================
TRAINING vs LIVE COMPARISON — W170 Champion
======================================================================
Metric          Training(Val) Training(Test) Live(30d)  Status
─────────────────────────────────────────────────────────────────
PF              2.238         2.818          1.85       ● YELLOW (-17%)
Win Rate        52.0%         52.0%          55.3%      ● GREEN
MaxDD%          20.8%         13.6%          4.2%       ● GREEN
Trades/Day      2.4           1.3            1.8        ● GREEN
TIM%            8.9%          11.1%          6.1%       ● GREEN
Avg Hold Bars   8.7           8.7            8.2        ● GREEN
Sharpe          4.091         4.215          0.92       ● YELLOW (expected lower live)

NOTE: Live PF < training PF is expected due to real execution costs.
Compare against val_2x_spread (PF 2.013) for realistic expectation.

VERDICT: Model performing within expected range. No retraining needed.
======================================================================
```

---

## 14. Enhanced Logging

### Overview

The live dashboard produces 6 log files for comprehensive post-hoc analysis and retraining data collection:

| File | Content | Frequency | Purpose |
|------|---------|-----------|---------|
| `trades.jsonl` | Full trade details | Per trade close | Trade-by-trade analysis |
| `actions.jsonl` | All 4 action values + decision | Every bar | Model behavior tracking |
| `alerts.jsonl` | Warnings and errors | On event | Issue diagnosis |
| `observations.jsonl` | 670-dim obs + action + decision | Configurable (default: hourly) | Replay, retraining data |
| `feature_stats.jsonl` | Per-feature mean/std/min/max | Per session boundary | Drift detection |
| `weekly_summary.jsonl` | Weekly aggregated metrics | Sunday 23:59 UTC | Performance evolution |

### trades.jsonl — Full Trade Record

Logged on every trade closure:

```json
{
  "timestamp": "2026-03-15T14:25:00Z",
  "trade_id": 47,
  "ticket": 12345678,
  "side": "LONG",
  "entry_price": 2651.30,
  "exit_price": 2658.00,
  "lots": 0.02,
  "pnl": 5.00,
  "pnl_pct": 0.40,
  "hold_bars": 9,
  "close_reason": "TP_HIT",
  "conviction": 0.72,
  "direction_raw": 0.65,
  "entry_conditions": {
    "rsi_14": 42.3,
    "trend_dir": 0.8,
    "session": "London",
    "vol_regime": "normal",
    "atr_14": 3.45,
    "spread_at_entry": 0.21,
    "daily_trade_count": 3,
    "current_drawdown": 0.012
  },
  "sl_initial": 2648.50,
  "sl_final": 2649.80,
  "tp": 2658.00,
  "sl_trailed": true,
  "profit_locked": 2.10,
  "risk_analysis": {
    "intended_risk_pct": 1.44,
    "actual_risk_pct": 1.68,
    "intended_risk_amount": 1.80,
    "actual_risk_amount": 2.10,
    "raw_lots": 0.0162,
    "rounded_lots": 0.02,
    "overshoot_pct": 16.7,
    "sl_distance": 2.80,
    "atr_at_entry": 1.75,
    "observation_period": false
  },
  "lesson_type": "GOOD_TRADE",
  "paper_trade": false
}
```

### actions.jsonl — Every Action Decision

Logged every M5 bar:

```json
{
  "timestamp": "2026-03-15T14:35:00Z",
  "bar_time": "2026-03-15T14:30:00Z",
  "action_raw": [-0.32, 0.44, -0.88, 0.12],
  "direction": -0.32,
  "conviction": 0.72,
  "exit_urgency": 0.06,
  "sl_adjustment": 0.56,
  "decision": "HOLD_FLAT",
  "has_position": false,
  "balance": 1234.56,
  "equity": 1234.56,
  "trade_rejected": false,
  "reject_reason": null
}
```

When a trade signal is generated but rejected due to lot sizing:

```json
{
  "timestamp": "2026-03-15T15:00:00Z",
  "bar_time": "2026-03-15T14:55:00Z",
  "action_raw": [0.78, 0.82, -0.91, 0.33],
  "direction": 0.78,
  "conviction": 0.91,
  "decision": "OPEN_LONG",
  "has_position": false,
  "balance": 112.50,
  "equity": 112.50,
  "trade_rejected": true,
  "reject_reason": "lot_below_minimum",
  "reject_details": {
    "raw_lots": 0.0038,
    "vol_min": 0.01,
    "intended_risk_pct": 1.82,
    "atr": 2.15
  }
}
```

### observations.jsonl — Observation Dumps (Configurable)

Logged every N bars (default: 12 = hourly). Contains full 670-dim observation vector for replay:

```json
{
  "timestamp": "2026-03-15T14:35:00Z",
  "bar_idx": 1234,
  "observation_670": [0.12, -0.45, 1.23, ...],
  "action_raw": [-0.32, 0.44, -0.88, 0.12],
  "action_parsed": {"direction": -0.32, "conviction": 0.72, "exit": 0.06, "sl_adj": 0.56},
  "decision": "HOLD_FLAT",
  "has_position": false,
  "feature_snapshot": {
    "returns_20bar": 0.12,
    "atr_14_norm": 1.23,
    "rsi_14": 0.42
  }
}
```

**Config:** `log_observations_every_n_bars: 12` (set to 1 for full replay, but file grows ~2MB/day at full rate)

**Purpose:** Replay model decisions for debugging. Can also be used to build a training dataset from live experience for future domain adaptation.

### feature_stats.jsonl — Feature Distribution Snapshots

Logged at each session boundary (00:00, 07:00, 12:00, 20:00 UTC):

```json
{
  "timestamp": "2026-03-15T12:00:00Z",
  "session": "London",
  "bars_in_session": 60,
  "stats": {
    "returns_20bar": {"mean": 0.02, "std": 1.05, "min": -3.2, "max": 2.8, "nan_count": 0},
    "atr_14_norm": {"mean": 1.23, "std": 0.45, "min": 0.32, "max": 2.89, "nan_count": 0}
  },
  "drift_flags": ["atr_14_norm"],
  "drift_count": 1,
  "baselined_features": 54,
  "correlation_drift_score": 0.08
}
```

### weekly_summary.jsonl — Weekly Performance Aggregates

Auto-generated Sunday 23:59 UTC:

```json
{
  "week": "2026-W11",
  "start": "2026-03-09T00:00:00Z",
  "end": "2026-03-14T20:00:00Z",
  "trades": 15,
  "wins": 8,
  "losses": 7,
  "pnl": 22.10,
  "pf": 2.10,
  "sharpe": 1.45,
  "max_dd_pct": 2.8,
  "win_rate": 0.533,
  "avg_hold_bars": 7.2,
  "avg_conviction": 0.65,
  "session_breakdown": {
    "london": {"trades": 6, "wins": 4, "pf": 2.3, "pnl": 14.50},
    "ny_overlap": {"trades": 4, "wins": 2, "pf": 1.2, "pnl": 3.20},
    "asia": {"trades": 3, "wins": 1, "pf": 0.7, "pnl": -2.10},
    "off_hours": {"trades": 2, "wins": 1, "pf": 1.1, "pnl": 0.50}
  },
  "action_stats": {
    "direction_mean": 0.08,
    "direction_std": 0.52,
    "conviction_mean": 0.65,
    "flat_rate": 0.62,
    "close_rate": 0.08
  },
  "feature_drift_count": 2,
  "correlation_drift_score": 0.08,
  "lesson_breakdown": {
    "GOOD_TRADE": 4, "WRONG_DIRECTION": 3, "BAD_TIMING": 2,
    "CORRECT_DIR_EARLY": 2, "WHIPSAW": 1, "HELD_TOO_LONG": 1
  },
  "circuit_breaker_activations": 0,
  "emergency_stops": 0,
  "paper_trade": false
}
```

**Purpose:** Track performance evolution week-over-week. Primary input for deciding if/when to retrain. If weekly PF trends below 1.0 for 3+ consecutive weeks → retraining trigger.

---

## 15. Data Flow Diagram

### Per-Bar Cycle (Every 5 Minutes)

```
1. NEW M5 BAR CLOSES
   │
2. MT5 Bridge: Fetch latest bars
   ├─ XAUUSD M5 (1 bar)
   ├─ Correlated M5 (5 instruments, 1 bar each)
   ├─ H1/H4/D1 if new bar closed
   └─ Account info (balance, equity, margin)
   │
3. Feature Pipeline: Compute 54 precomputed features
   ├─ Update rolling indicator buffers
   ├─ Compute Groups A-F + Upgrades 1-5
   └─ Normalize 38 market features (z-score)
   │
4. Account Features: Compute 8 live features
   ├─ Position state (from MT5)
   └─ Drawdown, equity ratio, etc.
   │
5. Memory Features: Compute 5 live features
   ├─ Pattern match (SQLite)
   └─ Win rate, TP rate, prediction accuracy
   │
6. Frame Stack: Build 67-dim frame → append to 10-frame buffer
   │
7. Observation: Flatten 10 frames → 670-dim vector
   │
8. Model Inference: 670-dim → 4 actions
   ├─ direction [-1, 1]
   ├─ conviction [0, 1]
   ├─ exit_urgency [0, 1]
   └─ sl_adjustment [0, 1]
   │
9. Risk Check: Circuit breakers, DD limits, trade caps
   │
10. Trade Executor: Action → MT5 order
    ├─ OPEN: send_market_order()
    ├─ CLOSE: close_position()
    ├─ TRAIL: modify_position()
    └─ HOLD: do nothing
    │
11. Memory Update: Record trade if closed
    ├─ Update SQLite tables
    └─ Generate journal entry
    │
12. Dashboard: Refresh all panels (1 Hz)
    │
13. Wait for next M5 bar close → REPEAT
```

### Timing Budget (per bar)

```
Total budget: 300 seconds (5 minutes between bars)
Actual computation: ~500ms

Breakdown:
  MT5 data fetch:      ~100ms (6 API calls)
  Feature computation:  ~200ms (54 features + indicators)
  Normalization:         ~10ms (38 z-scores)
  Model inference:       ~20ms (670-dim → 4-dim, GPU)
  Risk checks:            ~5ms
  MT5 order execution:  ~100ms (if needed)
  Memory update:         ~50ms (SQLite)
  Dashboard refresh:    ~100ms (UI thread, 1 Hz)
  ─────────────────────────
  Total:               ~585ms << 300,000ms budget
```

---

## 16. Configuration

### File: `config/live_config.py`

```python
@dataclass
class LiveConfig:
    """Live trading configuration.

    Most values are auto-detected from MT5.
    User only needs to set: model_path and optional overrides.
    """

    # === Model ===
    model_path: str = "model/spartus_champion_W170.zip"  # Champion model package

    # === MT5 Connection ===
    mt5_symbol: str = "XAUUSD"
    mt5_terminal_path: str = ""  # Auto-detect if empty

    # === Symbol Mapping (broker-specific overrides) ===
    symbol_map: dict = field(default_factory=lambda: {
        "EURUSD": "EURUSD",
        "XAGUSD": "XAGUSD",
        "USDJPY": "USDJPY",
        "US500":  "US500",     # Override if your broker uses "SPX500"
        "USOIL":  "USOIL",    # Override if your broker uses "CL-OIL"
    })

    # === Auto-Detected (DO NOT SET — pulled from MT5) ===
    # account_currency: detected at startup
    # tick_value: detected at startup
    # tick_size: detected at startup
    # contract_size: detected at startup
    # volume_min/max/step: detected at startup

    # === Risk Limits (override training defaults if desired) ===
    max_risk_pct: float = 0.02      # 2% per trade
    max_dd: float = 0.10            # 10% total max DD
    max_daily_dd: float = 0.03      # 3% daily DD
    daily_trade_hard_cap: int = 20
    min_hold_bars: int = 3          # 15 min minimum hold

    # === Circuit Breakers ===
    consecutive_loss_pause: int = 3       # Pause after N consecutive losses
    consecutive_loss_pause_minutes: int = 30
    severe_loss_pause: int = 5            # Longer pause after N losses
    severe_loss_pause_minutes: int = 120
    daily_dd_halt_pct: float = 0.02       # Halt at 2% daily DD
    daily_dd_close_all_pct: float = 0.03  # Close all at 3% daily DD
    weekly_dd_halt_pct: float = 0.05      # Halt for week at 5% DD

    # === Weekend ===
    friday_close_utc_hour: int = 20       # Close all at Friday 20:00 UTC
    friday_block_new_utc_hour: int = 19   # Block new entries at 19:00
    monday_resume_utc_hour: int = 0       # Resume Monday 00:30 UTC
    monday_resume_utc_minute: int = 30

    # === Feature Pipeline ===
    warmup_bars: int = 500          # M5 bars to load at startup
    norm_window: int = 200          # Normalization rolling window
    norm_clip: float = 5.0          # Z-score clip
    frame_stack: int = 10           # Observation frame stack

    # === Calendar Data (3-tier: MQL5 bridge > user CSV > static events) ===
    calendar_bridge_path: str = "storage/state/calendar_events.json"  # From MQL5 service
    calendar_csv_path: str = "data/calendar/economic_calendar.csv"    # Optional user CSV
    calendar_static_path: str = "data/calendar/known_events.json"     # Built-in fallback
    calendar_bridge_max_age_s: int = 7200  # Consider MQL5 bridge stale after 2 hours

    # === Paper Trading ===
    paper_trading: bool = True      # Start in paper mode by default

    # === Observation Period (Initial Deployment — OPTIONAL, off by default) ===
    observation_period_enabled: bool = False  # Enable to cap lots during first N days
    observation_period_days: int = 14         # Cap lots for first N days after first trade
    observation_lot_cap: float = 0.01         # Max lot during observation period

    # === Optional Post-Rounding Risk Cap (OFF by default for training parity) ===
    # When enabled, rejects trades where lot rounding inflates risk beyond threshold.
    # Default OFF — the model was trained WITHOUT this cap. Enable only if you want
    # extra safety at the cost of rejecting ~11% of trades at £100 balance.
    enable_post_rounding_risk_cap: bool = False
    post_rounding_risk_cap: float = 1.5  # Reject if actual risk > N× intended (when enabled)
    absolute_risk_cap_pct: float = 0.03  # Reject if actual risk > N% of balance (when enabled)

    # === Daily Reset ===
    daily_reset_utc_hour: int = 0   # 00:00 UTC — resets daily DD, trade count, CB pauses

    # === Logging ===
    log_every_action: bool = True   # Log all 4 action values every bar
    log_observations_every_n_bars: int = 12  # Log full 670-dim obs every N bars (12=hourly)
    log_feature_stats: bool = True  # Log feature distribution stats at session boundaries

    # === Analytics & Drift Detection ===
    feature_drift_threshold_sigma: float = 2.0  # Flag features drifting > Nσ from training
    corr_drift_yellow_threshold: float = 0.15   # Normalized corr drift score for yellow
    corr_drift_red_threshold: float = 0.25      # Normalized corr drift score for red
    corr_drift_yellow_persist: int = 3           # Consecutive windows needed for yellow alert
    corr_drift_red_persist: int = 2              # Consecutive windows needed for red alert
    action_flat_rate_warn: float = 0.90          # Warn if flat rate > 90% for 2+ days
    action_std_collapse_warn: float = 0.10       # Warn if any action dimension std < 0.1
    weekly_report_auto: bool = True              # Auto-generate weekly report Sunday 23:59 UTC
    retrain_trigger_pf_weeks: int = 3            # Suggest retrain if PF < 1.0 for N weeks

    # === Normalizer Persistence ===
    normalizer_state_path: str = "storage/state/normalizer_state.json"
    normalizer_backup_interval_s: int = 3600     # Save normalizer buffers every hour
    normalizer_max_age_s: int = 3600             # Max age for restoring saved state (1 hour)
```

### File: `config/default_config.yaml`

User-editable YAML config (loaded at startup, overrides LiveConfig defaults):

```yaml
# Spartus Live Dashboard Configuration
# Edit this file to customize your setup.

# Path to your trained model (W170 champion)
model_path: "model/spartus_champion_W170.zip"

# Broker-specific symbol names (uncomment to override)
# symbol_map:
#   US500: "SPX500"    # Vantage uses US500, IC Markets uses SPX500
#   USOIL: "CL-OIL"   # Varies by broker

# Risk limits
max_risk_pct: 0.02    # 2% per trade
max_dd: 0.10          # 10% total max drawdown
max_daily_dd: 0.03    # 3% daily drawdown

# Circuit breakers
consecutive_loss_pause: 3  # Pause after 3 consecutive losses

# Weekend protection
friday_close_utc_hour: 20  # Close all positions Friday 20:00 UTC

# Start in paper trading mode (recommended for first week)
paper_trading: true

# Calendar data (MQL5 bridge is preferred — install CalendarBridge.mq5 in MT5)
# calendar_csv_path: "data/calendar/economic_calendar.csv"  # Uncomment for CSV override

# Observation period (first deployment)
observation_period_days: 14         # Cap lots for first 2 weeks
# observation_period_enabled: true   # Uncomment to enable lot cap
observation_lot_cap: 0.01            # Max lot during observation period (if enabled)
# Optional risk cap (OFF by default — enable for extra safety, costs ~11% trade rejection)
# enable_post_rounding_risk_cap: true
# post_rounding_risk_cap: 1.5
# absolute_risk_cap_pct: 0.03

# Logging & Monitoring
log_observations_every_n_bars: 12  # 12 = hourly, 1 = every bar (verbose)
log_feature_stats: true            # Feature drift snapshots at session boundaries
weekly_report_auto: true           # Auto-generate weekly report

# Analytics thresholds
feature_drift_threshold_sigma: 2.0  # Flag drifted features
corr_drift_yellow_threshold: 0.15   # Normalized score for yellow
corr_drift_red_threshold: 0.25      # Normalized score for red
corr_drift_yellow_persist: 3        # Consecutive windows for yellow
corr_drift_red_persist: 2           # Consecutive windows for red
retrain_trigger_pf_weeks: 3         # Alert if PF < 1.0 for 3+ weeks
```

---

## 17. Implementation Order & Dependencies

Build phases in dependency order. Each phase is testable independently.

| Phase | Component | Files | Depends On | Est. Lines |
|-------|-----------|-------|------------|------------|
| **1** | MT5 Bridge | `core/mt5_bridge.py`, `utils/symbol_mapper.py` | None | ~400 |
| **2** | Feature Pipeline | `core/feature_pipeline.py`, `core/live_normalizer.py`, `features/*.py` | Phase 1 | ~900 |
| **3** | Model Inference | `core/inference_engine.py`, `core/model_loader.py` | None | ~250 |
| **4** | Trade Executor | `core/trade_executor.py`, `core/position_manager.py` | Phase 1, 3 | ~400 |
| **5** | Risk & Safety | `core/risk_manager.py`, `safety/*.py` | Phase 1 | ~500 |
| **6** | Memory System | `memory/*.py` | None (standalone) | ~300 |
| **7** | Dashboard UI (Tabs 1-5) | `dashboard/*.py` | Phase 1-6 | ~1200 |
| **7a** | Dashboard Tab 6: Analytics | `dashboard/tab_analytics.py` | Phase 1-6 | ~350 |
| **8** | Model Package Loader | `core/model_loader.py` (+ feature_baseline, stress_results) | Phase 3 | ~200 |
| **9** | Paper Trading | Config flag + UI changes | Phase 7 | ~100 |
| **10** | Startup Validation | `core/startup_validator.py` | Phase 1-6 | ~200 |
| **11** | CLI Live Monitor | `scripts/live_monitor.py` | Phase 6 | ~400 |
| **12** | Enhanced Logging | `utils/logger.py` (obs, feature_stats, weekly) | Phase 2, 4 | ~200 |
| **13** | Feature Baseline Generation | Training exporter enhancement (already done — bundled in model ZIP) | N/A | ~100 |

**Total estimate: ~5,500 lines across ~35 files.**

### Critical Path

```
Phase 1 (MT5 Bridge)
  └─▶ Phase 2 (Features) ─┐
  └─▶ Phase 4 (Executor) ──┤
  └─▶ Phase 5 (Risk) ──────┤
                             └─▶ Phase 7 (Dashboard Tabs 1-5)
Phase 3 (Model) ────────────┘        │
Phase 6 (Memory) ───────────────────┘
                                      └─▶ Phase 7a (Tab 6: Analytics)
                                      └─▶ Phase 11 (CLI Monitor)
Phase 12 (Enhanced Logging) ← Phase 2 + Phase 4
Phase 13 (Feature Baseline) ← Phase 2 (training side, one-time)
```

**Phases 1, 3, and 6 can be built in parallel** (no dependencies).
**Phase 7 (Dashboard) comes after all backend components.**
**Phases 11-13 can be built after core is functional** — they add monitoring, not trading logic.

---

## 18. Testing Strategy

### Unit Tests

```
tests/
├── test_mt5_bridge.py           # Mock MT5 API, test connection/reconnect
├── test_feature_pipeline.py     # Verify all 67 features computed correctly
├── test_live_normalizer.py      # Verify z-scores match expected behavior
├── test_normalizer_persistence.py  # Save/load normalizer state round-trip
├── test_trade_executor.py       # Test action → order translation
├── test_risk_manager.py         # Test circuit breakers, DD limits, daily reset
├── test_model_loader.py         # Test model package load/validate (incl. feature_baseline, stress_results)
├── test_startup_validator.py    # Test all pre-flight checks
├── test_parity.py               # CRITICAL: observation parity test
├── test_feature_drift.py        # Test drift detection against baseline
├── test_logging.py              # Verify all 6 log files write correctly
└── test_weekly_report.py        # Verify weekly summary generation
```

### Feature Parity Test (Most Important)

```python
def test_observation_parity():
    """Verify live pipeline produces correct 670-dim observations.

    1. Load a week of historical M5 data
    2. Run through standalone FeatureBuilder → get reference observations
    3. Feed same data through full live pipeline (normalizer + frame stack)
    4. Assert all 670 dimensions match within tolerance (1e-6)

    This test uses saved reference observations from the training system
    (bundled in tests/fixtures/) to catch any feature computation drift.
    """
```

### Integration Tests

```python
def test_full_loop_paper():
    """Run one full bar cycle on demo account.

    1. Connect to MT5 demo
    2. Warmup features
    3. Load model
    4. Process one bar
    5. Get action
    6. Execute (on demo)
    7. Verify: order placed, position tracked, memory updated
    """
```

### P/L Calculation Verification

```python
def test_pnl_matches_mt5():
    """Verify our P/L math matches MT5 exactly.

    1. Open position on demo account
    2. Wait for price to move
    3. Close position
    4. Compare: our calculated P/L vs MT5 reported P/L
    5. Must match within 1 tick tolerance
    """
```

---

## Appendix A: Feature Reference (All 67)

| # | Name | Group | Normalized | Source |
|---|------|-------|------------|--------|
| 1 | close_frac_diff | A: Price | Z-score | M5 close |
| 2 | returns_1bar | A: Price | Z-score | M5 close |
| 3 | returns_5bar | A: Price | Z-score | M5 close |
| 4 | returns_20bar | A: Price | Z-score | M5 close |
| 5 | bar_range | A: Price | Z-score | M5 OHLC |
| 6 | close_position | A: Price | Z-score | M5 OHLC |
| 7 | body_ratio | A: Price | Z-score | M5 OHLC |
| 8 | atr_14_norm | B: Volatility | Z-score | M5 HLC |
| 9 | atr_ratio | B: Volatility | Z-score | M5 HLC |
| 10 | bb_width | B: Volatility | Z-score | M5 close |
| 11 | bb_position | B: Volatility | Z-score | M5 close |
| 12 | rsi_14 | C: Momentum | Z-score | M5 close |
| 13 | macd_signal | C: Momentum | Z-score | M5 close |
| 14 | adx_14 | C: Momentum | Z-score | M5 HLC |
| 15 | ema_cross | C: Momentum | Z-score | M5 close |
| 16 | price_vs_ema200 | C: Momentum | Z-score | M5 close |
| 17 | stoch_k | C: Momentum | Z-score | M5 HLC |
| 18 | volume_ratio | D: Volume | Z-score | M5 volume |
| 19 | obv_slope | D: Volume | Z-score | M5 close+vol |
| 20 | h1_trend_dir | E: MTF | Z-score | H1 close |
| 21 | h4_trend_dir | E: MTF | Z-score | H4 close |
| 22 | d1_trend_dir | E: MTF | Z-score | D1 close |
| 23 | h1_rsi | E: MTF | Z-score | H1 close |
| 24 | mtf_alignment | E: MTF | Z-score | H1/H4/D1 |
| 25 | htf_momentum | E: MTF | Z-score | H4 close |
| 26 | hour_sin | F: Time | Exempt | Timestamp |
| 27 | hour_cos | F: Time | Exempt | Timestamp |
| 28 | day_of_week | F: Time | Exempt | Timestamp |
| 29 | session_quality | F: Time | Exempt | Timestamp |
| 30 | eurusd_returns_20 | U1: Corr | Z-score | EURUSD M5 |
| 31 | eurusd_rsi_14 | U1: Corr | Z-score | EURUSD M5 |
| 32 | eurusd_trend | U1: Corr | Z-score | EURUSD M5 |
| 33 | xagusd_returns_20 | U1: Corr | Z-score | XAGUSD M5 |
| 34 | xagusd_rsi_14 | U1: Corr | Z-score | XAGUSD M5 |
| 35 | usdjpy_returns_20 | U1: Corr | Z-score | USDJPY M5 |
| 36 | usdjpy_trend | U1: Corr | Z-score | USDJPY M5 |
| 37 | us500_returns_20 | U1: Corr | Z-score | US500 M5 |
| 38 | us500_rsi_14 | U1: Corr | Z-score | US500 M5 |
| 39 | usoil_returns_20 | U1: Corr | Z-score | USOIL M5 |
| 40 | gold_silver_ratio_z | U1: Corr | Z-score | XAU/XAG |
| 41 | hours_to_next_high_impact | U2: Calendar | Exempt | Calendar CSV |
| 42 | hours_to_next_nfp_fomc | U2: Calendar | Exempt | Calendar CSV |
| 43 | in_event_window | U2: Calendar | Exempt | Calendar CSV |
| 44 | daily_event_density | U2: Calendar | Exempt | Calendar CSV |
| 45 | london_fix_proximity | U2: Calendar | Exempt | Calendar CSV |
| 46 | comex_session_active | U2: Calendar | Exempt | Calendar CSV |
| 47 | spread_estimate_norm | U3: Spread | Exempt | ATR + hour |
| 48 | volume_spike | U3: Spread | Exempt | M5 volume |
| 49 | corr_gold_usd_100 | U4: Regime | Z-score | XAU + EUR |
| 50 | corr_gold_spx_100 | U4: Regime | Z-score | XAU + US500 |
| 51 | asian_range_norm | U5: Session | Exempt | M5 OHLC |
| 52 | asian_range_position | U5: Session | Exempt | M5 OHLC |
| 53 | session_momentum | U5: Session | Exempt | M5 close |
| 54 | london_ny_overlap | U5: Session | Exempt | Timestamp |
| 55 | has_position | G: Account | Exempt | Live state |
| 56 | position_side | G: Account | Exempt | Live state |
| 57 | unrealized_pnl | G: Account | Exempt | Live state |
| 58 | position_duration | G: Account | Exempt | Live state |
| 59 | current_drawdown | G: Account | Exempt | Live state |
| 60 | equity_ratio | G: Account | Exempt | Live state |
| 61 | sl_distance_ratio | G: Account | Exempt | Live state |
| 62 | profit_locked_pct | G: Account | Exempt | Live state |
| 63 | recent_win_rate | H: Memory | Exempt | SQLite |
| 64 | similar_pattern_winrate | H: Memory | Exempt | SQLite |
| 65 | trend_prediction_accuracy | H: Memory | Exempt | SQLite |
| 66 | tp_hit_rate | H: Memory | Exempt | SQLite |
| 67 | avg_sl_trail_profit | H: Memory | Exempt | SQLite |

---

## Appendix B: MT5 Python API Quick Reference

```python
import MetaTrader5 as mt5

# Connection
mt5.initialize()
mt5.shutdown()

# Account
mt5.account_info()          # → AccountInfo (currency, balance, equity, etc.)

# Symbol
mt5.symbol_info("XAUUSD")  # → SymbolInfo (tick_value, spread, etc.)
mt5.symbol_info_tick("XAUUSD")  # → Tick (bid, ask, last, volume, time)

# Data
mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 500)  # Last 500 M5 bars

# Orders
mt5.order_send(request)     # Place/modify/close order
mt5.positions_get(symbol="XAUUSD")  # Open positions
mt5.orders_get(symbol="XAUUSD")     # Pending orders

# Order request structure:
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "XAUUSD",
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,  # or ORDER_TYPE_SELL
    "price": mt5.symbol_info_tick("XAUUSD").ask,
    "sl": 2648.50,
    "tp": 2658.00,
    "deviation": 20,         # Max slippage in points
    "magic": 234000,         # EA magic number
    "comment": "Spartus_LONG_0.72",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
```

---

## Appendix C: Requirements

```
# live_dashboard/requirements.txt
MetaTrader5>=5.0.45
PyQt6>=6.5
pyqtgraph>=0.13
numpy>=1.24
pandas>=2.0
stable-baselines3>=2.1
torch>=2.0
ta>=0.11
PyYAML>=6.0
```

---

*Document version: 2.1*
*Created: 2026-02-26*
*Updated: 2026-03-01 — v2.0: W170 champion deployment, Tab 6 Analytics, CLI Monitor, Enhanced Logging, Feature Drift, Normalizer Persistence, Stress Results Comparison*
*Updated: 2026-03-01 — v2.1: Fully standalone (no training repo dependency), MQL5 calendar bridge, Trading State Machine (Start/Stop/Wind-Down), no auto-start, standalone memory system, tests/fixtures for parity*
*Covers: Standalone GitHub-ready live trading dashboard for Spartus Trading AI with training-equivalent monitoring*
