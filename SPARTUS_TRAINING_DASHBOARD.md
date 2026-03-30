# Spartus Trading AI - Training Dashboard Specification

**Companion document to [SPARTUS_TRADING_AI.md](SPARTUS_TRADING_AI.md)**

> **This document is the single reference for the Training Engine Dashboard.**
> It defines the layout, panels, data sources, styling, logging, alerts, and implementation plan.
> Update this file when dashboard features are added, changed, or removed.

---

## 1. Purpose & Overview

The Training Dashboard is a **PyQt6 desktop application** that runs while the SAC agent trains on historical XAUUSD data. Training runs for hours or days — the dashboard lets you monitor whether the AI is learning, spot problems early, and understand every decision the AI makes. The Qt window runs in the main thread while training runs in a background thread — zero impact on training performance.

**What it answers at a glance:**
- Is the AI learning? (balance trending up, win rate improving)
- Is the AI trading? (not stuck at zero trades)
- How far along are we? (week progress, ETA)
- What is the AI doing right now? (decision log)
- Are there problems? (errors, unhealthy signals, drawdown alerts)

**What it does NOT do:**
- Live trading monitoring (that's the Phase 2 deployment dashboard)
- Hyperparameter tuning UI (use Optuna dashboard for that)
- Data pipeline monitoring (that's a separate script)

---

## 2. Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Desktop UI | `PyQt6` | Native window, tabs, panels, labels, buttons, progress bars |
| Real-time charts | `pyqtgraph` | GPU-accelerated balance curve (green up, red down) |
| Live refresh | `QTimer` at 1 Hz | Non-blocking UI updates from shared_metrics dict |
| Layout | `QTabWidget` with 3 tabs | Overview, Metrics, AI Internals — no scrolling needed |
| Theme | Dark theme (custom QSS) | Deep dark bg (#0d1117), bright green (#2dcc2d) / red (#ff3333) |
| Deep analysis | `tensorboard` | Detailed charts, loss curves, distributions (browser) |
| Logging | `json` + `.jsonl` files | Persistent logs for post-analysis |
| Controls | QPushButton | Clickable Pause/Resume and Quit buttons in header |

---

## 3. Dashboard Layout

### Tabbed Desktop Layout (v3.4 — PyQt6)

> The dashboard is a native PyQt6 desktop window (default 1400×900px, resizable) with a **3-tab
> layout** for clean, readable display. No scrolling needed — each tab fits on screen with
> large fonts and bright colors. Pause/Resume and Quit buttons are in the header bar.

#### Tab 1 — OVERVIEW (main at-a-glance view)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SPARTUS TRAINING ENGINE v3.3   W47/200 | Step 476K    [LEARNING] [Pause][Quit]│
├─────────────────────────────────────────────────────────────────────────────────┤
│  [ OVERVIEW ]   METRICS   AI INTERNALS                                         │
│ ┌─ PROGRESS ──────────────────────────────────────────────────────────────────┐ │
│ │ Week: 47/200  Stage: 2/3 (Normal)  [████████████░░░░] 23%  142 steps/sec  │ │
│ │ Steps: 476,842  This Week: 6,842/10,000  Time: 3h 00m  ETA: ~9h 46m      │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ ┌─ ACCOUNT ──────────────────────┐ ┌─ QUICK STATS ──────────────────────────┐ │
│ │ Starting:    £100.00            │ │ Win Rate:      54.2%    (green)        │ │
│ │ Current:     £138.50  (green)   │ │ Trades:        612                     │ │
│ │ Peak:        £145.20  (cyan)    │ │ Profit Factor: 1.24                    │ │
│ │ Drawdown:    4.6%     (yellow)  │ │ Sharpe:        0.820                   │ │
│ │ Return:      +38.5%   (green)   │ │ Week P/L:      £+2.80  (green)        │ │
│ │ Bankruptcies: 1       (yellow)  │ │ Speed:         142 sps                 │ │
│ └─────────────────────────────────┘ └───────────────────────────────────────┘ │
│ ┌─ BALANCE CURVE ────────────────────────────────────────────────── LARGE ──┐ │
│ │ £145 │        ╭─╮         ← GREEN line when balance is above start       │ │
│ │      │    ╭──╯  ╰╮  ╭──╮                                                │ │
│ │ £130 │  ╭╯       ╰──╯  ╰╮       ← RED line when balance is below start  │ │
│ │      │╭─╯                ╰─╮╭──    Dashed reference at £100              │ │
│ │ £100 ┼──────────────────────────                                         │ │
│ │      0          80        200                                            │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│ ┌─ AI DECISION LOG (latest 8) ───┐ ┌─ ALERTS & WARNINGS ──────────────────┐ │
│ │ [W47] OPEN SHORT 0.01 (green)  │ │ [!] CRITICAL W46: DD 6.2% (red)     │ │
│ │ [W47] HOLD signal: 0.18 (gray) │ │ [+] POSITIVE W45: Best WR (green)   │ │
│ │ [W47] CLOSE LONG +0.85 (green) │ │ [!] WARNING W44: zero vol (yellow)  │ │
│ │ [W47] OPEN LONG 0.02  (green)  │ │ [SAC] W43: grad spike (gray)        │ │
│ └─────────────────────────────────┘ └──────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│ GPU: RTX 3060 (0.1/6 GB)  |  RAM: 12.9 GB (83%)  |  Disk: 457 GB  | Chk: W45│
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Tab 2 — METRICS (detailed trading stats)

```
│  OVERVIEW   [ METRICS ]   AI INTERNALS                                         │
│ ┌─ LEARNING METRICS ────────────┐ ┌─ THIS WEEK ────────────────────────────┐ │
│ │ Win Rate:      54.2%  (green) │ │ Trades:     14                         │ │
│ │ Profit Factor: 1.24           │ │ Wins:       8 (57%)                    │ │
│ │ Avg Trade P/L: £+0.38 (green)│ │ P/L:        £+2.80  (green)            │ │
│ │ Sharpe Ratio:  0.820          │ │ Best Trade: +£1.20  (green)            │ │
│ │ Memory Trades: 612            │ │ Worst Trade:-£0.80  (red)              │ │
│ │ Patterns:      847            │ │ Avg Hold:   12 bars                    │ │
│ │ Avg Lot Size:  0.02           │ │ Commission: £0.98                      │ │
│ └────────────────────────────────┘ └────────────────────────────────────────┘ │
│ ┌─ PREDICTIONS & TP ACCURACY ───┐ ┌─ REWARD BREAKDOWN ─────────────────────┐ │
│ │ Trend Accuracy: 56.8% (green) │ │ R1 Pos P/L (0.40):  +0.0120           │ │
│ │ UP Accuracy:    58.1%         │ │ R2 Quality (0.20):  +0.0000           │ │
│ │ DOWN Accuracy:  55.3%         │ │ R3 DD Pen  (0.15):  +0.0000           │ │
│ │ Verified:       1847          │ │ R4 D.Sharpe(0.15):  +0.0080           │ │
│ │ Pending:        255           │ │ R5 Risk-Adj(0.10):  +0.0020           │ │
│ │ TP Hit Rate:    42.0% (green) │ │ Raw: +0.0080  Normalised: +0.3400     │ │
│ │ TP Reachable:   65.0% (green) │ │ μ (mean): 0.0024  σ (std): 0.0243    │ │
│ │ SL Hit Rate:    28.0% (green) │ │ Clip %: 2.0%                          │ │
│ └────────────────────────────────┘ └────────────────────────────────────────┘ │
```

#### Tab 3 — AI INTERNALS (SAC model internals)

```
│  OVERVIEW   METRICS   [ AI INTERNALS ]                                         │
│ ┌─ SAC INTERNALS ───────────────┐ ┌─ CONVERGENCE ──────────────────────────┐ │
│ │ α (Entropy Coeff): 0.0312     │ │ Status:        ● IMPROVING (green)     │ │
│ │ Policy Entropy:    75% of init│ │ Val Sharpe:    0.670                   │ │
│ │ Q̄ Mean Value:      4.52       │ │ Best Checkpoint: W42                   │ │
│ │ Q↑ Max Value:      12.31      │ │ Since Best:    8 weeks (green)         │ │
│ │ ∇π Actor Grad:     0.230/1.0  │ │ Action Std:    0.2300                  │ │
│ │ ∇Q Critic Grad:    0.460/1.0  │ │ Entropy Trend: stable                  │ │
│ │ Lπ Actor Loss:     -2.3410    │ └────────────────────────────────────────┘ │
│ │ LQ Critic Loss:    0.8920     │                                            │
│ │ Learning Rate:     3.00e-04   │                                            │
│ │ Buffer Fill:       67%        │                                            │
│ └────────────────────────────────┘                                            │
│ ┌─ CURRICULUM & REGIME ─────────┐ ┌─ SAFETY & ANTI-HACK ──────────────────┐ │
│ │ Stage:       2/3 (Normal)     │ │ Daily Trades:  6/10  (green)           │ │
│ │ Difficulty:  0.42             │ │ Conviction:    0.3   (green)           │ │
│ │ Regime:      TRENDING UP      │ │ Hold Blocks:   0                       │ │
│ │ ▲ Trending Up:   -            │ │ Obs Health:    ✓ All OK (green)        │ │
│ │ ▼ Trending Down: -            │ │ Dead Features: 0     (green)           │ │
│ │ ═ Ranging:       -            │ │ NaN Features:  0     (green)           │ │
│ │ ⚡ Volatile:      -            │ │ Grad Clip %:   4.0%  (green)           │ │
│ │ Pattern Conf:    -            │ │ Domain Noise:  ✓ Active (green)        │ │
│ └────────────────────────────────┘ └────────────────────────────────────────┘ │
```

### Layout Structure (Qt Tabs)

```
QMainWindow
├── header (QHBoxLayout)        title + info + [LEARNING] badge + [Pause] [Quit]
├── QTabWidget (3 tabs)
│   ├── Tab 1: OVERVIEW
│   │   ├── progress (4.2)       full width — week, steps, speed, ETA, session
│   │   ├── account (4.3) | quick_stats (win rate, trades, pf, sharpe, week P/L)
│   │   ├── balance_chart (4.8)  LARGE pyqtgraph — green up, red down, £100 ref line
│   │   └── decision_log (4.9) | alerts (4.10)
│   ├── Tab 2: METRICS
│   │   ├── learning_metrics (4.4) | this_week (4.7)
│   │   └── predictions_tp (4.5+4.6) | reward_breakdown (4.14)
│   └── Tab 3: AI INTERNALS
│       ├── sac_internals (4.12) | convergence (4.13)
│       └── curriculum_regime (4.15) | safety_anti_hack (4.17)
└── footer (QHBoxLayout)        GPU | RAM | Disk | Checkpoint
```

**Panel count:** 17 total across 3 tabs. Ensemble (4.16) hidden in single-agent mode.

---

## 4. Panel Specifications

### 4.1 Header Bar

| Field | Value | Source |
|-------|-------|--------|
| Title | `SPARTUS TRAINING ENGINE v1.0` | Static |
| Keyboard hints | `[P]ause  [Q]uit` | Static |

**Styling:** Catppuccin Mocha dark theme. Title in blue (#89b4fa), bold 18px. Health badge with colored border (green=learning, cyan=early, yellow=plateau, red=overfitting/collapsed, orange=paused).

---

### 4.2 Progress Panel

| Field | Format | Source |
|-------|--------|--------|
| Week counter | `Week: {current} / {total}` | `trainer.current_week`, `config.total_training_weeks` |
| Progress bar | Rich `Progress` bar with percentage | Calculated from week counter |
| Step counters | `Steps: {global} / {total_global}` and `This Week: {week_step} / {steps_per_week}` | `agent.num_timesteps`, `config.steps_per_week` |
| Training time | `Training Time: {elapsed}` | `time.time() - start_time`, formatted as `Xh Ym` |
| ETA | `ETA: ~{remaining}` | `(elapsed / weeks_done) * weeks_remaining` |
| Speed | `Speed: {steps_per_sec} steps/sec` | Rolling average over last 60 seconds |
| Health badge | `LEARNING`, `EARLY`, `PLATEAU`, `STRUGGLING`, `STUCK` | See Section 7 |
| Session status | `FRESH start` or `RESUMED from W{n} checkpoint` | Detected on startup from `training_state.json` |

**Update frequency:** Every step (progress bar), every 100 steps (counters), every 10 seconds (ETA/speed).

**Session status:** On startup, the dashboard checks if `storage/training_state.json` exists. If it does, the session is a resume and this is displayed. If not, it shows "FRESH start". This tells you at a glance whether the training continued from a previous run.

**Colors:**
- Progress bar: green fill on dark background
- ETA: dim white (estimate, not exact)
- Session status: cyan for RESUMED, green for FRESH

---

### 4.3 Account Panel

| Field | Format | Source | Color Rule |
|-------|--------|--------|------------|
| Starting | `£{initial_balance:.2f}` | `config.initial_balance` | White (static) |
| Current | `£{balance:.2f}` | `env.balance` | Green if > starting, red if < starting |
| Peak | `£{peak:.2f}` | `env.peak_balance` | Cyan |
| Drawdown | `{dd:.1f}%` | `(peak - equity) / peak * 100` | Green < 3%, yellow 3-7%, red > 7% |
| Return | `+{return:.1f}%` or `-{return:.1f}%` | `(balance - initial) / initial * 100` | Green if positive, red if negative |
| Bankruptcies | `{count}` | `memory.bankruptcy_count` | Green if 0, yellow if 1-2, red if 3+ |

**Update frequency:** Every step.

---

### 4.4 Learning Metrics Panel

| Field | Format | Source | Color Rule |
|-------|--------|--------|------------|
| Win Rate | `{wr:.1f}%  (↑ from {baseline:.1f}%)` | `memory.get_win_rate()`, baseline from week 10 | Green > 52%, yellow 48-52%, red < 48% |
| Trend Accuracy | `{ta:.1f}%  (↑ from {baseline:.1f}%)` | `memory.get_trend_accuracy()` | Green > 55%, yellow 50-55%, red < 50% |
| Profit Factor | `{pf:.2f}` | `gross_profit / gross_loss` | Green > 1.2, yellow 1.0-1.2, red < 1.0 |
| Avg Trade P/L | `+£{avg:.2f}` or `-£{avg:.2f}` | `total_pnl / total_trades` | Green if positive, red if negative |
| Sharpe Ratio | `{sharpe:.2f}` | Calculated from weekly returns | Green > 0.8, yellow 0.3-0.8, red < 0.3 |
| Memory Trades | `{count}` | `memory.total_trades()` | White (informational) |
| Patterns | `{count}` | `memory.total_patterns()` | White (informational) |
| Avg Lot Size | `{avg:.2f}` | Average lot across recent trades | White (informational) |

**Arrows:** `↑` shown in green when metric improved vs baseline (week 10 snapshot). `↓` shown in red when metric declined. Baseline is recalculated every 50 weeks.

**Update frequency:** End of each week (metrics are weekly aggregates). Win rate also updates after each trade close.

---

### 4.5 Trend Predictions Panel

> This panel tracks the AI's ability to read market direction — the core skill that improves over time.
> Every prediction is recorded, verified after 20 bars, and the result (correct/wrong) is stored permanently.

| Field | Format | Source | Color Rule |
|-------|--------|--------|------------|
| Overall Accuracy | `{acc:.1f}%  (↑ from {baseline:.1f}%)` | `memory.get_trend_accuracy(window=100)` | Green > 55%, yellow 50-55%, red < 50% |
| UP Accuracy | `{acc:.1f}%  correct` | Accuracy on predictions where direction="UP" | Green > 55%, yellow 50-55%, red < 50% |
| DOWN Accuracy | `{acc:.1f}%  correct` | Accuracy on predictions where direction="DOWN" | Green > 55%, yellow 50-55%, red < 50% |
| Verified | `{verified} / {total} predictions` | Count of verified vs total predictions in DB | White (informational) |
| Pending | `{pending} (awaiting verification)` | Predictions not yet verified (< 20 bars old) | Dim white |
| Trend sparkline | `▁▂▃▃▅▅▆▆▇▇` | Last 20 weekly accuracy values as sparkline | Green if trending up, red if trending down |

**How verification works (the predict → wait → verify cycle):**

1. AI outputs a direction signal > 0.3 (e.g., +0.72 = predicting UP)
2. Prediction stored: direction, confidence, price at that moment
3. 20 M5 bars later (~100 minutes): system checks actual price movement
4. If price moved in predicted direction → `correct = true`
5. If price moved opposite → `correct = false`
6. Result stored permanently in memory database
7. Rolling accuracy (last 100 verified predictions) feeds back as observation feature #38

**Why this matters:** This is the AI's self-awareness metric. When the AI sees its own accuracy is high, it can trade with more conviction. When accuracy is low, it should be cautious. This feedback loop drives improvement naturally.

**Data source:** `predictions` table in SQLite memory database.

**Update frequency:** After each prediction verification (every step where a 20-bar-old prediction exists).

---

### 4.6 TP Accuracy Panel

> This panel tracks how well the AI is placing take-profit levels.
> Are TPs realistic? Is the market reaching them? Is the AI learning to set better TPs?

| Field | Format | Source | Color Rule |
|-------|--------|--------|------------|
| TP Hit Rate | `{rate:.1f}%  (↑ from {baseline:.1f}%)` | `memory.get_tp_hit_rate(window=50)` | Green > 40%, yellow 25-40%, red < 25% |
| TP Reachable Rate | `{rate:.1f}%` | `memory.get_tp_reachable_rate(window=50)` — did price touch TP at any point? | Green > 60%, yellow 40-60%, red < 40% |
| SL Hit Rate | `{rate:.1f}%` | % of closed trades that hit SL | Green < 30%, yellow 30-45%, red > 45% |
| Manual Close Rate | `{rate:.1f}%` | % of trades AI closed via exit signal | White (informational) |
| Avg Bars to Close | `{bars}` | `memory.get_avg_bars_to_close(window=50)` | White (informational) |
| Trend | `▁▂▃▃▅▅▆▆▇▇` | Last 20 weekly TP hit rates as sparkline | Green if trending up, red if trending down |

**What this tells you:**
- **TP Hit Rate** = of all closed trades, how many hit TP. Low = TPs are too ambitious or AI exits too early.
- **TP Reachable Rate** = did the market reach the TP level at ANY point, even if the trade was already closed? If TP Reachable is high but TP Hit Rate is low, the AI is closing too early (cutting winners).
- **SL Hit Rate** = high means the AI is getting stopped out too often (bad entries or SL too tight).
- **Manual Close Rate** = AI choosing to exit before TP or SL. Not inherently good or bad — context matters.

**Key insight:** If TP Reachable >> TP Hit Rate, the AI is closing profitable trades too early. This directly addresses the "let winners run" goal.

**Data source:** `tp_tracking` table in SQLite memory database.

**Update frequency:** After each trade close.

---

### 4.7 This Week Panel

| Field | Format | Source |
|-------|--------|--------|
| Trades | `{count}` | `len(env.trades_history)` for current week |
| Wins | `{wins} ({win_pct:.1f}%)` | Count of trades with pnl > 0 |
| P/L | `+£{pnl:.2f}` or `-£{pnl:.2f}` | `env.balance - week_start_balance` |
| Best | `+£{best:.2f}` | `max(trade.pnl for trade in week_trades)` |
| Worst | `-£{worst:.2f}` | `min(trade.pnl for trade in week_trades)` |
| Lot Range | `{min:.2f} - {max:.2f}` | Min/max lot sizes used this week |
| Avg Hold | `{bars} bars` | Average position duration in bars |
| Commission | `£{total:.2f}` | Total commission paid this week |

**Update frequency:** After each trade close, and at end of week.

**Colors:** P/L field green if positive, red if negative. Win percentage uses same thresholds as Learning Metrics win rate.

---

### 4.8 Balance Curve Panel

A real-time pyqtgraph chart showing balance over training steps, plus a smaller reward chart below it.

**Chart specification:**
- **Balance chart** (top): X-axis = training steps, Y-axis = balance in £. Blue line (#89b4fa), 2px width. Up to 2000 data points (auto-trimmed). Anti-aliased.
- **Reward chart** (bottom, smaller): X-axis = recent steps, Y-axis = raw reward. Green line (#a6e3a1), 1px width. Up to 500 data points.
- Both charts: dark background (#1e1e2e), subtle grid (alpha 0.1), dim axis labels.

**Data source:** Balance and raw reward read from `shared_metrics` each update tick (1 Hz).

**Implementation:** `pyqtgraph.PlotWidget` with `PlotDataItem.setData()` called on each timer tick. GPU-accelerated rendering via OpenGL. Handles thousands of points without lag.

**Update frequency:** End of each week.

---

### 4.9 AI Decision Log Panel

A scrolling log of the most recent 8 AI decisions. Each entry shows what the AI did and why.

**Entry formats:**

```
OPEN LONG:
  [W{week} Bar {bar}] OPEN LONG {lots:.2f} lots @ {price:.2f} (conv: {conviction:.2f})
                       SL: {sl:.2f}  TP: {tp:.2f}  Spread: {spread:.1f} pips

OPEN SHORT:
  [W{week} Bar {bar}] OPEN SHORT {lots:.2f} lots @ {price:.2f} (conv: {conviction:.2f})
                       SL: {sl:.2f}  TP: {tp:.2f}  Spread: {spread:.1f} pips

CLOSE (profitable):
  [W{week} Bar {bar}] CLOSE LONG +£{pnl:.2f} (held {bars} bars, {minutes} min)

CLOSE (loss):
  [W{week} Bar {bar}] CLOSE SHORT -£{pnl:.2f} (held {bars} bars, SL hit)

CLOSE (exit signal):
  [W{week} Bar {bar}] CLOSE LONG -£{pnl:.2f} (held {bars} bars, exit signal: {urgency:.2f})

TRAIL SL:
  [W{week} Bar {bar}] TRAIL SL {old_sl:.2f} → {new_sl:.2f} (locked +£{profit_locked:.2f})

HOLD:
  [W{week} Bar {bar}] HOLD (signal: {direction:.2f}, below threshold 0.3)
```

**Note:** HOLD entries are only logged every 50th bar to avoid spam. OPEN and CLOSE entries are always logged.

**Colors:**
- OPEN LONG: green text
- OPEN SHORT: red text
- CLOSE with profit: bright green
- CLOSE with loss: bright red
- HOLD: dim white
- Prices/numbers: cyan

**Data source:** `TrainingLogger.log_decision()` called from `SpartusTradeEnv.step()` and `_execute_action()`.

**Update frequency:** Real-time (as decisions happen).

---

### 4.10 Alerts & Warnings Panel

A scrolling log (last 6 entries) of notable events and problems.

**Alert types and prefixes:**

| Prefix | Meaning | Color | Triggers |
|--------|---------|-------|----------|
| `[!!]` | Critical | Bright red | Bankruptcy, drawdown > 10%, environment crash |
| `[!]` | Warning | Yellow | Drawdown > 5%, zero trades in a week, lot size auto-reduced, data quality issues |
| `[+]` | Positive | Green | New best win rate, new peak balance, checkpoint saved |
| `[i]` | Info | Dim white | Training phase change, feature removed, config update |

**Alert conditions (auto-generated):**

**Core Trading Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `env.balance <= 0` | `[!!] Week {w}: Bankruptcy — balance reset to £100 (cause: {cause})` | Critical |
| `drawdown > 10%` | `[!!] Week {w}: Drawdown {dd}% — episode ended (reward = -4.0)` | Critical |
| `daily_dd > 3%` | `[!] Week {w} Day {d}: Daily DD {dd}% — positions force-closed` | Warning |
| `drawdown > 5%` | `[!] Week {w}: Drawdown reached {dd}% — lot size auto-reduced` | Warning |
| `week_trades == 0` | `[!] Week {w}: Zero trades — model may be stuck` | Warning |
| `week_trades > 50` | `[!] Week {w}: {n} trades — possible reward hacking` | Warning |
| `win_rate > best_win_rate` | `[+] Week {w}: Best win rate so far ({wr}%) — checkpoint saved` | Positive |
| `balance > peak_balance` | `[+] Week {w}: New peak balance £{bal}` | Positive |
| `zero_volume_bars > 0` | `[!] Week {w}: {n} bars with zero volume (skipped)` | Warning |
| `data_validation_error` | `[!] Week {w}: Data validation: {message}` | Warning |
| `trend_accuracy > 75%` | `[!!] Week {w}: Trend accuracy {ta}% — investigate possible data leakage` | Critical |
| `sharpe > 3.0` | `[!!] Week {w}: Sharpe {s} — investigate possible overfitting` | Critical |
| `win_rate > 80%` | `[!!] Week {w}: Win rate {wr}% — investigate, something is wrong` | Critical |

**TP/SL Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `tp_hit_rate < 15% after week 50` | `[!] Week {w}: TP hit rate only {r}% — TPs may be too aggressive` | Warning |
| `tp_reachable - tp_hit > 30%` | `[!] Week {w}: AI closing too early — TP reachable {r1}% but hit only {r2}%` | Warning |
| `sl_hit_rate > 50%` | `[!] Week {w}: SL hit rate {r}% — entries or SL placement needs work` | Warning |

**SAC Training Health Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `entropy_coef < 0.001` | `[!!] Week {w}: Entropy collapsed (α={val:.4f}) — exploration dead` | Critical |
| `entropy_coef > 10.0` | `[!!] Week {w}: Entropy exploded (α={val:.1f}) — acting randomly` | Critical |
| `policy_entropy < 20% of initial` | `[!!] Week {w}: Policy entropy at {pct}% of initial — premature convergence` | Critical |
| `q_value_mean > 100` | `[!!] Week {w}: Q-value explosion (Q̄={val:.1f}) — critic diverging` | Critical |
| `grad_norm > 10x running_avg` | `[!] Week {w}: Gradient spike (∇={val:.2f}, avg={avg:.2f}) — monitoring` | Warning |
| `grad_clip_rate > 30%` | `[!] Week {w}: Gradient clipping {pct}% of steps — learning may be unstable` | Warning |
| `critic_loss > 10x initial` | `[!] Week {w}: Critic loss diverging ({val:.1f}x initial)` | Warning |

**Anti-Reward-Hacking Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `daily_trades >= 10` | `[!] Week {w}: Daily trade cap hit — conviction threshold raised to 0.6` | Warning |
| `min_hold_blocks > 5 per week` | `[!] Week {w}: {n} min-hold blocks — agent trying rapid open/close` | Warning |
| `daily_trades >= 10 for 3+ days` | `[!!] Week {w}: Persistent trade spamming ({n} days at cap) — investigate reward` | Critical |

**Observation Health Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `any feature std < 0.01` | `[!] Week {w}: Dead features detected: {indices} (std<0.01)` | Warning |
| `any feature std > 3.0` | `[!] Week {w}: Exploding features: {indices} (std>{vals})` | Warning |
| `any feature NaN > 5%` | `[!!] Week {w}: NaN features: {indices} (>{pcts}% NaN) — data corruption` | Critical |

**Reward System Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `reward_clip_rate > 15%` | `[!] Week {w}: Reward clipping at {pct}% — reward scale may be wrong` | Warning |
| `reward_running_std < 0.1` | `[!] Week {w}: Reward variance collapsed (σ={val:.4f}) — normalizer broken` | Warning |
| `reward_running_std > 10.0` | `[!] Week {w}: Reward variance exploded (σ={val:.1f}) — rewards unstable` | Warning |
| `R3 firing > 10% of steps` | `[!] Week {w}: Drawdown penalty firing {pct}% of steps — chronic risk` | Warning |

**Curriculum & Regime Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `stage transition (1→2 or 2→3)` | `[i] Week {w}: Curriculum advanced to Stage {n} ({label})` | Info |
| `any regime < 10% in buffer` | `[!] Week {w}: Regime imbalance — {regime} only {pct}% in buffer` | Warning |
| `any regime < 15% in buffer` | `[i] Week {w}: Regime {regime} at {pct}% (target ≥15%)` | Info |
| `pattern_confidence avg < 30%` | `[i] Week {w}: Low pattern confidence ({pct}%) — need more samples` | Info |

**Ensemble Alerts (when n_agents > 1):**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `agent Sharpe > 30% below best for 50 weeks` | `[!!] Week {w}: Agent {i} dropped — Sharpe {pct}% below best for 50 weeks` | Critical |
| `agent Sharpe > 20% below best` | `[!] Week {w}: Agent {i} Sharpe {pct}% below best — monitor for dropout` | Warning |
| `agreement_rate < 50%` | `[!] Week {w}: Ensemble agreement only {pct}% — agents diverging` | Warning |
| `any agent entropy collapsed` | `[!!] Week {w}: Agent {i} entropy collapsed (α={val:.4f})` | Critical |

**Convergence Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `status == CONVERGED` | `[+] Week {w}: Training converged — use best checkpoint (W{best})` | Positive |
| `status == OVERFITTING` | `[!!] Week {w}: Overfitting detected — Q growing, val Sharpe declining` | Critical |
| `status == COLLAPSED` | `[!!] Week {w}: Policy collapsed — action std < 0.05 for 20 weeks` | Critical |
| `status == PLATEAU after week 80` | `[!] Week {w}: Plateau — win rate 49-51% for {n} weeks` | Warning |

**LSTM Switch Alerts:**

| Condition | Alert | Priority |
|-----------|-------|----------|
| `1 of 3 LSTM criteria met` | `[i] Week {w}: LSTM switch criterion met: {criterion}` | Info |
| `2+ of 3 LSTM criteria met` | `[!] Week {w}: LSTM switch recommended — {n}/3 criteria met` | Warning |

**Data source:** Generated by `TrainingDashboard.check_alerts()` called at end of each week, after each trade, and after each observation health check (every 1000 steps).

**Update frequency:** Event-driven (when alerts fire). SAC alerts check every 100 steps. Observation health every 1000 steps. All others at end of week or on trade events.

---

### 4.11 System Stats Footer

A horizontal bar at the bottom of the Qt window (QHBoxLayout).

| Field | Format | Source |
|-------|--------|--------|
| GPU | `GPU: {name}` or `GPU: None (CPU)` | `torch.cuda.get_device_name()` or fallback |
| RAM | `RAM: {usage:.1f} GB` | `psutil.Process().memory_info().rss / 1e9` |
| Disk | `Disk: {size:.1f} GB` | Total size of `storage/` directory |
| Checkpoint | `Checkpoint: W{week}` | Most recent saved checkpoint week number |

**Update frequency:** Every 30 seconds.

**Colors:** All dim gray (#6c7086). Also includes clickable **Pause** (yellow) and **Stop Training** (red) QPushButtons on the right side.

### 4.12 SAC Internals Panel

> **Why this panel exists:** SAC training can fail silently — Q-value explosion, entropy collapse,
> and critic divergence produce an agent that *appears* to train but learns nothing useful.
> This panel makes SAC's internal health visible at a glance.

| Field | Format | Source | Alert Threshold |
|-------|--------|--------|-----------------|
| Entropy Coef (α) | `α: {value:.4f}` | `model.ent_coef_tensor.item()` | Red if < 0.001 (collapsed) or > 10.0 (exploded) |
| Policy Entropy | `Ent: {value:.3f} ({pct:.0f}% of init)` | Computed from action distribution. `_initial_entropy` captured after first 1000 steps | Yellow if < 40% of initial, Red if < 20% |
| Q-Value Mean | `Q̄: {value:.2f}` | Mean of critic output over recent batch | Red if > 10x max episode return (max ≈ weekly balance change) |
| Q-Value Max | `Q↑: {value:.2f}` | Max of critic output over recent batch | Red if growing exponentially (Q-growth sparkline) |
| Q-Value Growth | `Q↑trend: {sparkline}` | 10-week sparkline of Q̄ | Red if exponentially increasing |
| Actor Grad Norm | `∇π: {value:.3f}/{max:.1f}` | L2 norm vs max_grad_norm (1.0) | Yellow if > 0.8 (near clip), Red if spike > 10x avg |
| Critic Grad Norm | `∇Q: {value:.3f}/{max:.1f}` | L2 norm vs max_grad_norm (1.0) | Yellow if > 0.8 (near clip), Red if spike > 10x avg |
| Actor Loss | `Lπ: {value:.4f}` | Rolling 100-step average | Informational |
| Critic Loss | `LQ: {value:.4f}` | Rolling 100-step average | Red if diverging (>10x initial) |
| Learning Rate | `LR: {phase} {value:.2e}` | Current LR from schedule + phase name | Phase shown: "Warm", "Hold", "Decay" |
| Replay Buffer | `Buf: {pct:.0f}%` | `buffer.size() / buffer.buffer_size * 100` | Informational |

**Reference values:** The panel captures `_initial_entropy` and `_initial_critic_loss` during the first 1000 training steps. These are used as baselines for percentage-based alerts throughout training.

**Update frequency:** Every 100 training steps (matches SpartusCallback logging frequency).

**Layout:** QGroupBox titled "SAC Internals" with QGridLayout of label-value pairs.

```
┌─ SAC Internals ─────────────────────────────┐
│  α: 0.0312      Ent: 1.847 (75% of init)    │
│  Q̄: 4.52        Q↑: 12.31  ▁▂▂▃▃▄ (stable)  │
│  ∇π: 0.234/1.0  ∇Q: 0.456/1.0               │
│  Lπ: -2.341     LQ: 0.892                   │
│  LR: Hold 3.00e-04  |  Buf: 67%             │
└──────────────────────────────────────────────┘
```

**Data source:** The `SpartusCallback` (defined in SPARTUS_TRAINING_METHODOLOGY.md Section 10) logs
all SAC metrics to TensorBoard under the `sac/` prefix. The dashboard reads from the same
callback data.

### 4.13 Convergence Detection Panel

> **Problem:** Without convergence criteria, training just runs for 200 weeks with no way to know
> if it's done, overfitting, or stuck. This panel provides clear signals.

| Signal | Condition | Meaning | Action |
|--------|-----------|---------|--------|
| CONVERGED | 50-week rolling validation Sharpe hasn't improved for 50 weeks AND entropy is stable | Training is complete | Stop training, use best checkpoint |
| OVERFITTING | Q-values still growing BUT validation Sharpe declining for 30+ weeks | Agent is memorizing training data | Stop, rollback to best validation checkpoint |
| COLLAPSED | Action std < 0.05 for 20 consecutive weeks | Policy has collapsed to deterministic | Investigate — may need to restart with higher entropy |
| PLATEAU | Win rate between 49-51% for 40+ weeks after week 80 | Agent stuck at random performance | Try: increase features, change reward weights, add LSTM |
| IMPROVING | Validation Sharpe has positive trend over last 50 weeks | Training is healthy | Continue |

```
┌─ Convergence ────────────────────────────────┐
│  Status: ● IMPROVING                         │
│  Val Sharpe (50w): 0.42 → 0.67  (↑ +0.25)   │
│  Best Checkpoint: Week 142 (Sharpe: 0.71)    │
│  Weeks Since Best: 8                          │
│  Action Std: 0.23 (healthy)                  │
│  Entropy Trend: stable                        │
└──────────────────────────────────────────────┘
```

**Update frequency:** Every week (computed at end of each training week).

**Colors:**
- CONVERGED: Bright green (achievement)
- IMPROVING: Green
- PLATEAU: Yellow
- OVERFITTING: Red (flashing)
- COLLAPSED: Red (flashing)

### 4.14 Reward Breakdown Panel

> **Why this panel exists:** The 5-component composite reward is the primary learning signal.
> If one component dominates, the agent learns a skewed strategy. If normalization is failing,
> all learning is compromised. This panel makes reward health visible.

| Field | Format | Source | Alert Threshold |
|-------|--------|--------|-----------------|
| R1: Position P/L | `R1: {val:+.3f} (w=0.40)` | `env._last_r1` | Informational |
| R2: Trade Quality | `R2: {val:+.3f} (w=0.20)` | `env._last_r2` | Informational |
| R3: Drawdown Penalty | `R3: {val:+.3f} (w=0.15)` | `env._last_r3` | Red if firing frequently (>10% of steps) |
| R4: Diff. Sharpe | `R4: {val:+.3f} (w=0.15)` | `env._last_r4` | Informational |
| R5: Risk-Adj. Bonus | `R5: {val:+.3f} (w=0.10)` | `env._last_r5` | Informational |
| Raw Reward | `Raw: {val:+.4f}` | Weighted sum before normalization | Informational |
| Normalized Reward | `Norm: {val:+.4f}` | After RewardNormalizer | Red if frequently clipped |
| Running Mean | `μ: {val:.4f}` | `reward_normalizer.running_mean` | Yellow if drifting far from 0 |
| Running Std | `σ: {val:.4f}` | `sqrt(reward_normalizer.running_var)` | Red if < 0.1 (rewards collapsed) or > 10.0 (unstable) |
| Clipping Rate | `Clip: {pct:.1f}%` | % of rewards clipped to [-5,+5] last 100 steps | Yellow if >5%, Red if >15% |

**Update frequency:** Every 100 steps (rolling averages of component values).

**Layout:** QGroupBox titled "Reward Breakdown" with label-value grid.

```
┌─ Reward Breakdown ───────────────────────────┐
│  R1 Pos P/L:  +0.012 (0.40)  Raw:  +0.0083  │
│  R2 Quality:  +0.000 (0.20)  Norm: +0.341    │
│  R3 DD Pen:    0.000 (0.15)  μ: 0.0024       │
│  R4 D.Sharpe: +0.008 (0.15)  σ: 0.0243       │
│  R5 Risk-Adj: +0.002 (0.10)  Clip: 2.1%      │
└──────────────────────────────────────────────┘
```

**Data source:** The environment stores `_last_r1` through `_last_r5` each step.
The callback reads these along with `reward_normalizer.running_mean` and `running_var`.

### 4.15 Curriculum & Regime Panel

> **Why this panel exists:** Curriculum learning controls data difficulty, and regime-tagged
> replay ensures the agent doesn't forget rare market conditions. Both are invisible without
> explicit monitoring.

| Field | Format | Source | Alert Threshold |
|-------|--------|--------|-----------------|
| Curriculum Stage | `Stage: {n}/3 ({label})` | Derived from current week vs stage boundaries | Informational |
| Stage Progress | `[████████░░░] 67%` | Week within current stage | Informational |
| Current Week Difficulty | `Diff: {score:.2f}` | `classify_week_difficulty()` score | Informational |
| Current Market Regime | `Regime: {label}` | From regime classifier | Color-coded by regime |
| Buffer: Trending Up | `▲Up: {pct:.0f}%` | `RegimeTaggedReplayBuffer._regime_indices` | Yellow if <15%, Red if <10% |
| Buffer: Trending Down | `▼Dn: {pct:.0f}%` | Same | Yellow if <15%, Red if <10% |
| Buffer: Ranging | `═Rng: {pct:.0f}%` | Same | Yellow if <15%, Red if <10% |
| Buffer: Volatile | `⚡Vol: {pct:.0f}%` | Same | Yellow if <15%, Red if <10% |
| Buffer Total | `Buf: {n}/{max}` | Total replay buffer transitions | Informational |
| Pattern Confidence | `Pat: {avg_cred:.0f}%` | Average Bayesian credibility across active patterns | Yellow if <30% |

**Update frequency:** Stage and regime every step; buffer distribution every 1000 steps; pattern confidence every week.

**Layout:**

```
┌─ Curriculum & Regime ────────────────────────┐
│  Stage: 2/3 (Normal)                         │
│  [████████████████░░░░░░░░░░] 64%  W56/80    │
│  Difficulty: 0.42  |  Regime: ▲ TRENDING UP   │
│                                               │
│  Replay Buffer Regime Mix:                    │
│  ▲Up: 28%  ▼Dn: 24%  ═Rng: 31%  ⚡Vol: 17%   │
│  [OK — all regimes ≥15%]                      │
│  Pattern Confidence: 45% avg (312 patterns)   │
└──────────────────────────────────────────────┘
```

**Regime colors:** Trending Up = Green, Trending Down = Red, Ranging = Yellow, Volatile = Magenta.

**Stage labels:** Stage 1 = "Easy", Stage 2 = "Normal", Stage 3 = "Full Realism".

### 4.16 Ensemble Agents Panel

> **Why this panel exists:** When training 3 SAC agents in parallel, each agent can diverge
> independently — one might collapse, another might overfit. Without per-agent monitoring,
> a failing agent silently corrupts ensemble decisions.
>
> **NOTE:** This panel is ONLY shown when ensemble training is active (n_agents > 1).
> In single-agent mode, this panel is hidden to save screen space.

| Field | Format | Source | Alert Threshold |
|-------|--------|--------|-----------------|
| Agent {i} Val Sharpe | `A{i}: S={val:.2f}` | Per-agent validation Sharpe (last 50 weeks) | Red if >30% below best agent |
| Agent {i} Entropy | `α={val:.3f}` | Per-agent entropy coefficient | Red if collapsed (<0.001) |
| Agent {i} Win Rate | `WR={val:.1f}%` | Per-agent win rate on current week | Informational |
| Agreement Rate | `Agree: {pct:.0f}%` | % of steps where 2+ agents agree on direction | Yellow if <60% |
| Disagreements/Week | `Disagree: {n}` | Steps where no majority vote achieved | Yellow if >40% of trade signals |
| Exit Triggers | `Exits: A1:{n} A2:{n} A3:{n}` | Which agent triggered each exit | Informational |
| Dropout Risk | `{agent}: {status}` | Per-agent dropout eligibility (>30% below best) | Red if dropout imminent |

**Update frequency:** Per-agent metrics every 100 steps; agreement rate per week.

**Layout:**

```
┌─ Ensemble (3 Agents) ────────────────────────┐
│  Agent 1 (seed 42):  S=0.67  α=0.031  WR=55% │
│  Agent 2 (seed 137): S=0.72  α=0.028  WR=57% │
│  Agent 3 (seed 2024):S=0.58  α=0.035  WR=52% │
│                                               │
│  Agreement: 78%  |  Disagree: 142 steps       │
│  Exits by: A1=12  A2=8  A3=15                 │
│  ⚠ Agent 3 Sharpe 19% below best (watch)      │
└──────────────────────────────────────────────┘
```

**Dropout logic:** If an agent's validation Sharpe is >30% below the best agent's Sharpe for 50 consecutive weeks, mark it for dropout. Dashboard shows a warning at 20% gap (yellow) and dropout trigger at 30% (red).

### 4.17 Anti-Hack & Safety Panel

> **Why this panel exists:** Anti-reward-hacking safeguards, observation health, and LSTM
> switch readiness are all safety-critical. They don't fit cleanly into other panels but must
> be visible to catch problems before they compound.

| Field | Format | Source | Alert Threshold |
|-------|--------|--------|-----------------|
| Daily Trades Today | `Today: {n}/10` | Count of trades opened today | Yellow at 8, Red at 10 (threshold raised) |
| Conviction Threshold | `Conv: {val:.1f}` | Current threshold (0.3 normal, 0.6 raised) | Yellow when raised to 0.6 |
| Min Hold Blocks | `Hold Block: {n}` | Attempted exits blocked by 3-bar min hold | Yellow if >5/week |
| Obs Health | `Obs: ✓` or `Obs: ✗ {n} issues` | Count of dead/exploding/NaN features | Green if 0, Red if any |
| Dead Features | `Dead: {list}` | Feature indices with std < 0.01 | Red (list shown) |
| NaN Features | `NaN: {list}` | Feature indices with >5% NaN | Red (list shown) |
| Grad Clip Rate | `Grad Clip: {pct:.0f}%` | % of steps where gradients were clipped | Yellow >10%, Red >30% |
| LR Phase | `LR: {phase} ({val:.1e})` | Current LR schedule phase + value | Informational |
| Noise Active | `Noise: ✓ (σ=0.02)` | Domain randomization status | Green when active |
| LSTM Readiness | `LSTM: {status}` | Check 3 measurable switch criteria | "Not needed" / "Investigate" / "Switch recommended" |

**Update frequency:** Daily trades every step; obs health every 1000 steps; LSTM readiness every week.

**Layout:**

```
┌─ Safety & Anti-Hack ─────────────────────────┐
│  Daily: 6/10 (conv: 0.3)  |  Hold Blocks: 2  │
│  Obs Health: ✓ all features OK                │
│  Grad Clip: 4%  |  LR: Hold (3.00e-04)        │
│  Noise: ✓ (spread±30% slip±50% obs σ=0.02)    │
│  LSTM: Not needed (trend_acc=56%, WR=54%)     │
└──────────────────────────────────────────────┘
```

**LSTM readiness logic:**
- "Not needed": All 3 switch criteria are false
- "Investigate": 1 of 3 criteria met (e.g., trend_accuracy < 52% after week 80)
- "Switch recommended": 2+ of 3 criteria met

---

## 5. Color Scheme

### Base Theme (Dark with Bright Signals)

| Element | Color | Hex |
|---------|-------|-----|
| Background | Deep dark | `#0d1117` |
| Panel surface | Slightly lighter | `#161b22` |
| Panel borders | Subtle border | `#30363d` |
| Panel titles | Bright cyan | `#39d5ff` |
| Labels (left column) | Readable gray | `#8b949e` |
| Values (normal) | Bright white | `#e6edf3` |
| Values (positive/profit) | **Bright green** | `#2dcc2d` |
| Values (negative/loss) | **Bright red** | `#ff3333` |
| Values (warning) | Bright yellow | `#ffcc00` |
| Values (informational) | Light gray | `#b1bac4` |
| Header title | Bright blue | `#58a6ff` |
| Footer labels | Readable gray | `#8b949e` |
| Tab (selected) | Cyan underline | `#39d5ff` |
| Tab (unselected) | Gray text | `#8b949e` |

**Design principle:** Bright green and bright red are the primary signal colors — instantly
visible against the dark background. Gray labels are readable (not too dim). Font sizes are
14-15px for values, 13px for labels. No squinting required.

### Threshold-Based Colors

These apply to metric values and change dynamically:

**Trading Metrics:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Win Rate | > 52% | 48-52% | < 48% |
| Trend Accuracy | > 55% | 50-55% | < 50% |
| Profit Factor | > 1.2 | 1.0-1.2 | < 1.0 |
| Sharpe Ratio | > 0.8 | 0.3-0.8 | < 0.3 |
| Drawdown | < 3% | 3-7% | > 7% |
| Trades/Week | 5-30 | 1-4 or 31-50 | 0 or > 50 |

**SAC Internals:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Entropy Coef (α) | 0.01-1.0 | 0.001-0.01 or 1.0-10.0 | < 0.001 or > 10.0 |
| Policy Entropy (% init) | > 40% | 20-40% | < 20% |
| Q-Value Mean | < 50 | 50-100 | > 100 |
| Actor Grad Norm (vs max) | < 50% of max | 50-80% of max | > 80% (near clip) |
| Critic Grad Norm (vs max) | < 50% of max | 50-80% of max | > 80% (near clip) |
| Gradient Clip Rate | < 5% | 5-30% | > 30% |
| Critic Loss (vs initial) | < 2x initial | 2-10x initial | > 10x initial |
| LR Phase | Hold (steady) | Warm (early) | Decay (late) — informational only |

**Reward System:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Reward Clip Rate | < 5% | 5-15% | > 15% |
| Running Std (σ) | 0.1-5.0 | 0.05-0.1 or 5.0-10.0 | < 0.05 or > 10.0 |
| Running Mean (μ) | -0.5 to +0.5 | -2.0 to -0.5 or +0.5 to +2.0 | < -2.0 or > +2.0 |
| R3 Fire Rate | < 5% | 5-10% | > 10% |

**Curriculum & Regime:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Regime Buffer % (each) | ≥ 15% | 10-15% | < 10% |
| Pattern Confidence | > 50% | 30-50% | < 30% |
| Week Difficulty | — | — | — (informational, no threshold) |

**Ensemble:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Agreement Rate | > 70% | 50-70% | < 50% |
| Agent Sharpe Gap (vs best) | < 15% | 15-30% | > 30% (dropout) |
| Per-Agent Entropy | 0.01-1.0 | 0.001-0.01 | < 0.001 (collapsed) |

**Anti-Hack & Safety:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Daily Trades | < 8 | 8-9 | 10 (cap hit) |
| Min Hold Blocks/Week | 0-2 | 3-5 | > 5 |
| Obs Health | All OK (✓) | — | Any issues (✗) |
| LSTM Criteria Met | 0 (not needed) | 1 (investigate) | 2+ (switch recommended) |

**Convergence:**

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Status | IMPROVING / CONVERGED / STABLE | PLATEAU / WARMING_UP | OVERFITTING / COLLAPSED |
| Action Std | > 0.10 | 0.05-0.10 | < 0.05 |
| Weeks Since Best | < 20 | 20-50 | > 50 |

---

## 6. Controls

| Control | Action | Notes |
|---------|--------|-------|
| **Pause** button | Pause/resume training | Sets `shared_metrics["_paused"]`. Callback blocks in a sleep loop. Dashboard stays live showing PAUSED badge. Button text toggles to "Resume". |
| **Stop Training** button | Graceful quit | Sets `shared_metrics["_quit_requested"]`. Callback returns False, trainer saves checkpoint and breaks loop. |
| Window close (X) | Same as Stop Training | `closeEvent` sets `_quit_requested = True` |
| `Ctrl+C` (terminal) | Emergency quit | Kills training thread. Summary printed to stdout. |

**Implementation:** Qt buttons directly write to the shared_metrics dict. The `SpartusCallback._on_step()` checks `_paused` (blocks in sleep loop) and `_quit_requested` (returns False to stop SB3's `learn()`). The trainer's `run()` loop also checks `_quit_requested` between weeks.

**Paused state display:** When paused, the health badge in the header changes to an orange **PAUSED** badge. The Pause button text changes to **Resume**.

---

## 7. Health Indicators

These are synthesized from multiple metrics and displayed as an overall training health status in the header or progress panel.

### Healthy Training

```
Indicators that training is working:
  - Balance curve trending upward (with natural dips)
  - Win rate slowly increasing over weeks (not monotonically — but trending)
  - Trend accuracy slowly increasing over weeks
  - Memory growing steadily (10-20 trades per week)
  - Drawdown staying below 10%
  - Trade frequency 5-30 per week
  - Lot sizes scaling with account balance
```

### Unhealthy Training

```
Indicators that something is wrong:
  - Balance monotonically decreasing    → Reward signal might be wrong
  - Win rate stuck at 50% after 50 weeks → Features may not be predictive
  - Zero trades for multiple weeks       → Model is stuck (check action thresholds)
  - 50+ trades per week consistently     → Reward hacking / spam
  - Drawdown > 15%                       → Risk rules need tightening
  - Trend accuracy < 48% after 100 weeks → Model is not learning
  - Lot sizes always at minimum          → AI not learning conviction scaling

SAC Internals (silent failures):
  - Entropy coef α < 0.001              → Exploration dead, policy stuck in one mode
  - Entropy coef α > 10.0               → Agent acting randomly, not learning
  - Policy entropy < 20% of initial     → Premature convergence
  - Q-value mean > 100                  → Q-value explosion, critic diverging
  - Q-values growing exponentially      → TD-learning unstable
  - Gradient norms spiking > 10x avg    → Single bad batch dominating updates
  - Action std < 0.05 for 20 weeks      → Policy collapsed to deterministic
```

### Overall Health Badge

Displayed in the progress panel as a colored badge:

| Badge | Condition | Color |
|-------|-----------|-------|
| `LEARNING` | Win rate trending up AND balance trending up | Green |
| `EARLY` | Fewer than 20 weeks completed | Cyan |
| `PLATEAU` | Win rate flat for last 20 weeks | Yellow |
| `STRUGGLING` | Balance declining for last 10 weeks | Red |
| `STUCK` | Zero trades in last 3 weeks | Bright red |
| `PAUSED` | User pressed P | White on dark background |

---

## 8. TensorBoard Integration

The dashboard handles the terminal UI. TensorBoard provides deep drill-down analysis and is launched separately.

### Metrics Logged to TensorBoard

Logged via `SpartusCallback._on_step()` and `_on_rollout_end()`:

**Per-step metrics (every 100 steps):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `spartus/balance` | `env.balance` | Track balance over training steps |
| `spartus/equity` | `env._get_equity()` | Track equity including unrealized P/L |
| `spartus/drawdown` | `env.current_drawdown` | Monitor risk |
| `spartus/reward_raw` | Weighted sum before normalization | Raw reward signal health |
| `spartus/reward_normalized` | After RewardNormalizer | Normalized reward signal |
| `spartus/reward_running_mean` | `reward_normalizer.running_mean` | Normalizer drift |
| `spartus/reward_running_std` | `sqrt(reward_normalizer.running_var)` | Normalizer scale |

**Reward component metrics (every 100 steps):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `reward/r1_position_pnl` | R1 value (rolling avg) | Dense position signal |
| `reward/r2_trade_quality` | R2 value (rolling avg) | Sparse trade quality signal |
| `reward/r3_drawdown_penalty` | R3 value (rolling avg) | Emergency penalty frequency |
| `reward/r4_diff_sharpe` | R4 value (rolling avg) | Sharpe improvement signal |
| `reward/r5_risk_adjusted` | R5 value (rolling avg) | Risk-adjusted bonus |
| `reward/clip_rate` | % of rewards clipped to [-5,+5] | Reward distribution health |
| `reward/r3_fire_rate` | % of steps where R3 < 0 | Chronic drawdown indicator |

**Per-week metrics (end of each week):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `spartus/win_rate` | `env._get_win_rate()` | Learning curve |
| `spartus/trend_accuracy` | `memory.get_trend_accuracy()` | Prediction skill |
| `spartus/trades_per_week` | `len(env.trades_history)` | Activity level |
| `spartus/avg_trade_pnl` | Mean trade P/L | Profit per trade |
| `spartus/memory_size` | `memory.total_trades()` | Experience growth |
| `spartus/sharpe` | Weekly Sharpe calculation | Risk-adjusted return |
| `spartus/profit_factor` | Gross profit / gross loss | Edge quality |
| `spartus/avg_lot_size` | Mean lot size this week | Position sizing evolution |
| `spartus/bankruptcies_total` | Cumulative bankruptcy count | Risk management learning |
| `spartus/tp_hit_rate` | % of trades hitting TP (last 50) | TP placement quality |
| `spartus/tp_reachable_rate` | % of trades where TP was reachable | TP ambition vs reality |
| `spartus/sl_hit_rate` | % of trades hitting SL (last 50) | Entry/SL quality |
| `spartus/avg_sl_trail_profit` | Avg profit locked by SL trailing | Trailing SL effectiveness |

**Curriculum & regime metrics (end of each week):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `curriculum/stage` | 1, 2, or 3 | Current curriculum stage |
| `curriculum/week_difficulty` | `classify_week_difficulty()` score | How hard this week was |
| `regime/dominant` | Encoded: 0=up, 1=down, 2=range, 3=volatile | This week's dominant regime |
| `regime/buffer_trending_up` | % in buffer | Regime balance tracking |
| `regime/buffer_trending_down` | % in buffer | Regime balance tracking |
| `regime/buffer_ranging` | % in buffer | Regime balance tracking |
| `regime/buffer_volatile` | % in buffer | Regime balance tracking |
| `regime/min_representation` | min(all 4 regime %) | Quick imbalance check |
| `regime/pattern_confidence_avg` | Average Bayesian credibility | Pattern memory quality |

**Anti-reward-hacking metrics (end of each week):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `anti_hack/daily_trade_max` | Max trades in any single day this week | Trade spam detection |
| `anti_hack/conviction_raised_days` | Days where threshold was raised to 0.6 | Cap frequency |
| `anti_hack/min_hold_blocks` | Exits blocked by 3-bar minimum | Rapid close attempts |
| `anti_hack/avg_hold_bars` | Average position duration | Trade quality proxy |

**Ensemble metrics (end of each week, when n_agents > 1):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `ensemble/agent_{i}_sharpe` | Per-agent validation Sharpe | Individual agent health |
| `ensemble/agent_{i}_win_rate` | Per-agent win rate | Individual agent skill |
| `ensemble/agent_{i}_entropy` | Per-agent entropy coefficient | Individual exploration |
| `ensemble/agreement_rate` | % of steps with majority vote | Ensemble coherence |
| `ensemble/disagreements` | Steps with no majority | Ensemble divergence |
| `ensemble/sharpe_spread` | max(Sharpe) - min(Sharpe) | Agent dispersion |

**Convergence metrics (end of each week):**

| Metric Key | Value | Purpose |
|------------|-------|---------|
| `convergence/val_sharpe_50w` | 50-week rolling validation Sharpe | Training progress |
| `convergence/action_std` | Average action standard deviation | Policy exploration |
| `convergence/weeks_since_best` | Weeks since best val Sharpe | Stagnation detection |
| `convergence/status` | Encoded: 0=warming, 1=improving, 2=plateau, 3=overfitting, 4=collapsed, 5=converged, 6=stable | Machine-readable status |

**SAC internal metrics (SB3 built-in + custom via SpartusCallback):**

| Metric Key | Purpose | Source |
|------------|---------|--------|
| `train/actor_loss` | Policy network loss | SB3 built-in |
| `train/critic_loss` | Q-network loss | SB3 built-in |
| `train/ent_coef` | Auto-tuned entropy coefficient | SB3 built-in |
| `train/entropy` | Current policy entropy | SB3 built-in |
| `train/learning_rate` | Current learning rate (from schedule) | SB3 built-in |
| `sac/entropy_coef_alpha` | Entropy coefficient α value | Custom callback |
| `sac/policy_entropy_pct` | Policy entropy as % of initial | Custom callback |
| `sac/q_value_mean` | Mean of critic output | Custom callback |
| `sac/q_value_max` | Max of critic output | Custom callback |
| `sac/actor_grad_norm` | L2 norm of actor gradients | Custom callback |
| `sac/critic_grad_norm` | L2 norm of critic gradients | Custom callback |
| `sac/grad_clip_rate` | % of steps where grad was clipped | Custom callback |
| `sac/lr_phase` | Encoded: 0=warm, 1=hold, 2=decay | Custom callback |
| `sac/replay_buffer_pct` | Buffer utilization % | Custom callback |
| `sac/ALERT_gradient_spike` | Fires when gradient > 10x running average | Custom callback |
| `obs_health/dead_feature_{i}` | Feature std < 0.01 (dead/constant) | Custom callback |
| `obs_health/exploding_feature_{i}` | Feature std > 3.0 (normalization broken) | Custom callback |
| `obs_health/nan_feature_{i}` | Feature NaN rate > 5% | Custom callback |
| `obs_health/total_issues` | Count of unhealthy features | Custom callback |

**Domain randomization metrics (every 100 steps):**

| Metric Key | Purpose | Source |
|------------|---------|--------|
| `noise/spread_jitter` | Actual spread jitter applied | Custom callback |
| `noise/slippage_jitter` | Actual slippage jitter applied | Custom callback |
| `noise/commission_jitter` | Actual commission jitter applied | Custom callback |
| `noise/start_offset` | Episode start offset in bars | Custom callback |

### Launching TensorBoard

```bash
tensorboard --logdir=storage/logs/tensorboard/ --port=6006
# Open http://localhost:6006 in browser
```

---

## 9. Log File Formats

All logs are written to `storage/logs/` as JSONL (one JSON object per line) for easy parsing and post-analysis.

### 9.1 `training_log.jsonl` — Step-Level Data

Written every 10th step to balance detail vs file size.

```json
{
    "week": 47,
    "step": 892,
    "global_step": 470892,
    "timestamp": "2026-02-22T14:35:12",
    "balance": 138.50,
    "equity": 139.80,
    "reward": 0.032,
    "reward_normalized": 0.341,
    "reward_components": {
        "r1_position_pnl": 0.012,
        "r2_trade_quality": 0.000,
        "r3_drawdown_penalty": 0.000,
        "r4_diff_sharpe": 0.008,
        "r5_risk_adjusted": 0.002
    },
    "reward_running_mean": 0.0024,
    "reward_running_std": 0.0243,
    "reward_clipped": false,
    "action_direction": 0.72,
    "action_conviction": 0.68,
    "action_exit": 0.12,
    "action_sl_mgmt": 0.35,
    "has_position": true,
    "position_side": "LONG",
    "unrealized_pnl": 1.30,
    "drawdown": 0.046,
    "current_price": 2648.50,
    "regime": "trending_up",
    "curriculum_stage": 2,
    "noise_applied": {
        "spread_jitter": 0.12,
        "slippage_jitter": -0.08,
        "commission_jitter": 0.05,
        "obs_noise_applied": true
    }
}
```

**File size estimate:** ~350 bytes per entry, logged every 10 steps. At 2M total steps: ~70 MB.

### 9.2 `weekly_summary.jsonl` — Week-Level Aggregates

Written once at the end of each completed week.

```json
{
    "week": 47,
    "timestamp": "2026-02-22T15:02:30",
    "start_balance": 135.70,
    "final_balance": 138.50,
    "peak_balance": 145.20,
    "trades": 14,
    "wins": 8,
    "losses": 6,
    "win_rate": 0.571,
    "total_pnl": 2.80,
    "best_trade": 1.20,
    "worst_trade": -0.80,
    "max_drawdown": 0.046,
    "avg_lot_size": 0.02,
    "lot_range_min": 0.01,
    "lot_range_max": 0.03,
    "avg_hold_bars": 12,
    "commission_total": 0.98,
    "trend_accuracy": 0.568,
    "profit_factor": 1.24,
    "sharpe": 0.82,
    "memory_trades_total": 612,
    "patterns_total": 847,
    "bankruptcies_total": 1,
    "tp_hit_rate": 0.357,
    "tp_reachable_rate": 0.643,
    "sl_hit_rate": 0.286,
    "manual_close_rate": 0.357,
    "avg_sl_trail_profit": 0.45,
    "checkpoint_saved": true,
    "checkpoint_path": "storage/models/spartus_week_0047",
    "health_badge": "LEARNING",
    "training_speed_steps_sec": 142,
    "elapsed_seconds": 13320,
    "curriculum_stage": 2,
    "week_difficulty": 0.42,
    "regime_dominant": "trending_up",
    "reward_stats": {
        "avg_raw": 0.0083,
        "avg_normalized": 0.341,
        "running_mean": 0.0024,
        "running_std": 0.0243,
        "clip_rate": 0.021,
        "r1_avg": 0.012,
        "r2_avg": 0.000,
        "r3_avg": 0.000,
        "r4_avg": 0.008,
        "r5_avg": 0.002
    },
    "sac_health": {
        "entropy_coef": 0.0312,
        "policy_entropy": 1.847,
        "entropy_pct_of_initial": 75,
        "q_value_mean": 4.52,
        "q_value_max": 12.31,
        "actor_grad_norm_avg": 0.234,
        "critic_grad_norm_avg": 0.456,
        "grad_clip_rate": 0.04,
        "actor_loss": -2.341,
        "critic_loss": 0.892,
        "learning_rate": 3e-4,
        "lr_phase": "Hold",
        "replay_buffer_pct": 67
    },
    "regime_buffer_distribution": {
        "trending_up": 0.28,
        "trending_down": 0.24,
        "ranging": 0.31,
        "volatile": 0.17
    },
    "anti_hack": {
        "daily_trade_count_max": 8,
        "conviction_threshold_raised_days": 1,
        "min_hold_blocks": 2,
        "avg_pattern_confidence": 0.45
    },
    "obs_health": {
        "dead_features": [],
        "exploding_features": [],
        "nan_features": [],
        "all_healthy": true
    },
    "convergence": {
        "status": "IMPROVING",
        "val_sharpe_50w": 0.67,
        "best_val_sharpe": 0.71,
        "best_val_week": 142,
        "weeks_since_best": 8,
        "action_std": 0.23
    },
    "ensemble": {
        "active": true,
        "agent_sharpes": [0.67, 0.72, 0.58],
        "agent_win_rates": [0.55, 0.57, 0.52],
        "agreement_rate": 0.78,
        "disagreements": 142,
        "exit_triggers": [12, 8, 15],
        "dropout_warnings": ["Agent 3: 19% below best"]
    },
    "lstm_criteria": {
        "trend_acc_low": false,
        "no_upward_trend": false,
        "frame_stack_helps": false,
        "criteria_met": 0,
        "recommendation": "Not needed"
    }
}
```

**File size estimate:** ~1.8 KB per entry, 200 weeks: ~360 KB.

### 9.3 `decisions.jsonl` — AI Decision Log

Written for every OPEN and CLOSE action. HOLD logged every 50th bar only.

```json
{
    "week": 47,
    "step": 892,
    "global_step": 470892,
    "timestamp": "2026-02-22T14:35:12",
    "type": "OPEN_LONG",
    "price": 2647.30,
    "lots": 0.02,
    "conviction": 0.68,
    "direction_signal": 0.72,
    "exit_signal": 0.12,
    "sl": 2642.80,
    "tp": 2653.50,
    "spread_pips": 1.8,
    "slippage_pips": 0.3,
    "commission": 0.14,
    "rsi_at_action": 0.58,
    "trend_dir_at_action": 0.42,
    "session": "london_am",
    "balance_before": 137.65,
    "drawdown_at_action": 0.034,
    "regime_at_action": "trending_up",
    "curriculum_stage": 2,
    "conviction_threshold": 0.3,
    "daily_trade_count": 6,
    "ensemble_votes": {
        "long": 3,
        "short": 0,
        "abstain": 0,
        "agreed_direction": "LONG",
        "per_agent_conviction": [0.72, 0.65, 0.68]
    },
    "noise_at_action": {
        "spread_jitter_applied": 0.08,
        "slippage_jitter_applied": -0.12
    },
    "pattern_win_rate": 0.58,
    "pattern_confidence": 0.42
}
```

**Close entries add:** `pnl`, `duration_bars`, `close_reason` (SL_HIT, TP_HIT, EXIT_SIGNAL, FLIP_DIRECTION, WEEK_END, MIN_HOLD_BLOCK), `exit_triggered_by_agent` (agent index if ensemble exit), `reward_at_close` (breakdown of R1-R5 for the closing step).

**File size estimate:** ~650 bytes per entry, ~15 trades/week, 200 weeks: ~1.9 MB.

### 9.4 `alerts.log` — Human-Readable Alert Log

Plain text, one line per alert. Same content as the Alerts panel.

```
2026-02-22 14:02:30 [!]  Week 46: Drawdown reached 6.2% — lot size auto-reduced
2026-02-22 13:58:10 [+]  Week 45: Best win rate so far (58.3%) — checkpoint saved
2026-02-22 13:45:00 [!]  Week 44: 2 bars with zero volume (skipped)
2026-02-22 12:10:05 [!!] Week 40: Bankruptcy #1 — balance reset to £100 (cause: oversized_positions)
```

---

## 10. Data Flow

How data moves from the training loop into the dashboard and logs:

```
SpartusTradeEnv.step()
    │
    ├──→ Returns: obs, reward, done, truncated, info
    │    info = {
    │      balance, equity, position, trades, win_rate, trend_accuracy,
    │      reward_components: {r1, r2, r3, r4, r5},
    │      reward_raw, reward_normalized, reward_clipped,
    │      reward_running_mean, reward_running_std,
    │      regime, daily_trade_count, conviction_threshold,
    │      min_hold_blocked, noise_applied: {spread, slippage, commission},
    │      curriculum_stage, week_difficulty
    │    }
    │
    ▼
SpartusCallback._on_step()  (stable-baselines3 callback)
    │
    │  ┌─ Every step ─────────────────────────────────────────────────┐
    │  │ Anti-hack tracking: daily trade count, min-hold blocks       │
    │  │ Reward component accumulation for rolling averages           │
    │  └──────────────────────────────────────────────────────────────┘
    │
    │  ┌─ Every 100 steps ────────────────────────────────────────────┐
    │  │ TensorBoard: spartus/* (balance, equity, drawdown, reward)   │
    │  │ TensorBoard: reward/* (r1-r5, clip_rate, r3_fire_rate)       │
    │  │ TensorBoard: noise/* (jitter values)                         │
    │  │ SAC Internals: extract α, entropy, Q-values, grad norms      │
    │  │ TensorBoard: sac/* (all SAC health metrics)                  │
    │  │ Gradient spike detection (compare to 10x running avg)         │
    │  │ LR phase detection (warm/hold/decay from schedule)            │
    │  └──────────────────────────────────────────────────────────────┘
    │
    │  ┌─ Every 1000 steps ───────────────────────────────────────────┐
    │  │ Observation health check: dead/exploding/NaN features         │
    │  │ TensorBoard: obs_health/* (per-feature alerts)                │
    │  │ Regime buffer distribution check                              │
    │  │ TensorBoard: regime/buffer_* (distribution %)                 │
    │  └──────────────────────────────────────────────────────────────┘
    │
    ├──→ TrainingLogger.log_step(week, step, obs, action, reward, info)
    │    Writes to: training_log.jsonl (every 10th step)
    │    Includes: reward components, regime, noise, curriculum stage
    │
    ├──→ TrainingLogger.log_decision(week, step, type, details)
    │    Writes to: decisions.jsonl (on OPEN/CLOSE events)
    │    Includes: ensemble votes, regime, conviction threshold, pattern confidence
    │
    └──→ TrainingDashboard.update(week, step, env_info, agent_info)
         Updates in-memory state for next Rich Live refresh
         │
         ├──→ Updates core panels: progress, account, metrics, this_week, chart
         ├──→ Updates SAC internals panel (α, entropy, Q, gradients, LR, buffer)
         ├──→ Updates reward breakdown panel (R1-R5, μ, σ, clip rate)
         ├──→ Updates curriculum & regime panel (stage, difficulty, buffer mix)
         ├──→ Updates anti-hack panel (daily count, hold blocks, obs health)
         ├──→ Updates ensemble panel (per-agent stats, agreement, exits)
         ├──→ Updates convergence panel (status, val Sharpe, action std)
         ├──→ Checks alert conditions → TrainingDashboard.check_alerts()
         └──→ Rich Live auto-refreshes at 2 Hz

Ensemble flow (when n_agents > 1):
    │
    ├──→ Each agent.predict(obs) independently
    │    ├── Agent 1 (seed 42)  → action_1 = [dir, conv, exit, sl]
    │    ├── Agent 2 (seed 137) → action_2 = [dir, conv, exit, sl]
    │    └── Agent 3 (seed 2024)→ action_3 = [dir, conv, exit, sl]
    │
    ├──→ ensemble_predict() combines:
    │    ├── Direction: majority vote (2/3 agree, else HOLD)
    │    ├── Conviction: average of agreeing agents
    │    ├── Exit: max(exit_signals) — conservative (any agent can exit)
    │    └── SL mgmt: average of agreeing agents
    │
    └──→ Final action → env.step(action) → same flow as above

End of week:
    │
    ├──→ TrainingLogger.log_weekly_summary(week, results)
    │    Writes to: weekly_summary.jsonl
    │    Includes: sac_health, reward_stats, regime_buffer_distribution,
    │    anti_hack, obs_health, convergence, ensemble, lstm_criteria
    │
    ├──→ TrainingDashboard.append_balance(week, balance)
    │    Adds point to balance curve chart
    │
    ├──→ TrainingDashboard.update_baselines()
    │    Recalculates trend arrows (↑/↓) for metrics
    │
    ├──→ ConvergenceDetector.update(week, val_sharpe, entropy, action_std, q_values)
    │    Returns: status (IMPROVING/CONVERGED/OVERFITTING/COLLAPSED/PLATEAU)
    │
    ├──→ Curriculum stage check: advance if week crosses stage boundary
    │    Stage 1→2 at week 31, Stage 2→3 at week 81
    │
    ├──→ TrainingDashboard.check_alerts()
    │    Fires: trading alerts, SAC alerts, reward alerts,
    │    anti-hack alerts, obs health alerts, regime alerts,
    │    ensemble alerts, convergence alerts, LSTM alerts
    │
    └──→ TrainingReportGenerator (conditional):
         ├── Every report_interval weeks (default 10) → generate full report
         ├── On convergence event → generate convergence report
         └── On 'R' keypress → generate manual report
         Reads: weekly_summary.jsonl, decisions.jsonl, alerts.log
         Writes: storage/reports/training_report_W{week}.md
```

---

## 11. Implementation Structure

### File Location

```
src/training/dashboard.py         — TrainingDashboard class: 17 panels, layout, alerts (~600 lines)
src/training/logger.py            — TrainingLogger class: 4 log files, JSONL schemas (~200 lines)
src/training/callback.py          — SpartusCallback class: multi-frequency logging, SAC extraction (~350 lines)
src/training/convergence.py       — ConvergenceDetector class: rolling windows, status signals (~100 lines)
src/training/ascii_chart.py       — AsciiChart class: balance curve renderer (~80 lines)
src/training/report_generator.py  — TrainingReportGenerator: LLM analysis reports (~250 lines)
src/training/health_check.py      — CLI quick health check entry point (~80 lines)
```

### Class Overview

#### `TrainingDashboard` (dashboard.py)

```python
class TrainingDashboard:
    """Real-time Rich terminal dashboard for training monitoring.

    Manages 17 panels across 8 row groups, updating from SpartusCallback
    data at 2 Hz via Rich Live. Handles single-agent and ensemble modes.
    """

    def __init__(self, config, total_weeks, n_agents=1):
        """Initialize layout, panels, state, and convergence detector.

        Args:
            config: Training config (initial_balance, steps_per_week, etc.)
            total_weeks: Total training weeks (200)
            n_agents: Number of ensemble agents (1 = single, 3 = ensemble)
        """

    # --- State Updates ---
    def update(self, week, step, env_info, agent_info, sac_info=None, ensemble_info=None):
        """Update all 17 panel data dicts from latest training step.

        Args:
            env_info: From env step() info dict (balance, reward_components, regime, etc.)
            agent_info: From SpartusCallback (SAC metrics, obs health, etc.)
            sac_info: SAC internals (α, entropy, Q-values, gradients, LR phase)
            ensemble_info: Per-agent metrics (when n_agents > 1)
        """

    def append_balance(self, week, balance):
        """Add a data point to the balance curve."""

    def log_decision(self, week, bar, decision_type, details, ensemble_votes=None):
        """Add an entry to the AI decision log.

        Args:
            ensemble_votes: Dict with {long, short, abstain, per_agent_conviction}
                           when ensemble mode active. Displayed as 'votes:3/0/0 → LONG'
        """

    def log_alert(self, level, message):
        """Add an alert to the alerts panel. Level: '!!', '!', '+', 'i'."""

    # --- Alert Checking ---
    def check_alerts(self, week, env_info, memory, sac_info=None, ensemble_info=None):
        """Evaluate ALL alert conditions (46 total) and fire if triggered.

        Alert categories checked:
          - Core trading (12): bankruptcy, drawdown, zero trades, etc.
          - TP/SL (3): hit rates, early closing
          - SAC health (7): entropy collapse, Q explosion, grad spikes
          - Anti-hack (3): daily cap, hold blocks, persistent spam
          - Obs health (3): dead, exploding, NaN features
          - Reward system (4): clip rate, std collapse/explosion, R3 fire rate
          - Curriculum/regime (4): stage transitions, regime imbalance
          - Ensemble (4): agent dropout, agreement, entropy collapse
          - Convergence (4): converged, overfitting, collapsed, plateau
          - LSTM switch (2): criteria met count
        """

    # --- Rendering ---
    def build_layout(self):
        """Build the full Rich Layout with all panels.

        Layout structure:
          header → progress → middle(account+week | metrics+predictions+tp+chart)
          → sac_row(sac_internals | convergence)
          → reward_regime_row(reward_breakdown | curriculum_regime)
          → safety_ensemble_row(anti_hack | ensemble)  [ensemble hidden if n_agents==1]
          → decisions → alerts → footer
        """

    # --- Core Panels ---
    def _build_progress_panel(self):
        """Progress bar with curriculum stage indicator."""

    def _build_account_panel(self):
        """Account status: balance, peak, drawdown, return, bankruptcies."""

    def _build_metrics_panel(self):
        """Learning metrics: win rate, trend accuracy, Sharpe, profit factor."""

    def _build_predictions_panel(self):
        """Trend prediction tracking with verify cycle."""

    def _build_tp_panel(self):
        """TP/SL accuracy tracking."""

    def _build_week_panel(self):
        """Current week summary: trades, P/L, hold time, commission."""

    def _build_chart_panel(self):
        """ASCII balance curve (last 20 weeks) via AsciiChart."""

    def _build_decisions_panel(self):
        """AI decision log (last 8) with ensemble vote display."""

    def _build_alerts_panel(self):
        """Alerts & warnings (last 6) with color-coded prefixes."""

    def _build_footer(self):
        """System stats: GPU, RAM, disk, checkpoint."""

    # --- v3.3 New Panels ---
    def _build_sac_internals_panel(self):
        """SAC health: α, entropy (% init), Q-values, gradient norms (vs max),
        actor/critic loss, LR phase (Warm/Hold/Decay), replay buffer %."""

    def _build_convergence_panel(self):
        """Convergence detection: status badge, val Sharpe trend, best checkpoint,
        weeks since best, action std, entropy trend."""

    def _build_reward_breakdown_panel(self):
        """5-component reward: R1-R5 values with weights, raw reward, normalized
        reward, running mean (μ), running std (σ), clipping rate."""

    def _build_curriculum_regime_panel(self):
        """Curriculum stage (1-3) with progress bar, week difficulty score,
        current market regime with color, replay buffer regime distribution
        (4 regimes with ≥15% check), pattern confidence (Bayesian avg)."""

    def _build_ensemble_panel(self):
        """Per-agent validation Sharpe/entropy/win rate, agreement rate,
        disagreements, exit triggers by agent, dropout risk warnings.
        Returns None when n_agents == 1 (panel hidden)."""

    def _build_anti_hack_panel(self):
        """Daily trade count/10 with conviction state, min-hold blocks,
        observation health (✓/✗ with details), gradient clip rate,
        LR phase, noise config status, LSTM readiness (3 criteria)."""

    # --- Lifecycle ---
    def start(self, trainer):
        """Start the Live display loop. Called once at training start."""

    def stop(self):
        """Stop the Live display. Save final state."""
```

#### `TrainingLogger` (logger.py)

```python
class TrainingLogger:
    """Persistent JSONL logging for post-analysis.

    Writes 4 log files:
      - training_log.jsonl: Step-level data (every 10th step, ~350 bytes/entry)
      - decisions.jsonl: OPEN/CLOSE actions (~650 bytes/entry)
      - weekly_summary.jsonl: Week aggregates (~1.8 KB/entry)
      - alerts.log: Human-readable alert log
    """

    def __init__(self, log_dir="storage/logs"):
        """Open log file handles."""

    def log_step(self, week, step, global_step, action, reward, info):
        """Write a step entry to training_log.jsonl (every 10th step).
        Includes: reward_components (r1-r5), reward_normalized,
        reward_running_mean/std, regime, curriculum_stage, noise_applied."""

    def log_decision(self, week, step, global_step, decision_type, details,
                     ensemble_votes=None):
        """Write a decision entry to decisions.jsonl.
        Includes: ensemble_votes, regime_at_action, conviction_threshold,
        daily_trade_count, pattern_win_rate, pattern_confidence, noise_at_action.
        Close entries add: pnl, duration_bars, close_reason, exit_triggered_by_agent,
        reward_at_close."""

    def log_weekly_summary(self, week, results, sac_health=None,
                           convergence=None, ensemble=None):
        """Write a week summary to weekly_summary.jsonl.
        Includes: curriculum_stage, week_difficulty, reward_stats,
        sac_health, regime_buffer_distribution, anti_hack, obs_health,
        convergence, ensemble, lstm_criteria."""

    def log_alert(self, level, message):
        """Write an alert to alerts.log."""

    def close(self):
        """Flush and close all file handles."""
```

#### `SpartusCallback` (callback.py)

```python
class SpartusCallback(BaseCallback):
    """Stable-baselines3 callback bridging training to dashboard + logs.

    Handles multi-frequency logging:
      - Every step: anti-hack tracking, reward accumulation
      - Every 100 steps: TensorBoard (spartus/*, reward/*, sac/*, noise/*)
      - Every 1000 steps: obs health check, regime buffer distribution
      - End of week: weekly summary, convergence check, LSTM criteria
    """

    def __init__(self, dashboard, logger, config):
        """Store references to dashboard and logger.
        Initialize running averages, gradient EMA, initial baselines."""
        # self._grad_norm_ema = 1.0
        # self._initial_entropy = None  (captured after first 1000 steps)
        # self._initial_critic_loss = None
        # self._reward_clip_count = 0
        # self._r3_fire_count = 0

    def _on_step(self):
        """Called every training step.

        Every step: accumulate reward components, track anti-hack counters
        Every 100 steps: extract SAC internals, log to TensorBoard + dashboard
        Every 1000 steps: check obs health, regime buffer balance
        """

    def _on_rollout_end(self):
        """Called at end of rollout. Logs aggregate metrics to TensorBoard."""

    # --- SAC Internals Extraction ---
    def _extract_sac_metrics(self):
        """Extract α, entropy, Q-values from model internals.
        Returns dict: {entropy_coef, policy_entropy, q_mean, q_max}"""

    def _compute_grad_norm(self, network):
        """Compute L2 norm of gradients for actor or critic network."""

    def _detect_lr_phase(self):
        """Determine current LR phase (Warm/Hold/Decay) from progress."""

    # --- Health Checks ---
    def _check_observation_health(self):
        """Check all 42 features for dead (std<0.01), exploding (std>3.0),
        NaN (>5%). Returns dict of issues."""

    def _check_gradient_spike(self, current_norm):
        """Compare gradient norm to 10x running EMA. Fire alert if spike."""

    # --- Convergence ---
    def _update_convergence(self, week, val_sharpe):
        """Delegate to ConvergenceDetector. Returns status string."""

    # --- LSTM Criteria ---
    def _check_lstm_criteria(self, week, trend_acc, win_rate):
        """Evaluate 3 measurable LSTM switch criteria.
        Returns: (criteria_met: int, recommendation: str)"""
```

#### `ConvergenceDetector` (callback.py or convergence.py)

```python
class ConvergenceDetector:
    """Tracks training convergence signals over time.

    Maintains rolling windows of validation Sharpe, action std, and entropy
    to determine if training is IMPROVING, CONVERGED, OVERFITTING, COLLAPSED,
    or in PLATEAU.
    """

    def __init__(self, window=50, patience=50):
        """Initialize rolling buffers."""

    def update(self, week, val_sharpe, action_std, entropy, q_mean):
        """Add week's metrics and recompute status."""

    def get_status(self, current_week):
        """Return current convergence status and supporting data.
        Returns: {status, val_sharpe_trend, best_week, weeks_since_best, action_std}
        """
```

#### `AsciiChart` (ascii_chart.py)

```python
class AsciiChart:
    """Renders a simple ASCII line chart for the balance curve."""

    def __init__(self, width=45, height=6):
        """Set chart dimensions."""

    def render(self, values, labels=None):
        """Render a list of float values as an ASCII chart string."""
        # Returns a multi-line string using box-drawing characters

    def _scale_values(self, values):
        """Map values to row indices within the chart height."""

    def _draw_line(self, scaled):
        """Connect points with line-drawing characters."""
```

---

## 12. Integration with Training Loop

The dashboard plugs into the training loop via the callback and direct calls:

```python
def train_spartus(config):
    """Main training loop with full v3.3 dashboard integration.

    Supports: curriculum learning, ensemble training, convergence detection,
    regime-tagged replay, anti-reward-hacking, observation health monitoring.
    """

    # Initialize components
    n_agents = config.ensemble.get("n_agents", 1)
    dashboard = TrainingDashboard(config, total_weeks=len(train_weeks), n_agents=n_agents)
    logger = TrainingLogger(config.log_dir)
    convergence = ConvergenceDetector(window=50, patience=50)
    callback = SpartusCallback(dashboard, logger, config, convergence)

    # Initialize agents (single or ensemble)
    agents = []
    for i in range(n_agents):
        agent = SAC("MlpPolicy", env=None, learning_rate=lr_schedule,
                     seed=config.ensemble["seeds"][i], max_grad_norm=1.0, ...)
        agents.append(agent)

    # Curriculum: sort weeks by difficulty for stages 1-2
    easy_weeks, normal_weeks, full_weeks = build_curriculum(train_weeks, config)

    # Start Rich Live display
    with Live(dashboard.build_layout(), refresh_per_second=2, console=Console()) as live:
        for week_idx, week_data in enumerate(curriculum_ordered_weeks):
            # Determine curriculum stage
            stage = get_curriculum_stage(week_idx)

            # Create environment with domain randomization
            env = SpartusTradeEnv(week_data, config, memory,
                                  noise_config=config.noise_config)
            for agent in agents:
                agent.set_env(env)

            # Train each agent (callback handles dashboard + logging)
            for agent in agents:
                agent.learn(
                    total_timesteps=config.steps_per_week,
                    reset_num_timesteps=False,
                    callback=callback
                )

            # End of week: collect results
            results = collect_week_results(env, memory)
            sac_health = callback.get_sac_health()
            obs_health = callback.get_obs_health()

            # End of week: ensemble metrics
            ensemble_info = None
            if n_agents > 1:
                ensemble_info = collect_ensemble_metrics(agents, env)

            # End of week: convergence check
            val_sharpe = compute_validation_sharpe(agents, val_data)
            conv_status = convergence.update(week_idx, val_sharpe,
                                              sac_health["action_std"],
                                              sac_health["policy_entropy"],
                                              sac_health["q_value_mean"])

            # End of week: LSTM criteria check
            lstm_criteria = callback.check_lstm_criteria(
                week_idx, results["trend_accuracy"], results["win_rate"])

            # Log everything
            logger.log_weekly_summary(week_idx, results,
                                       sac_health=sac_health,
                                       convergence=conv_status,
                                       ensemble=ensemble_info)
            dashboard.append_balance(week_idx, env.balance)
            dashboard.check_alerts(week_idx, results, memory,
                                    sac_info=sac_health,
                                    ensemble_info=ensemble_info)

            # Refresh display
            live.update(dashboard.build_layout())

            # Convergence: auto-stop if converged or overfitting
            if conv_status["status"] in ("CONVERGED", "OVERFITTING"):
                logger.log_alert("!!", f"Training stopped: {conv_status['status']}")
                break

    # Cleanup
    logger.close()
    dashboard.stop()
```

---

## 13. Build Checklist

Implementation order for building the dashboard:

```
Phase A: Logging Foundation
  [ ] Create src/training/logger.py with TrainingLogger class
  [ ] Implement log_step() writing to training_log.jsonl
      - Include reward_components (r1-r5), reward_normalized, reward_running_mean/std
      - Include regime, curriculum_stage, noise_applied
  [ ] Implement log_decision() writing to decisions.jsonl
      - Include ensemble_votes, regime_at_action, conviction_threshold
      - Include pattern_win_rate, pattern_confidence, noise_at_action
  [ ] Implement log_weekly_summary() writing to weekly_summary.jsonl
      - Include sac_health, reward_stats, regime_buffer_distribution
      - Include anti_hack, obs_health, convergence, ensemble, lstm_criteria
  [ ] Implement log_alert() writing to alerts.log
  [ ] Create storage/logs/ directory structure
  [ ] Test: verify JSONL output format matches Section 9 schemas exactly

Phase B: Callback — Core
  [ ] Create src/training/callback.py with SpartusCallback
  [ ] Implement _on_step() routing to logger and TensorBoard
  [ ] Implement _on_rollout_end() for aggregate metrics
  [ ] Capture _initial_entropy and _initial_critic_loss during first 1000 steps
  [ ] Test: verify callback fires correctly during SAC training

Phase B2: Callback — SAC Internals Extraction
  [ ] Implement _extract_entropy_coef() from model.ent_coef_tensor
  [ ] Implement _compute_grad_norm(network) for actor and critic
  [ ] Implement gradient spike detection (10x running average alert)
  [ ] Implement LR phase detection (warm/hold/decay from progress)
  [ ] Log all sac/* metrics to TensorBoard every 100 steps
  [ ] Test: verify SAC metric extraction produces valid values

Phase B3: Callback — Reward Component Logging
  [ ] Read env._last_r1 through env._last_r5 each step
  [ ] Compute rolling averages of each reward component
  [ ] Track reward clipping rate (% clipped to [-5,+5])
  [ ] Track R3 fire rate (% of steps where drawdown penalty < 0)
  [ ] Log all reward/* metrics to TensorBoard every 100 steps
  [ ] Test: verify reward component logging matches env output

Phase B4: Callback — Observation Health
  [ ] Implement _check_observation_health() (every 1000 steps)
  [ ] Check each of 42 features: std < 0.01 (dead), std > 3.0 (exploding), NaN > 5%
  [ ] Log obs_health/* alerts to TensorBoard
  [ ] Fire dashboard alerts when issues detected
  [ ] Test: inject synthetic dead/NaN features and verify detection

Phase B5: Callback — Regime & Curriculum Tracking
  [ ] Read regime classification from env each step
  [ ] Track regime buffer distribution from RegimeTaggedReplayBuffer
  [ ] Log curriculum/* and regime/* metrics to TensorBoard
  [ ] Fire alerts on stage transitions and regime imbalance
  [ ] Test: verify regime tracking across stage transitions

Phase B6: Callback — Anti-Reward-Hacking Tracking
  [ ] Track daily_trade_count, conviction_threshold_raised_days
  [ ] Track min_hold_blocks (exits blocked by 3-bar rule)
  [ ] Log anti_hack/* metrics to TensorBoard
  [ ] Fire alerts when daily cap hit or persistent spamming
  [ ] Test: simulate rapid trading and verify cap activation

Phase B7: Callback — Ensemble Metrics (conditional)
  [ ] Implement per-agent metric extraction (Sharpe, entropy, win rate)
  [ ] Track agreement rate and disagreement count
  [ ] Track exit triggers per agent
  [ ] Implement dropout detection (>30% below best for 50 weeks)
  [ ] Log ensemble/* metrics to TensorBoard
  [ ] Test: verify ensemble metrics with 3 mock agents

Phase B8: Callback — Convergence Detection
  [ ] Implement ConvergenceDetector class (from methodology spec)
  [ ] Track 50-week rolling validation Sharpe
  [ ] Track action std, entropy trend, Q-value growth
  [ ] Return status: CONVERGED/OVERFITTING/COLLAPSED/PLATEAU/IMPROVING
  [ ] Log convergence/* metrics to TensorBoard
  [ ] Test: simulate various convergence scenarios

Phase C: ASCII Chart
  [ ] Create src/training/ascii_chart.py with AsciiChart class
  [ ] Implement render() with box-drawing characters
  [ ] Test: verify chart renders correctly for sample data

Phase D: Dashboard Panels — Core
  [ ] Create src/training/dashboard.py with TrainingDashboard class
  [ ] Implement _build_progress_panel() (with curriculum stage in progress bar)
  [ ] Implement _build_account_panel()
  [ ] Implement _build_metrics_panel()
  [ ] Implement _build_predictions_panel()
  [ ] Implement _build_tp_panel()
  [ ] Implement _build_week_panel()
  [ ] Implement _build_chart_panel() (uses AsciiChart)
  [ ] Implement _build_decisions_panel() (with ensemble vote display)
  [ ] Implement _build_alerts_panel() (with all 60+ alert conditions)
  [ ] Implement _build_footer()
  [ ] Test: verify core panels render correctly

Phase D2: Dashboard Panels — v3.3 New Panels
  [ ] Implement _build_sac_internals_panel()
      - α, entropy (% of init), Q-values, gradient norms (vs max), losses, LR phase, buffer %
  [ ] Implement _build_convergence_panel()
      - Status badge, val Sharpe trend, best checkpoint, action std, entropy trend
  [ ] Implement _build_reward_breakdown_panel()
      - R1-R5 values with weights, raw/normalized, μ, σ, clip rate
  [ ] Implement _build_curriculum_regime_panel()
      - Stage indicator, progress bar, difficulty, regime, buffer distribution, pattern confidence
  [ ] Implement _build_ensemble_panel()
      - Per-agent stats, agreement rate, exits by agent, dropout warnings
      - Conditional display: hidden when n_agents == 1
  [ ] Implement _build_anti_hack_panel()
      - Daily trades/10, conviction state, hold blocks, obs health, grad clip, LR, noise, LSTM
  [ ] Implement build_layout() composing all 17 panels with proper row arrangement
  [ ] Test: verify all panels render, ensemble panel hides in single-agent mode

Phase E: State Management & Alerts
  [ ] Implement update() method routing data to all 17 panels
  [ ] Implement append_balance() for chart data
  [ ] Implement log_decision() with ensemble vote formatting
  [ ] Implement log_alert() for all alert types
  [ ] Implement check_alerts() with ALL alert conditions from Section 4.10:
      - Core trading alerts (12 conditions)
      - TP/SL alerts (3 conditions)
      - SAC training health alerts (7 conditions)
      - Anti-reward-hacking alerts (3 conditions)
      - Observation health alerts (3 conditions)
      - Reward system alerts (4 conditions)
      - Curriculum & regime alerts (4 conditions)
      - Ensemble alerts (4 conditions)
      - Convergence alerts (4 conditions)
      - LSTM switch alerts (2 conditions)
  [ ] Test: verify all 46 alert conditions fire correctly

Phase F: Integration
  [ ] Implement start() and stop() lifecycle methods
  [ ] Implement keyboard handlers (P for pause, Q for quit)
  [ ] Integrate with training loop (wrap in Rich Live context)
  [ ] Integrate ensemble flow: ensemble_predict() → action → env.step()
  [ ] Integrate ConvergenceDetector into end-of-week flow
  [ ] Integrate curriculum stage advancement at week boundaries
  [ ] Test: run dashboard with mock training data end-to-end
  [ ] Test: verify TensorBoard logs contain all metric categories
  [ ] Test: verify JSONL logs match schemas from Section 9

Phase G: Polish
  [ ] Apply base color scheme from Section 5
  [ ] Apply threshold-based coloring to ALL metrics (8 category tables from Section 5)
  [ ] Add health badge to progress panel
  [ ] Apply regime-specific colors (green/red/yellow/magenta)
  [ ] Apply convergence status colors (green/yellow/red with flashing)
  [ ] Verify terminal width handling (graceful degradation on narrow terminals)
  [ ] Verify ensemble panel hide/show logic
  [ ] Test: full training run with dashboard active, all 17 panels updating
```

---

## 14. LLM Training Analysis System

> **Why this system exists:** The dashboard is designed for a human to watch in real-time. But the human may not know
> whether what they see is healthy or problematic. The LLM (Claude) that designed the Spartus system understands every
> metric, threshold, and failure mode — it's the best diagnostician available. This system gives Claude access to
> training data so it can analyze, diagnose, and recommend corrections.
>
> **Two access modes:**
> 1. **Report mode:** Training auto-generates structured markdown reports. User hands them to Claude for analysis.
> 2. **Direct mode:** User pauses training and asks Claude to read the JSONL logs directly within a session.

### 14.1 Training Report Generator

**File:** `src/training/report_generator.py` (~250 lines)

**Input sources:**
- `storage/logs/weekly_summary.jsonl` — Primary data (has everything)
- `storage/logs/training_log.jsonl` — Step-level detail when needed
- `storage/logs/decisions.jsonl` — Trade-level analysis
- `storage/logs/alerts.log` — Alert history

**Output:** `storage/reports/training_report_W{week:04d}.md`

**Design principle:** Every metric includes its threshold context inline. Instead of:
```
win_rate: 51.2%
```
The report outputs:
```
win_rate: 51.2% [YELLOW — GREEN >52%, YELLOW 48-52%, RED <48%]
```
This lets the LLM immediately assess whether a value is concerning without looking up thresholds separately.

### 14.2 Report Format Specification

Below is the exact template the report generator produces. Example values shown for Week 47.

````markdown
# SPARTUS TRAINING REPORT — Week 47/200

**Generated:** 2026-02-22 15:02:30 | **Trigger:** auto (10-week interval)
**Report version:** 1.0 | **System version:** v3.3

---

## 1. EXECUTIVE SUMMARY

| Field | Value |
|-------|-------|
| Week | 47 / 200 (23.5%) |
| Curriculum Stage | 2/3 (Normal) — weeks 31-80 |
| Health Badge | LEARNING |
| Convergence | IMPROVING |
| Top Concern | Gradient clip rate at 12% [YELLOW] |

**One-line assessment:** Training is progressing normally. Win rate and Sharpe trending up. Minor gradient instability detected but within acceptable bounds.

---

## 2. METRICS SNAPSHOT (Current Values)

| Metric | Value | Trend (10w) | Status | Thresholds |
|--------|-------|-------------|--------|------------|
| Win Rate | 54.2% | ↑ +3.1% | GREEN | G >52%, Y 48-52%, R <48% |
| Trend Accuracy | 56.8% | ↑ +2.4% | GREEN | G >55%, Y 50-55%, R <50% |
| Sharpe Ratio | 0.82 | ↑ +0.15 | GREEN | G >0.8, Y 0.3-0.8, R <0.3 |
| Profit Factor | 1.24 | ↑ +0.08 | GREEN | G >1.2, Y 1.0-1.2, R <1.0 |
| Drawdown (max) | 4.6% | → flat | GREEN | G <3%, Y 3-7%, R >7% |
| Trades/Week (avg) | 14 | → stable | GREEN | G 5-30, Y 1-4/31-50, R 0/>50 |
| Balance | £138.50 | ↑ +£12.30 | GREEN | Positive = green |
| Bankruptcies | 1 | — | YELLOW | G 0, Y 1-2, R 3+ |

---

## 3. LEARNING CURVE (Last 20 Weeks)

| Week | Win Rate | Trend Acc | Sharpe | Balance | Trades | Drawdown |
|------|----------|-----------|--------|---------|--------|----------|
| 28 | 48.1% | 50.2% | 0.21 | £106.20 | 11 | 3.2% |
| 29 | 49.3% | 51.0% | 0.28 | £108.40 | 13 | 2.8% |
| 30 | 48.7% | 50.8% | 0.25 | £107.10 | 12 | 3.5% |
| ... | | | | | | |
| 45 | 53.8% | 55.2% | 0.76 | £135.70 | 15 | 5.1% |
| 46 | 52.1% | 56.1% | 0.79 | £135.70 | 12 | 6.2% |
| 47 | 54.2% | 56.8% | 0.82 | £138.50 | 14 | 4.6% |

**Trend analysis:**
- Win rate: Positive trend (+3.1% over 10 weeks). Consistent improvement, not noise.
- Sharpe: Positive trend (+0.15 over 10 weeks). Risk-adjusted performance improving.
- Balance: Net upward with healthy dips. No monotonic increase (good — not overfitting).
- Drawdown: Peaked at 6.2% in W46, recovered. Within normal range.

---

## 4. SAC HEALTH ASSESSMENT

| Metric | Value | Status | Thresholds |
|--------|-------|--------|------------|
| Entropy Coef (α) | 0.0312 | GREEN | G 0.01-1.0, Y 0.001-0.01/1.0-10.0, R <0.001/>10.0 |
| Policy Entropy | 1.847 (75% of init) | GREEN | G >40%, Y 20-40%, R <20% |
| Q-Value Mean | 4.52 | GREEN | G <50, Y 50-100, R >100 |
| Q-Value Max | 12.31 | GREEN | Not exponentially growing |
| Actor Grad Norm | 0.234 / 1.0 | GREEN | G <50%, Y 50-80%, R >80% of max |
| Critic Grad Norm | 0.456 / 1.0 | GREEN | G <50%, Y 50-80%, R >80% of max |
| Grad Clip Rate | 12% | YELLOW | G <5%, Y 5-30%, R >30% |
| Actor Loss | -2.341 | INFO | Rolling 100-step average |
| Critic Loss | 0.892 | GREEN | G <2x init, Y 2-10x, R >10x init |
| Learning Rate | 3.00e-04 (Hold phase) | INFO | Warm → Hold → Decay |
| Replay Buffer | 67% full | INFO | — |

**SAC verdict:** Healthy. All core metrics within normal bounds. Gradient clipping at 12% is elevated but below the 30% danger threshold — likely caused by volatile training weeks in Stage 2. Monitor for escalation.

---

## 5. REWARD SYSTEM HEALTH

| Component | Avg Value (10w) | Weight | Status |
|-----------|----------------|--------|--------|
| R1: Position P/L | +0.012 | 0.40 | Active — primary learning signal |
| R2: Trade Quality | +0.003 | 0.20 | Sparse (fires on trade close only) |
| R3: Drawdown Penalty | -0.001 | 0.15 | Firing 3% of steps [GREEN, threshold R >10%] |
| R4: Diff. Sharpe | +0.008 | 0.15 | Active — Sharpe improving |
| R5: Risk-Adj. Bonus | +0.002 | 0.10 | Active — low-DD trades rewarded |

| Normalizer | Value | Status | Thresholds |
|-----------|-------|--------|------------|
| Running Mean (μ) | 0.0024 | GREEN | G -0.5 to +0.5, Y -2 to +2, R outside |
| Running Std (σ) | 0.0243 | GREEN | G 0.1-5.0, Y edges, R <0.05/>10.0 |
| Clip Rate | 2.1% | GREEN | G <5%, Y 5-15%, R >15% |

**Reward verdict:** Healthy. All 5 components active and contributing. Normalizer stable. R3 not over-firing. No single component dominating.

---

## 6. TRADING BEHAVIOR ANALYSIS

| Metric | Value | Context |
|--------|-------|---------|
| Avg trades/week (10w) | 14.2 | Normal range (5-30) |
| Avg hold duration | 12.3 bars (61.5 min) | Healthy — above 3-bar minimum |
| Win avg P/L | +£0.85 | — |
| Loss avg P/L | -£0.52 | Win/loss ratio: 1.63 (good — winners > losers) |
| TP hit rate | 35.7% | [GREEN] |
| TP reachable rate | 64.3% | Gap: 28.6% — AI may be closing winners slightly early |
| SL hit rate | 28.6% | [GREEN, threshold R >50%] |
| Exit signal rate | 35.7% | AI actively managing exits |
| Daily trade cap hits | 1 day in last 10 weeks | Minimal — not spam trading |
| Min-hold blocks | 2 in last 10 weeks | Minimal — not trying rapid close |

**Regime performance breakdown (last 20 weeks):**

| Regime | Win Rate | Trades | Avg P/L | Notes |
|--------|----------|--------|---------|-------|
| Trending Up | 61.2% | 48 | +£0.95 | Best regime — expected |
| Trending Down | 52.3% | 35 | +£0.42 | Decent — learning shorts |
| Ranging | 47.8% | 42 | -£0.12 | Weakest — ranging is hardest |
| Volatile | 50.0% | 18 | +£0.28 | Small sample — more data needed |

---

## 7. REGIME & CURRICULUM STATUS

| Field | Value | Status |
|-------|-------|--------|
| Curriculum Stage | 2/3 (Normal) | Weeks 31-80 |
| Stage Progress | 64% (Week 56 of 80) | — |
| Week Difficulty | 0.42 (moderate) | — |
| Current Regime | Trending Up | — |

**Replay buffer regime distribution:**

| Regime | Buffer % | Status | Threshold |
|--------|----------|--------|-----------|
| Trending Up | 28% | GREEN | ≥15% required |
| Trending Down | 24% | GREEN | ≥15% required |
| Ranging | 31% | GREEN | ≥15% required |
| Volatile | 17% | GREEN | ≥15% required |

**Pattern memory:** 312 patterns, average Bayesian confidence 45% [YELLOW, threshold G >50%]. Confidence will naturally increase as more samples accumulate.

---

## 8. ENSEMBLE STATUS

| Agent | Seed | Val Sharpe | Entropy (α) | Win Rate | Gap vs Best |
|-------|------|-----------|-------------|----------|-------------|
| A1 | 42 | 0.67 | 0.031 | 55% | -7% |
| A2 | 137 | 0.72 | 0.028 | 57% | best |
| A3 | 2024 | 0.58 | 0.035 | 52% | -19% [YELLOW] |

| Metric | Value | Status |
|--------|-------|--------|
| Agreement Rate | 78% | GREEN (threshold Y <70%, R <50%) |
| Disagreements/Week | 142 steps | — |
| Exit Triggers | A1:12, A2:8, A3:15 | A3 triggering most exits |
| Dropout Risk | A3: 19% below best (watch) | YELLOW at 20%, RED at 30% |

**Ensemble verdict:** Agent 3 (seed 2024) underperforming. Gap at 19% — approaching 20% warning threshold. Not yet at 30% dropout level. Monitor for next 10 weeks.

---

## 9. OBSERVATION HEALTH

| Check | Result | Details |
|-------|--------|---------|
| Dead features (std < 0.01) | PASS | 0 features |
| Exploding features (std > 3.0) | PASS | 0 features |
| NaN features (>5% NaN) | PASS | 0 features |

**All 42 features healthy.**

---

## 10. ALERT HISTORY (Last 10 Weeks)

| Week | Alert | Priority |
|------|-------|----------|
| 47 | Gradient clip rate at 12% — monitoring | Warning |
| 46 | Drawdown reached 6.2% — lot size auto-reduced | Warning |
| 45 | Best win rate so far (58.3%) — checkpoint saved | Positive |
| 44 | 2 bars with zero volume (skipped) | Warning |
| 43 | Gradient spike detected (∇π=4.2, 10x avg) | Warning |
| 40 | Curriculum advanced to Stage 2 (Normal) | Info |

**Alert summary:** 4 warnings, 1 positive, 1 info. No critical alerts in last 10 weeks. Gradient-related warnings are the most frequent — consistent with Stage 2 introducing harder market data.

---

## 11. ANOMALY FLAGS

Automated checks against all system thresholds. Only flags that are YELLOW or RED are listed.

| # | Anomaly | Severity | Details |
|---|---------|----------|---------|
| 1 | Grad clip rate elevated | YELLOW | 12% (threshold: GREEN <5%) |
| 2 | Ensemble Agent 3 lagging | YELLOW | 19% below best (threshold: YELLOW at 15%) |
| 3 | Pattern confidence low | YELLOW | 45% avg (threshold: GREEN >50%) |
| 4 | TP gap elevated | INFO | TP reachable 64% but hit 36% — gap of 28% suggests early closing |

**No RED anomalies detected.**

---

## 12. RECOMMENDED ACTIONS

Based on anomaly flags and training state:

| Priority | Recommendation | Rationale |
|----------|---------------|-----------|
| Low | Monitor gradient clip rate over next 5 weeks | Currently 12% — if it rises above 20%, consider reducing learning rate or checking for data quality issues |
| Low | Monitor Agent 3 (seed 2024) | If gap exceeds 30% for 50 consecutive weeks, it will be auto-dropped from ensemble |
| None | No code changes recommended at this time | Training is progressing normally with positive trends in all key metrics |

---

*Report generated by TrainingReportGenerator v1.0*
*System: Spartus Trading AI v3.3*
*Next auto-report: Week 50*
````

### 14.3 Quick Health Check

For fast mid-session check-ins, a lightweight alternative to the full report.

**File:** `src/training/health_check.py` (~80 lines)

**Usage:**
```bash
python -m src.training.health_check
# or
python -m src.training.health_check --week 47
```

**Output format** (compact, ~40 lines, designed to be pasted directly into a Claude conversation):

```
═══════════════════════════════════════════════════════
 SPARTUS HEALTH CHECK — Week 47/200 — Stage 2 (Normal)
═══════════════════════════════════════════════════════
 Status: LEARNING | Convergence: IMPROVING
───────────────────────────────────────────────────────
 METRIC          VALUE     TREND   STATUS
 Win Rate        54.2%     ↑ +3.1  GREEN
 Trend Accuracy  56.8%     ↑ +2.4  GREEN
 Sharpe          0.82      ↑ +0.15 GREEN
 Balance         £138.50   ↑       GREEN
 Entropy (α)     0.0312    →       GREEN
 Q-Value Mean    4.52      →       GREEN
 Grad Clip %     12%       ↑       YELLOW ⚠
 Reward Clip %   2.1%      →       GREEN
 Action Std      0.23      →       GREEN
───────────────────────────────────────────────────────
 ALERTS (last 3):
 [!] W47: Grad clip rate 12%
 [!] W46: Drawdown 6.2%
 [+] W45: Best win rate 58.3%
───────────────────────────────────────────────────────
 ENSEMBLE: A1=0.67 A2=0.72 A3=0.58 | Agree:78%
 REGIME: ▲28% ▼24% ═31% ⚡17% [OK]
 OBS HEALTH: ✓ all 42 features OK
───────────────────────────────────────────────────────
 RECOMMENDATION: Continue training. Monitor grad clip rate.
═══════════════════════════════════════════════════════
```

**When to use which:**
- **Health Check:** Quick "is everything OK?" — takes 2 seconds
- **Full Report:** Deep dive when something looks off, or periodic review — takes ~30 seconds to generate

### 14.4 Auto-Generation Schedule

Reports are generated automatically at these triggers:

| Trigger | What Generates | File |
|---------|---------------|------|
| Every 10 weeks (configurable: `report_interval`) | Full report | `training_report_W{week}.md` |
| Convergence event (CONVERGED, OVERFITTING, COLLAPSED) | Full report | `training_report_W{week}_CONVERGED.md` (etc.) |
| End of training (final week) | Full report | `training_report_FINAL.md` |
| User presses `R` key | Full report | `training_report_W{week}_manual.md` |
| User runs health_check.py | Health check | Printed to stdout (not saved) |

**Config:**
```python
REPORT_CONFIG = {
    "report_interval": 10,        # Generate full report every N weeks
    "report_dir": "storage/reports",
    "auto_on_convergence": True,   # Auto-report on convergence events
    "auto_on_finish": True,        # Auto-report when training ends
    "include_learning_curve": 20,  # Last N weeks in learning curve table
    "include_alert_history": 10,   # Last N weeks of alerts
}
```

### 14.5 Direct Log Reading Protocol

When Claude is in an active session and the user wants a live diagnosis:

**Workflow:**

```
User:  [Presses P to pause training]
User:  "Hey Claude, check how the training is going"

Claude reads (in this order):
  1. storage/logs/weekly_summary.jsonl  (last 10 entries — ~18 KB)
  2. storage/logs/alerts.log            (last 30 lines — ~2 KB)
  3. storage/reports/training_report_W{latest}.md  (if exists — most recent auto-report)

Claude analyzes and responds with:
  - Overall assessment (healthy/concerning/critical)
  - Specific findings with evidence
  - Recommended actions (if any)
  - Whether to continue, pause, or stop training

User:  [Presses P to resume, or follows Claude's recommendations]
```

**Why this works:**
- `weekly_summary.jsonl` contains ALL metrics in every entry (SAC health, reward stats, convergence, ensemble, obs health, etc.)
- JSONL files are append-only — safe to read while training is paused
- Claude can parse JSON natively and cross-reference thresholds from memory
- The weekly summary schema (Section 9.2) was specifically designed with enough context for LLM analysis

**What Claude looks for when reading logs:**

1. **Trend direction** — Are win rate, Sharpe, trend accuracy going up over last 10-20 weeks?
2. **SAC stability** — Is entropy stable? Are Q-values bounded? Any gradient spikes?
3. **Reward health** — Is the normalizer stable? Is any component dominating? Is R3 over-firing?
4. **Convergence signals** — Is training converging, overfitting, or stuck?
5. **Ensemble coherence** — Are agents agreeing? Any dropout candidates?
6. **Observation quality** — Any dead, exploding, or NaN features?
7. **Anti-hack triggers** — Is the agent hitting trade caps or hold blocks frequently?
8. **Alert patterns** — Are the same warnings repeating? Are they escalating?

### 14.6 LLM Analysis Prompt Template

When handing a report to Claude (in a new conversation or mid-session), use this template:

```
I'm running the Spartus Trading AI training system (v3.3). Here is the latest
training report [attached/pasted below].

Please analyze it and tell me:
1. Is the AI learning properly? What evidence supports your assessment?
2. Are there any red flags or silent failures I should worry about?
3. Are the SAC internals (entropy, Q-values, gradients) healthy?
4. What specific code or config changes would you recommend, if any?
5. Should training continue, or should we stop and adjust something?

If you spot any issues, please be specific about:
- What the problem is
- Which metric or metrics indicate it
- What the likely cause is
- What code change would fix it (file and section)
```

**For a quick check, shorter version:**

```
Spartus training health check — Week {N}. Is everything OK?
[paste health check output]
```

### 14.7 Report Generator Class

```python
class TrainingReportGenerator:
    """Generates structured markdown training reports optimized for LLM analysis.

    Reads JSONL logs, computes trends, flags anomalies, and produces a
    comprehensive markdown report that an LLM can parse to diagnose training
    health and recommend corrections.
    """

    # All thresholds mirrored from Section 5 Color Scheme
    THRESHOLDS = {
        "win_rate": {"green": 52, "yellow": 48, "unit": "%", "higher_is_better": True},
        "trend_accuracy": {"green": 55, "yellow": 50, "unit": "%", "higher_is_better": True},
        "sharpe": {"green": 0.8, "yellow": 0.3, "unit": "", "higher_is_better": True},
        "profit_factor": {"green": 1.2, "yellow": 1.0, "unit": "", "higher_is_better": True},
        "drawdown": {"green": 3, "yellow": 7, "unit": "%", "higher_is_better": False},
        "entropy_coef": {"green_min": 0.01, "green_max": 1.0, "red_min": 0.001, "red_max": 10.0},
        "entropy_pct": {"green": 40, "yellow": 20, "unit": "%", "higher_is_better": True},
        "q_value_mean": {"green": 50, "yellow": 100, "higher_is_better": False},
        "grad_clip_rate": {"green": 5, "yellow": 30, "unit": "%", "higher_is_better": False},
        "reward_clip_rate": {"green": 5, "yellow": 15, "unit": "%", "higher_is_better": False},
        "reward_std": {"green_min": 0.1, "green_max": 5.0, "red_min": 0.05, "red_max": 10.0},
        "regime_min_pct": {"green": 15, "yellow": 10, "unit": "%", "higher_is_better": True},
        "pattern_confidence": {"green": 50, "yellow": 30, "unit": "%", "higher_is_better": True},
        "agreement_rate": {"green": 70, "yellow": 50, "unit": "%", "higher_is_better": True},
        "agent_sharpe_gap": {"green": 15, "yellow": 30, "unit": "%", "higher_is_better": False},
        "action_std": {"green": 0.10, "yellow": 0.05, "higher_is_better": True},
    }

    def __init__(self, log_dir="storage/logs", report_dir="storage/reports"):
        """Set log and report directories. Create report_dir if needed."""

    def generate_report(self, current_week, trigger="auto"):
        """Read all JSONL logs, analyze trends, flag anomalies, produce full
        markdown report. Returns path to generated report file.

        Args:
            current_week: Current training week number
            trigger: "auto", "manual", "convergence", "final"
        """

    def generate_health_check(self):
        """Quick 40-line health check from latest weekly_summary entry.
        Returns formatted string (printed to stdout, not saved to file)."""

    def _load_weekly_summaries(self, last_n=20):
        """Load last N entries from weekly_summary.jsonl. Returns list of dicts."""

    def _load_recent_decisions(self, last_n=200):
        """Load last N decisions for trade behavior analysis."""

    def _load_recent_alerts(self, last_n=50):
        """Load recent entries from alerts.log. Returns list of strings."""

    def _compute_trend(self, values, window=10):
        """Compute trend direction and magnitude over window.
        Returns: (direction: '↑'/'↓'/'→', magnitude: float)"""

    def _assess_metric(self, metric_name, value):
        """Check a metric against THRESHOLDS dict.
        Returns: (status: 'GREEN'/'YELLOW'/'RED', threshold_str: str)"""

    def _compute_regime_performance(self, summaries):
        """Break down win rate and P/L by market regime."""

    def _flag_anomalies(self, latest_summary, trends):
        """Check all metrics against thresholds. Returns list of anomaly dicts:
        [{metric, value, severity, details, threshold_context}]"""

    def _generate_recommendations(self, anomalies, convergence_status, week):
        """Rule-based recommendations from anomaly flags.

        Rules include:
        - Entropy collapse → increase ent_coef_init or restart
        - Q-value explosion → reduce learning rate
        - Gradient clip rate high → check data quality, reduce LR
        - Agent dropout imminent → may need different seed
        - Persistent plateau → try LSTM or increase features
        - Reward clip rate high → adjust reward component weights
        - Regime imbalance → check data distribution
        """

    def _format_report(self, data):
        """Assemble all sections into the final markdown report string."""
```

### 14.8 Integration with Training Loop

The report generator plugs into the end-of-week flow:

```python
# In train_spartus() — after existing end-of-week processing:

    report_gen = TrainingReportGenerator(config.log_dir, config.report_dir)

    # ... inside the week loop, after all logging ...

    # Auto-generate report on interval
    if week_idx > 0 and week_idx % config.report_interval == 0:
        report_path = report_gen.generate_report(week_idx, trigger="auto")
        dashboard.log_alert("i", f"Training report saved: {report_path}")

    # Auto-generate on convergence events
    if conv_status["status"] in ("CONVERGED", "OVERFITTING", "COLLAPSED"):
        report_path = report_gen.generate_report(week_idx, trigger="convergence")
        dashboard.log_alert("i", f"Convergence report saved: {report_path}")

    # ... after training loop ends ...

    # Final report
    report_path = report_gen.generate_report(week_idx, trigger="final")
    dashboard.log_alert("+", f"Final training report saved: {report_path}")
```

**Keyboard handler for `R`:**
```python
# In keyboard listener thread:
if key == 'r':
    report_path = report_gen.generate_report(current_week, trigger="manual")
    dashboard.log_alert("i", f"Report generated: {report_path}")
```

### 14.9 Build Checklist Addition

```
Phase H: LLM Analysis System
  [ ] Create src/training/report_generator.py with TrainingReportGenerator
  [ ] Implement THRESHOLDS dict mirroring all Section 5 thresholds
  [ ] Implement _load_weekly_summaries() reading JSONL
  [ ] Implement _load_recent_decisions() for trade analysis
  [ ] Implement _load_recent_alerts() reading alerts.log
  [ ] Implement _compute_trend() with direction + magnitude
  [ ] Implement _assess_metric() checking against thresholds
  [ ] Implement _compute_regime_performance() breakdown
  [ ] Implement _flag_anomalies() checking all metrics
  [ ] Implement _generate_recommendations() with rule-based advice
  [ ] Implement _format_report() assembling all 12 sections
  [ ] Implement generate_report() orchestrating full flow
  [ ] Implement generate_health_check() for quick summaries
  [ ] Create src/training/health_check.py CLI entry point
  [ ] Add 'R' keyboard shortcut to dashboard
  [ ] Add auto-generation every report_interval weeks
  [ ] Add auto-generation on convergence events
  [ ] Add final report generation at end of training
  [ ] Create storage/reports/ directory structure
  [ ] Test: generate report from mock weekly_summary.jsonl
  [ ] Test: verify health check output format
  [ ] Test: verify all anomaly flags trigger correctly
  [ ] Test: verify recommendations match anomaly patterns
```

---

**Document Version:** 1.4
**Created:** 2026-02-22
**Updated:** 2026-02-23
**Status:** Design Complete — Ready for Implementation
**Changes in v1.1:** Added Trend Predictions panel (Section 4.5) with full predict→wait→verify cycle, added session resume/recovery status to Progress panel, added health badge to Progress panel
**Changes in v1.2:** Added TP Accuracy panel (Section 4.6) with TP hit rate, TP reachable rate, SL hit rate, manual close rate. Added SL trail entries to decision log. Added TP/SL metrics to TensorBoard logging and weekly summary JSONL. Added TP-related alerts (early closing, aggressive TPs, high SL hit rate). Updated action logging to include action[3] (SL management). Renumbered panels 4.7-4.11.
**Changes in v1.3:** Major overhaul to support all v3.3 system features. Added 6 new panels: SAC Internals (4.12), Convergence Detection (4.13), Reward Breakdown (4.14), Curriculum & Regime (4.15), Ensemble Agents (4.16), Anti-Hack & Safety (4.17). Expanded alerts from 12 to 46 conditions across 10 categories (SAC health, anti-hack, obs health, reward, curriculum, ensemble, convergence, LSTM). Updated all 3 log format schemas with reward components, regime, ensemble votes, convergence, SAC health. Expanded TensorBoard from 18 to 60+ metrics across 8 categories. Added comprehensive color thresholds for all new metrics. Rebuilt data flow diagram with ensemble flow, SAC extraction, reward component flow. Expanded build checklist from 7 to 16 phases. Updated class overview with full method signatures and docstrings. Total: 17 panels, 46 alert conditions, 60+ TensorBoard metrics, 4 log files.
**Changes in v1.4:** Added LLM Training Analysis System (Section 14). New TrainingReportGenerator class produces structured markdown reports optimized for Claude/LLM consumption — every metric includes inline threshold context (GREEN/YELLOW/RED) so the LLM can immediately assess health without looking up thresholds. Full 12-section report template with example values. Quick Health Check script for 40-line summaries. Auto-generation every 10 weeks, on convergence events, and on demand via `R` key. Direct Log Reading Protocol documenting how Claude reads JSONL logs mid-session. LLM Analysis Prompt Template for consistent analysis requests. Added `R` keyboard shortcut. Added report_generator.py and health_check.py to file structure. Added Phase H to build checklist (23 tasks). Updated data flow diagram with report generation flow.
**Changes in v1.4.1:** Cross-reference alignment audit. Added STABLE to convergence status encoding (0-6 scale now matches METHODOLOGY). Added STABLE to convergence color scheme.
**Changes in v1.4.2:** Design review alignment. Added daily DD > 3% alert (positions force-closed) to Core Trading Alerts. Updated drawdown > 10% alert to show terminal penalty value. Aligns with v3.3.2 changes: circuit breaker penalties use SET (not ADD), 10% DD episode termination, daily DD 3% check.
**Companion Documents:** [SPARTUS_TRADING_AI.md](SPARTUS_TRADING_AI.md) | [SPARTUS_TRAINING_METHODOLOGY.md](SPARTUS_TRAINING_METHODOLOGY.md) | [SPARTUS_DATA_PIPELINE.md](SPARTUS_DATA_PIPELINE.md)
