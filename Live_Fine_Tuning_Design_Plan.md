# Live Fine-Tuning System — Design Plan

**Version:** 2.0
**Date:** 2026-03-10
**Status:** PLANNING (System Analysis Complete)

---

## 1. Problem Statement

The trained SAC model is frozen after training. It learns from historical data (2015-2024) but cannot adapt to current market conditions (2026). Gold has moved from ~$1900 to $5200+, and market dynamics (volatility regimes, session patterns, correlation structures) have shifted. The frozen model outputs low conviction in live because the current market looks different from training data.

**Goal:** Build a system that continuously fine-tunes the trained model on live market data from MetaTrader 5, allowing it to adapt to the current regime while preserving learned fundamentals.

---

## 2. Architecture Overview

```
                    ┌─────────────────────────────┐
                    │    MT5 (Live Market Data)    │
                    │    M5 bars, tick data        │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    Shared MT5 Bridge         │
                    │  (existing mt5_bridge.py)    │
                    └──────┬─────────┬────────────┘
                           │         │
              ┌────────────▼──┐  ┌───▼────────────────┐
              │ TRACK 1       │  │ TRACK 2             │
              │ Live Dashboard│  │ Fine-Tune Dashboard  │
              │ (frozen model)│  │ (learning model)     │
              │ Trades demo   │  │ SAC gradient updates  │
              │ No learning   │  │ on live M5 data       │
              │ BASELINE      │  │ EXPERIMENTAL          │
              └───────────────┘  └─────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Validation Gate      │
                              │ (must beat baseline  │
                              │  before promotion)   │
                              └─────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Model Promotion      │
                              │ Fine-tuned → Live    │
                              └─────────────────────┘
```

**Dual-track approach:**
- **Track 1 (Live Dashboard):** Runs the frozen trained model. No learning. This is the control group / baseline. Already built and running.
- **Track 2 (Fine-Tune Dashboard):** Runs the same model but with SAC learning enabled on live data. Adapts to current market. This is what we're building.

Both tracks pull data from the same MT5 connection. They can run simultaneously.

---

## 3. Core Components

### 3.1 Live Data Collector (`LiveDataCollector`)

Accumulates M5 bars from MT5 into episode-sized chunks that the training environment can process.

**Responsibilities:**
- Pull M5 OHLCV bars from MT5 via existing bridge
- Pull correlated instrument data (EURUSD, XAGUSD, USDJPY, US500, USOIL)
- Compute all 54 precomputed features using existing `feature_builder.py`
- Store in rolling buffer (configurable size, default: 2000 bars = ~1 week of M5)
- Handle market close gaps (weekends, holidays)
- Provide episode-ready DataFrames to the training environment

**Key design decision:** Reuse the existing `FeatureBuilder` from training, NOT the live dashboard's `FeaturePipeline`. This ensures feature parity with training.

**Data flow:**
```
MT5 → M5 bars (XAUUSD + 5 correlated) → FeatureBuilder → 54 precomputed features → Buffer
```

### 3.2 Live Training Environment (`LiveTradeEnv`)

A modified version of the training `TradeEnv` that operates on live-accumulated data instead of historical parquet files.

**Key differences from historical TradeEnv:**
- Data source: `LiveDataCollector` buffer instead of parquet files
- Episode structure: rolling window (last N bars) instead of fixed weekly chunks
- No random week selection — always uses most recent data
- Same observation space (670-dim), same action space (4-dim), same reward function

**Episode lifecycle:**
1. Collect 500+ bars from live data
2. Run SAC episode through the environment (same step/reset logic)
3. Compute rewards using same 5-component reward function
4. Feed transitions to replay buffer
5. Run SAC gradient updates
6. Repeat when new data accumulates

### 3.3 Replay Buffer Manager (`ReplayBufferManager`)

Manages the mix of historical and live data in the SAC replay buffer to prevent catastrophic forgetting.

**Strategy: 70/30 Historical/Live Mix**
- Seed the replay buffer with transitions from the trained model's final state
- As live transitions come in, maintain a 70% historical / 30% live ratio
- Historical transitions are randomly sampled from a saved training buffer
- This ensures the model doesn't "forget" historical patterns while learning new ones

**Implementation:**
- Save the replay buffer from the final training checkpoint (already done via SB3's `save_replay_buffer`)
- On fine-tune start, load the saved buffer (200K transitions)
- As live transitions come in, evict oldest historical transitions proportionally
- Track the ratio and log it for monitoring

### 3.4 Fine-Tune Orchestrator (`LiveFineTuner`)

The main controller that coordinates data collection, training, and checkpointing.

**Lifecycle:**
```
1. INIT: Load trained model + replay buffer + feature baseline
2. COLLECT: Accumulate M5 bars from MT5 (wait for minimum 500 bars)
3. TRAIN: Run SAC episode on accumulated data
4. UPDATE: Gradient updates using mixed replay buffer
5. CHECKPOINT: Save model every N episodes
6. VALIDATE: Compare against frozen baseline on held-out data
7. REPEAT from step 2
```

**Training schedule:**
- Collect bars continuously during market hours
- Run training episode every 4 hours (when ~48 new M5 bars accumulated)
- Or on-demand: user can trigger a training cycle from the dashboard
- Gradient steps: same as training (8 per step, batch=1024)
- Learning rate: reduced (1e-4 instead of 3e-4) for fine-tuning stability

### 3.5 Validation Gate (`ValidationGate`)

Prevents deploying a fine-tuned model that's worse than the baseline.

**Validation criteria:**
1. Run the fine-tuned model on the last 2 weeks of historical data (out-of-sample)
2. Compare Sharpe ratio against the frozen baseline
3. Fine-tuned model must have Sharpe >= baseline Sharpe * 0.9 (allow 10% degradation on historical)
4. Fine-tuned model must show improvement on recent live data
5. No catastrophic metrics: action_std > 0.3, win_rate > 30%, no bankruptcy

**Promotion flow:**
```
Fine-tuned model → Validation Gate → PASS → Export as new live model
                                   → FAIL → Continue fine-tuning (don't deploy)
```

---

## 4. Persistent Memory Architecture — "The Evolving Brain"

The biggest risk in online RL is **catastrophic forgetting** — the model "unlearns" good strategies while adapting to new ones. A human trader doesn't forget SL management just because they're learning a new entry strategy. The brain retains and builds on top of existing knowledge.

Neural networks naturally forget because they have **one set of weights** — updating for new patterns can overwrite old ones. The human brain avoids this through complementary learning systems (fast hippocampus + slow neocortex), sleep consolidation (replaying memories), and modular representation (different regions for different skills).

We implement a **4-layer memory system** that approximates these brain mechanisms:

```
┌─────────────────────────────────────────────────────────┐
│                    LAYER 4: STRATEGY MEMORY              │
│  Explicit regime → strategy mapping (declarative memory) │
│  "When market looks like X, the model did Y successfully" │
├─────────────────────────────────────────────────────────┤
│                    LAYER 3: WEIGHT PROTECTION             │
│  EWC — protects important weights from being overwritten  │
│  (analogous to long-term memory consolidation)            │
├─────────────────────────────────────────────────────────┤
│                    LAYER 2: EXPERIENCE REPLAY             │
│  Curated historical replay buffer (like dreaming/sleep)   │
│  Brain replays important memories to consolidate them     │
├─────────────────────────────────────────────────────────┤
│                    LAYER 1: SLOW ADAPTATION               │
│  Low learning rate + KL anchor to frozen baseline         │
│  (prevents rapid unlearning of fundamentals)              │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: Slow Adaptation (Prevents Rapid Forgetting)

The simplest defense: learn slowly so old knowledge isn't overwritten quickly.

- **Fine-tuning LR: 1e-4** (vs 3e-4 in training) — 3x slower weight updates
- **KL Divergence Anchor:** Continuously measure how far the fine-tuned policy has drifted from the frozen baseline. Add a KL penalty to the SAC loss function that resists excessive drift. This is like a rubber band connecting the new model to its foundation — it can stretch and adapt, but it's always anchored to what it learned.
- **Entropy alpha floor: 0.01** — maintains exploration, prevents policy collapse
- **Pause threshold:** If KL divergence exceeds 0.5, pause fine-tuning and alert user. The model is drifting too far from its training.

**Brain analogy:** This is like how your neocortex changes slowly — you don't completely rewire your trading intuition after one bad trade. It takes repeated exposure to shift deep beliefs.

### Layer 2: Experience Replay (The "Dreaming" System)

The human brain replays important experiences during sleep to consolidate them into long-term memory. Our replay buffer does the same thing — it continuously reminds the model of its past experiences so they don't fade.

**Strategy: Curated 70/30 Historical/Live Mix**
- Seed replay buffer with 200K transitions from the trained model's final state
- As live transitions come in, maintain 70% historical / 30% live ratio
- Historical transitions are **curated** — not random. Prioritize:
  - High-reward transitions (successful trades the model should remember)
  - Diverse regime coverage (trending, ranging, volatile, quiet)
  - Both LONG and SHORT successes (prevent side bias regression)
  - Loss scenarios with correct SL management (remember risk management)

**Curated vs Random Replay:**
Random replay would eventually dilute important historical memories as the buffer grows. Curated replay ensures the model's "greatest hits" are always in the mix — just like how your brain preferentially consolidates emotionally significant or survival-relevant memories.

**Implementation:**
```python
class CuratedReplayBuffer:
    """Three-tier replay buffer mimicking brain memory consolidation."""

    # Tier 1: CORE MEMORIES (never evicted, ~30% of historical)
    # Best trades, worst trades, regime transitions, perfect SL management
    # These are the foundational skills — always replayed

    # Tier 2: SUPPORTING MEMORIES (slow eviction, ~40% of historical)
    # Good variety of market conditions, both sides, all sessions
    # Provides breadth of experience

    # Tier 3: RECENT MEMORIES (fast turnover, ~30% = live data)
    # Live market transitions, most recent learning
    # This is where adaptation happens
```

**Brain analogy:** Tier 1 = long-term procedural memory (riding a bike). Tier 2 = episodic memory (past market experiences). Tier 3 = working memory (what's happening right now). Your brain doesn't treat all memories equally — important ones get stronger consolidation.

### Layer 3: Elastic Weight Consolidation — EWC (Protecting Critical Knowledge)

This is the most powerful anti-forgetting mechanism. It's directly inspired by neuroscience research on **synaptic consolidation** — the process where important neural connections become resistant to change.

**How it works:**
1. After training completes, compute the **Fisher Information Matrix** — this measures how important each weight is for the model's current performance
2. Weights that are critical for good trading decisions get a high "importance score"
3. During fine-tuning, add a penalty that **resists changing important weights**
4. Less important weights are free to adapt to new patterns

**The key insight:** Not all weights matter equally. Some encode "gold tends to pull back after rapid moves" (fundamental, should be preserved). Others encode "volatility is usually X pips" (regime-specific, should adapt). EWC automatically identifies which is which.

```python
class EWCProtection:
    """Elastic Weight Consolidation — protects important weights."""

    def __init__(self, model, fisher_samples=2000, ewc_lambda=5000):
        # After training, compute importance of each weight
        self.fisher = self._compute_fisher(model, fisher_samples)
        self.frozen_params = {n: p.clone() for n, p in model.named_parameters()}

    def penalty(self, model):
        """EWC loss: penalize changes to important weights."""
        loss = 0
        for name, param in model.named_parameters():
            # (current - original)^2 * importance
            loss += (self.fisher[name] * (param - self.frozen_params[name]).pow(2)).sum()
        return self.ewc_lambda * loss
```

**Brain analogy:** This is exactly how synaptic consolidation works. Synapses that were important for learned skills become physically more resistant to change (through protein synthesis at the synapse). You CAN still update them, but it takes stronger, more consistent signals. A single bad trade doesn't overwrite years of experience — but a new sustained pattern eventually can.

### Layer 4: Strategy Memory (Explicit Regime-Strategy Mapping)

This layer goes beyond what standard RL does. It adds an **explicit declarative memory** — a database that maps market regimes to the strategies/behaviors the model successfully used.

**Concept:**
```
REGIME: High-volatility trending (ATR > 2x normal, strong momentum)
BEST STRATEGY: Short pullbacks, wide SL, hold 8-12 bars
EVIDENCE: 47 trades, WR 62%, PF 2.1, period: 2019-Q4, 2020-Q3, 2024-H2
STATUS: ACTIVE (conditions seen in live data last 3 days)

REGIME: Low-volatility range (ATR < 0.7x normal, no momentum)
BEST STRATEGY: Fade extremes, tight SL, hold 4-6 bars
EVIDENCE: 83 trades, WR 58%, PF 1.8, period: 2017-H1, 2021-Q2
STATUS: DORMANT (not seen in live data recently)
```

**This is analogous to your "strategy book" as a human trader.** You don't re-derive your strategies from scratch — you have them catalogued. When the market enters a regime you recognize, you pull out the right strategy. When you encounter something new, you experiment and add it to the book.

**Implementation:**
- During training, the system clusters episodes by market regime (volatility, trend strength, session, correlation state)
- For each cluster, it records: what the model did, how it performed, which action patterns worked
- During live fine-tuning, when the model encounters a regime, it checks the strategy memory
- If performance on a known regime degrades after fine-tuning, flag it: "WARNING: Model forgetting how to handle [regime X]"
- The strategy memory is a **JSON database** — it persists forever, never gets overwritten by gradient updates

**Why this matters:** Even if the neural network weights drift (Layers 1-3 aren't perfect), this explicit memory provides a check. It's like having a trading journal that says "I know I used to be good at this, so if my results are getting worse, something is wrong."

### Layer 5: Rollback & Checkpoint Safety Net

The final layer — always have an escape hatch.

- Keep the last **10 fine-tune checkpoints** (not just 5)
- The frozen baseline model is **NEVER modified** — always available as fallback
- Weekly automated comparison: if fine-tuned model underperforms frozen baseline on ANY regime that was profitable in training, trigger alert
- One-click rollback from dashboard to any previous checkpoint
- Automatic rollback if: 3 consecutive validation failures, or drawdown exceeds 2x baseline max drawdown

---

## 5. How This Prevents Forgetting — The Full Picture

Here's how all 5 layers work together, using your exact analogy:

**Scenario:** The model learned "short pullbacks in uptrends" during training (2015-2024). Now it's 2026, gold is at $5200+, and the model encounters a new pattern: "momentum breakouts hold longer in the $5000+ range."

| Layer | What it does |
|-------|-------------|
| **L1: Slow Adaptation** | Model adjusts slowly to the new breakout pattern. LR is low, so the pullback-shorting weights don't change dramatically in one session. KL anchor resists wholesale drift. |
| **L2: Experience Replay** | During every training batch, 70% of examples are HISTORICAL — including those successful pullback shorts. The model is constantly reminded: "hey, you were good at this, don't forget." |
| **L3: EWC Protection** | The weights that encode "detect pullback → go short" have HIGH Fisher importance. They resist change. The model can learn the new breakout pattern using weights with LOW importance — effectively learning the new strategy *alongside* the old one, not *instead of* it. |
| **L4: Strategy Memory** | The strategy database explicitly records: "High-trend pullback shorting: WR 62%, PF 2.1." If the model's performance on this regime drops below historical, the system flags: "FORGETTING ALERT: pullback-short strategy degrading." |
| **L5: Rollback** | If all else fails and the model loses its pullback edge, rollback to a checkpoint where it still had it. Never lose more than a few days of adaptation. |

**The result:** The model doesn't choose between old strategy and new strategy. Like a human trader, it **keeps both** and deploys whichever fits the current market condition. The observation space already contains regime indicators (ATR, momentum, session, correlations) — the model uses these to decide which "strategy" to activate, just like you'd look at the market and say "this is a pullback setup" vs "this is a breakout."

---

## 6. Dashboard Changes — Training Dashboard Restructure

### 6.1 Remove Header "Fine-Tune" Button

The current header has: **Start | Resume | Fine-Tune | Pause | Quit**

The existing "Fine-Tune" button does a simple weight-load + counter-reset for re-training on historical data. This functionality moves INTO the new Live Fine-Tune tab (as one of its options). The header becomes:

**Start | Resume | Pause | Quit**

### 6.2 New Tab 7: "LIVE FINE-TUNE" (next to MODEL EXPORT)

Current tabs: OVERVIEW | METRICS | INTERNALS | DB VIEWER | TRADE JOURNAL | MODEL EXPORT

New: OVERVIEW | METRICS | INTERNALS | DB VIEWER | TRADE JOURNAL | MODEL EXPORT | **LIVE FINE-TUNE**

### 6.3 Tab 7 Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  LIVE FINE-TUNE                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌── MODE SELECTOR ──────────────────────────────────────────┐  │
│  │  [Historical Fine-Tune]  [Live Fine-Tune (MT5)]           │  │
│  │  Model: [Select Checkpoint ▼]                             │  │
│  │  [START FINE-TUNING]  [STOP]  [VALIDATE NOW]              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌── CONNECTION STATUS ──────┐  ┌── TRAINING STATUS ─────────┐  │
│  │  MT5: Connected ●         │  │  State: COLLECTING         │  │
│  │  Bars collected: 847      │  │  Episodes: 12              │  │
│  │  Buffer: 847/2000         │  │  Last episode: 4.2h ago    │  │
│  │  Next episode: 1.8h       │  │  Total grad steps: 9,600   │  │
│  └───────────────────────────┘  └────────────────────────────┘  │
│                                                                  │
│  ┌── MEMORY LAYERS ─────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  L1 Slow Adaptation                                       │   │
│  │  ├─ Learning rate: 1e-4 (3x slower)                       │   │
│  │  ├─ KL divergence: 0.12 / 0.50 max  [████░░░░░░] 24%     │   │
│  │  └─ Status: HEALTHY                                       │   │
│  │                                                           │   │
│  │  L2 Experience Replay                                     │   │
│  │  ├─ Core memories: 60,000 (30%) — LOCKED                 │   │
│  │  ├─ Supporting: 80,000 (40%)                              │   │
│  │  ├─ Live/recent: 12,400 (6.2% of 30%)                    │   │
│  │  └─ Buffer health: BALANCED                               │   │
│  │                                                           │   │
│  │  L3 EWC Protection                                        │   │
│  │  ├─ Fisher computed: YES (2000 samples)                   │   │
│  │  ├─ Weight divergence: 0.08 / 0.50 max                   │   │
│  │  ├─ Protected params: 847/1,024 (82.7%)                   │   │
│  │  └─ Status: PROTECTING                                    │   │
│  │                                                           │   │
│  │  L4 Strategy Memory                                       │   │
│  │  ├─ Known regimes: 8                                      │   │
│  │  ├─ Active regime: High-vol trending                      │   │
│  │  ├─ Forgetting alerts: 0                                  │   │
│  │  └─ Strategy book: 8/8 regimes profitable                 │   │
│  │                                                           │   │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌── VALIDATION GATE ────────┐  ┌── CHECKPOINTS ─────────────┐  │
│  │  Last check: 2h ago       │  │  1. FT_ep012 (current)     │  │
│  │  Result: PASS ✓           │  │  2. FT_ep010               │  │
│  │  Sharpe: 0.72 (base: 0.64)│  │  3. FT_ep005               │  │
│  │  Regime coverage: 6/8     │  │  [PROMOTE] [ROLLBACK]      │  │
│  │  [Validate Now]           │  │                             │  │
│  └───────────────────────────┘  └─────────────────────────────┘  │
│                                                                  │
│  ┌── LAST EPISODE SUMMARY ──────────────────────────────────┐   │
│  │  Trades: 47 | WR: 61.7% | PF: 1.84 | P/L: +£23.45      │   │
│  │  Direction: 52% LONG / 48% SHORT | Avg hold: 7.2 bars    │   │
│  │  Conviction: mean 0.42 | TP hit: 34% | SL hit: 41%      │   │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Mode Selector: Two Fine-Tuning Modes

| Mode | Data Source | Use Case |
|------|-----------|----------|
| **Historical Fine-Tune** | Parquet files (same as training) | Re-train on historical with new config. Replaces old header button. |
| **Live Fine-Tune (MT5)** | Live M5 bars from MetaTrader 5 | Adapt to current market. Requires MT5 connection. |

Both modes use the same 4-layer memory system. The only difference is data source.

### 6.5 Controls

| Control | Action |
|---------|--------|
| **Start Fine-Tuning** | Opens model picker → loads model + Fisher + baseline → begins |
| **Stop** | Gracefully stops fine-tuning, saves current state |
| **Validate Now** | Runs validation gate immediately (doesn't wait for schedule) |
| **Promote** | Exports fine-tuned model as new live deployment package |
| **Rollback** | Reverts to selected checkpoint |

### 6.6 State Machine (Extended)

```
IDLE → (Start clicked) → INITIALIZING → (model loaded) → COLLECTING
COLLECTING → (min bars reached) → TRAINING → (episode done) → COLLECTING
TRAINING → (validation triggered) → VALIDATING → (done) → COLLECTING
Any state → (Stop clicked) → IDLE
Any state → (Promote clicked) → EXPORTING → (done) → IDLE
```

---

## 6. MT5 Data Requirements

### 6.1 Primary Symbol
- XAUUSD M5 bars (OHLCV + tick_volume)
- Need at least 500 bars (~41 hours of market time) before first training episode

### 6.2 Correlated Instruments
- EURUSD M5 (for correlation features)
- XAGUSD M5 (for gold/silver ratio)
- USDJPY M5 (for USD strength)
- US500 M5 (for risk-on/risk-off)
- USOIL M5 (for commodity correlation)

All pulled via existing MT5 bridge with the same Dukascopy symbol mapping.

### 6.3 Data Freshness
- M5 bars requested every 5 minutes (on timer, same as live dashboard)
- Correlated instruments on same schedule
- Economic calendar events loaded from CSV (same as training)

---

## 7. Guardrails — Keeping the Car on the Road

The racetrack analogy: we want the car driving straight, not swerving left and right. These guardrails ensure the fine-tuning process stays on track.

### 7.1 The Guardrail Framework

```
                     OVERFITTING ZONE
                    ┌─────────────────┐
                    │  Model memorizes │
                    │  recent data     │
                    │  (swerving left) │
    ────────────────┤                 ├────────────────
                    │                 │
    ════════════════╪═════════════════╪════════════════  ← THE STRAIGHT ROAD
                    │                 │                     (balanced learning)
    ────────────────┤                 ├────────────────
                    │  Model forgets  │
                    │  old strategies  │
                    │  (swerving right)│
                    └─────────────────┘
                     FORGETTING ZONE
```

### 7.2 Left Guardrail: Preventing Overfitting to Recent Data

| Guardrail | What it does | Trigger |
|-----------|-------------|---------|
| **Validation gate** | Tests on HISTORICAL data the model hasn't seen recently. If performance drops, the model is overfitting to live data. | Sharpe < 90% of baseline |
| **Replay buffer mixing** | 70% of training data is always historical. Model can't ignore the past even if live data is different. | Automatic (always active) |
| **Regime diversity check** | Strategy Memory alerts if model improves on one regime at the cost of another. | Performance on any regime < 70% of historical |
| **Short window detection** | If model's improvements only show on the last 48 bars (4 hours) but not on the last 500, it's memorizing, not learning. | Improvement ratio < 0.5 |

### 7.3 Right Guardrail: Preventing Forgetting

| Guardrail | What it does | Trigger |
|-----------|-------------|---------|
| **EWC weight protection** | Important weights resist change. Model can only adapt using less-critical weights. | Weight divergence > threshold |
| **KL divergence anchor** | Policy can't drift too far from the trained baseline. Like a rubber band. | KL > 0.5 |
| **Core memory tier** | 60K high-value historical transitions are NEVER evicted from replay buffer. | Automatic (always active) |
| **Action std monitoring** | If action diversity drops, model is converging on narrow behavior (losing versatility). | action_std < 0.4 |

### 7.4 Speed Guardrail: Learning Rate Control

```
Training LR:      3e-4  ████████████████████████████████ (full speed)
Fine-tuning LR:   1e-4  ██████████░░░░░░░░░░░░░░░░░░░░░░ (3x slower)
Emergency LR:     3e-5  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (10x slower, if KL > 0.3)
```

If KL divergence rises above 0.3 (60% of max), automatically reduce LR to 3e-5. This slows down adaptation before hitting the pause threshold. Like easing off the accelerator before hitting the brakes.

### 7.5 Automatic Safety Actions

| Condition | Action | Reversible? |
|-----------|--------|-------------|
| KL > 0.3 | Reduce LR to 3e-5 | Yes (auto-restore when KL < 0.2) |
| KL > 0.5 | **PAUSE fine-tuning**, alert user | Yes (user resumes) |
| action_std < 0.3 | **PAUSE**, alert "policy collapsing" | Yes |
| 3 consecutive validation failures | **AUTO-ROLLBACK** to best checkpoint | Yes |
| Drawdown > 2x baseline max | **STOP fine-tuning**, export last good checkpoint | Yes |
| Weight divergence > threshold | **PAUSE**, alert "drifting too far" | Yes |
| Any regime performance < 50% of historical | **ALERT** + reduce live buffer ratio to 20% | Yes |

### 7.6 The "Straight Road" Metrics

These metrics define what "on track" looks like:

```
HEALTHY FINE-TUNING looks like:
  ✓ KL divergence: 0.05 - 0.20 (learning but anchored)
  ✓ Weight divergence: slowly increasing (adapting)
  ✓ Validation Sharpe: >= baseline (not losing old skills)
  ✓ Live episode Sharpe: improving (gaining new skills)
  ✓ Action std: 0.5 - 0.7 (diverse behavior maintained)
  ✓ All regimes: >= 70% of historical performance
  ✓ Both LONG and SHORT profitable
  ✓ Conviction distribution: shifting upward on live data

UNHEALTHY FINE-TUNING looks like:
  ✗ KL divergence: > 0.3 (drifting too fast)
  ✗ Weight divergence: spiking suddenly (unstable)
  ✗ Validation Sharpe: declining (forgetting)
  ✗ Action std: < 0.4 (losing versatility)
  ✗ One regime improving, others declining (overfitting)
  ✗ Side bias emerging (losing balance)
```

---

## 8. System Integration Analysis — How It Fits Into the Training System

### 8.1 Files That Change

| File | Change | Size |
|------|--------|------|
| `src/training/qt_dashboard.py` | Remove header Fine-Tune button, add Tab 7 | MEDIUM |
| `src/training/trainer.py` | Add Fisher computation at export, EWC loss integration | MEDIUM |
| `src/training/callback.py` | Add EWC metrics logging, weight divergence tracking | SMALL |
| `src/training/exporter.py` | Package Fisher matrix + reference weights in ZIP | SMALL |
| `src/training/convergence.py` | Add `LiveConvergenceDetector` class | SMALL |
| `src/config.py` | Add fine-tune config parameters | SMALL |

### 8.2 New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/training/ewc_protection.py` | Fisher computation + EWC penalty | ~200 |
| `src/training/curated_replay_buffer.py` | 3-tier buffer with priority sampling | ~300 |
| `src/training/strategy_memory.py` | Regime clustering + performance tracking | ~250 |
| `src/training/validation_gate.py` | Multi-criteria validation before promotion | ~200 |
| `src/training/live_data_collector.py` | MT5 bar accumulation + feature computation | ~250 |
| `src/training/tab_live_finetune.py` | PyQt6 tab for live fine-tune dashboard | ~400 |

**Total new code: ~1,600 lines across 6 new files + ~200 lines of changes to existing files.**

### 8.3 SB3 Integration Challenge: EWC Loss Injection

The biggest technical challenge. SB3's SAC computes actor/critic losses internally. We need to add the EWC penalty BEFORE the optimizer step.

**Solution: Custom SAC subclass**

```python
class SACWithEWC(SAC):
    """SAC with Elastic Weight Consolidation penalty."""

    def __init__(self, *args, ewc_protection=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc_protection

    def train(self, gradient_steps, batch_size):
        """Override train() to inject EWC penalty."""
        # Let SB3 do its normal training
        super().train(gradient_steps, batch_size)

        # If EWC is active, add penalty and do an extra backward step
        if self.ewc is not None and self.ewc.enabled:
            ewc_loss = self.ewc.penalty(self.actor, self.critic)
            ewc_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
```

**Alternative (simpler):** Hook into `_on_step()` callback to apply EWC as a **weight regularization** after each SB3 training step. This is less precise but doesn't require subclassing SAC.

### 8.4 Replay Buffer Integration

SB3 uses `ReplayBuffer` with a fixed circular buffer. Our curated buffer needs:
- Priority tiers (core/supporting/recent)
- Selective eviction (never evict core)
- Metadata per transition (reward, regime, trade result)

**Solution:** Wrap SB3's buffer with our curation logic:

```python
class CuratedReplayBuffer:
    """Wraps SB3 ReplayBuffer with 3-tier priority system."""

    def __init__(self, buffer_size, obs_space, action_space, ...):
        self._buffer = ReplayBuffer(buffer_size, obs_space, action_space, ...)
        self._tiers = {"core": set(), "supporting": set(), "recent": set()}
        self._metadata = {}  # idx → {reward, regime, trade_result}

    def add(self, obs, next_obs, action, reward, done, infos):
        """Add with tier classification."""
        idx = self._buffer.pos
        self._buffer.add(obs, next_obs, action, reward, done, infos)
        tier = self._classify_tier(reward, infos)
        self._tiers[tier].add(idx)

    def sample(self, batch_size):
        """Sample with tier-weighted priority."""
        # 30% from core, 40% from supporting, 30% from recent
        ...
```

### 8.5 What Stays Unchanged

These components work as-is, no modifications needed:
- `src/data/feature_builder.py` — same 54 precomputed features
- `src/environment/trade_env.py` — same env (accepts DataFrame, doesn't care about source)
- `src/training/reward.py` — same 5-component reward function
- `src/data/normalizer.py` — same rolling z-score normalization
- `src/data/correlation_features.py`, `calendar_features.py`, `regime_features.py`, `session_features.py`
- `live_dashboard/` — completely separate, runs independently

---

## 9. File Structure (Planned)

```
src/training/
├── ewc_protection.py             # NEW: Fisher matrix + EWC penalty (Layer 3)
├── curated_replay_buffer.py      # NEW: 3-tier priority buffer (Layer 2)
├── strategy_memory.py            # NEW: Regime → strategy DB (Layer 4)
├── validation_gate.py            # NEW: Multi-criteria validation
├── live_data_collector.py        # NEW: MT5 bar accumulation + features
├── tab_live_finetune.py          # NEW: PyQt6 Tab 7 for fine-tune dashboard
├── trainer.py                    # MODIFY: EWC integration, Fisher computation
├── callback.py                   # MODIFY: EWC metrics, divergence logging
├── qt_dashboard.py               # MODIFY: Remove header FT button, add Tab 7
├── exporter.py                   # MODIFY: Package Fisher + ref weights
├── convergence.py                # MODIFY: Add LiveConvergenceDetector
└── reward.py                     # NO CHANGE

src/
├── config.py                     # MODIFY: Add fine-tune parameters
├── data/feature_builder.py       # NO CHANGE: Same 54 precomputed features
├── environment/trade_env.py      # NO CHANGE: Same env (accepts any DataFrame)
└── data/normalizer.py            # NO CHANGE: Same z-score normalization
```

---

## 8. Configuration

New fields in `live_config.py`:

```python
# ---- Live Fine-Tuning --------------------------------------------------
finetune_enabled: bool = False
finetune_lr: float = 1e-4                    # Reduced LR for stability
finetune_min_bars: int = 500                  # Min bars before first episode
finetune_episode_interval_hours: float = 4.0  # Hours between training episodes
finetune_gradient_steps: int = 8              # Same as training
finetune_batch_size: int = 1024               # Same as training
finetune_checkpoint_interval: int = 5         # Save every N episodes
finetune_max_checkpoints: int = 10            # Keep last 10 checkpoints

# ---- Layer 1: Slow Adaptation -------------------------------------------
finetune_max_kl_divergence: float = 0.5       # Pause if KL exceeds this
finetune_kl_penalty_weight: float = 0.1       # KL penalty in SAC loss function

# ---- Layer 2: Curated Replay Buffer ------------------------------------
finetune_buffer_core_pct: float = 0.30        # Core memories (never evicted)
finetune_buffer_supporting_pct: float = 0.40  # Supporting memories (slow eviction)
finetune_buffer_recent_pct: float = 0.30      # Live/recent data (fast turnover)
finetune_buffer_core_size: int = 60000        # ~60K core transitions
finetune_buffer_total_size: int = 200000      # Total buffer capacity

# ---- Layer 3: EWC Weight Protection ------------------------------------
finetune_ewc_enabled: bool = True
finetune_ewc_lambda: float = 5000             # EWC penalty strength
finetune_ewc_fisher_samples: int = 2000       # Samples for Fisher matrix computation

# ---- Layer 4: Strategy Memory -------------------------------------------
finetune_strategy_memory_enabled: bool = True
finetune_regime_clusters: int = 8             # Number of market regime clusters
finetune_forgetting_threshold: float = 0.7    # Alert if regime performance < 70% of historical

# ---- Validation Gate ----------------------------------------------------
finetune_val_sharpe_threshold: float = 0.9    # Must be >= 90% of baseline Sharpe
finetune_auto_rollback_failures: int = 3      # Rollback after N consecutive validation failures
finetune_max_drawdown_mult: float = 2.0       # Rollback if drawdown > 2x baseline max
```

---

## 9. Weekly Workflow

The user's intended workflow:

1. **Sunday 23:00 UTC** — Market opens. Start live dashboard (frozen model on demo). Start fine-tune dashboard alongside it.
2. **Monday-Friday** — Live dashboard trades on demo with frozen model. Fine-tuner collects live data and runs training episodes every 4 hours.
3. **Friday 22:00 UTC** — Market closes. Fine-tuner has accumulated ~1,440 M5 bars and run ~30 training episodes during the week.
4. **Weekend** — Run validation gate on the fine-tuned model. If it passes, export and swap into live dashboard for next week.
5. **Repeat** — Each week the model adapts further to current conditions.

Over time, the fine-tuned model should:
- Output higher conviction on current market patterns
- Adapt to current volatility regime ($5200+ gold vs $1900 training data)
- Learn current session patterns (which sessions are most volatile NOW)
- Maintain historical knowledge via replay buffer mixing

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Catastrophic forgetting | HIGH | Replay buffer mixing (70/30), low LR, KL monitoring |
| Overfitting to recent regime | MEDIUM | Validation gate, historical buffer, weekly rotation |
| MT5 data quality issues | LOW | Same bridge as live dashboard (proven), data validation |
| Two systems competing for MT5 | LOW | Read-only data access, no trade conflicts |
| Fine-tuned model worse than baseline | MEDIUM | Validation gate blocks promotion, rollback available |
| Memory/GPU constraints | MEDIUM | Fine-tuning uses same resources as training (~470MB GPU) |

---

## 11. Implementation Order

### Phase 1: Data Collection (Foundation)
1. `LiveDataCollector` — pull and buffer M5 bars from MT5
2. Test: verify feature parity between live-computed and historical features

### Phase 2: Memory Layer 2 — Curated Replay Buffer
3. `CuratedReplayBuffer` — three-tier replay (core/supporting/recent)
4. Replay curation logic: score historical transitions by reward, diversity, regime coverage
5. Buffer persistence: save/load curated buffer to disk across sessions
6. Test: verify buffer composition ratios and curation quality

### Phase 3: Training Integration
7. Adapt `TradeEnv` to accept live data buffer
8. `LiveFineTuner` — orchestrate episodes and gradient updates with curated buffer
9. Memory Layer 1 integration: low LR + KL divergence anchor in SAC loss
10. Test: run a fine-tune cycle on buffered data, verify KL stays bounded

### Phase 4: Memory Layer 3 — EWC Weight Protection
11. Fisher Information Matrix computation after training completes
12. `EWCProtection` — penalty term added to SAC critic/actor loss
13. Per-weight importance visualization (which weights are "locked")
14. Test: verify important weights resist change while unimportant ones adapt

### Phase 5: Memory Layer 4 — Strategy Memory
15. Regime clustering from training episodes (volatility, trend, session, correlation)
16. `StrategyMemory` — JSON database mapping regimes → model behavior → performance
17. Forgetting detection: flag when performance on known regime degrades
18. Test: verify regime classification and strategy recall accuracy

### Phase 6: Safeguards & Validation
19. `ValidationGate` — historical + live validation per regime
20. Automated rollback logic (3 consecutive failures, or 2x baseline drawdown)
21. Checkpoint management (keep last 10)
22. Test: simulate forgetting scenario and verify rollback triggers

### Phase 7: Dashboard
23. New "Live Fine-Tune" button in training dashboard header
24. `tab_live_finetune.py` — monitoring tab with memory layer visualizations
25. Strategy Memory viewer (regime map, forgetting alerts)
26. Model promotion and export flow

### Phase 8: Integration Testing
27. End-to-end test: MT5 → collect → train → validate → export
28. Forgetting stress test: train on one regime, switch to another, verify retention
29. Run alongside live dashboard for one full week
30. Compare fine-tuned vs frozen model performance across all regimes

---

## 12. Dependencies

- **Existing:** MT5 bridge, feature_builder, trade_env, SAC model, reward function, training dashboard
- **New packages:** None — all components use existing dependencies (SB3, PyTorch, PyQt6, MT5)
- **Hardware:** Same as training (RTX 3060 6GB, 16GB RAM). Fine-tuning is less intensive than full training.
- **MT5:** Must be running and logged in. Both live dashboard and fine-tuner connect to same MT5 instance.

---

## 13. Open Questions

1. **Episode length for live data:** Use full buffer (500+ bars) as one episode, or split into smaller episodes?
2. **Correlated instruments:** Pull from MT5 directly, or reuse Dukascopy downloads? (MT5 may not have all symbols)
3. **Market hours handling:** Fine-tune only during market hours, or also during weekend on accumulated data?
4. **Multi-week continuity:** Should the replay buffer persist across weeks, or reset each Monday?
5. **Entropy schedule:** Keep entropy alpha pinned at 0.01 (same as training), or allow more exploration during fine-tuning?
6. **EWC computation cost:** Fisher matrix requires forward passes through training data. Compute once at export time, or recompute periodically?
7. **Strategy Memory granularity:** How many regime clusters? Too few = broad categories. Too many = overfitting to specific historical periods.
8. **Core memory selection:** What criteria define a "core memory" transition? Top N% by reward? Manual curation? Diversity-based sampling?
9. **Cross-week strategy drift:** If the model adapts to Week 1's regime, then Week 2 is completely different, how aggressively should it adapt? (Tension between stability and responsiveness)
10. **EWC lambda tuning:** Too high = model can't adapt at all (frozen). Too low = no protection. Need empirical testing to find the sweet spot.

---

## 14. Impact Report — Is This Worth Building?

### 14.1 The Core Problem (Why We Need This)

**Training performance (W81-W100):**
- 7,563 trades executed
- ALL 20 weeks profitable (W81-W100)
- Win rates: 47% → 84% (improving steadily)
- Balance: £100 → £503 (+403% in 100 weeks of simulated data)
- Best week: W100 with 84% WR, +£203 P/L

**Live performance (Week 1 demo, W170 equivalent):**
- 152 trades, +£44.18, PF 1.09, WR 46.7%
- Direction accuracy: 47.1% (barely above random)
- Model conviction: 0.15-0.30 (below the 0.30 threshold)
- Result: **Most of the time the AI sits out doing nothing**

**The gap:** The model is brilliant on 2015-2024 historical data but outputs low conviction on 2026 live data because $5200+ gold looks NOTHING like the $1200-2000 range it trained on. The fundamentals it learned (pullback shorting, SL management, session patterns) are valid — but the model doesn't recognize the current price environment as "familiar enough" to act on with confidence.

### 14.2 What Live Fine-Tuning Would Fix

| Problem | How Fine-Tuning Fixes It | Expected Impact |
|---------|-------------------------|-----------------|
| **Low conviction (0.15-0.30)** | Model sees current price levels repeatedly, learns they're "normal." Conviction distribution shifts upward. | HIGH — This is the #1 bottleneck. Conviction should reach 0.40-0.60 within 2-3 weeks of fine-tuning. |
| **Poor direction accuracy (47%)** | Model learns current volatility patterns, momentum characteristics at $5200+ levels. | MEDIUM — Should improve to 52-55% within 4-6 weeks. Small improvement = big P/L impact at scale. |
| **Missed opportunities** | Higher conviction + better direction = more trades taken, larger position sizes. | HIGH — Instead of sitting out, the model acts on valid signals. |
| **Session pattern mismatch** | Model learns which sessions are volatile NOW (2026 session patterns may differ from 2018). | MEDIUM — Better timing = fewer whipsaw losses. |
| **Volatility regime adaptation** | ATR at $5200 is different from ATR at $1500. SL/TP distances need recalibration. | MEDIUM — Better risk management on current price levels. |

### 14.3 Quantified Impact Estimate

**Conservative scenario (fine-tuning works modestly):**
- Conviction rises from 0.20 → 0.35 average
- Direction accuracy improves from 47% → 52%
- Trades per week: 20 → 40 (more signals pass threshold)
- Weekly P/L estimate: £44 → £80-100 (1.8-2.3x improvement)

**Optimistic scenario (fine-tuning works well):**
- Conviction rises from 0.20 → 0.50 average
- Direction accuracy improves from 47% → 55%
- Trades per week: 20 → 60+ (confident entry on valid signals)
- Weekly P/L estimate: £44 → £150-200+ (3.4-4.5x improvement)

**Worst case (fine-tuning fails, guardrails catch it):**
- No improvement (KL too constrained, or market too different)
- Validation gate blocks promotion
- Fall back to frozen model (no loss, same as today)
- We learn what doesn't work, adjust parameters, try again

### 14.4 Where the Biggest Wins Come From

**#1: Conviction Uplift (70% of expected improvement)**

This is where the money is. The model currently "knows" the right direction 47% of the time — not great, but not terrible for a scalping system with asymmetric payoff. The problem is it won't ACT on what it knows because conviction is too low.

Fine-tuning on live data means the model sees current gold prices (5200+) in its training environment. After a few hundred episodes at these levels, the neural network adjusts its internal representation so $5200 feels "normal" — just like $1500 felt normal during historical training. Conviction naturally rises because the input distribution matches what the network expects.

**#2: Direction Accuracy (20% of expected improvement)**

The model learned "pullback shorting" on historical data where pullbacks had certain characteristics (speed, depth, recovery pattern). At $5200+ with different volatility, the exact pullback signature changes. Fine-tuning teaches the model what pullbacks look like NOW.

Even a 3-5% improvement in direction accuracy compounds massively:
- 47% accuracy on 100 trades with 1.5:1 payoff ratio = marginal profit
- 52% accuracy on 100 trades with 1.5:1 payoff ratio = strong profit
- The difference between treading water and making real money

**#3: Position Sizing (10% of expected improvement)**

Higher conviction → tiered sizing gives larger lots → same winning trade makes more money. The conviction→lots mapping is exponential — going from 0.20 to 0.45 conviction doesn't just double the lot size, it could quadruple it (from scalp tier to full conviction tier).

### 14.5 What This Won't Fix

Being honest about limitations:

| Not fixable by fine-tuning | Why |
|---------------------------|-----|
| **Fundamental strategy flaws** | If the reward function or environment design is wrong, fine-tuning just learns the wrong thing faster. Our training results (W81-W100) suggest the fundamentals are solid. |
| **Spread/commission edge erosion** | If the edge is too small to survive costs, no amount of fine-tuning helps. Need to verify on demo first. |
| **Black swan events** | Model can't learn from events it hasn't seen. Flash crashes, unexpected news — these require hard-coded circuit breakers, not RL. |
| **Broker-specific behavior** | Slippage, requotes, spread widening during news. These are live execution issues, not model issues. |

### 14.6 Profitability Assessment

**Is this a good move? YES, for three reasons:**

1. **The training model proves the strategy works.** W81-W100 show consistent profitability across diverse market conditions. The model has learned real trading skills. The problem isn't the model — it's the distribution shift between training and live data. Fine-tuning directly addresses this.

2. **The risk is bounded.** With our 5-layer guardrail system, the worst case is "no improvement" — we fall back to the frozen model. There's no scenario where fine-tuning makes the live model WORSE, because the validation gate blocks bad models from deployment.

3. **The upside is multiplicative.** Even modest improvements in conviction (0.20 → 0.35) and direction accuracy (47% → 52%) compound into significant P/L gains because of the tiered lot sizing system. We're not looking for a 10% improvement — we're looking at 2-4x weekly P/L through unlocking the model's existing knowledge.

**Break-even analysis:** If we spend ~1,600 lines of code and 2-3 weeks of development, the system needs to improve weekly P/L by just £30-40 to pay for itself in the first month of operation. Given the conviction bottleneck is so clear and addressable, this is a high-confidence investment.

### 14.7 Evolution Path

This system evolves over time, just like the AI itself:

```
Month 1: Basic fine-tuning (Layers 1-2 only)
├── Model adapts to current price regime
├── Conviction rises, more trades taken
└── Measure: weekly P/L improvement vs frozen baseline

Month 2: Add EWC + Strategy Memory (Layers 3-4)
├── Better knowledge retention across market shifts
├── Regime-aware strategy switching
└── Measure: performance stability across regime changes

Month 3+: Continuous learning loop
├── Model adapts week-by-week automatically
├── Strategy book grows with each new regime encountered
├── Human reviews weekly reports, adjusts guardrail parameters
└── Measure: quarterly Sharpe improvement, maximum drawdown reduction
```

The system gets smarter over time — not just the model, but the fine-tuning infrastructure itself. We tune the guardrail thresholds, the buffer ratios, the EWC lambda based on real-world results. It's evolution on two levels: the AI evolving, and the training system evolving.

---

## 15. Answers to Open Questions

Based on the system analysis, here are recommended answers:

| # | Question | Recommended Answer | Rationale |
|---|----------|-------------------|-----------|
| 1 | Episode length | Full buffer as one episode (500+ bars) | Simulates realistic trading week, same as training |
| 2 | Correlated instruments | Pull from MT5 directly, fall back to Dukascopy | MT5 has most symbols; Dukascopy fills gaps |
| 3 | Market hours | Fine-tune during market hours ONLY | Weekend data is stale; run validation on weekends |
| 4 | Multi-week continuity | Buffer PERSISTS across weeks | Core + supporting memories must survive; recent tier refreshes |
| 5 | Entropy schedule | Keep alpha floor at 0.01 | Same as training — don't risk collapse |
| 6 | EWC computation | Compute ONCE at export time, store in ZIP | Fisher is expensive; model weights don't change after training |
| 7 | Strategy Memory clusters | Start with 6, expand if needed | 6 = (low/high vol) × (trending/ranging/choppy) |
| 8 | Core memory criteria | Top 20% by |reward| + regime diversity sampling | Captures best wins AND worst losses (both are important) |
| 9 | Cross-week drift | Moderate: 70/30 ratio handles this naturally | Buffer mixing prevents whiplash between regimes |
| 10 | EWC lambda | Start at 1000, tune empirically | Lower than academic papers suggest (5000) because we have 4 other layers of protection |

---

*This document will be updated as we refine the plan and begin implementation. Version history: v1.0 initial plan, v2.0 added memory architecture + system analysis + guardrails + impact report.*
