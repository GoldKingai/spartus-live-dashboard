# Spartus Trading AI - Complete System Blueprint v3.3

**Project:** Spartus Trading AI
**Purpose:** Train a single unified AI to learn how to trade through experience on historical data
**Target Market:** XAUUSD (Gold) - extensible to other instruments
**Execution:** MetaTrader 5 via Python API
**Training:** Local machine (GPU-accelerated when available)
**Author:** Calvindmp J + AI Co-Developer
**Date:** 2026-02-22

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy: Learn by Trading](#2-core-philosophy)
3. [Lessons Learned from GoldCoTrader](#3-lessons-learned)
4. [Two Separate Systems: Training Engine vs Live Interface](#4-two-separate-systems)
5. [Architecture Overview](#5-architecture-overview)
6. [The Unified Model](#6-the-unified-model)
7. [Account Management: A Core Learning Objective](#7-account-management)
8. [Training Environment (with Realistic Market Simulation)](#8-training-environment)
9. [Data Pipeline & Sources (PRIORITY #1)](#9-data-pipeline)
10. [Feature Engineering](#10-feature-engineering)
11. [Reward Design](#11-reward-design)
12. [Memory System](#12-memory-system)
13. [Training Methodology](#13-training-methodology)
14. [Validation Framework](#14-validation-framework)
15. [Risk Management](#15-risk-management)
16. [Model Portability & Export](#16-model-portability)
17. [Training Engine Dashboard](#17-training-engine-dashboard)
18. [Live Trading Interface (Phase 2)](#18-live-trading-interface)
19. [Project Structure](#19-project-structure)
20. [Technology Stack](#20-technology-stack)
21. [Training Schedule & Milestones](#21-training-schedule)
22. [Success Criteria](#22-success-criteria)

---

## 1. Executive Summary

Spartus Trading AI is a **single unified AI agent** that learns to trade by doing. It is trained on years of historical market data, trading week by week, learning from every win and loss. Over time it builds up experience - it gets better at reading market trends, timing entries, managing positions, and making consistent profit.

**This is NOT two separate models.** This is ONE AI that learns everything through trading experience:
- Reading market flow and predicting where price is going
- Timing entries to catch moves early
- Setting proper stop-losses and take-profits based on what it has learned
- Closing trades quickly when the market turns against it
- Taking consistent small profits - even 1-2 pounds per trade is good if consistent
- Building confidence over weeks/months of simulated trading

**Trend prediction is a METRIC, not a separate system.** The more the AI trades, the better it gets at reading the market. We track its trend prediction accuracy as a measure of how well it understands the market - but it's not a standalone predictor, it's a skill the AI develops through trading.

### How It Works (Simple Version)

```
Week 1: AI starts trading with £100 on historical data
         Makes mistakes, learns what works, what doesn't
         Builds initial memory of patterns and outcomes

Week 2: AI carries forward what it learned
         Starts recognizing: "Oh, when the market does THIS, it usually goes UP"
         Places better trades, fewer mistakes

Week 10: AI has seen 10 different market conditions
          Memory database growing with pattern outcomes
          Starting to time entries better, set smarter SL/TP

Week 50: AI has deep experience across many market regimes
          Trend prediction accuracy improving (tracked as metric)
          Consistently taking small profits, avoiding big losses

Week 100+: Mature AI with rich trading memory
            Knows when to trade and when to sit out
            Quick to react to market reversals
            Ready for live deployment testing
```

### Core Principles

1. **Learn by doing:** The AI trades on historical data and learns from outcomes
2. **Build memory over time:** Every trade outcome is stored, patterns are tracked
3. **Trend reading develops naturally:** The better the AI trades, the better it reads the market
4. **Consistent profit over big wins:** Even small wins matter if they're consistent
5. **Quick reactions:** Close losing trades fast, ride winners as long as safe
6. **No data leakage:** The AI never sees future data during training
7. **Simple foundation, smart execution:** Few features, clean reward, proper validation
8. **Account management is learned, not hardcoded:** AI learns to size positions based on balance, scale down in drawdown, grow from £100
9. **Model is portable:** Trained model can be exported, shared, and loaded by anyone on any machine

---

## 2. Core Philosophy: Learn by Trading

### How Humans Learn to Trade

A human trader learns by:
1. Watching the market, seeing patterns develop
2. Placing a trade based on what they see
3. Watching the outcome - did it work?
4. Remembering: "Last time the market looked like this and I went long, I won/lost"
5. Over months/years, building intuition about market flow
6. Getting faster at recognizing setups and reacting to reversals

### How Spartus Learns to Trade

Spartus follows the same path, but on historical data:
1. **Observes:** Sees current price, indicators, volume, multi-timeframe context
2. **Decides:** Buy, sell, hold, close, adjust stop-loss
3. **Experiences outcome:** Trade hits TP, hits SL, or gets closed manually
4. **Remembers:** Stores pattern + context + outcome in memory database
5. **Improves:** Over weeks of training, learns which setups work in which conditions
6. **Develops trend sense:** As memory grows, it can "feel" market direction better

### What Makes This Different from GoldCoTrader

| Aspect | GoldCoTrader (Failed) | Spartus (New) |
|--------|----------------------|---------------|
| Architecture | RL agent with 3000+ features | RL agent with 30-40 focused features |
| Learning | Confused by 20+ reward signals | Clear reward: make money or don't |
| Memory | Pattern databases existed but were disconnected | Unified memory system that directly feeds observations |
| Trend prediction | Dead code (Phase 4 never activated) | Tracked as accuracy metric, naturally improves with experience |
| Position management | Hardcoded trailing overrode AI decisions | AI controls everything, hard rules only for risk limits |
| Targets | Fixed pip values regardless of volatility | AI learns what targets work in different conditions |
| Validation | None (trained and tested on same data) | Walk-forward + purged cross-validation |
| Complexity | 71 files, 4600-line environment | ~25 files, <500 lines each |

---

## 3. Lessons Learned from GoldCoTrader

Every design decision in Spartus exists to prevent mistakes made in the previous system.

### FATAL MISTAKES (caused complete system failure)

| # | Mistake | Impact | Spartus Prevention |
|---|---------|--------|-------------------|
| 1 | **Look-ahead normalization**: Min/max used entire dataset including future prices | 15-30% fake performance boost | Expanding-window normalization only. Unit test verifies no future data leakage |
| 2 | **No train/test split**: Model evaluated on same data it trained on | No way to know if model actually learned anything | Walk-forward validation + Purged Cross-Validation + paper trading |
| 3 | **3,000+ observation features**: LSTM couldn't extract signal from noise | Underfitting despite appearing to train | 30-40 curated features max, validated with SHAP importance |
| 4 | **Insufficient training data**: 45K steps for 3000-dim space = 0.003% of needed data | Model memorized noise instead of learning | Match data to model capacity. More data per feature |

### CRITICAL MISTAKES (severely hurt learning)

| # | Mistake | Impact | Spartus Prevention |
|---|---------|--------|-------------------|
| 5 | **20+ conflicting reward components** | AI couldn't figure out what "good" meant | 5-component composite reward with normalization, no conflicts |
| 6 | **Fixed pip targets (10/15/25 pips)** | Got stopped out in volatile markets, targets unreachable in quiet ones | ATR-scaled targets OR let AI learn its own targets |
| 7 | **Automatic trailing overrode AI** | AI couldn't learn stop management because environment ignored its decisions | AI controls entries/exits. Hard rules only for max risk limits |
| 8 | **ent_coef=0.10 (entropy 5-100x too high)** | Policy stayed random, never converged | Use SAC (auto-tunes entropy) or start PPO at 0.01 |
| 9 | **Phase 4 market state rewards never activated** | Built entire system that produced zero signal | Every code path tested. No dead features |
| 10 | **71 Python files, 4600-line environment** | Impossible to debug or isolate issues | <30 files, <500 lines each, modular design |

### DESIGN MISTAKES (wrong approach)

| # | Mistake | Impact | Spartus Prevention |
|---|---------|--------|-------------------|
| 11 | **Discrete action space** (8 fixed actions) | Couldn't adapt position sizing or targets to conditions | Continuous action space for nuanced decisions |
| 12 | **Multi-position support was fake** | Code tracked list but only used positions[0] | Either proper multi-position or single-position cleanly |
| 13 | **gamma=0.99 for weekly episodes** | Discounted distant rewards to near-zero | gamma=0.95-0.97 for non-stationary markets |
| 14 | **LSTM hidden state reset every 512 steps** | Lost temporal context mid-trade | Proper episode boundaries aligned with LSTM |

---

## 4. Two Separate Systems: Training Engine vs Live Interface

**CRITICAL DESIGN DECISION:** Spartus is TWO separate applications that share ONE model.

```
┌──────────────────────────────────────┐     ┌───────────────────────────────────┐
│       SYSTEM 1: TRAINING ENGINE      │     │   SYSTEM 2: LIVE INTERFACE        │
│       (This is built FIRST)          │     │   (Built AFTER training works)    │
│                                      │     │                                   │
│  PURPOSE:                            │     │  PURPOSE:                          │
│  Train the AI model on historical    │     │  Load a trained model and let it   │
│  data, week by week, building up     │     │  trade LIVE on MetaTrader 5       │
│  experience and memory.              │     │                                   │
│                                      │     │                                   │
│  RUNS:                               │     │  RUNS:                            │
│  On YOUR machine, for days/weeks     │     │  On any machine with MT5 + model  │
│  while training the model.           │     │  Runs continuously during market  │
│  NOT connected to live market.       │     │  hours on real money.             │
│                                      │     │                                   │
│  COMPONENTS:                         │     │  COMPONENTS:                      │
│  • Data loader (historical data)     │     │  • Model loader (loads .zip)      │
│  • Training environment (simulator)  │     │  • MT5 connector (live trades)    │
│  • SAC agent (learning)              │     │  • Feature builder (live data)    │
│  • Memory system (SQLite)            │     │  • Risk manager (hard rules)      │
│  • Monitoring dashboard              │     │  • Qt6 Dashboard UI               │
│  • Validation framework              │     │  • Telegram integration           │
│                                      │     │                                   │
│  OUTPUTS:                            │     │  INPUTS:                          │
│  ✓ Trained model (.zip file)         │────>│  ✓ Trained model (.zip file)      │
│  ✓ Memory databases (.db)            │────>│  ✓ Memory databases (.db)         │
│  ✓ Training reports & metrics        │     │  ✓ Live MT5 market data           │
│  ✓ Feature scaler configs (.json)    │────>│  ✓ Feature scaler configs (.json) │
│                                      │     │                                   │
│  HAS ITS OWN DASHBOARD:             │     │  HAS ITS OWN DASHBOARD:           │
│  • Training progress (weeks done)    │     │  • Live positions & P/L           │
│  • Balance curve & drawdown          │     │  • AI confidence & trend read     │
│  • Win rate evolution                │     │  • Account status                 │
│  • AI decision log (what it's doing) │     │  • Activity log                   │
│  • Errors & warnings                 │     │  • Start/Stop/Kill switch         │
│  • Memory stats (patterns learned)   │     │  • Mode: Advisor / Auto           │
│                                      │     │                                   │
└──────────────────────────────────────┘     └───────────────────────────────────┘
```

### Why Two Systems?

1. **Training takes days/weeks** - you don't want live market connection during training
2. **Training needs simulation** - fake trades on historical data, not real money
3. **Live needs real execution** - MT5 orders, real spreads, real money
4. **Model is portable** - train on one machine, deploy on another (or share with others)
5. **Different monitoring needs** - training cares about learning progress; live cares about profit & risk

### The Handoff: Training → Live

```
1. Training engine trains model for 100+ weeks of historical data
2. Model passes validation (walk-forward + held-out test)
3. Training engine EXPORTS:
   ├── spartus_model_v1.zip         (SAC model weights)
   ├── spartus_memory.db            (trade history, patterns, trend data)
   ├── spartus_scaler_config.json   (normalization parameters)
   └── spartus_training_report.json (performance metrics)

4. Live interface IMPORTS the above files
5. Live interface connects to MT5, starts trading with the trained model
6. Live interface continues building memory from real trades
```

---

## 5. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SPARTUS TRADING AI                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │  MT5 Data    │───>│  Data Engine  │───>│  Feature Builder        │ │
│  │  (OHLCV +    │    │  • Validate   │    │  • 30-40 features       │ │
│  │   Tick Vol)  │    │  • Clean      │    │  • Expanding normalize  │ │
│  └─────────────┘    └──────────────┘    │  • Fractional diff      │ │
│                                          └──────────┬──────────────┘ │
│                                                      │                │
│                                                      ▼                │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │              SPARTUS UNIFIED TRADING AGENT                    │   │
│  │              (Single RL Model - SAC or PPO)                   │   │
│  │                                                               │   │
│  │  OBSERVES:                                                    │   │
│  │  ├── Market features (price, volume, indicators)  [~25]       │   │
│  │  ├── Multi-timeframe context (H1/H4/D1 trends)   [~5]        │   │
│  │  ├── Account state (equity, position, drawdown)   [~5]        │   │
│  │  ├── Memory context (pattern history, win rates)  [~5]        │   │
│  │  └── Time context (session, day of week)          [~3]        │   │
│  │                                           Total: ~40 features │   │
│  │                                                               │   │
│  │  DECIDES (4 Continuous Actions):                               │   │
│  │  ├── Direction: -1.0 (short) to +1.0 (long), 0 = flat        │   │
│  │  ├── Conviction: 0.0 (min size) to 1.0 (max size)            │   │
│  │  ├── Exit urgency: 0.0 (hold) to 1.0 (close immediately)    │   │
│  │  └── SL management: 0.0 (hold SL) to 1.0 (trail tight)      │   │
│  │                                                               │   │
│  │  LEARNS OVER TIME:                                            │   │
│  │  ├── When to enter (reads market flow)                        │   │
│  │  ├── When to exit (quick exits on reversals)                  │   │
│  │  ├── How to trail SL (lock in profit, let winners run)        │   │
│  │  ├── How much to risk (scales with confidence)                │   │
│  │  ├── What patterns work (stores outcomes in memory)           │   │
│  │  └── Market trend direction (tracked as accuracy metric)      │   │
│  │                                                               │   │
│  └──────────────────────────────┬────────────────────────────────┘   │
│                                 │                                     │
│                    ┌────────────┴────────────┐                       │
│                    ▼                         ▼                        │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  HARD RISK RULES     │  │  MEMORY SYSTEM                       │ │
│  │  (Not learned -      │  │  (Persistent across training weeks)  │ │
│  │   enforced by code)  │  │                                      │ │
│  │                      │  │  • Pattern outcomes database          │ │
│  │  • Max loss per trade│  │  • Trend prediction tracker           │ │
│  │  • Max daily DD: 3%  │  │  • Trade history & statistics         │ │
│  │  • Max total DD: 10% │  │  • Market condition memory            │ │
│  │  • Max positions: 1  │  │  • Model checkpoints                  │ │
│  │  • Kill switch       │  │                                      │ │
│  └──────────────────────┘  └──────────────────────────────────────┘ │
│                                 │                                     │
│                                 ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │  MT5 EXECUTION (Live Mode)                                       ││
│  │  • Convert AI decisions to MT5 orders                            ││
│  │  • Position monitoring & management                               ││
│  │  • Emergency flatten capability                                   ││
│  └──────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │  DASHBOARD                                                        ││
│  │  • Live trend prediction accuracy tracking                        ││
│  │  • Account status, open positions, P/L                            ││
│  │  • AI confidence levels, memory stats                             ││
│  │  • Training progress (weeks completed, win rate evolution)        ││
│  └──────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Unified Model

### Why One Model, Not Two

The previous analysis suggested splitting prediction and execution into two models. After re-examining how GoldCoTrader actually worked - and understanding the core vision - a single unified model is the right approach:

1. **Trading IS prediction.** When the AI opens a buy, it's predicting the market will go up. When it closes, it's predicting the move is over. Every trade IS a prediction.
2. **Memory bridges prediction and execution.** As the AI trades and stores outcomes, it naturally develops trend-reading ability. The memory system is what connects past experience to future decisions.
3. **LLMs learn this way.** Large language models learn language by predicting the next token - not by having separate grammar and vocabulary models. Similarly, a trading AI learns market flow by trading - not by separating prediction from execution.
4. **The GoldCoTrader was right about this.** The core concept of learning-by-trading was correct. What was wrong was the implementation (leaky data, too many features, broken rewards, etc.).

### Algorithm Choice: SAC (Soft Actor-Critic)

**Why SAC for this unified agent:**

| Requirement | Why SAC Fits |
|-------------|-------------|
| Continuous actions (position size, direction) | SAC natively supports continuous action spaces |
| Auto-tuned exploration | SAC automatically balances exploration vs exploitation (no manual ent_coef) |
| Sample efficient | Uses replay buffer - learns from past experiences multiple times |
| Off-policy learning | Can learn from historical data, not just current policy |
| Stable training | Maximum entropy framework prevents collapse to single strategy |
| Memory-compatible | Replay buffer acts like short-term memory; external memory adds long-term |

**Alternative: PPO with proper tuning**
- If SAC proves unstable for trading, PPO (Proximal Policy Optimization) is the fallback
- PPO was used in GoldCoTrader - the algorithm itself wasn't the problem, the implementation was
- Key fixes for PPO if used: ent_coef=0.01, n_steps=2048, n_epochs=2 (not 4)

### Action Space (Continuous)

```python
# 4 continuous outputs from the policy network
action_space = gymnasium.spaces.Box(
    low=np.array([-1.0, 0.0, 0.0, 0.0]),
    high=np.array([1.0, 1.0, 1.0, 1.0]),
    dtype=np.float32
)

# action[0]: Direction signal
#   -1.0 = strong short signal
#    0.0 = no signal (stay flat or hold current position)
#   +1.0 = strong long signal
#   THRESHOLD: |signal| > 0.3 to open a new position (prevents noise trades)

# action[1]: Conviction / position sizing
#   0.0 = minimum size (if opening)
#   1.0 = maximum size (capped by risk rules)
#   Scales linearly within risk limits

# action[2]: Exit urgency
#   0.0 = hold current position (no exit)
#   > 0.5 = close position (the higher, the more urgent)
#   1.0 = emergency close (AI detected reversal)

# action[3]: Stop-loss management (ONLY active when in a trade)
#   0.0 = leave SL where it is (give trade room to breathe)
#   0.3 = move SL to breakeven (lock in zero-loss once in profit)
#   0.5 = trail SL moderately (protect some profit, still give room)
#   0.8 = trail SL tight (lock in most of the profit)
#   1.0 = trail SL to minimum distance (maximum profit lock, risk getting stopped)
#
#   HOW THIS WORKS:
#   When the trade is in PROFIT and action[3] > 0.2:
#     → new_sl = entry_price + (current_profit × action[3])  (for longs)
#     → SL can ONLY move in the AI's favor (tighten), never loosen
#     → Minimum SL distance: 0.5 ATR (prevent noise stop-outs)
#
#   This teaches the AI to:
#     → Hold winners: keep action[3] low while trend continues
#     → Lock in profit: raise action[3] when reversal signs appear
#     → Trail intelligently: balance between protecting profit and giving room
```

### How the AI Interprets Actions

```python
def interpret_action(action, current_position, market_state):
    direction = action[0]       # -1 to +1
    conviction = action[1]      # 0 to 1
    exit_signal = action[2]     # 0 to 1
    sl_adjustment = action[3]   # 0 to 1

    # CASE 1: No position open
    if current_position is None:
        if abs(direction) > 0.3:  # Strong enough signal
            side = "LONG" if direction > 0 else "SHORT"
            size = calculate_size(conviction, account_equity, current_atr)

            # MANDATORY: Every trade MUST have a stop-loss (hard rule)
            sl = calculate_sl(side, current_price, current_atr, conviction)
            tp = calculate_tp(side, current_price, current_atr, conviction)

            return OpenTrade(side, size, sl, tp)
        else:
            return Hold()  # Signal too weak, wait

    # CASE 2: Position open
    else:
        # Priority 1: Exit check
        if exit_signal > 0.5:
            return CloseTrade(current_position)  # AI wants out

        # Priority 2: Direction flip check
        if current_position.side == "LONG" and direction < -0.5:
            return CloseTrade(current_position)  # Close long, might open short next step
        if current_position.side == "SHORT" and direction > 0.5:
            return CloseTrade(current_position)  # Close short, might open long next step

        # Priority 3: Stop-loss management (trail / lock in profit)
        if sl_adjustment > 0.2:
            new_sl = adjust_stop_loss(current_position, sl_adjustment, market_state)
            if new_sl is not None:
                return ModifyStopLoss(current_position, new_sl)

        return Hold()  # Maintain position, SL unchanged


def adjust_stop_loss(position, sl_adjustment, market_state):
    """
    AI-controlled stop-loss trailing.

    The AI learns WHEN and HOW MUCH to trail its stop-loss:
    - Low sl_adjustment (0.0-0.2): Leave SL alone → let the trade breathe
    - Medium (0.3-0.5): Move SL to breakeven or modest trail → protect capital
    - High (0.6-0.8): Trail tightly → lock in most profit
    - Max (0.9-1.0): Trail to minimum distance → maximum protection, risk stop-out

    RULES (hard-coded, not learned):
    - SL can ONLY move in the AI's favor (tighten), NEVER loosen
    - Minimum SL distance: 0.5 × ATR (prevent noise stop-outs)
    - SL only trails when position is in profit (no breakeven in a loss)
    """
    current_price = market_state['current_price']
    atr = market_state['atr_14']
    min_sl_distance = atr * 0.5  # Hard minimum

    if position['side'] == 'LONG':
        current_profit = current_price - position['entry_price']
        if current_profit <= 0:
            return None  # Not in profit, don't trail

        # Calculate new SL based on AI's adjustment signal
        # Higher sl_adjustment = SL moves closer to current price
        trail_amount = current_profit * sl_adjustment
        new_sl = position['entry_price'] + trail_amount

        # Enforce: SL can only tighten (move UP for longs)
        new_sl = max(new_sl, position['stop_loss'])

        # Enforce: minimum distance from current price
        max_sl = current_price - min_sl_distance
        new_sl = min(new_sl, max_sl)

        if new_sl > position['stop_loss']:
            return new_sl  # SL improved

    elif position['side'] == 'SHORT':
        current_profit = position['entry_price'] - current_price
        if current_profit <= 0:
            return None  # Not in profit, don't trail

        trail_amount = current_profit * sl_adjustment
        new_sl = position['entry_price'] - trail_amount

        # Enforce: SL can only tighten (move DOWN for shorts)
        new_sl = min(new_sl, position['stop_loss'])

        # Enforce: minimum distance from current price
        min_sl = current_price + min_sl_distance
        new_sl = max(new_sl, min_sl)

        if new_sl < position['stop_loss']:
            return new_sl  # SL improved

    return None  # No improvement possible


def calculate_sl(side, price, atr, conviction):
    """
    MANDATORY stop-loss at entry.
    AI conviction influences SL distance: higher conviction → tighter SL (more risk).
    Hard rules ensure minimum distance.

    EVERY trade MUST have an SL. This is non-negotiable.
    """
    # AI conviction scales SL distance: 1.0-2.5 ATR
    # High conviction → tighter (1.0 ATR) → believes in the trade
    # Low conviction → wider (2.5 ATR) → less sure, needs room
    sl_atr_multiple = 2.5 - (conviction * 1.5)  # Maps 0→2.5, 1→1.0
    sl_atr_multiple = max(sl_atr_multiple, 1.0)  # Hard minimum: 1.0 ATR

    sl_distance = atr * sl_atr_multiple

    if side == "LONG":
        return price - sl_distance
    else:
        return price + sl_distance


def calculate_tp(side, price, atr, conviction):
    """
    Take-profit at entry. AI conviction influences TP distance.
    Higher conviction → further TP (expects bigger move).
    """
    # AI conviction scales TP distance: 1.5-4.0 ATR
    tp_atr_multiple = 1.5 + (conviction * 2.5)  # Maps 0→1.5, 1→4.0

    tp_distance = atr * tp_atr_multiple

    if side == "LONG":
        return price + tp_distance
    else:
        return price - tp_distance
```

### What the AI Learns About Trade Management

```
LESSON 1: Let winners run
  → When in profit and trend continues, keep sl_adjustment LOW (0.0-0.2)
  → The per-step P/L reward keeps giving positive signal
  → AI learns: "market still moving my way, don't touch the SL"

LESSON 2: Lock in profit when reversal signs appear
  → When in profit but momentum fading, raise sl_adjustment (0.5-0.8)
  → SL moves into profit territory, guaranteeing a win
  → AI learns: "trend weakening, protect what I've gained"

LESSON 3: Cut losers immediately
  → When trade goes against AI, raise exit_signal above 0.5
  → Trade closes at market with controlled loss
  → Per-step negative reward teaches: "don't sit in losing trades"

LESSON 4: Stop-loss is non-negotiable
  → EVERY trade has an SL at entry (hard rule, not learned)
  → SL based on ATR × conviction factor
  → Even if AI outputs nonsense, the hard rule places an SL
  → This guarantees the account cannot be blown by a single trade
```

### Network Architecture

```python
# Policy network for SAC
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],    # Actor (policy) network: 2 layers of 256
        "qf": [256, 256]     # Critic (Q-value) network: 2 layers of 256
    },
    "activation_fn": torch.nn.ReLU,
}

# For temporal awareness, we have two options:

# OPTION A: SAC with frame stacking (simple, recommended to start)
# Stack last 10 observations as input → 42 features × 10 = 420 input dims
# This gives the MLP temporal context without LSTM complexity

# OPTION B: Custom LSTM policy for SAC (if Option A is insufficient)
# Requires custom policy class with LSTM layer before actor/critic heads
# More complex but handles long-term dependencies better
# Implement only if Option A proves insufficient

# START WITH OPTION A. Only move to B if needed.

# OPTION C: SimbaV2-style normalization (Phase 2 enhancement)
# Based on SimbaV2 (arXiv 2502.15280, Feb 2026) which achieved SOTA on 57
# continuous control tasks using SAC with hyperspherical normalization.
# Key changes from standard MLP:
#   1. Replace LayerNorm with L2-normalization on weight matrices
#      (projects weights onto unit hypersphere — prevents co-adaptation)
#   2. Add distributional critic (model full return distribution, not just mean)
#      This gives richer gradient signal, especially for tail risk awareness
#   3. Reward scaling: normalize TD-targets by running statistics
#      (complementary to our RewardNormalizer — this normalizes at the critic level)
#
# Implementation requires custom SB3 policy class:
#   class SimbaV2Policy(SACPolicy):
#       def _build(self):
#           # Replace nn.Linear layers with L2-normalized version
#           # Replace critic head with distributional output (N quantiles)
#
# When to try Option C:
#   - After Option A has been validated and shows promise
#   - If critic loss is unstable or Q-values are noisy
#   - If the agent struggles with tail risk (rare large losses)
#
# Expected benefits: more stable critic, better tail risk handling,
# faster convergence on complex reward landscapes
```

---

## 7. Account Management: A Core Learning Objective

### Why This Matters

The AI isn't just learning to trade - **it's learning to manage an account**. Starting from £100, it must:

1. **Size positions based on what it can afford** - not place crazy lot sizes
2. **Scale DOWN when losing** - smaller lots during drawdown to survive
3. **Scale UP when winning** - gradually increase size as account grows
4. **Never risk too much on one trade** - hard rules cap it, but the AI should LEARN to be conservative
5. **Survive bankruptcy scenarios** - if balance hits £0, the episode ends, strong negative reward

### How the AI Learns Account Management

The account state is **directly observable** - the AI sees:

```python
# Account & position features in observation vector (Section 10.G, features #30-37)
# 8 features total — NOT normalized (already ratios/flags)
account_features = {
    'has_position':      1.0 if position_open else 0.0,    # #30: Binary position flag
    'position_side':     +1.0 if long, -1.0 if short, 0.0, # #31: Direction awareness
    'unrealized_pnl':    position_pnl / balance,            # #32: Current position P/L
    'position_duration': bars_since_entry / 100,             # #33: How long held
    'current_drawdown':  (peak_equity - current_equity) / peak_equity,  # #34: Risk awareness
    'equity_ratio':      current_equity / initial_balance,   # #35: Overall performance
    'sl_distance_ratio': (current_price - stop_loss) / atr,  # #36: SL proximity (0 if flat)
    'profit_locked_pct': (stop_loss - entry_price) / atr,  # #37: Profit locked by trail (ATR-scaled)
    # NOTE: Using ATR as denominator, not entry_price. For XAUUSD at ~$2650,
    # entry_price denominator gives ~0.0004 (invisible to network).
    # ATR denominator gives ~0.5-2.5 (meaningful signal).
    # Positive = SL is above entry (profit locked). Zero when flat.
}
```

Combined with the **conviction output** (action[1]), the AI learns:
- High equity_ratio → can afford more conviction → larger position
- High current_drawdown → should reduce conviction → smaller position
- Very low balance → barely any conviction → minimum size or flat

### Position Sizing: AI Conviction + Hard Rules

```python
def calculate_lot_size(conviction, balance, atr, symbol_info):
    """
    AI's conviction (0-1) determines position size WITHIN hard limits.

    The AI learns:
    - Low conviction (0.0-0.3) → minimum lot (0.01)
    - Medium conviction (0.3-0.7) → moderate lot based on balance
    - High conviction (0.7-1.0) → larger lot (still capped by risk rules)

    Hard rules ALWAYS apply:
    - Max 2% of equity risked per trade
    - Minimum lot: 0.01 (broker minimum)
    - Maximum lot: scaled by account size
    """
    # AI's desired risk level
    ai_risk_pct = conviction * 0.02  # 0% to 2% (conviction scales within max)

    # Hard cap: never risk more than 2%
    risk_pct = min(ai_risk_pct, 0.02)
    risk_amount = balance * risk_pct

    # Convert to lots using ATR for stop distance
    sl_distance = atr * 1.5  # 1.5 ATR stop
    point_value = symbol_info['trade_tick_value'] / symbol_info['trade_tick_size']

    lots = risk_amount / (sl_distance * point_value)

    # Broker limits
    lots = max(0.01, lots)                    # Minimum lot
    lots = min(lots, symbol_info['volume_max'])  # Broker max

    # ACCOUNT SIZE SCALING: Scale max lot by account size
    # £100 account → max 0.05 lots
    # £500 account → max 0.25 lots
    # £1000 account → max 0.50 lots
    account_max_lot = balance / 2000  # £2000 per 1 lot max
    lots = min(lots, max(0.01, account_max_lot))

    # Round to broker step
    lot_step = symbol_info.get('volume_step', 0.01)
    lots = round(lots / lot_step) * lot_step

    return lots

def lot_size_during_drawdown(conviction, balance, peak_balance, atr, symbol_info):
    """
    During drawdown, FURTHER reduce position size.
    This is a hard rule (not learned), but the AI ALSO learns
    to output lower conviction during drawdown periods.

    IMPORTANT: Drawdown reduction is applied as a SINGLE scaling factor
    (not stacked multipliers) to avoid going below broker minimum.
    After scaling, we verify the risk at 0.01 lots doesn't exceed 2%.
    """
    base_lots = calculate_lot_size(conviction, balance, atr, symbol_info)

    drawdown_pct = (peak_balance - balance) / peak_balance

    # Single scaling factor based on worst drawdown tier (NOT stacked)
    if drawdown_pct > 0.10:       # 10% drawdown
        dd_scale = 0.0            # No new trades — too risky
    elif drawdown_pct > 0.08:     # 8% drawdown
        dd_scale = 0.30           # 70% reduction
    elif drawdown_pct > 0.05:     # 5% drawdown
        dd_scale = 0.60           # 40% reduction
    else:
        dd_scale = 1.0            # No reduction

    # At 10%+ drawdown: no new trades at all
    if dd_scale == 0.0:
        return 0.0  # Signal: DO NOT TRADE

    scaled_lots = base_lots * dd_scale
    scaled_lots = max(0.01, scaled_lots)  # Broker minimum

    # SAFETY CHECK: Verify 0.01 lots doesn't exceed 2% risk
    # If even minimum lot exceeds risk limit, DO NOT TRADE
    sl_distance = atr * 1.5  # Approximate SL distance
    point_value = symbol_info['trade_tick_value'] / symbol_info['trade_tick_size']
    risk_at_min_lot = 0.01 * sl_distance * point_value
    if risk_at_min_lot > balance * 0.02:
        return 0.0  # Even 0.01 lots exceeds 2% risk — sit out

    return scaled_lots
```

### Bankruptcy & Recovery

```
SCENARIO: AI blows the account (balance reaches £0 or near it)

TRAINING BEHAVIOR:
    → Episode ends immediately
    → Terminal penalty: reward = -5.0 with done=True (within normalizer range)
    → Balance resets to £100 for next training episode
    → Memory KEEPS all the trades (including the bad ones)
    → AI learns from the bankruptcy: "what I did wrong last time"
    → Over many episodes, AI learns to avoid this situation

THIS IS IMPORTANT:
    → Early in training, the AI WILL blow accounts. This is normal.
    → Each bankruptcy teaches the AI about risk management.
    → Over time, bankruptcies become rarer as the AI learns lot sizing.
    → A mature model should have ZERO bankruptcies in validation.

HARD SAFETY NET (all implemented in step()):
    → If total drawdown >= 10% → force close + episode ends (reward = -4.0)
    → If daily drawdown > 3% → force close + no new trades for rest of day
    → If account blown (balance <= 0) → episode ends (reward = -5.0)
    → All terminal penalties use SET (=), not ADD (-=), to stay within
      the RewardNormalizer's [-5, +5] clip range. done=True zeros out
      bootstrapped future returns, making -5.0 terminal already very severe.
```

### What Good Account Management Looks Like

```
WEEK 1 (Balance: £100):
    → AI opens 0.01 lot (minimum) because conviction is low
    → Learns basic entries/exits
    → Might lose £5-10, might gain £2-3

WEEK 20 (Balance: £115):
    → AI opens 0.01-0.02 lots (growing confidence)
    → Starting to scale based on conviction
    → Better at cutting losses early

WEEK 50 (Balance: £140):
    → AI opens 0.01-0.03 lots depending on setup quality
    → During drawdown, drops back to 0.01
    → Taking £1-2 profit trades consistently

WEEK 100 (Balance: £200):
    → AI opens 0.02-0.05 lots (account supports more)
    → Still conservative during drawdown
    → Knows when to sit out (flat periods)
    → Consistent small gains, rare big losses
```

---

## 8. Training Environment

### Environment Design (< 500 lines)

**CRITICAL: The training environment MUST simulate real market conditions accurately.**

#### Realistic Market Simulation

The training environment is NOT a simplified game - it must mirror real trading:

```python
# Simulation realism configuration
SIMULATION_CONFIG = {
    # SPREAD SIMULATION (XAUUSD typical spreads)
    "spread_base_pips": 2.0,          # Base spread: 2 pips (20 cents on gold)
    "spread_london_pips": 1.5,        # London session: tighter spreads
    "spread_ny_pips": 2.0,            # NY session: normal spreads
    "spread_asia_pips": 3.0,          # Asia session: wider spreads
    "spread_news_multiplier": 3.0,    # During high-vol events: 3x spread
    "spread_off_hours_pips": 5.0,     # Off-hours: widest spreads

    # SLIPPAGE SIMULATION
    "slippage_mean_pips": 0.5,        # Average slippage: 0.5 pips
    "slippage_std_pips": 0.3,         # Slippage variation
    "slippage_high_vol_mean": 2.0,    # High volatility: more slippage

    # COMMISSION (broker-dependent)
    "commission_per_lot": 7.0,        # $7 per lot round-trip (typical ECN)

    # EXECUTION DELAY
    "execution_delay_bars": 0,        # For M5: execute on same bar (conservative)
    # Note: We could add 1-bar delay for even more realism

    # FILL QUALITY
    "partial_fills": False,           # Keep simple for now
    "requotes": False,                # Keep simple for now

    # DOMAIN RANDOMIZATION (Anti-Overfitting)
    # Randomizes simulation parameters each episode to prevent memorization.
    # The AI must learn robust strategies that work across a range of conditions,
    # not strategies tuned to exact historical spread/slippage values.
    "noise_config": {
        "spread_jitter": 0.30,        # ±30% random variation on spreads
        "slippage_jitter": 0.50,      # ±50% random variation on slippage
        "commission_jitter": 0.20,    # ±20% random variation on commissions
        "observation_noise_std": 0.02, # Gaussian noise σ=0.02 added to market features
                                       # (NOT applied to position/account/memory features)
        "start_offset_max": 50,       # Randomly skip 0-50 bars at episode start
                                       # (prevents memorizing exact starting patterns)
    }
}
```

```python
class SpartusTradeEnv(gymnasium.Env):
    """
    Unified trading environment for Spartus AI.

    The AI trades on historical M5/M15/H1 data one week at a time.
    It observes market features, makes trading decisions, and receives
    rewards based on trading outcomes.

    REALISTIC SIMULATION:
    - Variable spreads based on session/volatility
    - Random slippage on entries and exits
    - Commissions deducted from balance
    - Stop-loss hit detection uses HIGH/LOW (not just close)
    - No partial fills or requotes (kept simple)

    Key differences from GoldCoTrader:
    - 42 features, not 3000
    - 4 continuous actions, not 8 discrete
    - 5-component composite reward with normalization, not 20+ components
    - Expanding-window normalization (no look-ahead)
    - Memory context from persistent databases
    - Realistic simulation of spreads, slippage, commissions
    """

    def __init__(self, market_data, config):
        super().__init__()

        self.data = market_data          # Dict of {M5: df, H1: df, H4: df, D1: df}
        self.config = config
        self.feature_builder = FeatureBuilder(config)
        self.memory = TradingMemory(config.memory_db_path)
        self.sim_config = SIMULATION_CONFIG

        # Observation: 42 features (420 with frame stacking)
        obs_dim = config.num_features * config.frame_stack
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # Action: 4 continuous values (direction, conviction, exit, SL management)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Trading state
        self.balance = config.initial_balance  # £100 starting
        self.peak_balance = config.initial_balance
        self.position = None              # Single position (keep it simple)
        self.trades_history = []
        self.current_step = 0

    def _get_spread(self, bar):
        """Calculate realistic spread based on session and volatility."""
        hour = bar['time'].hour if hasattr(bar['time'], 'hour') else 12
        if 8 <= hour < 12:
            spread = self.sim_config['spread_london_pips']
        elif 13 <= hour < 17:
            spread = self.sim_config['spread_ny_pips']
        elif 0 <= hour < 8:
            spread = self.sim_config['spread_asia_pips']
        else:
            spread = self.sim_config['spread_off_hours_pips']
        # Convert pips to price (gold: 1 pip = $0.10)
        return spread * 0.10

    def _get_slippage(self):
        """Random slippage simulation."""
        return np.random.normal(
            self.sim_config['slippage_mean_pips'],
            self.sim_config['slippage_std_pips']
        ) * 0.10  # Convert to price

    def _get_entry_price(self, bar, side):
        """Realistic entry price: close + spread + slippage."""
        spread = self._get_spread(bar)
        slippage = abs(self._get_slippage())
        if side == 'LONG':
            return bar['close'] + spread / 2 + slippage  # Buy at ask + slippage
        else:
            return bar['close'] - spread / 2 - slippage  # Sell at bid - slippage

    def _get_exit_price(self, bar, side):
        """Realistic exit price: close - spread - slippage (opposite of entry)."""
        spread = self._get_spread(bar)
        slippage = abs(self._get_slippage())
        if side == 'LONG':
            return bar['close'] - spread / 2 - slippage  # Sell at bid - slippage
        else:
            return bar['close'] + spread / 2 + slippage  # Buy at ask + slippage

    def _deduct_commission(self, lots):
        """Deduct round-trip commission."""
        commission = lots * self.sim_config['commission_per_lot']
        self.balance -= commission

    def step(self, action):
        """Execute one trading step."""
        # 1. Get current market state
        current_bar = self.data['M5'].iloc[self.current_step]
        current_price = current_bar['close']

        # 2. Check SL/TP using HIGH/LOW (realistic - not just close)
        if self.position:
            self._check_sl_tp_realistic(current_bar)  # Uses high/low
            self._update_position_pnl(current_price)

        # 3. Execute AI's decision (with spread + slippage)
        reward = self._execute_action(action, current_bar)

        # 4. Advance time
        self.current_step += 1
        done = self.current_step >= len(self.data['M5']) - 1
        truncated = False

        # 5. Check circuit breakers (bankruptcy, drawdown limits)
        #
        # IMPORTANT: Terminal penalties use SET (=), not ADD (-=).
        # The reward from _execute_action has already been normalized to [-5, +5].
        # Using -= would create outlier rewards (e.g., -15.0) that destabilize
        # SAC's auto-entropy tuning. Instead, we SET the reward to boundary values.
        # Combined with done=True (which zeros out bootstrapped future returns),
        # a terminal reward of -5.0 is already a massive penalty.
        #
        current_equity = self._get_equity()
        current_dd = (self.peak_balance - current_equity) / self.peak_balance if self.peak_balance > 0 else 0.0

        if self._check_account_blown():
            reward = -5.0   # Maximum penalty (normalizer clip boundary)
            done = True
        elif current_dd >= 0.10:
            # Total drawdown >= 10% → end episode (hard rule)
            # Lot sizing already prevents new trades at 10% DD, but open
            # positions could push DD further. Force close and terminate.
            if self.position:
                self._force_close_position(current_bar)
            reward = -4.0   # Severe penalty (within normalized range)
            done = True
        elif self._daily_dd_exceeded():
            # Daily drawdown > 3% → no more trading today (hard rule)
            # Don't end the episode, but force close any open position
            if self.position:
                self._force_close_position(current_bar)
            reward = -3.0   # Strong warning (within normalized range)
            # Note: done stays False — episode continues, but AI must survive

        # Track peak balance
        self.peak_balance = max(self.peak_balance, current_equity)

        # 6. Build next observation
        obs = self._build_observation()

        # 7. Track trend prediction accuracy
        self._track_trend_prediction(action, current_bar)

        info = {
            'balance': self.balance,
            'equity': current_equity,
            'position': self.position,
            'trades': len(self.trades_history),
            'win_rate': self._get_win_rate(),
            'trend_accuracy': self.memory.get_trend_accuracy(),
            'peak_balance': self.peak_balance,
            'drawdown': current_dd,
        }

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset for new episode (new trading week)."""
        self.current_step = self.config.lookback  # Skip warmup period
        # NOTE: Balance carries forward between weeks (compounding)
        # NOTE: Memory persists (learning accumulates)
        self.position = None
        self.frame_buffer = deque(maxlen=self.config.frame_stack)
        self.daily_start_equity = self._get_equity()  # Track daily DD
        self.current_date = None  # Reset daily tracking

        # Fill frame buffer with initial observations
        for i in range(self.config.frame_stack):
            obs = self._build_single_observation()
            self.frame_buffer.append(obs)

        return self._get_stacked_observation(), {}

    def _check_sl_tp_realistic(self, bar):
        """
        Check if SL or TP was hit during this bar using HIGH/LOW prices.

        This is critical for realistic simulation — real SL/TP orders trigger
        intra-bar, not just at close. We use the bar's high/low to detect hits.

        SAME-BAR CONFLICT: When both SL and TP fall within a bar's range
        (both could have been hit), we ALWAYS assume SL was hit first.
        Rationale: conservative simulation prevents learning overly optimistic
        strategies. In live trading, adverse moves typically happen faster
        than favorable ones (markets fall faster than they rise).
        """
        if not self.position:
            return

        sl = self.position['stop_loss']
        tp = self.position['take_profit']

        if self.position['side'] == 'LONG':
            sl_hit = bar['low'] <= sl     # Price dropped to SL
            tp_hit = bar['high'] >= tp    # Price rose to TP
        else:  # SHORT
            sl_hit = bar['high'] >= sl    # Price rose to SL
            tp_hit = bar['low'] <= tp     # Price dropped to TP

        if sl_hit and tp_hit:
            # Both hit in same bar — conservative: SL hit first
            self._close_at_price(sl, reason='SL_HIT')
        elif sl_hit:
            self._close_at_price(sl, reason='SL_HIT')
        elif tp_hit:
            self._close_at_price(tp, reason='TP_HIT')

    def _daily_dd_exceeded(self):
        """
        Check if daily drawdown exceeds 3% (hard rule).

        Tracks the equity at the start of each trading day. If equity drops
        more than 3% from the day's starting equity, returns True.
        Forces position close and prevents new trades for the rest of the day.
        """
        current_bar = self.data['M5'].iloc[self.current_step]
        bar_date = current_bar['time'].date() if hasattr(current_bar['time'], 'date') else None

        # Detect new trading day → reset daily tracking
        if bar_date != self.current_date:
            self.current_date = bar_date
            self.daily_start_equity = self._get_equity()
            return False

        # Check daily DD
        current_equity = self._get_equity()
        if self.daily_start_equity > 0:
            daily_dd = (self.daily_start_equity - current_equity) / self.daily_start_equity
            return daily_dd > 0.03  # 3% daily DD limit

        return False
```

### Episode Structure

```
TRAINING FLOW:

Week 1 Data → Environment loads M5/H1/H4/D1 bars for that week
           → AI trades through the entire week (~1300-1500 M5 bars)
           → Episode ends when week data runs out OR account blown
           → Balance carries forward to next week (compounding)
           → Memory persists across all weeks

Week 2 Data → Same thing, but AI remembers what it learned
           → Pattern memory has outcomes from Week 1
           → Continues building experience

...

Week N      → AI has traded N weeks of historical data
           → Rich memory of patterns and outcomes
           → Trend prediction accuracy tracked over all weeks
```

### What the AI Controls vs. What Is Hard-Coded

| Decision | Who Controls | Why |
|----------|-------------|-----|
| When to enter a trade | **AI** | This is what it's learning |
| Direction (long/short) | **AI** | Core trading decision |
| Position size | **AI** (within limits) | AI scales conviction, hard rules cap risk |
| When to exit | **AI** | AI decides when move is over |
| Initial stop-loss distance | **AI** (conviction-scaled) | AI learns SL distance per setup, hard rule enforces minimum |
| Stop-loss trailing / profit lock | **AI** (action[3]) | AI learns when/how much to trail SL into profit |
| Take-profit level | **AI** (conviction-scaled) | AI learns TP distance per setup |
| Maximum loss per trade | **Hard rule** | Cannot risk more than X% per trade |
| SL is mandatory on every trade | **Hard rule** | Every trade MUST have an SL. No exceptions. |
| SL can only tighten, never loosen | **Hard rule** | SL moves in AI's favor only. Prevents removing protection. |
| Minimum SL distance (0.5 ATR) | **Hard rule** | Prevents noise stop-outs from SL too close to price |
| Maximum daily drawdown | **Hard rule** | Trading stops if daily DD > 3% |
| Maximum total drawdown | **Hard rule** | Training ends if total DD > 10% |
| Maximum positions | **Hard rule** | Phase 1: 1 position max (single position). Multi-position is a Phase 2 enhancement |
| Kill switch | **Hard rule** | Emergency flatten all positions |
| Minimum hold duration (3 bars) | **Hard rule** | No closing before 3 bars (15 min) unless SL/TP hit. Prevents reward hacking via rapid open/close |
| Per-day trade limit (10) | **Hard rule** | Soft cap: after 10 trades/day, conviction threshold rises from 0.3→0.6. Prevents spam trading |

---

## 9. Data Pipeline & Sources (PRIORITY #1)

> **DATA IS THE FIRST THING WE SECURE.** Before writing any training code,
> we need years of clean XAUUSD historical data. This is Step 1.

**Detailed specs in [SPARTUS_DATA_PIPELINE.md](SPARTUS_DATA_PIPELINE.md)**

### Data Flow

```
MT5 Historical Data (via Python API)
    │
    ├── XAUUSD M5  (5-minute bars) ── Primary trading timeframe
    ├── XAUUSD M15 (15-minute bars) ── Confirmation
    ├── XAUUSD H1  (1-hour bars) ──── Trend context
    ├── XAUUSD H4  (4-hour bars) ──── Major structure
    └── XAUUSD D1  (daily bars) ───── Strategic direction
    │
    ▼
Data Validation
    • No gaps > 2 hours (during trading hours)
    • OHLC integrity (High >= Open, Close, Low)
    • No price spikes > 3% in one bar
    • Volume > 0 for all bars
    • At least 1200 M5 bars per trading week
    │
    ▼
Feature Calculation
    • All 30-40 features calculated (see Section 8)
    • Technical indicators via TA-Lib / pandas-ta
    • Fractional differentiation on price features (d ≈ 0.3-0.5)
    │
    ▼
Normalization (CRITICAL - NO LOOK-AHEAD)
    • Expanding window: at bar T, normalize using bars 0 to T only
    • Method: Rolling z-score (mean/std of past 200+ bars)
    • Alternative: Quantile transform on past data only
    • UNIT TEST: Verify no future data touches any feature at any bar
    │
    ▼
Storage: CSV files per week, organized by year
    storage/data/
    ├── 2020/week_01.parquet
    ├── 2020/week_02.parquet
    └── ...
```

### Data Sources & Depth (RESEARCH COMPLETED)

**Priority order for acquiring data:**

#### Source 1: MetaTrader 5 (Your Broker) — EASIEST, DO THIS FIRST

| Timeframe | Typical Depth | Notes |
|-----------|--------------|-------|
| D1 | 10-20 years | Best depth |
| H4 | 5-10 years | Good for context |
| H1 | 3-5 years | Primary context |
| M15 | 2-3 years | Confirmation |
| M5 | 1-3 years | Primary trading TF |

```python
# Pull all available data from your MT5 broker
import MetaTrader5 as mt5

mt5.initialize()
# Pull maximum available M5 data
rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 999999)
# This pulls everything the broker has available
```

**Limitation:** Most brokers only have 1-3 years of M5 data. That's enough to START training but not enough for deep training (need 5+ years).

#### Source 2: Dukascopy — BEST FREE SOURCE FOR DEEP HISTORY

| What | Depth | Quality |
|------|-------|---------|
| Tick data | 2010-present | Excellent (Swiss bank) |
| M1-D1 OHLCV | 2010-present | Excellent |
| Free to download | Unlimited | No API key needed |

**How to download Dukascopy data:**

```python
# Option A: Manual download from https://www.dukascopy.com/swiss/english/marketwatch/historical/
# Select XAUUSD, choose timeframe, download CSV

# Option B: Python library
pip install duka

# Download M5 data for 2020-2025
from duka.app import app as duka_app
duka_app("XAUUSD", "2020-01-01", "2025-12-31", 5)  # 5 = M5

# Option C: tick_vault library (more reliable)
pip install tick-vault

from tick_vault import TickVault
vault = TickVault()
vault.download("XAUUSD", "2015-01-01", "2025-12-31", timeframe="M5")
```

#### Source 3: Kaggle Datasets — DEEPEST FREE HISTORY

| Dataset | Depth | Format |
|---------|-------|--------|
| XAUUSD Historical 2004-2024 | 20 years | CSV M1-D1 |
| Gold Price Historical Data | 2000-2024 | CSV D1 |

```bash
# Install kaggle CLI
pip install kaggle

# Download (need Kaggle API key in ~/.kaggle/kaggle.json)
kaggle datasets download -d xauusd-gold-price-historical-data
```

#### Source 4: HistData.com — RELIABLE FREE M1 DATA

```
URL: https://www.histdata.com/download-free-forex-historical-data/
Format: CSV, M1 bars
Depth: 2000-present
Download: Manual (monthly ZIP files)
```

#### Source 5: yfinance (Yahoo Finance) — QUICK D1 DATA

```python
pip install yfinance

import yfinance as yf
gold = yf.download("GC=F", start="2010-01-01", end="2025-12-31", interval="1d")
# NOTE: yfinance only has D1 data for gold. Good for validation, not for M5 training.
```

### Data Acquisition Strategy

```
STEP 1: Pull MT5 broker data (immediate - you already have MT5)
        → Gets 1-3 years of M5/M15, 5-10 years of H1/H4/D1
        → Enough to START building and testing the environment

STEP 2: Download Dukascopy M5 data (same day)
        → Gets 10+ years of M5 data
        → This is your MAIN training dataset
        → Cross-validate with MT5 data where they overlap

STEP 3: Supplement with Kaggle for very deep history
        → Gets 20 years of D1/H4/H1 for context
        → Use for higher timeframe features

STEP 4: Validate all data sources against each other
        → Prices should match within 0.1% between sources
        → Any mismatches → investigate and fix
```

### Data Quality Comparison

| Source | Reliability | M5 Depth | Free? | Ease of Download |
|--------|------------|----------|-------|-----------------|
| MT5 Broker | High (your actual broker) | 1-3 years | Yes | Easiest (Python API) |
| Dukascopy | Very High (Swiss bank) | 10+ years | Yes | Medium (scripts) |
| Kaggle | Medium (user-uploaded) | 20 years | Yes | Easy (CLI/API) |
| HistData | High (institutional) | 20+ years | Yes | Manual (ZIP) |
| yfinance | Medium | D1 only | Yes | Easiest |

### Anti-Leakage Guarantee

**The #1 rule of Spartus data pipeline:**

> At time T, the AI can ONLY see data from times <= T.
> This applies to EVERY feature, EVERY normalization, EVERY indicator.

**How this is enforced:**
1. Feature builder only uses `.rolling()` and `.expanding()` on past data
2. Normalization uses expanding window (not full-dataset min/max)
3. Unit test runs on every build: feeds random bar T, checks no feature uses data from T+1 or later
4. Labels (for trend accuracy tracking) can look forward - but labels never enter the observation space

---

## 10. Feature Engineering

### Philosophy

> Start with 30-40 features. Validate each with SHAP importance on held-out data.
> Remove anything that doesn't help. Add new features only if backed by evidence.

### Complete Feature List (~42 features)

#### A. Price & Returns (7 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 1 | `close_frac_diff` | Fractionally differentiated close price (d≈0.35) | Stationary price that preserves memory |
| 2 | `returns_1bar` | log(close_t / close_{t-1}) | Current price momentum |
| 3 | `returns_5bar` | log(close_t / close_{t-5}) | Short-term trend |
| 4 | `returns_20bar` | log(close_t / close_{t-20}) | Medium-term trend |
| 5 | `bar_range` | (high - low) / close | Current bar volatility |
| 6 | `close_position` | (close - low) / (high - low + 1e-8) | Where price closed within bar |
| 7 | `body_ratio` | abs(close - open) / (high - low + 1e-8) | Candle body strength |

#### B. Volatility (4 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 8 | `atr_14_norm` | ATR(14) / close × 100 | Current volatility level |
| 9 | `atr_ratio` | ATR(7) / ATR(21) | Volatility expanding or contracting |
| 10 | `bb_width` | (BB_upper - BB_lower) / BB_middle | Bollinger squeeze/expansion |
| 11 | `bb_position` | (close - BB_lower) / (BB_upper - BB_lower) | Position within bands |

#### C. Momentum & Trend (6 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 12 | `rsi_14` | RSI(14) / 100 | Overbought/oversold (0-1 normalized) |
| 13 | `macd_signal` | MACD_histogram / ATR(14) | Momentum vs volatility |
| 14 | `adx_14` | ADX(14) / 100 | Trend strength (0-1) |
| 15 | `ema_cross` | (EMA_20 - EMA_50) / ATR(14) | Moving average crossover signal |
| 16 | `price_vs_ema200` | (close - EMA_200) / ATR(14) | Long-term trend position |
| 17 | `stoch_k` | Stochastic %K / 100 | Momentum oscillator (0-1) |

#### D. Volume (2 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 18 | `volume_ratio` | volume / SMA(volume, 20) | Volume relative to average |
| 19 | `obv_slope` | LinearRegSlope(OBV, 10) normalized | Volume-price divergence |

#### E. Multi-Timeframe Context (6 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 20 | `h1_trend_dir` | Sign of EMA(20) slope on H1, scaled by strength | Hourly trend |
| 21 | `h4_trend_dir` | Sign of EMA(20) slope on H4, scaled by strength | 4-hour trend |
| 22 | `d1_trend_dir` | Sign of EMA(20) slope on D1, scaled by strength | Daily trend |
| 23 | `h1_rsi` | H1 RSI(14) / 100 | HTF overbought/oversold |
| 24 | `mtf_alignment` | Average of h1/h4/d1 trend directions (-1 to +1) | Multi-TF confluence |
| 25 | `htf_momentum` | H4 MACD_hist / H4 ATR | HTF momentum context |

#### F. Time & Session (4 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 26 | `hour_sin` | sin(2π × hour / 24) | Cyclical time encoding |
| 27 | `hour_cos` | cos(2π × hour / 24) | Cyclical time encoding |
| 28 | `day_of_week` | day_number / 4 (Mon=0, Fri=1) | Weekly pattern |
| 29 | `session_quality` | London=1.0, NY=0.9, Asia=0.5, Off=0.2 | Trading session importance |

#### G. Account & Position State (6 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 30 | `has_position` | 1 if position open, 0 if flat | Position awareness |
| 31 | `position_side` | +1 long, -1 short, 0 flat | Direction awareness |
| 32 | `unrealized_pnl` | Current unrealized P/L as % of balance | Profit/loss awareness |
| 33 | `position_duration` | Bars since entry / 100 (normalized) | How long position held |
| 34 | `current_drawdown` | Current DD from peak equity (0-1) | Risk awareness |
| 35 | `equity_ratio` | Current equity / initial balance | Overall performance |
| 36 | `sl_distance_ratio` | (current_price - stop_loss) / ATR, 0 if flat | How far SL is from price (trail context) |
| 37 | `profit_locked_pct` | (stop_loss - entry_price) / ATR, 0 if SL below entry or flat | How much profit is locked in by current SL (ATR-scaled for meaningful signal) |

#### H. Memory Context (5 features)

| # | Name | Calculation | Why |
|---|------|-------------|-----|
| 38 | `recent_win_rate` | Win rate of last 20 trades (0-1) | Recent performance awareness |
| 39 | `similar_pattern_winrate` | Memory DB: win rate for similar market conditions | Experience-based confidence |
| 40 | `trend_prediction_accuracy` | Rolling accuracy of trend calls (0-1) | Self-awareness of prediction quality |
| 41 | `tp_hit_rate` | % of recent trades where TP was hit (0-1) | TP setting quality awareness |
| 42 | `avg_sl_trail_profit` | Average profit locked by SL trailing (rolling 20 trades) | SL management quality awareness |

**Total: 42 features** (with 10-frame stacking = 420 input dimensions)

### Feature Validation

Before any feature is used in training:

```python
def validate_feature(feature_series, target, test_data):
    """Every feature must pass ALL checks."""

    # 1. No future leakage (CRITICAL)
    assert no_future_data_used(feature_series)

    # 2. Stationarity (ADF test at 5%)
    adf_pvalue = adfuller(feature_series.dropna())[1]
    assert adf_pvalue < 0.05, f"Feature not stationary: p={adf_pvalue}"

    # 3. Information value (permutation importance > 0 on test data)
    importance = permutation_importance(model, test_data, feature)
    assert importance > 0, "Feature adds no value"

    # 4. Not redundant (correlation < 0.8 with existing features)
    for existing in current_features:
        corr = feature_series.corr(existing)
        assert abs(corr) < 0.8, f"Redundant with {existing.name}: corr={corr}"
```

---

## 11. Reward Design

### The Cardinal Rule

> **The reward function must be simple enough that you can explain it in one sentence.**
>
> Spartus reward: "Make money on trades, don't blow the account."

### Reward Function (5 Components + Normalization)

> **Upgrade from v3.2:** The original 3-component reward was too simplistic. Based on research
> (arXiv 2506.04358 — Risk-Aware RL Reward for Financial Trading, 2025), we use a 5-component
> composite reward with running normalization. Each component provides a different learning signal.

```python
class RewardNormalizer:
    """
    Running reward normalization using exponential moving average.
    SAC's auto-tuned entropy works best with unit-scale rewards.
    Without this, reward magnitudes ranging from ±0.01 to -10.0
    destabilize entropy coefficient tuning.
    """
    def __init__(self, tau=0.001):
        self.running_mean = 0.0
        self.running_var = 1.0
        self.tau = tau

    def normalize(self, reward):
        self.running_mean = (1 - self.tau) * self.running_mean + self.tau * reward
        self.running_var = (1 - self.tau) * self.running_var + self.tau * (reward - self.running_mean) ** 2
        normalized = (reward - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        return np.clip(normalized, -5.0, 5.0)  # Hard clip for stability


def calculate_reward(self, action, current_price, previous_price):
    """
    5-component composite reward with running normalization.
    Each component captures a different aspect of good trading.
    Final reward is normalized to unit scale for SAC stability.
    """
    reward = 0.0

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 1: Position P/L (dense, every step in a trade)
    #   Weight: 0.40 — Primary learning signal
    #   Teaches: being on the right side of the market = good
    # ─────────────────────────────────────────────────────────────
    r1 = 0.0
    if self.position is not None:
        price_change = (current_price - previous_price) / previous_price
        if self.position['side'] == 'SHORT':
            price_change = -price_change
        # Normalize by balance so reward scales appropriately as account grows
        position_pnl = price_change * self.position['lots'] * self._pip_value()
        r1 = (position_pnl / self.balance) * 100  # % of balance, scaled

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 2: Trade completion quality (sparse, at close)
    #   Weight: 0.20 — Reinforces good trade outcomes
    #   Scaled by quality, not binary: big win > small win > small loss > big loss
    # ─────────────────────────────────────────────────────────────
    r2 = 0.0
    if self._trade_just_closed:
        trade_pnl = self._last_trade_pnl
        commission = self._last_trade_commission
        net_pnl = trade_pnl - commission  # Must beat commissions

        if net_pnl > 0:
            # Reward scaled by profit quality (bigger net wins get more reward)
            r2 = min(net_pnl / self.balance * 50, 2.0)  # Cap at 2.0
        else:
            # Penalty scaled by loss size (bigger losses get more penalty)
            r2 = max(net_pnl / self.balance * 50, -2.0)  # Floor at -2.0

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 3: Drawdown penalty (emergency brake)
    #   Weight: 0.15 — Only fires in serious trouble
    # ─────────────────────────────────────────────────────────────
    r3 = 0.0
    current_dd = (self.peak_equity - self._get_equity()) / self.peak_equity
    if current_dd > 0.10:
        r3 = -3.0  # Strong warning
    elif current_dd > 0.07:
        r3 = -1.0  # Moderate warning

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 4: Differential Sharpe ratio (dense, every step)
    #   Weight: 0.15 — Rewards improving risk-adjusted returns
    #   Based on: Moody & Saffell (2001) differential Sharpe ratio
    #   This gives step-by-step signal about risk-adjusted performance
    # ─────────────────────────────────────────────────────────────
    r4 = 0.0
    if self.position is not None:
        step_return = (current_price - previous_price) / previous_price
        if self.position['side'] == 'SHORT':
            step_return = -step_return
        # Update running Sharpe statistics
        self._sharpe_A = self._sharpe_A + self._sharpe_eta * (step_return - self._sharpe_A)
        self._sharpe_B = self._sharpe_B + self._sharpe_eta * (step_return**2 - self._sharpe_B)
        # Differential Sharpe: how much this step improved/hurt the running Sharpe
        denom = (self._sharpe_B - self._sharpe_A**2) ** 1.5 + 1e-8
        r4 = (self._sharpe_B * (step_return - self._sharpe_A) -
               0.5 * self._sharpe_A * (step_return**2 - self._sharpe_B)) / denom

    # ─────────────────────────────────────────────────────────────
    # COMPONENT 5: Risk-adjusted position reward (dense)
    #   Weight: 0.10 — Rewards profit during low drawdown more
    #   Teaches: safe profits > risky profits
    #
    #   DESIGN NOTE: The risk_factor uses linear decay (1.0 - dd/0.10)
    #   across the actual DD range [0, 0.10]. The episode terminates at
    #   10% DD, so this factor smoothly goes from 1.0 (no DD, full bonus)
    #   to 0.0 (at DD limit, no bonus). This gives real differentiation:
    #     - Profit at 1% DD: r5 = r1 * 0.90 (nearly full bonus)
    #     - Profit at 5% DD: r5 = r1 * 0.50 (half bonus)
    #     - Profit at 9% DD: r5 = r1 * 0.10 (minimal bonus)
    #   Previous formula min(1/max(dd, 0.01), 5.0) was always capped at 5.0
    #   for DD < 20%, providing zero differentiation within the [0, 0.10] range.
    # ─────────────────────────────────────────────────────────────
    r5 = 0.0
    if self.position is not None and r1 > 0:
        # Profit is worth more when drawdown is low
        max_dd = 0.10  # Episode termination threshold
        risk_factor = max(0.0, 1.0 - current_dd / max_dd)  # Linear: 1.0 (safe) → 0.0 (at limit)
        r5 = r1 * risk_factor

    # ─────────────────────────────────────────────────────────────
    # COMBINE with weights
    # ─────────────────────────────────────────────────────────────
    raw_reward = (0.40 * r1) + (0.20 * r2) + (0.15 * r3) + (0.15 * r4) + (0.10 * r5)

    # Normalize for SAC stability
    return self.reward_normalizer.normalize(raw_reward)
```

### Reward Component Summary

| # | Component | Weight | Signal Type | What It Teaches |
|---|-----------|--------|-------------|-----------------|
| R1 | Position P/L | 0.40 | Dense (every step in trade) | Being right about market direction |
| R2 | Trade completion quality | 0.20 | Sparse (at trade close) | Bigger net wins > small wins; commission matters |
| R3 | Drawdown penalty | 0.15 | Rare (only when DD>7%) | Avoid catastrophic losses |
| R4 | Differential Sharpe | 0.15 | Dense (every step in trade) | Improve risk-adjusted returns, not just raw P/L |
| R5 | Risk-adjusted bonus | 0.10 | Dense (when in profit) | Safe profits > risky profits. Linear decay: r5 = r1 × (1 - DD/0.10) |

### Why This Composite Design (Research Basis)

The original 3-component reward had critical weaknesses:
- **Binary trade completion** (+1/-1) gave equal reward for a £0.01 win and a £5.00 win
- **No risk-adjusted signal** — AI could learn high-return-high-drawdown strategies
- **No normalization** — 1000x magnitude range (±0.01 to -10.0) destabilized SAC entropy tuning
- **Reward not scaled by balance** — same £1 trade was valued equally on £100 vs £200 account

The 5-component design addresses all of these. The differential Sharpe ratio (R4) is based on Moody & Saffell (2001) and has been validated in modern financial RL (arXiv 2506.04358, 2025). The running normalization ensures SAC's auto-entropy tuning operates on unit-scale rewards.

### What This Does NOT Include (On Purpose)

| Removed Component | Why Removed |
|-------------------|-------------|
| Exploration bonuses | SAC handles exploration automatically via entropy |
| Inactivity penalties | Not trading IS a valid strategy in ranging markets |
| Pattern match bonuses | Patterns tracked in memory, rewarded indirectly through P/L |
| Milestone bonuses (25%, 50% to TP) | Creates perverse incentives, distorts exit decisions |
| Trend alignment bonuses | Captured naturally through P/L and differential Sharpe |

### Anti-Reward-Hacking Safeguards

> **Problem:** The dashboard detects reward hacking (e.g., >50 trades/week), but nothing
> *prevents* it structurally. Without hard rules, the AI could discover that spamming tiny
> trades on favorable noise generates small positive R2 signals faster than holding positions.

```python
# Anti-reward-hacking rules (enforced in environment, NOT learned)
ANTI_HACK_CONFIG = {
    # 1. MINIMUM HOLD DURATION
    #    No voluntary close before 3 bars (15 minutes on M5).
    #    SL/TP hits still execute immediately — this only prevents AI from
    #    using action[2] (exit urgency) to close within 3 bars.
    "min_hold_bars": 3,

    # 2. NET PROFIT CHECK FOR TRADE COMPLETION BONUS
    #    R2 (trade completion quality) only gives positive reward if
    #    net_pnl > commission. Already implemented in R2 above, but
    #    additionally: R2 = 0 (not negative) for trades closed exactly
    #    at break-even after costs. Only genuinely losing trades get negative R2.

    # 3. PER-DAY TRADE LIMIT (soft cap)
    #    After 10 trades in a single trading day (00:00-23:59 UTC),
    #    the conviction threshold rises from 0.3 to 0.6.
    #    This means the AI must be TWICE as confident to open new positions.
    #    Effect: AI can still trade in strong setups, but can't spam.
    "daily_trade_soft_cap": 10,
    "normal_conviction_threshold": 0.3,
    "elevated_conviction_threshold": 0.6,
}

def _get_conviction_threshold(self):
    """Dynamic conviction threshold — rises after heavy trading."""
    today_trades = sum(1 for t in self.trades_history
                       if t['close_time'].date() == self.current_date)
    if today_trades >= self.config.daily_trade_soft_cap:
        return self.config.elevated_conviction_threshold  # 0.6
    return self.config.normal_conviction_threshold  # 0.3

def _can_close_position(self):
    """Check if minimum hold duration is met."""
    if self.position is None:
        return False
    bars_held = self.current_step - self.position['entry_step']
    if bars_held < self.config.min_hold_bars:
        return False  # Too soon — SL/TP can still trigger, but AI can't voluntarily close
    return True
```

**Implementation notes:**
- `min_hold_bars` is checked in `_execute_action()` before processing exit signals
- SL/TP hits bypass `min_hold_bars` — hard stops always execute immediately
- The daily trade counter resets at the start of each new trading day in the data
- These rules are NOT part of the reward function — they are environment constraints

---

## 12. Memory System

### Why Memory Matters

The memory system is what makes Spartus **learn over time**. Without it, each week of training starts fresh. With it, the AI carries experience forward.

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SPARTUS MEMORY SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─ Trade History (SQLite) ───────────────────────────────┐ │
│  │  Every trade recorded: entry, exit, side, PnL, duration│ │
│  │  Features at entry stored (what market looked like)     │ │
│  │  Used to: calculate win rate, track improvement         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Pattern Memory (SQLite) ──────────────────────────────┐ │
│  │  Market conditions at trade entry → trade outcome       │ │
│  │  Key fields: rsi, macd, trend_dir, session, outcome     │ │
│  │  Queried each step to get "similar_pattern_winrate"     │ │
│  │  Grows with every trade - AI gets more accurate         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Trend Predictions (SQLite) ───────────────────────────┐ │
│  │  Each step: AI's implied trend prediction + actual move  │ │
│  │  Prediction inferred from action direction              │ │
│  │  Actual move recorded N bars later                       │ │
│  │  Accuracy calculated rolling (last 100/500/all-time)    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Model Checkpoints (File System) ─────────────────────┐ │
│  │  Best model saved after each week if profitable         │ │
│  │  Tracks which week it's from, performance stats         │ │
│  │  Allows rollback if model degrades                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### How Memory Feeds into Observations

Every step, the environment queries the memory system:

```python
def _get_memory_features(self, current_market_state):
    """Get memory-based features for observation vector."""

    # 1. Recent win rate (last 20 trades)
    recent_trades = self.memory.get_recent_trades(n=20)
    recent_win_rate = sum(1 for t in recent_trades if t.pnl > 0) / max(len(recent_trades), 1)

    # 2. Similar pattern win rate
    # Find past trades where market conditions were similar to now
    similar = self.memory.find_similar_conditions(
        rsi=current_market_state['rsi'],
        trend_dir=current_market_state['trend_dir'],
        session=current_market_state['session'],
        volatility_regime=current_market_state['vol_regime'],
        tolerance=0.15  # 15% similarity band
    )
    similar_win_rate = sum(1 for s in similar if s.outcome > 0) / max(len(similar), 1)

    # 3. Trend prediction accuracy (rolling 100)
    trend_acc = self.memory.get_trend_accuracy(window=100)

    return [recent_win_rate, similar_win_rate, trend_acc]
```

### Trend Prediction Tracking

> **CRITICAL:** The AI must learn from every prediction outcome. Each prediction is recorded,
> verified after N bars, and the result (correct/incorrect) is stored permanently in memory.
> This creates a feedback loop: predict → wait → verify → learn.

**The prediction-verification cycle:**

```
Step 1: AI outputs direction signal > 0.3 (e.g., +0.72)
        → Prediction recorded: "UP" with confidence 0.72, price £2647.30

Step 2: 20 bars pass (~100 minutes for M5)

Step 3: System checks: What actually happened?
        → Price is now £2651.80 → Market moved UP → Prediction CORRECT ✓
        OR
        → Price is now £2643.10 → Market moved DOWN → Prediction WRONG ✗

Step 4: Result stored in memory database:
        → predicted_direction: "UP"
        → actual_direction: "UP" or "DOWN"
        → actual_move: +4.50 or -4.20 (price difference)
        → correct: true or false
        → This record is PERMANENT — the AI's accuracy history is never lost
```

**How trend prediction accuracy is measured:**

```python
def _track_trend_prediction(self, action, current_bar):
    """
    Infer trend prediction from AI's action and check against actual outcome.

    If AI goes LONG → it's predicting market goes UP
    If AI goes SHORT → it's predicting market goes DOWN
    If AI is FLAT → it's predicting SIDEWAYS/UNCERTAIN

    We check N bars later if it was right.
    """
    direction_signal = action[0]  # -1 to +1

    if abs(direction_signal) > 0.3:
        predicted_direction = "UP" if direction_signal > 0 else "DOWN"

        # Store prediction with timestamp
        self.memory.store_prediction(
            step=self.current_step,
            prediction=predicted_direction,
            confidence=abs(direction_signal),
            price_at_prediction=current_bar['close']
        )

    # Check predictions from N bars ago
    self.memory.verify_old_predictions(
        current_step=self.current_step,
        current_price=current_bar['close'],
        lookforward=20  # Check 20 bars later (about 1.5 hours for M5)
    )


def verify_old_predictions(self, current_step, current_price, lookforward=20):
    """
    Find all unverified predictions from N bars ago and check if they were right.
    This is the LEARNING FEEDBACK — the AI knows whether its reads were correct.
    """
    unverified = self.db.execute("""
        SELECT id, predicted_direction, price_at_prediction, step
        FROM predictions
        WHERE verified_at_step IS NULL
        AND step <= ?
    """, (current_step - lookforward,)).fetchall()

    for pred_id, predicted_dir, pred_price, pred_step in unverified:
        # What actually happened?
        actual_move = current_price - pred_price
        actual_direction = "UP" if actual_move > 0 else "DOWN"
        was_correct = (predicted_dir == actual_direction)

        # Store the verified result
        self.db.execute("""
            UPDATE predictions SET
                actual_direction = ?,
                actual_move = ?,
                correct = ?,
                verified_at_step = ?
            WHERE id = ?
        """, (actual_direction, actual_move, was_correct, current_step, pred_id))

    self.db.commit()
```

**What the AI learns from this:**
- The `trend_prediction_accuracy` feature (observation #38) feeds this accuracy back into the AI's observation vector
- When accuracy is high → the AI knows its reads are good → can trade with more conviction
- When accuracy is low → the AI knows its reads are poor → should be more cautious
- This creates a self-aware feedback loop that improves naturally over training weeks

This means trend prediction accuracy **automatically improves** as the AI gets better at trading. We track it but don't reward it directly - it's a natural outcome of better trading decisions.

### Take-Profit Accuracy Tracking

> **Did the market reach the TP level the AI set?** If the AI is good at setting TPs, it means it
> understands how far a move will go. If TPs are rarely hit, the AI is setting unrealistic targets.

**How TP accuracy works:**

```python
def track_tp_accuracy(self, trade):
    """
    After every completed trade, record whether the TP was reached.
    This tells us: Is the AI good at predicting HOW FAR the market will move?

    Possible outcomes:
    1. TP HIT → AI correctly predicted the move size → tp_correct = True
    2. SL HIT → Market reversed before reaching TP → tp_correct = False
    3. EXIT SIGNAL → AI closed early (may or may not have reached TP)
    4. WEEK END → Trade closed at end of episode

    We track:
    - TP hit rate: % of trades where the market reached the TP level
    - TP overshoot: did price go PAST the TP? (AI could have set higher)
    - TP time: how many bars to reach TP (faster = better market read)
    """
    tp_record = {
        'trade_id': trade['id'],
        'tp_level': trade['take_profit'],
        'entry_price': trade['entry_price'],
        'exit_price': trade['exit_price'],
        'side': trade['side'],
        'tp_hit': trade['close_reason'] == 'TP_HIT',
        'sl_hit': trade['close_reason'] == 'SL_HIT',
        'manual_close': trade['close_reason'] in ('EXIT_SIGNAL', 'FLIP_DIRECTION'),
        'bars_to_close': trade['duration_bars'],
        # Did price reach TP at any point during the trade? (using high/low)
        'tp_was_reachable': trade.get('price_reached_tp', False),
    }
    self.db.execute("INSERT INTO tp_tracking (...) VALUES (...)", tp_record)
```

**What this teaches the AI:**
- If TP hit rate is high → AI understands move sizes → keep setting similar TPs
- If TP hit rate is low → AI is too optimistic → should set closer TPs or let conviction scale TP
- If price often overshoots TP → AI is too conservative → could set further TPs
- TP accuracy feeds into observation as `tp_hit_rate` feature

---

## 13. Training Methodology

### Overview

```
PHASE 1: Setup & Data (1 week)
    → Pull data, build features, validate no leakage

PHASE 2: Initial Training (2-3 weeks)
    → Train SAC agent through 2-3 years of weekly data
    → Memory system accumulates experience
    → Track trend accuracy, win rate, profit curves

PHASE 3: Extended Training (2-4 weeks)
    → Continue through 5+ years of data
    → Feature importance analysis (remove weak features)
    → Hyperparameter tuning with Optuna
    → Multiple training runs with different seeds

PHASE 4: Validation (1 week)
    → Walk-forward test on held-out year(s)
    → Paper trading on MT5 demo
    → Compare live vs backtest performance

PHASE 5: Live Deployment (ongoing)
    → Deploy on real account with minimal size
    → Continue learning from live trades
    → Periodic retraining on new data
```

### Week-by-Week Training Flow

```python
def train_spartus(config):
    """Main training loop."""

    # Load all available weeks of data
    all_weeks = load_all_weekly_data(config.data_path)

    # Split: 70% train, 15% validation, 15% test (temporal order)
    # 2-week purge gap between splits to prevent data leakage
    # See SPARTUS_DATA_PIPELINE.md Section 8 for rationale
    n = len(all_weeks)
    train_end = int(0.7 * n)
    val_start = train_end + 2   # 2-week purge gap
    val_end = int(0.85 * n)
    test_start = val_end + 2    # 2-week purge gap

    train_weeks = all_weeks[:train_end]
    val_weeks = all_weeks[val_start:val_end]
    test_weeks = all_weeks[test_start:]

    # Initialize memory system (persists across all training)
    memory = TradingMemory(config.memory_db_path)

    # Initialize SAC agent
    # See SPARTUS_TRAINING_METHODOLOGY.md Section 2 for lr_schedule and full config rationale
    agent = SAC(
        "MlpPolicy",
        env=None,  # Set per-week
        learning_rate=lr_schedule,     # Warm-up → hold → cosine decay (see methodology doc)
        buffer_size=200_000,
        batch_size=256,
        gamma=0.97,
        tau=0.005,
        ent_coef="auto",
        max_grad_norm=1.0,             # Gradient clipping by global L2 norm
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Train through weeks sequentially
    balance = config.initial_balance  # Carries forward

    for week_idx, week_data in enumerate(train_weeks):
        print(f"\n{'='*60}")
        print(f"WEEK {week_idx + 1}/{len(train_weeks)}")
        print(f"Balance: £{balance:.2f}")
        print(f"Memory size: {memory.total_trades()} trades")
        print(f"Trend accuracy: {memory.get_trend_accuracy():.1%}")
        print(f"{'='*60}")

        # Create environment for this week
        env = SpartusTradeEnv(
            market_data=week_data,
            config=config,
            memory=memory,
            initial_balance=balance
        )

        # Train on this week's data
        agent.set_env(env)
        agent.learn(
            total_timesteps=config.steps_per_week,
            reset_num_timesteps=False,  # Continue global step count
            progress_bar=True
        )

        # Update balance (compounding)
        balance = env.balance

        # Save checkpoint if profitable
        if balance > config.initial_balance:
            agent.save(f"storage/models/spartus_week_{week_idx}")

        # Log progress
        log_weekly_progress(week_idx, env, memory)

    # Validate on held-out weeks
    validate_on_holdout(agent, val_weeks, memory, config)
```

### Training Resilience & Crash Recovery

> **CRITICAL:** Training runs for days. If the system crashes, loses power, or is interrupted,
> we MUST be able to resume without losing progress. Every piece of learned knowledge persists.

**What is saved automatically during training:**

| Component | Save Method | Frequency | Recovery |
|-----------|------------|-----------|----------|
| Model weights (SAC) | `agent.save()` to `.zip` | Every completed week | Load latest checkpoint |
| Replay buffer | `agent.save_replay_buffer()` | Every completed week | Load with model |
| Memory database | SQLite (auto-commits) | After every trade | Already on disk |
| Training state | `training_state.json` | Every completed week | Load and resume |
| Balance | In `training_state.json` | Every completed week | Restored on resume |
| Week progress | In `training_state.json` | Every completed week | Skip completed weeks |

**Training state file:**

```python
# Saved to storage/training_state.json after every completed week
training_state = {
    "last_completed_week": 47,
    "global_step": 470_000,
    "current_balance": 138.50,
    "peak_balance": 145.20,
    "bankruptcies": 1,
    "best_win_rate": 0.583,
    "best_sharpe": 0.92,
    "model_checkpoint_path": "storage/models/spartus_week_0047.zip",
    "replay_buffer_path": "storage/models/spartus_week_0047_buffer.pkl",
    "memory_db_path": "storage/memory/spartus_memory.db",
    "config_hash": "a3f8c1...",  # Detect if config changed between runs
    "timestamp": "2026-02-22T15:02:30"
}
```

**Resume logic:**

```python
def resume_or_start_training(config):
    """Resume from last checkpoint if available, otherwise start fresh."""
    state_path = "storage/training_state.json"

    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)

        print(f"RESUMING from week {state['last_completed_week'] + 1}")
        print(f"Balance: £{state['current_balance']:.2f}")
        print(f"Global steps: {state['global_step']:,}")

        # Load model from checkpoint
        agent = SAC.load(state['model_checkpoint_path'])

        # Load replay buffer (preserves all past experience)
        agent.load_replay_buffer(state['replay_buffer_path'])

        # Memory DB is already on disk — just connect
        memory = TradingMemory(state['memory_db_path'])

        # Resume from next week
        start_week = state['last_completed_week'] + 1
        balance = state['current_balance']

    else:
        print("STARTING fresh training run")
        agent = SAC("MlpPolicy", ...)
        memory = TradingMemory(config.memory_db_path)
        start_week = 0
        balance = config.initial_balance

    return agent, memory, start_week, balance
```

**What this guarantees:**
- The model's neural network weights are preserved (learned trading skill)
- The replay buffer is preserved (all past experiences for off-policy learning)
- The memory database is preserved (all trade history, patterns, trend predictions)
- Training resumes from the exact next week — no repeated work, no lost progress
- The AI continues evolving and building knowledge from where it left off

### Hyperparameter Configuration

```python
config = {
    # === TRAINING ===
    "initial_balance": 100.0,       # Start with £100
    "steps_per_week": 10_000,       # ~7x the week length for multiple passes
    "total_training_weeks": 200,     # 4+ years of weekly data

    # === SAC AGENT ===
    "learning_rate": 3e-4,
    "buffer_size": 200_000,          # Large replay buffer for off-policy learning
    "batch_size": 256,
    "gamma": 0.97,                   # Discount factor (lower for non-stationary)
    "tau": 0.005,                    # Soft update for target networks
    "ent_coef": "auto",              # SAC auto-tunes this
    "learning_starts": 5_000,        # Fill buffer before training starts

    # === NETWORK ===
    "net_arch_pi": [256, 256],       # Policy network
    "net_arch_qf": [256, 256],       # Q-network
    "activation": "ReLU",

    # === FEATURES ===
    "num_features": 42,
    "frame_stack": 10,               # Stack last 10 bars as input
    "lookback": 200,                 # Bars needed for indicator warmup
    "normalization": "rolling_zscore", # Rolling z-score, window=200

    # === RISK (HARD RULES) ===
    "max_risk_per_trade": 0.02,      # 2% max risk per trade
    "max_daily_drawdown": 0.03,      # 3% daily DD → stop trading
    "max_total_drawdown": 0.10,      # 10% total DD → end episode
    "max_positions": 1,              # Keep it simple: 1 position at a time (to start)
    "min_sl_atr_multiple": 1.0,      # SL must be at least 1 ATR away

    # === MEMORY ===
    "memory_db_path": "storage/memory/spartus_memory.db",
    "pattern_similarity_tolerance": 0.15,
    "trend_check_horizon": 20,       # Check prediction 20 bars ahead
}
```

### Hyperparameter Tuning with Optuna

```python
def optuna_objective(trial):
    """Tune key hyperparameters."""
    config = {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.93, 0.99),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "frame_stack": trial.suggest_int("frame_stack", 5, 20),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
        "net_width": trial.suggest_categorical("net_width", [128, 256, 512]),
    }

    # Train for N weeks, return Sharpe ratio on validation weeks
    sharpe = train_and_validate(config)
    return sharpe

study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=50)
```

---

## 14. Validation Framework

### Three Layers of Validation

```
LAYER 1: Purged Walk-Forward (During Training)
    → Every 20 training weeks, test on next 5 unseen weeks
    → Purge: remove 1 week between train/test (embargo)
    → Tracks: is the model improving over time?

LAYER 2: Held-Out Test Set (After Training)
    → 15% of data never touched during training
    → Final go/no-go check
    → Must show: positive Sharpe, acceptable drawdown

LAYER 3: Demo Account Testing (Before Live — THE REAL TEST)
    → Load trained model onto MT5 DEMO account
    → Demo account runs on LIVE market data in real-time
    → The only difference from real trading: the money is not real
    → This is the ultimate test — if it works here, it works live
    → 2-4 weeks minimum observation period
    → Compare: demo performance vs backtest performance
    → Alert if demo < 80% of backtest (overfitting likely)

    HOW IT WORKS:
    → MetaTrader 5 demo accounts mirror the live market exactly
    → Same prices, same spreads, same execution conditions
    → The AI trades with fake money but real market conditions
    → Zero risk: if the model is bad, you lose nothing
    → This catches problems that backtesting cannot:
      - Real spread widening during news events
      - Real execution delays and requotes
      - Real market gaps and flash moves
      - Conditions the historical data didn't cover
```

### Key Metrics Tracked

```python
tracked_metrics = {
    # Trading Performance
    "total_profit_pct": float,       # Total return as % of starting balance
    "sharpe_ratio": float,           # Risk-adjusted returns
    "max_drawdown_pct": float,       # Worst peak-to-trough decline
    "win_rate": float,               # % of trades that were profitable
    "profit_factor": float,          # Gross profit / gross loss
    "avg_trade_pnl": float,          # Average P/L per trade
    "trades_per_week": float,        # Trading frequency

    # Trend Prediction (TRACKED, not optimized)
    "trend_accuracy_overall": float, # % of correct trend calls
    "trend_accuracy_up": float,      # Accuracy when predicting UP
    "trend_accuracy_down": float,    # Accuracy when predicting DOWN
    "trend_accuracy_evolution": list, # How accuracy changes over training

    # Learning Progress
    "memory_size": int,              # Total trades in memory
    "pattern_coverage": float,       # How many market conditions seen
    "balance_curve": list,           # Balance over time (should trend up)
    "win_rate_evolution": list,      # How win rate improves over weeks
}
```

### Red Flags

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| Trend accuracy > 75% | Investigate | Possible data leakage |
| Sharpe > 3.0 in backtest | Investigate | Likely overfitting |
| Live accuracy < 80% of backtest | Stop | Overfitting confirmed |
| Win rate > 80% | Investigate | Something is wrong |
| 0 trades in a week | Check | Model might be stuck |
| 50+ trades in a week | Check | Model might be spamming |
| Drawdown > 10% | Halt | Risk rules should have caught this |

---

## 15. Risk Management

### Hard Rules (Enforced by Environment Code, NOT Learned)

```python
class RiskManager:
    """Hard risk rules that cannot be overridden by the AI."""

    def __init__(self, config):
        self.max_risk_per_trade = config.max_risk_per_trade  # 2%
        self.max_daily_dd = config.max_daily_drawdown         # 3%
        self.max_total_dd = config.max_total_drawdown          # 10%
        self.max_positions = config.max_positions               # Phase 1: 1
        self.min_sl_distance_atr = config.min_sl_atr_multiple  # 1.0 ATR

    def validate_trade(self, proposed_trade, account_state, market_state):
        """Check if proposed trade passes all risk rules."""

        # 1. Position limit
        if account_state.open_positions >= self.max_positions:
            return False, "Max positions reached"

        # 2. Daily drawdown check
        if account_state.daily_dd >= self.max_daily_dd:
            return False, "Daily DD limit reached"

        # 3. Total drawdown check
        if account_state.total_dd >= self.max_total_dd:
            return False, "Total DD limit reached"

        # 4. Position sizing (ATR-based, capped by max risk)
        atr = market_state.atr_14
        risk_amount = account_state.equity * self.max_risk_per_trade
        sl_distance = max(atr * self.min_sl_distance_atr, atr)  # At least 1 ATR
        max_size = risk_amount / (sl_distance * dollar_per_point)
        proposed_trade.size = min(proposed_trade.size, max_size)

        # 5. Minimum SL distance
        if proposed_trade.sl_distance < sl_distance:
            proposed_trade.sl = adjust_sl_to_minimum(proposed_trade, sl_distance)

        return True, "Trade approved"
```

### Position Sizing: ATR-Based with Fractional Kelly

```python
def calculate_position_size(conviction, equity, atr, win_rate, avg_win_loss_ratio):
    """
    Position size scales with:
    - AI conviction (from action[1])
    - Current volatility (higher ATR → smaller position)
    - Historical win rate (Kelly fraction)

    Capped by max risk per trade (hard rule).
    """
    # Kelly fraction (conservative 0.25x)
    if win_rate > 0 and avg_win_loss_ratio > 0:
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly_fraction = max(0, kelly * 0.25)  # Quarter Kelly for safety
    else:
        kelly_fraction = 0.01  # Minimum during early learning

    # Risk amount
    risk_pct = kelly_fraction * conviction  # Scale by AI conviction
    risk_pct = min(risk_pct, 0.02)          # Hard cap at 2%
    risk_amount = equity * risk_pct

    # Convert to lots
    sl_distance = atr * 1.5
    lots = risk_amount / (sl_distance * dollar_per_point)

    # Enforce broker limits
    lots = max(min_lot, min(lots, max_lot))
    return round_to_step(lots)
```

---

## 16. Model Portability & Export

### The Goal

> **Anyone can download the trained model and run it on their own MT5 account.**
> The model file is self-contained: model weights + memory + config.

### Export Format

```python
def export_model(model, memory, config, output_path="spartus_model_v1.zip"):
    """
    Export trained model as a single portable .zip file.

    The .zip contains everything needed to run the model:
    - model_weights/          SAC model (PyTorch state dict)
    - memory.db               Trade history & pattern memory (SQLite)
    - scaler_config.json      Feature normalization parameters
    - model_config.json       Hyperparameters & feature list
    - training_report.json    Performance metrics & validation results
    - README.txt              How to use this model
    """
    import zipfile
    import json
    import shutil

    # 1. Save model weights
    model.save("temp_export/model_weights")

    # 2. Copy memory database
    shutil.copy2(config.memory_db_path, "temp_export/memory.db")

    # 3. Save scaler config (normalization params)
    scaler_config = {
        "method": config.normalization,
        "window": config.normalization_window,
        "feature_columns": config.feature_columns,
        "frame_stack": config.frame_stack,
    }
    with open("temp_export/scaler_config.json", "w") as f:
        json.dump(scaler_config, f, indent=2)

    # 4. Save model config
    model_config = {
        "algorithm": "SAC",
        "num_features": config.num_features,
        "frame_stack": config.frame_stack,
        "action_space": [4],
        "net_arch_pi": [256, 256],
        "net_arch_qf": [256, 256],
        "training_weeks": config.total_training_weeks,
        "initial_balance": config.initial_balance,
        "symbol": "XAUUSD",
        "version": "1.0",
    }
    with open("temp_export/model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # 5. Save training report
    training_report = {
        "win_rate": memory.get_overall_win_rate(),
        "sharpe_ratio": memory.get_sharpe(),
        "total_trades": memory.total_trades(),
        "trend_accuracy": memory.get_trend_accuracy(),
        "max_drawdown": memory.get_max_drawdown(),
        "final_balance": memory.current_balance,
        "weeks_trained": config.total_training_weeks,
        "validation_sharpe": config.validation_sharpe,
    }
    with open("temp_export/training_report.json", "w") as f:
        json.dump(training_report, f, indent=2)

    # 6. Create README
    readme = f"""
SPARTUS TRADING AI MODEL v{model_config['version']}
=====================================
Trained on {config.total_training_weeks} weeks of XAUUSD historical data.
Starting balance: £{config.initial_balance}
Final balance: £{memory.current_balance:.2f}
Win rate: {training_report['win_rate']:.1%}
Sharpe: {training_report['sharpe_ratio']:.2f}

TO USE THIS MODEL:
1. Install requirements: pip install -r requirements.txt
2. Have MetaTrader 5 running with XAUUSD symbol
3. Run: python live_engine.py --model spartus_model_v1.zip
"""
    with open("temp_export/README.txt", "w") as f:
        f.write(readme)

    # 7. Create ZIP
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("temp_export"):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, "temp_export")
                zf.write(filepath, arcname)

    # Cleanup
    shutil.rmtree("temp_export")
    print(f"Model exported to: {output_path}")
    return output_path
```

### Import / Load Model

```python
def load_exported_model(zip_path):
    """
    Load an exported Spartus model from a .zip file.
    Anyone with this file can run the model.
    """
    import zipfile
    import json

    # Extract ZIP
    extract_dir = "loaded_model"
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # Load config
    with open(f"{extract_dir}/model_config.json") as f:
        model_config = json.load(f)

    with open(f"{extract_dir}/scaler_config.json") as f:
        scaler_config = json.load(f)

    # Load SAC model
    from stable_baselines3 import SAC
    model = SAC.load(f"{extract_dir}/model_weights")

    # Load memory
    memory = TradingMemory(f"{extract_dir}/memory.db")

    return model, memory, model_config, scaler_config
```

### Sharing Models

```
HOW TO SHARE:
1. Train your model → produces spartus_model_v1.zip (~50-100 MB)
2. Upload to GitHub releases, Google Drive, Dropbox, etc.
3. Other person downloads the .zip
4. They run: python live_engine.py --model spartus_model_v1.zip
5. Model loads with all memory and starts trading on THEIR MT5 account

THE MODEL FILE CONTAINS:
✓ Trained neural network weights
✓ Complete trading memory (all patterns & outcomes)
✓ Normalization config (so features are calculated correctly)
✓ Model architecture config (so the right network is built)
✓ Training report (so you know what you're getting)

THE MODEL FILE DOES NOT CONTAIN:
✗ API keys or credentials
✗ Historical market data (too large)
✗ Broker-specific configuration
✗ Personal information
```

---

## 17. Training Engine Dashboard

### Why a Dashboard?

While the model trains (potentially for days), you need to SEE:
- Is it learning? (balance going up over weeks)
- Is it making trades? (not stuck at 0)
- Are there errors? (environment crashes, data issues)
- What is the AI thinking? (its decisions each step)
- When should you stop? (convergence or divergence detected)

### Dashboard Layout (Terminal-Based for Training Engine)

The training engine uses a **terminal/TUI dashboard** (not Qt6 - that's for the live interface).

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SPARTUS TRAINING ENGINE                           [Ctrl+C to stop]    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ PROGRESS ─────────────────────────────────────────────────────────┐ │
│  │  Week: 47 / 200  [████████████░░░░░░░░░░░░░░░░░░░░░░░] 23.5%     │ │
│  │  Training Time: 3h 42m  |  ETA: ~12h 15m                          │ │
│  │  Steps: 470,000 / 2,000,000                                       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ ACCOUNT ──────────┐  ┌─ LEARNING METRICS ────────────────────────┐ │
│  │ Starting:  £100.00  │  │ Win Rate:      54.2% (↑ from 48.1%)      │ │
│  │ Current:   £138.50  │  │ Trend Acc:     56.8% (↑ from 50.2%)      │ │
│  │ Peak:      £145.20  │  │ Profit Factor: 1.24                      │ │
│  │ Drawdown:  4.6%     │  │ Avg Trade:     +£0.38                    │ │
│  │ Return:    +38.5%   │  │ Sharpe:        0.82                      │ │
│  └─────────────────────┘  │ Memory Trades: 612                       │ │
│                           │ Patterns:      847                       │ │
│  ┌─ THIS WEEK ─────────┐ └────────────────────────────────────────────┘ │
│  │ Trades: 14           │                                               │
│  │ Wins:   8 (57.1%)    │  ┌─ BALANCE CURVE (last 20 weeks) ─────────┐ │
│  │ P/L:    +£2.80       │  │ £145 │        ╭─╮                        │ │
│  │ Best:   +£1.20       │  │      │    ╭──╯  ╰╮  ╭──╮                │ │
│  │ Worst:  -£0.80       │  │ £130 │  ╭╯       ╰──╯  ╰╮              │ │
│  │ Lot sizes: 0.01-0.03 │  │      │╭─╯                ╰─╮╭──        │ │
│  └──────────────────────┘  │ £115 ╯                      ╰╯          │ │
│                            │ £100 ┼──────────────────────────          │ │
│                            │      Week 28        Week 38   Week 47    │ │
│                            └──────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ AI DECISION LOG (latest) ──────────────────────────────────────────┐│
│  │ [W47 Bar 892] OPEN LONG 0.02 lots @ 2647.30 (conv: 0.68)          ││
│  │               Spread: 1.8 pips | Slippage: 0.3 pips                ││
│  │ [W47 Bar 905] CLOSE LONG +£0.85 (held 13 bars, 65 min)            ││
│  │ [W47 Bar 912] HOLD (signal: 0.18, below threshold 0.3)            ││
│  │ [W47 Bar 920] OPEN SHORT 0.01 lots @ 2651.80 (conv: 0.42)        ││
│  │ [W47 Bar 928] EXIT SIGNAL 0.72 → CLOSE SHORT -£0.30              ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  ┌─ ERRORS / WARNINGS ────────────────────────────────────────────────┐│
│  │ [!] Week 44: 2 bars with zero volume (skipped)                     ││
│  │ [i] Week 45: Best win rate so far (58.3%) - checkpoint saved       ││
│  │ [!] Week 46: Drawdown reached 6.2% - lot size reduced             ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│ GPU: None (CPU) | RAM: 2.1 GB | Speed: 142 steps/sec | Disk: 1.2 GB   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation: Rich Terminal Dashboard

```python
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

class TrainingDashboard:
    """Real-time terminal dashboard for monitoring training."""

    def __init__(self):
        self.console = Console()
        self.metrics = {}
        self.decision_log = []
        self.error_log = []

    def update(self, week, step, env_info, agent_info):
        """Update dashboard with latest training state."""
        self.metrics = {
            'week': week,
            'balance': env_info['balance'],
            'peak': env_info.get('peak_balance', env_info['balance']),
            'win_rate': env_info.get('win_rate', 0),
            'trend_acc': env_info.get('trend_accuracy', 0),
            'trades': env_info.get('trades', 0),
            'drawdown': env_info.get('drawdown', 0),
        }

    def log_decision(self, week, bar, decision_text):
        """Log an AI decision for display."""
        self.decision_log.append(f"[W{week} Bar {bar}] {decision_text}")
        if len(self.decision_log) > 20:
            self.decision_log = self.decision_log[-20:]

    def log_error(self, message):
        """Log an error or warning."""
        self.error_log.append(message)
        if len(self.error_log) > 10:
            self.error_log = self.error_log[-10:]

    def render(self):
        """Render the full dashboard to terminal."""
        # Uses rich library for beautiful terminal output
        # See implementation in src/training/dashboard.py
        pass
```

### TensorBoard Integration (Detailed Charts)

For deeper analysis, TensorBoard runs alongside the terminal dashboard:

```python
# Launch with: tensorboard --logdir=storage/logs/
# Tracks:
# - spartus/balance (per step and per week)
# - spartus/win_rate (rolling)
# - spartus/trend_accuracy (rolling)
# - spartus/drawdown (current)
# - spartus/trades_per_week
# - spartus/avg_trade_pnl
# - spartus/lot_sizes (what sizes the AI chooses)
# - spartus/entropy (SAC exploration level)
# - spartus/actor_loss, critic_loss
```

### Log Files

All training data is also saved to log files for post-analysis:

```
storage/logs/
├── training_log.jsonl          # One JSON line per step (compact)
├── weekly_summary.jsonl        # One JSON line per week (key metrics)
├── decisions.jsonl             # Every AI decision with context
├── errors.log                  # All errors and warnings
└── tensorboard/                # TensorBoard event files
```

---

## 18. Live Trading Interface (Phase 2)

> **This is built AFTER the training engine works and produces a profitable model.**
> The live interface loads the exported model and trades on real MT5.

### MT5 Connector

```python
class MT5Connector:
    """Interface to MetaTrader 5 for data and execution."""

    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 terminal not running")

    # DATA METHODS
    def get_bars(self, symbol, timeframe, start, count):
        rates = mt5.copy_rates_from(symbol, timeframe, start, count)
        return pd.DataFrame(rates)

    def get_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

    # EXECUTION METHODS
    def open_trade(self, symbol, side, volume, sl, tp):
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if side == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 777777,
            "comment": "Spartus AI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return mt5.order_send(request)

    def close_trade(self, ticket, symbol, volume, side):
        close_type = mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if side == "BUY" else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 777777,
            "comment": "Spartus AI Close",
        }
        return mt5.order_send(request)

    def modify_sl_tp(self, ticket, sl, tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        return mt5.order_send(request)

    def emergency_close_all(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        results = []
        for pos in positions:
            result = self.close_trade(pos.ticket, symbol, pos.volume,
                                      "BUY" if pos.type == 0 else "SELL")
            results.append(result)
        return results
```

### Live Trading Loop

```python
class SpartusLiveEngine:
    """Runs the trained Spartus model on live MT5 data."""

    def __init__(self, model_path, config):
        self.model = SAC.load(model_path)
        self.mt5 = MT5Connector()
        self.memory = TradingMemory(config.memory_db_path)
        self.risk_mgr = RiskManager(config)
        self.feature_builder = FeatureBuilder(config)

    def run(self, symbol="XAUUSD", timeframe_minutes=5):
        """Main loop: pull data → predict → execute → repeat."""
        while self.running:
            # 1. Pull latest bars
            bars = self.mt5.get_bars(symbol, mt5.TIMEFRAME_M5, datetime.utcnow(), 250)

            # 2. Build features
            features = self.feature_builder.build(bars, self.memory)

            # 3. Get AI decision
            action, _ = self.model.predict(features, deterministic=True)

            # 4. Interpret and execute
            decision = interpret_action(action, self.current_position, self.risk_mgr)
            self.execute_decision(decision)

            # 5. Update memory with any trade outcomes
            self.memory.update_from_live(self.mt5.get_positions())

            # 6. Wait for next bar
            sleep(timeframe_minutes * 60)
```

### Live Dashboard (Qt6 - Phase 2)

```
┌─────────────────────────────────────────────────────────────────┐
│  SPARTUS TRADING AI                              [Settings] [X]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ TREND READING ──────────────────────────────────────────┐   │
│  │  AI's Market Read:  BULLISH (confidence: 0.72)            │   │
│  │  Trend Accuracy (last 100 calls):  62.0%                  │   │
│  │  Trend Accuracy (all time):        58.4%                  │   │
│  │  Accuracy Evolution: ▁▂▃▃▅▅▆▆▇▇  (improving over time)  │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ POSITIONS ──────────────────────────────────────────────┐   │
│  │  LONG XAUUSD  0.05 lot  Entry: 2645.30  Duration: 35min  │   │
│  │  SL: 2639.80  |  Current: 2648.50  |  P/L: +£1.60        │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ PERFORMANCE ─────┐  ┌─ MEMORY ──────────────────────────┐  │
│  │ Balance:  £147.30  │  │ Total trades: 892                 │  │
│  │ Today P/L: +£3.20  │  │ Win rate: 57.3%                   │  │
│  │ Week P/L:  +£8.50  │  │ Patterns stored: 1,240            │  │
│  │ Max DD:    2.1%    │  │ Similar to now: 23 (65% won)      │  │
│  │ Sharpe:    1.34    │  │ Weeks of experience: 47            │  │
│  └────────────────────┘  └───────────────────────────────────┘  │
│                                                                  │
│  ┌─ CONTROLS ───────────────────────────────────────────────┐   │
│  │  [START]  [STOP]  [EMERGENCY CLOSE ALL]                   │   │
│  │  Mode: [Auto] [Advisor]    Status: RUNNING                │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ ACTIVITY LOG ───────────────────────────────────────────┐   │
│  │ 14:35 Opened LONG 0.05 at 2645.30 (confidence: 0.72)     │   │
│  │ 14:30 Market read: BULLISH, waiting for entry signal      │   │
│  │ 14:15 Closed SHORT +£1.20 (held 25min, SL trailed)       │   │
│  │ 13:50 Opened SHORT 0.04 at 2649.10 (confidence: 0.61)    │   │
│  └───────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│ MT5: Connected │ Model: Loaded │ Memory: 892 trades │ Up: 3h    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 19. Project Structure

```
SpartusTradeAI/
├── SPARTUS_TRADING_AI.md           # This blueprint (main reference)
├── SPARTUS_TRAINING_METHODOLOGY.md  # Detailed training specs
├── SPARTUS_DATA_PIPELINE.md         # Data engineering specs
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   │
│   │   ══════════════════════════════════════════════
│   │   SYSTEM 1: TRAINING ENGINE (Built First)
│   │   ══════════════════════════════════════════════
│   │
│   ├── data/                         # Data pipeline (~5 files)
│   │   ├── __init__.py
│   │   ├── mt5_loader.py             # Pull data from MT5
│   │   ├── dukascopy_loader.py       # Pull data from Dukascopy
│   │   ├── data_validator.py         # Quality & leakage checks
│   │   ├── feature_builder.py        # Calculate all 42 features
│   │   └── normalizer.py             # Expanding-window normalization
│   │
│   ├── environment/                  # Training environment (~4 files)
│   │   ├── __init__.py
│   │   ├── trade_env.py              # SpartusTradeEnv (<500 lines)
│   │   ├── reward.py                 # 5-component composite reward with normalization
│   │   └── market_simulator.py       # Spreads, slippage, commissions
│   │
│   ├── memory/                       # Memory system (~3 files)
│   │   ├── __init__.py
│   │   ├── trading_memory.py         # SQLite memory database
│   │   └── trend_tracker.py          # Trend prediction tracking
│   │
│   ├── training/                     # Training pipeline (~11 files)
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── validation.py             # Walk-forward + purged CV
│   │   ├── dashboard.py              # Rich terminal training dashboard (~600 lines)
│   │   ├── logger.py                 # JSONL logging (4 log files)
│   │   ├── callback.py               # SB3 callback: SAC extraction, multi-freq logging
│   │   ├── convergence.py            # Convergence detection (IMPROVING/OVERFITTING/etc.)
│   │   ├── ascii_chart.py            # Balance curve renderer
│   │   ├── report_generator.py       # LLM-optimized training reports
│   │   ├── health_check.py           # Quick CLI health check
│   │   └── exporter.py               # Export model to portable .zip
│   │
│   ├── risk/                         # Risk management (~2 files)
│   │   ├── __init__.py
│   │   └── risk_manager.py           # Hard risk rules + lot sizing
│   │
│   │   ══════════════════════════════════════════════
│   │   SYSTEM 2: LIVE INTERFACE (Built After Training)
│   │   ══════════════════════════════════════════════
│   │
│   ├── execution/                    # Live trading (~4 files)
│   │   ├── __init__.py
│   │   ├── mt5_connector.py          # MT5 API wrapper
│   │   ├── live_engine.py            # Live trading loop
│   │   └── model_loader.py           # Load exported .zip model
│   │
│   └── interface/                    # Live Dashboard (Phase 2, ~3 files)
│       ├── __init__.py
│       ├── main_window.py            # Qt6 main window
│       └── components.py             # UI components
│
├── storage/
│   ├── models/                       # Saved model checkpoints & exports
│   ├── data/                         # Historical market data (parquet)
│   │   ├── raw/                      # Raw downloaded data
│   │   └── processed/                # Cleaned & validated data by week
│   ├── features/                     # Pre-computed feature files (parquet)
│   ├── memory/                       # Memory databases (SQLite)
│   ├── reports/                      # LLM training analysis reports (.md)
│   └── logs/                         # Training & execution logs
│       ├── tensorboard/              # TensorBoard event files
│       ├── training_log.jsonl        # Step-by-step training log
│       ├── weekly_summary.jsonl      # Per-week summary
│       ├── decisions.jsonl           # AI decision log
│       └── alerts.log                # Human-readable alert log
│
├── tests/
│   ├── test_features.py              # Feature calculation correctness
│   ├── test_no_leakage.py            # CRITICAL: verify no future data leaks
│   ├── test_reward.py                # Reward function tests
│   ├── test_risk_rules.py            # Risk management + lot sizing tests
│   ├── test_memory.py                # Memory system tests
│   ├── test_environment.py           # End-to-end environment tests
│   ├── test_market_simulator.py      # Spread/slippage simulation tests
│   └── test_model_export.py          # Export/import round-trip tests
│
├── notebooks/                        # Jupyter exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_training_results.ipynb
│
└── scripts/
    ├── download_data.py              # Script to download all historical data
    ├── train.py                      # Entry point: start training
    ├── validate.py                   # Entry point: run validation
    ├── export_model.py               # Entry point: export trained model
    └── live.py                       # Entry point: run live trading
```

**Target: ~25 Python files, <500 lines each.**

---

## 20. Technology Stack

```bash
# Python 3.10+ required

# === CORE ===
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn

# === RL AGENT ===
pip install stable-baselines3>=2.6.0    # SAC implementation
pip install gymnasium>=0.29.0           # RL environments

# === TRADING ===
pip install MetaTrader5                 # MT5 API (Windows only)

# === INDICATORS ===
pip install TA-Lib                      # C-based, fast (needs wheel on Windows)
pip install pandas-ta                   # Pure Python fallback

# === FEATURE ENGINEERING ===
pip install fracdiff                    # Fractional differentiation
pip install shap                        # Feature importance analysis

# === VALIDATION ===
pip install timeseriescv                # Purged cross-validation

# === HYPERPARAMETER TUNING ===
pip install optuna                      # Bayesian optimization
pip install optuna-dashboard            # Web UI

# === MONITORING & DASHBOARD ===
pip install tensorboard                 # Training visualization
pip install rich                        # Rich terminal dashboard for training engine

# === DATA SOURCES ===
pip install duka                        # Dukascopy data downloader
pip install kaggle                      # Kaggle dataset download (optional)
pip install yfinance                    # Yahoo Finance (D1 only, for validation)

# === UI (Phase 2 - Live Interface) ===
pip install PySide6                     # Qt6 live trading dashboard
```

---

## 21. Training Schedule & Milestones

### Phase 0: Data Acquisition (PRIORITY #1 - Day 1-3)
- [ ] Set up project structure, install dependencies
- [ ] Pull all available data from MT5 broker (M5 through D1)
- [ ] Download Dukascopy XAUUSD M5 data (2015-2025)
- [ ] Download supplementary data from Kaggle / HistData
- [ ] Validate and cross-check data sources
- [ ] Organize into storage/data/raw/ and storage/data/processed/
- [ ] **GATE: Have at least 5 years of clean M5 data before proceeding**

### Phase 1: Data Pipeline & Features (Week 1)
- [ ] Implement data validator (quality + gap detection)
- [ ] Implement feature builder (all 42 features)
- [ ] Implement expanding-window normalizer
- [ ] Implement multi-timeframe alignment
- [ ] Pre-compute features for all available weeks
- [ ] **Unit test: verify no feature uses future data (CRITICAL)**

### Phase 2: Environment & Memory (Week 2)
- [ ] Implement SpartusTradeEnv (<500 lines)
- [ ] Implement realistic market simulation (spreads, slippage, commissions)
- [ ] Implement 5-component composite reward with RewardNormalizer
- [ ] Implement risk manager (hard rules + lot sizing by account size)
- [ ] Implement memory system (SQLite databases)
- [ ] Implement trend prediction tracker
- [ ] Implement training dashboard (Rich terminal UI)
- [ ] **Smoke test: AI can trade 1 week without crashing**

### Phase 3: Initial Training (Weeks 3-4)
- [ ] First SAC training run (50 weeks of data, starting from £100)
- [ ] Verify: balance curve trending up (even slowly)
- [ ] Verify: memory system growing with trade outcomes
- [ ] Verify: trend accuracy being tracked
- [ ] Verify: lot sizing scales with account balance
- [ ] Verify: drawdown → reduced lot sizes
- [ ] Feature importance analysis - remove weak features
- [ ] **Checkpoint: Is win rate > 45% after 50 weeks?**

### Phase 4: Full Training (Weeks 5-7)
- [ ] Train through all available data (200+ weeks)
- [ ] Optuna hyperparameter tuning (50 trials)
- [ ] Multiple training seeds for robustness
- [ ] Walk-forward validation on held-out years
- [ ] Implement model export (.zip file)
- [ ] **Target: Win rate > 52%, Sharpe > 0.5 on validation**

### Phase 5: Validation & Demo Account Testing (Week 8+)
- [ ] Test on completely held-out year of data
- [ ] Export model as portable .zip
- [ ] Set up MT5 demo account (same broker as intended live account)
- [ ] Load trained model into live interface connected to demo account
- [ ] Run on demo account for 2-4 weeks (real market data, fake money)
- [ ] Monitor: does demo performance match backtest? (must be > 80%)
- [ ] Monitor: does the AI handle real spreads, gaps, and execution?
- [ ] Monitor: does the AI continue learning from demo trades?
- [ ] Build live dashboard interface (Qt6 - Phase 2)
- [ ] **Go/No-go decision for real account based on demo results**

---

## 22. Success Criteria

### The AI is working when:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Win rate (out-of-sample) | > 52% | > 55% | > 60% |
| Sharpe ratio | > 0.5 | > 1.0 | > 1.5 |
| Max drawdown | < 15% | < 10% | < 7% |
| Trend accuracy (rolling 100) | > 53% | > 58% | > 63% |
| Profit factor | > 1.1 | > 1.3 | > 1.5 |
| Consistent small profits | Most weeks green | 70%+ weeks green | 80%+ weeks green |
| Live vs backtest ratio | > 0.75 | > 0.85 | > 0.90 |

### The AI is NOT working if:

- Trend accuracy < 50% (worse than coin flip)
- Win rate < 48% (losing more than winning)
- Max drawdown > 20% (risk management failed)
- Zero trades for multiple weeks (model frozen)
- 50+ trades per day (reward hacking / spam)
- Balance declining over training weeks (not learning)

---

## Appendix: Key Differences from GoldCoTrader

| Aspect | GoldCoTrader | Spartus |
|--------|-------------|---------|
| Models | 1 confused RL model | 1 focused RL model with clear purpose |
| Features | 3,000+ (mostly noise) | 42 (curated, validated) |
| Reward | 20+ conflicting components | 5-component composite with normalization |
| Normalization | Look-ahead (FATAL bug) | Expanding window only |
| Validation | None | Walk-forward + purged CV + paper trading |
| Action space | 8 discrete actions | 4 continuous actions |
| Position targets | Fixed pips (10/15/25) | ATR-scaled or AI-learned |
| Memory | Existed but disconnected | Core feature, feeds observations |
| Trend tracking | Dead code (never activated) | Automatic, tracked every step |
| File count | 71 files | ~20 files |
| Env size | 4,600 lines | <500 lines |
| Entropy tuning | Manual (0.10, way too high) | Auto (SAC handles it) |
| Stop management | Hardcoded trailing overrode AI | AI controls, hard rules only for max risk |
| Algorithm | RecurrentPPO (wrong for continuous) | SAC (designed for continuous) |

---

**Document Version:** 3.3
**Created:** 2026-02-22
**Updated:** 2026-02-23
**Status:** Blueprint Complete - Ready for Implementation
**Key Changes in v3.0:** Added Two Systems separation, Account Management as core objective, Model Portability, Training Engine Dashboard, Realistic Market Simulation, Comprehensive Data Sources, Data Acquisition as Priority #1
**Key Changes in v3.1:** Expanded trend prediction verification cycle with explicit predict→wait→verify→learn feedback loop, added Training Resilience & Crash Recovery (auto-checkpoint + resume), expanded Demo Account Testing phase with MT5 demo details
**Key Changes in v3.2:** Expanded action space from 3 to 4 dimensions (added SL management action[3] for trailing stop-loss). Added adjust_stop_loss(), calculate_sl(), calculate_tp() functions. Added 4 new features (sl_distance_ratio, profit_locked_pct, tp_hit_rate, avg_sl_trail_profit) → 42 total features, 420 input dims. Added TP accuracy tracking with tp_tracking table. Added mandatory SL hard rule. Updated appendix and all references.
**Key Changes in v3.3:** Major system-wide upgrade based on 37-item gap analysis and 2025-2026 RL research. Fixed 5 bugs (position sizing stacking, walk-forward leakage, position count, cyclic normalization, regime detector). Upgraded reward from 3→5 components with running normalization and anti-reward-hacking safeguards. Added gradient clipping, LR schedule (warm-up + cosine decay), SAC internals monitoring, observation health checks, LSTM switch criteria. Added curriculum learning, domain randomization, data augmentation. Added Bayesian shrinkage for pattern memory, regime-tagged replay buffer, ensemble SAC. Added SimbaV2 and DR-SAC as Phase 2 architecture options. Added SAC internals and convergence detection dashboard panels.
**Key Changes in v3.3.1:** Cross-reference alignment audit. Fixed: account features (Sec 7 aligned with Sec 10.G: 8 features not 5), max positions diagram (2→1), export action_space (3→4), reward references ("3 clean"→"5-component"), added purge gaps to train/val/test split code, updated project structure (11 training files, added storage/features/ and storage/reports/), updated appendix tables.
**Key Changes in v3.3.2:** Deep critical design review — 6 design flaws fixed:
1. Circuit breaker penalties now use SET (=) not ADD (-=) to stay within normalizer's [-5,+5] range (was creating -15.0 outliers that destabilize SAC entropy).
2. Episode termination DD threshold fixed from 80% to 10% (matching spec hard rules table).
3. Added `_check_sl_tp_realistic()` method spec with same-bar SL/TP conflict rule (conservative: assume SL hit first).
4. `profit_locked_pct` feature (#37) changed from entry-price-based to ATR-based denominator (was producing ~0.0004 invisible values for gold).
5. R5 component formula changed from `min(1/max(dd,0.01), 5.0)` (always capped) to `max(0, 1.0 - dd/0.10)` (smooth linear decay across actual DD range).
6. Added `_daily_dd_exceeded()` method implementing the 3% daily DD hard rule (was specified but not implemented in step()).
**Next Step:** Phase 0 data acquisition complete. Phase 1: Build feature engineering pipeline
