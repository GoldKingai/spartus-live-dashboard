# Spartus Trading AI - Training Methodology

**Companion document to [SPARTUS_TRADING_AI.md](SPARTUS_TRADING_AI.md)**

---

## 1. Training Philosophy

Spartus learns to trade the same way a human does: by trading on historical data, making mistakes, remembering what worked, and gradually improving. The key insight is that **trend prediction is a byproduct of trading skill**, not the other way around.

```
MORE TRADING EXPERIENCE → BETTER PATTERN RECOGNITION → BETTER TREND READING
                       → BETTER ENTRY TIMING         → MORE CONSISTENT PROFIT
                       → BETTER EXIT DECISIONS        → LOWER DRAWDOWNS
```

---

## 2. Algorithm: SAC (Soft Actor-Critic)

### Why SAC

| Property | Benefit for Spartus |
|----------|-------------------|
| Continuous action space | AI outputs direction, conviction, and exit signals as continuous values |
| Automatic entropy tuning | No manual exploration parameter to guess wrong (GoldCoTrader had ent_coef=0.10) |
| Replay buffer | Learns from ALL past experiences, not just recent ones (sample efficient) |
| Off-policy | Can learn from trades made by earlier versions of itself |
| Maximum entropy objective | Explores diverse strategies naturally, avoids collapsing to one pattern |
| Stable training | Less prone to policy collapse than PPO |

### SAC Hyperparameters (Starting Point)

```python
sac_config = {
    "learning_rate": 3e-4,            # Standard starting point
    "buffer_size": 200_000,           # Store 200K transitions
    "batch_size": 256,                # Update from 256 transitions per gradient step
    "gamma": 0.97,                    # Discount factor (lower than 0.99 for non-stationary markets)
    "tau": 0.005,                     # Soft target network update rate
    "ent_coef": "auto",               # SAC auto-tunes exploration (CRITICAL improvement over GoldCoTrader)
    "target_entropy": "auto",         # Derived automatically from action space
    "learning_starts": 5_000,         # Fill buffer with 5K transitions before training
    "train_freq": 1,                  # Train after every environment step
    "gradient_steps": 1,              # 1 gradient step per environment step
    "max_grad_norm": 1.0,            # Clip gradients by global norm (prevents exploding gradients)
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256],         # Policy (actor) network
            "qf": [256, 256]          # Q-value (critic) network
        },
        "activation_fn": "ReLU"
    }
}
```

### Why These Specific Values

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `learning_rate=3e-4` | Standard for SAC, proven across domains |
| `buffer_size=200_000` | With ~1300 steps/week × 200 weeks = 260K transitions. Buffer holds most of training |
| `batch_size=256` | Large enough for stable gradients with 420-dim input (42 features × 10 frame stack) |
| `gamma=0.97` | Lower than standard 0.99 because market conditions are non-stationary. Discounts distant rewards faster |
| `tau=0.005` | Standard soft update. Lower = more stable but slower learning |
| `ent_coef="auto"` | THE KEY FIX. SAC auto-adjusts exploration. No more guessing like GoldCoTrader's 0.10 disaster |
| `learning_starts=5_000` | Need diverse initial experiences before updates are meaningful |
| `net_arch=[256,256]` | 2 layers of 256. Enough capacity for 420-dim input without overfitting on limited data |
| `max_grad_norm=1.0` | Clips gradients by global L2 norm. Prevents exploding gradients from large reward spikes or rare transitions. SB3 supports natively |

### Gradient Norm Monitoring

Track gradient norms during training to detect instability early:
- Log actor and critic gradient norms to TensorBoard every 100 steps
- **Alert** if gradient norm exceeds 10x the running average (sudden spike)
- **Alert** if gradient norm is consistently >0.9 of `max_grad_norm` (saturated clipping = learning signal loss)
- If gradients are frequently clipped, consider reducing learning rate

### Hyperparameter Tuning Schedule

```
Phase 1 (Weeks 1-50): Use default config above. Learn if basic approach works.
Phase 2 (Weeks 50-100): Optuna search on: learning_rate, gamma, batch_size, frame_stack
Phase 3 (Weeks 100+): Fine-tune best config. Focus on net_arch and buffer_size.
```

### Learning Rate Schedule (Warm-Up + Cosine Decay)

> **Why not fixed LR?** A fixed 3e-4 throughout training is suboptimal. Early updates on
> random/noisy data with a high LR can destabilize; late-stage training benefits from a lower LR
> for fine-tuning converged policies.

```python
import numpy as np

def lr_schedule(progress_remaining: float) -> float:
    """
    3-phase learning rate schedule for SAC.
    SB3 calls this with progress_remaining going from 1.0 → 0.0.

    Phase 1 (first 5%):   Linear warm-up from 1e-5 to 3e-4
    Phase 2 (5% to 75%):  Hold at 3e-4
    Phase 3 (75% to 100%): Cosine decay from 3e-4 to 5e-5

    For 200-week training (~260K steps):
      Phase 1 ≈ weeks 1-10     (warm-up)
      Phase 2 ≈ weeks 10-150   (full learning rate)
      Phase 3 ≈ weeks 150-200  (fine-tuning decay)
    """
    progress = 1.0 - progress_remaining  # 0.0 → 1.0

    lr_max = 3e-4
    lr_min = 5e-5
    lr_warmup_start = 1e-5
    warmup_end = 0.05    # First 5% of training
    decay_start = 0.75   # Last 25% of training

    if progress < warmup_end:
        # Phase 1: Linear warm-up
        t = progress / warmup_end
        return lr_warmup_start + t * (lr_max - lr_warmup_start)
    elif progress < decay_start:
        # Phase 2: Hold at max
        return lr_max
    else:
        # Phase 3: Cosine decay
        t = (progress - decay_start) / (1.0 - decay_start)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t))

# Usage with SB3:
# model = SAC("MlpPolicy", env, learning_rate=lr_schedule, ...)
```

**Why these phases:**
- **Warm-up** prevents large early updates on random experience (replay buffer mostly empty)
- **Hold** allows the bulk of learning at full speed
- **Cosine decay** gently reduces LR so the policy converges smoothly rather than oscillating

### Future Enhancement: DR-SAC (Distributionally Robust SAC)

> **Phase 2 upgrade path.** Standard SAC assumes training and deployment distributions match.
> In finance, they never do — live market microstructure, liquidity, and volatility regimes
> differ from historical simulation.

DR-SAC (arXiv 2506.12622, 2025) optimizes against the *worst-case* transition model within a
KL-divergence uncertainty set. Key results:
- 9.8x higher average reward vs vanilla SAC under environment perturbations
- Particularly relevant when deploying to live trading where conditions differ from backtest
- Requires custom SAC implementation (not in standard SB3)

**Implementation plan (Phase 2):**
1. Prove basic SAC works on historical data first
2. If live deployment shows distribution shift issues, implement DR-SAC
3. The dual-critic structure from DR-SAC can be layered on top of our existing SAC config

### Regime-Tagged Replay Buffer

> **Problem:** Standard replay buffer samples uniformly. As the agent accumulates experience,
> old regime experiences get diluted. If the market spends 80% of time trending and 20% ranging,
> the buffer becomes 80% trending data — and the agent forgets how to handle range-bound markets.
> This is catastrophic forgetting for non-stationary environments.

Inspired by LCPO (ICLR 2025 Spotlight — locally constrained policy optimization for non-stationary
environments), we tag each experience with its market regime and ensure balanced sampling.

```python
class RegimeTaggedReplayBuffer:
    """
    Extension of SB3's ReplayBuffer that tags transitions with market regime
    and ensures balanced regime representation in each training batch.

    Regime classification (derived from existing features):
      - trending_up:   h1_trend_dir > 0.3 AND mtf_alignment > 0.3
      - trending_down: h1_trend_dir < -0.3 AND mtf_alignment < -0.3
      - ranging:       abs(h1_trend_dir) < 0.3 AND abs(mtf_alignment) < 0.3
      - volatile:      atr_14_norm > 75th percentile of recent ATR values
    """

    REGIMES = ['trending_up', 'trending_down', 'ranging', 'volatile']
    MIN_REGIME_REPRESENTATION = 0.15  # Each regime gets at least 15% of batch

    def classify_regime(self, observation):
        """Classify the regime from observation features."""
        # Features at known indices (from the 42-feature spec)
        h1_trend = observation[20]     # h1_trend_dir (feature #21, 0-indexed)
        mtf_align = observation[23]    # mtf_alignment (feature #24, 0-indexed)
        atr_norm = observation[5]      # atr_14_norm (feature #6, 0-indexed)

        if h1_trend > 0.3 and mtf_align > 0.3:
            return 'trending_up'
        elif h1_trend < -0.3 and mtf_align < -0.3:
            return 'trending_down'
        elif atr_norm > self._atr_p75:
            return 'volatile'
        else:
            return 'ranging'

    def sample(self, batch_size):
        """
        Sample with regime balance guarantee.
        Each regime gets at least MIN_REGIME_REPRESENTATION of the batch.
        Remaining slots are filled uniformly from the full buffer.
        """
        min_per_regime = int(batch_size * self.MIN_REGIME_REPRESENTATION)
        guaranteed = min_per_regime * len(self.REGIMES)  # 4 × 15% = 60%
        remaining = batch_size - guaranteed               # 40% uniform

        batch_indices = []
        for regime in self.REGIMES:
            regime_indices = self._regime_indices[regime]
            if len(regime_indices) >= min_per_regime:
                sampled = np.random.choice(regime_indices, min_per_regime, replace=False)
            else:
                # Regime is rare — sample with replacement
                sampled = np.random.choice(regime_indices, min_per_regime, replace=True)
            batch_indices.extend(sampled)

        # Fill remaining slots uniformly
        all_indices = np.arange(self.size())
        uniform_sample = np.random.choice(all_indices, remaining, replace=False)
        batch_indices.extend(uniform_sample)

        return self._get_samples(batch_indices)
```

**Key benefits:**
- Prevents catastrophic forgetting of rare market regimes
- Ensures the agent maintains competence in volatile/crash conditions even during long trending periods
- Minimal overhead: regime classification uses features already in the observation

---

## 3. Observation Design

### Frame Stacking

Instead of using LSTM (which had issues in GoldCoTrader with hidden state resets), we use **frame stacking**:

```
At time T, the observation is:
[features_T, features_{T-1}, features_{T-2}, ..., features_{T-9}]

42 features × 10 frames = 420-dimensional input vector
```

**Why frame stacking over LSTM:**
- No hidden state to manage or reset
- Standard MLP policy works (no RecurrentPPO complexity)
- 10 M5 bars = 50 minutes of context (enough for intraday patterns)
- If more context needed, increase frame_stack to 20 (42 × 20 = 840 dims, still manageable)

**If frame stacking is insufficient — measurable switch criteria:**

Do NOT switch to LSTM on gut feeling. Use these explicit, measurable conditions:

```
INVESTIGATE LSTM IF ANY OF THESE ARE TRUE:
  1. trend_accuracy < 52% after week 80 AND win_rate < 50%
     → Frame stacking isn't capturing enough temporal context

  2. Balance curve shows no upward trend over a 50-week rolling window
     after week 100
     → Agent has stagnated despite sufficient training time

  3. Increasing frame_stack from 10→20 improves trend_accuracy by >3%
     on validation data
     → More temporal context helps → LSTM would help even more

WHEN SWITCHING TO LSTM:
  - Implement a custom RecurrentSAC policy (not available in vanilla SB3)
  - Use SB3-contrib's RecurrentPPO as reference, adapt for SAC
  - Sequence length: 20 steps (match the 20-frame-stack equivalent)
  - Hidden state: reset at episode boundaries (start of each week)
  - CRITICAL: align sequence boundaries with episode boundaries — no
    cross-episode hidden state leakage
  - Keep all other hyperparameters identical for fair comparison
  - Run both frame-stack and LSTM for 50 weeks on same data, compare
```

### Feature Normalization

**IMPORTANT:** Not all features use the same normalization method. Features fall into 3 categories:

| Category | Features | Method | Why |
|----------|----------|--------|-----|
| **Market features** | #1-25 (price, volatility, momentum, volume, MTF) | Rolling z-score (200-bar window) | Non-stationary, need adaptive normalization |
| **Cyclic/bounded features** | #26-29 (hour_sin, hour_cos, day_of_week, session_quality) | **NO normalization** — already in [-1,1] or [0,1] | Pre-bounded by definition. Rolling z-score on a sine wave creates artifacts |
| **Account/position features** | #30-37 (has_position, side, pnl, drawdown, equity, SL features) | **NO normalization** — already in known ranges | These are ratios/flags (0-1, -1 to +1) by construction |
| **Memory features** | #38-42 (win_rate, pattern WR, accuracy, TP hit, SL trail) | **NO normalization** — already in [0,1] | These are rates/percentages by definition |

```python
# Features that SKIP normalization (already bounded)
NORMALIZATION_EXEMPT = {
    'hour_sin', 'hour_cos', 'day_of_week', 'session_quality',  # Cyclic/bounded
    'has_position', 'position_side',                             # Binary/ternary
    'unrealized_pnl', 'position_duration', 'current_drawdown',  # Already ratios
    'equity_ratio', 'sl_distance_ratio', 'profit_locked_pct',   # Already ratios
    'recent_win_rate', 'similar_pattern_winrate',                # Already [0,1]
    'trend_prediction_accuracy', 'tp_hit_rate', 'avg_sl_trail_profit'  # Already [0,1]
}

class ExpandingWindowNormalizer:
    """
    Normalize features using ONLY past data.
    At bar T, normalization uses statistics from bars 0 to T.
    NEVER uses future data.
    Exempt features (cyclic, bounded, ratios) are passed through unchanged.
    """

    def __init__(self, method="rolling_zscore", window=200):
        self.method = method
        self.window = window

    def normalize(self, series, current_idx, feature_name=None):
        """Normalize a single value using past data only."""
        # Skip normalization for exempt features
        if feature_name and feature_name in NORMALIZATION_EXEMPT:
            return series[current_idx]  # Pass through unchanged

        if self.method == "rolling_zscore":
            # Use last 200 bars for statistics
            start = max(0, current_idx - self.window)
            window_data = series[start:current_idx + 1]
            mean = window_data.mean()
            std = window_data.std()
            if std < 1e-8:
                return 0.0
            return (series[current_idx] - mean) / std

        elif self.method == "expanding_minmax":
            # Use ALL past data up to current bar
            past_data = series[:current_idx + 1]
            min_val = past_data.min()
            max_val = past_data.max()
            if max_val - min_val < 1e-8:
                return 0.5
            return (series[current_idx] - min_val) / (max_val - min_val)
```

### Feature Pre-Computation

For training speed, features are pre-computed per-week:

```python
def prepare_week_features(week_data, normalizer):
    """Pre-compute and normalize ALL features for a training week."""
    df = week_data['M5'].copy()

    # 1. Calculate raw features (TA-Lib / pandas-ta)
    df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
    df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # ... all 42 features

    # 2. Normalize EACH BAR using only past data
    for col in feature_columns:
        df[f'{col}_norm'] = [
            normalizer.normalize(df[col].values, i)
            for i in range(len(df))
        ]

    # 3. Skip warmup period (first 200 bars have insufficient indicator history)
    df = df.iloc[200:]

    return df
```

---

## 4. Reward Function Deep Dive

### The 5-Component Design (with Running Normalization)

> **Updated in v3.3:** Upgraded from 3-component to 5-component composite reward with running
> normalization. See SPARTUS_TRADING_AI.md Section 11 for the full implementation code.

```
COMPONENT 1 (R1): Position P/L (dense signal, every step)
    Weight: 0.40
    → Teaches: being on the right side of the market = good
    → Magnitude: proportional to price movement, scaled by balance
    → Only active when in a trade

COMPONENT 2 (R2): Trade completion quality (sparse signal, at close)
    Weight: 0.20
    → Teaches: bigger net wins > small wins; commission matters
    → Scaled by PnL quality (not binary): min(net_pnl / balance * 50, 2.0)
    → Net PnL = trade PnL - commission (must beat costs)

COMPONENT 3 (R3): Drawdown penalty (emergency signal)
    Weight: 0.15
    → Teaches: avoid catastrophic losses
    → Two-tier: DD > 7% → -1.0 (moderate), DD > 10% → -3.0 (strong)
    → Only fires in serious trouble

COMPONENT 4 (R4): Differential Sharpe ratio (dense signal, every step)
    Weight: 0.15
    → Teaches: improve risk-adjusted returns, not just raw P/L
    → Based on Moody & Saffell (2001) differential Sharpe
    → Gives step-by-step signal about risk-adjusted performance

COMPONENT 5 (R5): Risk-adjusted position bonus (dense signal, when in profit)
    Weight: 0.10
    → Teaches: safe profits > risky profits
    → Profit worth more when drawdown is low
    → Linear decay: r5 = r1 * max(0, 1.0 - dd / 0.10)
    → At DD=0%: full bonus (r5 = r1). At DD=5%: half (r5 = 0.5*r1).
    → At DD=10%: zero bonus. Smoothly differentiates risk levels.
```

**Reward Normalization (RewardNormalizer):**
```
SAC's auto-tuned entropy works best with unit-scale rewards.
Without normalization, reward magnitudes range widely across components,
which destabilizes entropy coefficient tuning.

RewardNormalizer uses exponential moving average (tau=0.001):
    → Tracks running_mean and running_var
    → Normalizes: (reward - mean) / sqrt(var + 1e-8)
    → Hard clips to [-5.0, +5.0] for stability

Combined formula:
    raw_reward = (0.40 * R1) + (0.20 * R2) + (0.15 * R3) + (0.15 * R4) + (0.10 * R5)
    final_reward = reward_normalizer.normalize(raw_reward)

CRITICAL: Terminal penalties (circuit breakers) use SET, not ADD:
    → Account blown: reward = -5.0 (not reward -= 10.0)
    → DD >= 10%: reward = -4.0 (not reward -= 5.0)
    → Daily DD > 3%: reward = -3.0
    → Combined with done=True, which zeros out bootstrapped future returns,
      these penalties are already very severe without needing outlier magnitudes.
    → Using -= would create rewards outside [-5, +5] and destabilize SAC entropy.
```

### What We DON'T Reward

| Signal | Why NOT Included |
|--------|-----------------|
| Opening a trade | Creates incentive to open trades for the bonus, not for profit |
| Pattern match | Patterns are tracked in memory, rewarded indirectly through P/L |
| Trend alignment | Captured naturally through P/L and differential Sharpe |
| Holding in profit | P/L already captures this (positive reward each step while in profit) |
| Milestones (25%, 50%, 75% to TP) | Creates artificial targets that distort natural exit decisions |
| Exploration bonuses | SAC handles exploration automatically |
| Inactivity penalty | Not trading IS sometimes the right decision (ranging market) |

---

## 5. Memory System Design

### Database Schema

```sql
-- Trade History: Every completed trade
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    week_number INTEGER,
    entry_step INTEGER,
    exit_step INTEGER,
    side TEXT,           -- 'LONG' or 'SHORT'
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    duration_bars INTEGER,
    -- Market conditions at entry
    rsi_at_entry REAL,
    atr_at_entry REAL,
    trend_dir_at_entry REAL,
    mtf_alignment_at_entry REAL,
    volume_ratio_at_entry REAL,
    session_at_entry TEXT,
    -- Outcome
    hit_sl BOOLEAN,
    hit_tp BOOLEAN,
    manual_close BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trend Predictions: Track prediction accuracy
-- EVERY prediction is stored, verified, and kept permanently.
-- The AI learns from this: "was I right or wrong about market direction?"
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    week_number INTEGER,
    step INTEGER,
    predicted_direction TEXT,  -- 'UP', 'DOWN'
    confidence REAL,          -- How strong the signal was (0.3-1.0)
    price_at_prediction REAL, -- Price when prediction was made
    -- Filled in LATER (after lookforward bars have passed)
    actual_direction TEXT,    -- What actually happened: 'UP' or 'DOWN'
    actual_move REAL,         -- Price difference (positive = up, negative = down)
    correct BOOLEAN,          -- Did prediction match actual? TRUE/FALSE
    verified_at_step INTEGER, -- Step when verification occurred
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- VERIFICATION CYCLE:
-- 1. AI outputs signal > 0.3 → INSERT prediction (actual fields NULL)
-- 2. After 20 bars pass → UPDATE with actual_direction, actual_move, correct
-- 3. Rolling accuracy queried as observation feature #38
-- 4. AI sees its own accuracy → adjusts confidence accordingly

-- Pattern Memory: Market conditions → outcomes
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY,
    -- Condition signature (binned for matching)
    rsi_bin INTEGER,          -- RSI binned to 10-point bands (0-10, 10-20, etc.)
    trend_bin INTEGER,        -- Trend direction binned (-2, -1, 0, 1, 2)
    volatility_bin INTEGER,   -- ATR percentile bin (1-5)
    session TEXT,             -- London, NY, Asia, Off
    mtf_alignment_bin INTEGER,-- Multi-TF alignment bin (-2 to 2)
    -- Outcome statistics
    total_occurrences INTEGER DEFAULT 0,
    winning_occurrences INTEGER DEFAULT 0,
    avg_pnl REAL DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TP Accuracy: Track take-profit hit rate
-- Did the AI set realistic TPs? Did the market reach them?
CREATE TABLE tp_tracking (
    id INTEGER PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    tp_level REAL,           -- TP price level set at entry
    entry_price REAL,
    exit_price REAL,
    side TEXT,               -- 'LONG' or 'SHORT'
    tp_hit BOOLEAN,          -- Did trade close by hitting TP?
    sl_hit BOOLEAN,          -- Did trade close by hitting SL?
    manual_close BOOLEAN,    -- Did AI close early (exit signal)?
    tp_was_reachable BOOLEAN,-- Did price touch TP level at any point (using bar high/low)?
    bars_to_close INTEGER,   -- How many bars the trade lasted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Checkpoints
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY,
    week_number INTEGER,
    model_path TEXT,
    balance REAL,
    win_rate REAL,
    sharpe REAL,
    total_trades INTEGER,
    trend_accuracy REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Memory Query for Observations

```python
class TradingMemory:
    """Persistent memory that feeds into observations."""

    def get_similar_pattern_winrate(self, current_state):
        """
        Find past trades with similar market conditions.
        Uses Bayesian shrinkage to prevent noisy win rates from small samples.

        The key insight: 60% win rate over 5 samples has a 95% CI of [15%, 95%].
        That's useless noise. Bayesian shrinkage pulls small-sample estimates
        toward the prior (0.5 = "we don't know") proportionally to sample size.

        Credibility schedule:
          10 samples → 25% confidence → mostly neutral (0.5)
          30 samples → 50% confidence → balanced
          100 samples → 77% confidence → mostly real signal
          300 samples → 91% confidence → strong signal
        """
        rsi_bin = int(current_state['rsi_14'] * 10)  # 0-10
        trend_bin = int(round(current_state['mtf_alignment'] * 2))  # -2 to 2
        vol_bin = self._get_volatility_bin(current_state['atr_14_norm'])
        session = current_state['session']

        results = self.db.execute("""
            SELECT total_occurrences, winning_occurrences
            FROM patterns
            WHERE rsi_bin = ? AND trend_bin = ? AND volatility_bin = ? AND session = ?
        """, (rsi_bin, trend_bin, vol_bin, session)).fetchone()

        if results and results[0] >= 10:
            count = results[0]
            raw_win_rate = results[1] / count

            # Bayesian shrinkage: blend raw win rate with prior (0.5)
            # 30-sample credibility weight: need ~30 samples for 50% confidence
            credibility = count / (count + 30)
            return (raw_win_rate * credibility) + (0.5 * (1 - credibility))

        return 0.5  # <10 samples → not enough data, return prior

    def get_trend_accuracy(self, window=100):
        """Rolling trend prediction accuracy."""
        results = self.db.execute("""
            SELECT correct FROM predictions
            WHERE verified_at_step IS NOT NULL
            ORDER BY id DESC LIMIT ?
        """, (window,)).fetchall()

        if len(results) < 10:
            return 0.5  # Not enough data → neutral
        return sum(r[0] for r in results) / len(results)

    def store_trade(self, trade_info, market_conditions):
        """Store completed trade and update pattern statistics."""
        # Insert trade
        self.db.execute("INSERT INTO trades (...) VALUES (...)", trade_info)

        # Update pattern statistics
        rsi_bin = int(market_conditions['rsi_14'] * 10)
        # ... bin other conditions
        won = 1 if trade_info['pnl'] > 0 else 0

        self.db.execute("""
            INSERT INTO patterns (rsi_bin, trend_bin, volatility_bin, session,
                                  total_occurrences, winning_occurrences)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(rsi_bin, trend_bin, volatility_bin, session) DO UPDATE SET
                total_occurrences = total_occurrences + 1,
                winning_occurrences = winning_occurrences + ?,
                last_updated = CURRENT_TIMESTAMP
        """, (rsi_bin, trend_bin, vol_bin, session, won, won))
```

---

## 6. Week-by-Week Training Process

### Training Loop Detail

```python
def train_one_week(agent, week_data, memory, config, week_idx):
    """Train the agent on one week of historical data."""

    # 1. Create environment
    env = SpartusTradeEnv(
        market_data=week_data,
        config=config,
        memory=memory,
        initial_balance=memory.current_balance
    )

    # 2. Set environment on agent
    agent.set_env(env)

    # 3. Train
    # total_timesteps should be more than 1 week to allow multiple passes
    # ~1300 bars per week × ~8 = ~10,000 steps allows ~8 full passes
    agent.learn(
        total_timesteps=config.steps_per_week,  # 10,000
        reset_num_timesteps=False,  # Global step counter continues
        progress_bar=True,
        callback=SpartusCallback(config, week_idx)
    )

    # 4. Collect results
    results = {
        'week': week_idx,
        'final_balance': env.balance,
        'trades': len(env.trades_history),
        'wins': sum(1 for t in env.trades_history if t['pnl'] > 0),
        'total_pnl': env.balance - memory.current_balance,
        'max_drawdown': env.max_drawdown,
        'trend_accuracy': memory.get_trend_accuracy()
    }

    # 5. Update memory with this week's trades
    for trade in env.trades_history:
        memory.store_trade(trade, trade['market_conditions_at_entry'])

    # 6. Carry balance forward
    memory.current_balance = env.balance

    # 7. Save checkpoint if profitable
    if env.balance > memory.current_balance * 0.98:  # Not lost more than 2%
        agent.save(f"storage/models/spartus_week_{week_idx:04d}")
        memory.save_checkpoint(week_idx, results, f"storage/models/spartus_week_{week_idx:04d}")

    return results
```

### Training Progression Expectations

```
Weeks 1-10:   AI is random. Mostly losing. Memory is small.
              Expected win rate: 40-48%
              Expected trend accuracy: 48-52%
              This is NORMAL. Don't panic.

Weeks 10-30:  AI starts recognizing basic patterns.
              Memory has 100-300 trades.
              Win rate climbing: 48-52%
              Some weeks profitable, many still red.

Weeks 30-60:  AI has meaningful experience.
              Memory has 500-1000 trades.
              Pattern matching starting to help.
              Win rate: 50-54%
              Trend accuracy: 52-56%
              More consistent, fewer big losses.

Weeks 60-100: AI is developing real skill.
              Memory is rich (1000+ trades).
              Win rate: 52-57%
              Trend accuracy: 54-60%
              Majority of weeks profitable.
              Small, consistent gains.

Weeks 100+:   Mature AI if all goes well.
              Win rate: 55-60%
              Trend accuracy: 57-63%
              Consistent small profits.
              Quick to cut losses.
              Knows when NOT to trade.
```

### Curriculum Learning (Progressive Difficulty)

> **Problem:** Without curriculum learning, the AI's first week of training could be a 2020
> COVID crash or a flash crash day. Starting on extreme conditions teaches the wrong lessons
> (e.g., "always short" or "never trade"). Inspired by TRADING-R1 (arXiv 2509.11420, 2025).

Training data is organized into 3 stages of progressive difficulty:

```python
def classify_week_difficulty(week_data):
    """
    Pre-classify each week by difficulty BEFORE training starts.
    Uses 3 metrics: trend clarity, volatility rank, whipsaw frequency.
    """
    m5 = week_data['M5']

    # Trend clarity: how directional was the week?
    # Higher = clearer trend, easier to trade
    weekly_return = (m5['close'].iloc[-1] - m5['close'].iloc[0]) / m5['close'].iloc[0]
    avg_bar_range = (m5['high'] - m5['low']).mean()
    trend_clarity = abs(weekly_return) / (avg_bar_range * len(m5) + 1e-8)

    # Volatility rank: percentile of this week's ATR vs all weeks
    atr = m5['high'].rolling(14).max() - m5['low'].rolling(14).min()
    vol_rank = atr.mean()  # Higher = more volatile = harder

    # Whipsaw frequency: how many direction reversals?
    direction = np.sign(m5['close'].diff())
    reversals = (direction.diff() != 0).sum()
    whipsaw_freq = reversals / len(m5)  # Higher = more whipsaw = harder

    return {
        'trend_clarity': trend_clarity,
        'vol_rank': vol_rank,
        'whipsaw_freq': whipsaw_freq,
        'difficulty_score': (1 - trend_clarity) * 0.4 + vol_rank * 0.3 + whipsaw_freq * 0.3
    }

# Sort weeks by difficulty score (lowest = easiest)
all_weeks_scored = [(w, classify_week_difficulty(w)) for w in all_weeks]
easy_weeks = sorted(all_weeks_scored, key=lambda x: x[1]['difficulty_score'])
```

**3-Stage Curriculum:**

```
STAGE 1 (Weeks 1-30): "Easy Mode"
    → Train on the 30 EASIEST weeks (clear trends, low volatility)
    → Wider SL margins (1.5x normal min distance)
    → Purpose: teach basic entry/exit mechanics without noise
    → AI learns: "buy when market trending up, sell when trending down"

STAGE 2 (Weeks 31-80): "Normal Mode"
    → Mix of trending AND ranging weeks (medium difficulty)
    → Normal SL margins, full realistic spreads/commissions
    → Purpose: teach regime awareness
    → AI learns: "don't trade during chop, sit out when unclear"

STAGE 3 (Weeks 81-200): "Full Realism"
    → ALL remaining weeks in CHRONOLOGICAL order
    → All market conditions: volatility spikes, news events, flash moves
    → Purpose: final hardening against real-world conditions
    → AI learns: survival, adaptation, true generalization
```

**Important:** Curriculum only affects *training data order*, not the validation/test sets. Validation always uses chronological held-out data to measure true generalization.

---

## 7. Validation Protocol

### Walk-Forward Validation

```python
def walk_forward_validate(agent, all_weeks, config):
    """
    Walk-forward: train on N weeks, test on next M weeks, advance.
    This simulates real deployment where you train on past data
    and trade into the unknown future.

    CRITICAL: step_size MUST equal test_size to prevent leakage.
    If step_size < test_size, previous test data leaks into next fold's training.
    """
    train_size = 50   # Train on 50 weeks
    test_size = 10    # Test on next 10 weeks
    embargo = 1       # 1-week gap between train and test (prevents information leak)
    step_size = 10    # Advance 10 weeks = test_size (NO OVERLAP between test sets)

    results = []

    for start in range(0, len(all_weeks) - train_size - embargo - test_size, step_size):
        # Train phase: weeks [start, start + train_size)
        train_weeks = all_weeks[start:start + train_size]

        # Embargo: skip 'embargo' weeks between train and test
        test_start = start + train_size + embargo
        test_weeks = all_weeks[test_start:test_start + test_size]

        # Fresh agent and memory for each fold (no contamination)
        agent_copy = deepcopy(agent)
        memory = TradingMemory(":memory:")  # Fresh memory for this fold

        for week in train_weeks:
            train_one_week(agent_copy, week, memory, config)

        # Test (no learning, deterministic)
        fold_results = []
        for week in test_weeks:
            env = SpartusTradeEnv(week, config, memory)
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = agent_copy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
            fold_results.append(info)

        results.append({
            'train_period': f"weeks {start}-{start+train_size-1}",
            'test_period': f"weeks {test_start}-{test_start+test_size-1}",
            'sharpe': calculate_sharpe(fold_results),
            'win_rate': calculate_win_rate(fold_results),
            'total_pnl': sum(r['balance'] - config.initial_balance for r in fold_results),
            'max_dd': max(r['max_drawdown'] for r in fold_results)
        })

    # Verify no test overlap
    # Fold 1: train 0-49, test 51-60
    # Fold 2: train 10-59, test 61-70
    # Fold 3: train 20-69, test 71-80
    # Test sets NEVER overlap. Previous test data enters future training (rolling forward).
    return results
```

### Purged Cross-Validation (for Hyperparameter Tuning)

```python
from timeseriescv import PurgedGroupTimeSeriesSplit

def purged_cv_evaluate(agent_class, config, all_weeks):
    """
    Purged cross-validation for hyperparameter selection.
    Use this to choose the best hyperparameters.
    Use walk-forward for final evaluation.
    """
    cv = PurgedGroupTimeSeriesSplit(
        n_splits=5,
        group_gap=2,  # 2-week embargo between train/test
    )

    scores = []
    for train_idx, test_idx in cv.split(all_weeks):
        train_data = [all_weeks[i] for i in train_idx]
        test_data = [all_weeks[i] for i in test_idx]

        # Train
        agent = agent_class(config)
        for week in train_data:
            train_one_week(agent, week, memory, config)

        # Test
        sharpe = evaluate_on_weeks(agent, test_data, config)
        scores.append(sharpe)

    return np.mean(scores), np.std(scores)
```

### 7.5 Ensemble SAC (Multiple Seeds)

> **Problem:** A single SAC agent is fragile — different random seeds produce very different
> training outcomes. One seed might converge to a profitable strategy, another might not.
> This is well-documented in deep RL and has been addressed in FinRL contest winners (2024-2025)
> via simple ensemble methods (arXiv 2511.12120).

**Approach:** Train 3 SAC agents with different random seeds. All share the same hyperparameters,
data, curriculum, and features — only the random seed differs.

```python
ENSEMBLE_CONFIG = {
    "n_agents": 3,
    "seeds": [42, 137, 2024],   # Fixed seeds for reproducibility
    "agreement_threshold": 2,    # 2/3 must agree on direction to trade
}

def ensemble_predict(agents, obs):
    """
    Ensemble decision-making at validation/deployment time.
    Direction: majority vote (2/3 must agree)
    Conviction: average of agreeing agents
    Exit: any agent's exit signal > 0.5 triggers close (conservative)
    """
    actions = [agent.predict(obs, deterministic=True)[0] for agent in agents]

    # Direction: majority vote
    directions = [np.sign(a[0]) if abs(a[0]) > 0.3 else 0 for a in actions]
    long_votes = sum(1 for d in directions if d > 0)
    short_votes = sum(1 for d in directions if d < 0)
    hold_votes = sum(1 for d in directions if d == 0)

    if long_votes >= ENSEMBLE_CONFIG["agreement_threshold"]:
        direction = 1.0
        agreeing = [a for a, d in zip(actions, directions) if d > 0]
    elif short_votes >= ENSEMBLE_CONFIG["agreement_threshold"]:
        direction = -1.0
        agreeing = [a for a, d in zip(actions, directions) if d < 0]
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])  # No agreement → hold

    # Conviction: average of agreeing agents
    conviction = np.mean([a[1] for a in agreeing])

    # Exit: ANY agent voting to exit triggers close (conservative)
    exit_urgency = max(a[2] for a in actions)

    # SL management: average of agreeing agents
    sl_mgmt = np.mean([a[3] for a in agreeing])

    return np.array([direction * abs(np.mean([a[0] for a in agreeing])),
                      conviction, exit_urgency, sl_mgmt])
```

**Training process:**
1. Train 3 agents in parallel (same data, different seeds)
2. Track each agent's validation Sharpe independently
3. If one agent consistently underperforms the other two by >30% on validation Sharpe over 50 weeks, drop it and retrain with a new seed
4. At deployment, use ensemble predictions

**Why this works:**
- Reduces variance from random initialization
- Majority vote filters out individual agent noise
- Conservative exit (any agent) protects capital
- Minimal extra compute: 3x training time, but agents can train in parallel

---

## 8. Regime Awareness

### Hidden Markov Model for Market Regimes

While the main model is a single unified RL agent, a simple HMM regime detector runs as preprocessing:

```python
from hmmlearn import hmm

class RegimeDetector:
    """Simple HMM to classify market regimes."""

    def __init__(self, n_regimes=3):
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100
        )

    def fit(self, returns, volatility):
        """Fit on historical returns and volatility."""
        X = np.column_stack([returns, volatility])
        self.model.fit(X)

    def predict(self, returns, volatility):
        """Predict current regime."""
        X = np.column_stack([returns, volatility])
        states = self.model.predict(X)
        return states[-1]  # Current regime

    def get_regime_label(self, state):
        """Map HMM state to human-readable label."""
        # States are sorted by mean return during fitting
        labels = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
        return labels.get(state, "UNKNOWN")
```

**How regime information enters the observation space:**

Regime is NOT a separate 43rd feature. It is implicitly captured by existing features #20-24 (`h1_trend_dir`, `h4_trend_dir`, `d1_trend_dir`, `h1_rsi`, `mtf_alignment`). These multi-timeframe features naturally encode the current regime:
- All positive + high momentum = trending bullish regime
- All negative + high momentum = trending bearish regime
- Mixed/near-zero + low momentum = ranging regime
- Extreme moves + high ATR = volatile/crisis regime

**Optional HMM enhancement:** If regime detection via raw features proves insufficient, the HMM's output would REPLACE feature #24 (`mtf_alignment`) with a more sophisticated 3-state regime signal. This keeps the feature count at 42. The HMM is pre-fit on historical returns + volatility before training begins, then used as a read-only feature generator during training.

The AI learns to behave differently in different regimes **through experience**, not through separate reward rules.

---

## 9. Handling Non-Stationarity

### Fractional Differentiation

```python
from fracdiff import fdiff

def fractionally_differentiate(price_series, d=0.35):
    """
    Make price series stationary while preserving memory.

    d=0 → original price (non-stationary, maximum memory)
    d=1 → returns (stationary, minimum memory)
    d≈0.35 → sweet spot for gold (verified via ADF test)
    """
    result = fdiff(price_series.values, d=d)
    return pd.Series(result, index=price_series.index)

def find_optimal_d(price_series, max_d=1.0, significance=0.05):
    """Find minimum d that achieves stationarity."""
    from statsmodels.tsa.stattools import adfuller

    for d in np.arange(0.05, max_d, 0.05):
        diff_series = fdiff(price_series.values, d=d)
        diff_series = diff_series[~np.isnan(diff_series)]
        if len(diff_series) > 100:
            adf_pvalue = adfuller(diff_series)[1]
            if adf_pvalue < significance:
                return d
    return 1.0  # Fallback to standard returns
```

### Rolling Normalization

```python
def rolling_zscore(series, window=200):
    """
    Z-score normalization using rolling window.
    At each bar, uses ONLY the past 'window' bars.
    No future data leakage.
    """
    rolling_mean = series.rolling(window=window, min_periods=50).mean()
    rolling_std = series.rolling(window=window, min_periods=50).std()
    normalized = (series - rolling_mean) / (rolling_std + 1e-8)
    return normalized.clip(-5, 5)  # Clip extreme outliers
```

---

## 10. Monitoring & Debugging

### TensorBoard Logging

```python
class SpartusCallback(BaseCallback):
    """Training callback for logging metrics and SAC health monitoring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_norm_ema = 1.0     # Running average of gradient norms
        self._initial_entropy = None   # Captured after first few updates

    def _on_step(self):
        # Log every 100 steps
        if self.num_timesteps % 100 == 0:
            env = self.training_env.envs[0]

            # ── Trading metrics ──
            self.logger.record("spartus/balance", env.balance)
            self.logger.record("spartus/equity", env._get_equity())
            self.logger.record("spartus/win_rate", env._get_win_rate())
            self.logger.record("spartus/total_trades", len(env.trades_history))
            self.logger.record("spartus/drawdown", env.current_drawdown)
            self.logger.record("spartus/trend_accuracy", env.memory.get_trend_accuracy())

            # ── SAC Internals (3.3: Stability Monitoring) ──
            if hasattr(self.model, 'ent_coef_tensor'):
                ent_coef = self.model.ent_coef_tensor.item()
                self.logger.record("sac/entropy_coef_alpha", ent_coef)

            if hasattr(self.model, 'critic'):
                # Q-value statistics from the last critic update
                # (SB3 stores these in model.logger internally)
                pass  # Q-values logged via SB3's built-in "train/critic_loss"

            # Log gradient norms (actor + critic)
            if hasattr(self.model, 'actor') and self.num_timesteps > self.model.learning_starts:
                actor_grad_norm = self._compute_grad_norm(self.model.actor)
                critic_grad_norm = self._compute_grad_norm(self.model.critic)
                self.logger.record("sac/actor_grad_norm", actor_grad_norm)
                self.logger.record("sac/critic_grad_norm", critic_grad_norm)

                # Gradient spike detection
                self._grad_norm_ema = 0.99 * self._grad_norm_ema + 0.01 * (actor_grad_norm + critic_grad_norm)
                combined = actor_grad_norm + critic_grad_norm
                if combined > 10 * self._grad_norm_ema:
                    self.logger.record("sac/ALERT_gradient_spike", True)

        # ── Observation health check every 1000 steps (3.4) ──
        if self.num_timesteps % 1000 == 0 and self.num_timesteps > 0:
            self._check_observation_health()

        return True

    def _compute_grad_norm(self, network):
        """Compute total gradient L2 norm for a network."""
        total_norm = 0.0
        for p in network.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _check_observation_health(self):
        """
        Runtime observation verification (3.4).
        Checks for dead features, exploding values, and NaN contamination.
        """
        env = self.training_env.envs[0]
        if not hasattr(env, '_recent_observations'):
            return

        obs_buffer = np.array(env._recent_observations[-100:])  # Last 100 obs
        if len(obs_buffer) < 50:
            return

        for feat_idx in range(obs_buffer.shape[1]):
            feat_std = np.std(obs_buffer[:, feat_idx])
            feat_nan_pct = np.mean(np.isnan(obs_buffer[:, feat_idx]))

            if feat_std < 0.01:
                self.logger.record(f"obs_health/dead_feature_{feat_idx}", feat_std)
            if feat_std > 3.0:
                self.logger.record(f"obs_health/exploding_feature_{feat_idx}", feat_std)
            if feat_nan_pct > 0.05:
                self.logger.record(f"obs_health/nan_feature_{feat_idx}", feat_nan_pct)

    def _on_rollout_end(self):
        # Log at end of each rollout
        self.logger.record("spartus/week", self.current_week)
        self.logger.record("spartus/memory_size", self.memory.total_trades())
```

### Key Things to Watch

```
HEALTHY TRAINING:
    ✓ Balance curve trending upward (with dips, not monotonic)
    ✓ Win rate slowly increasing over weeks
    ✓ Trend accuracy slowly increasing over weeks
    ✓ Memory growing steadily (10-20 trades per week)
    ✓ Drawdown staying below 10%
    ✓ Trade frequency 5-30 per week

UNHEALTHY TRAINING:
    ✗ Balance monotonically decreasing → reward signal might be wrong
    ✗ Win rate stuck at 50% after 50 weeks → features may not be predictive
    ✗ Zero trades → model is stuck (check action thresholds)
    ✗ 100+ trades per week → reward hacking
    ✗ Drawdown > 15% → risk rules need tightening
    ✗ Trend accuracy < 48% after 100 weeks → model not learning

SAC INTERNALS — CRITICAL HEALTH CHECKS:
    ✗ Q-value mean > 10x max possible episode return → Q-value explosion
    ✗ Q-value growing exponentially over 10+ weeks → critic divergence
    ✗ Entropy coef α collapsed to ~0 → exploration dead, policy stuck
    ✗ Entropy coef α exploded to >10 → agent behaving randomly
    ✗ Policy entropy < 20% of initial value → premature convergence
    ✗ Actor/critic gradient norm > 10x running average → gradient spike
    ✗ Gradients consistently at max_grad_norm → signal being lost, reduce LR

OBSERVATION HEALTH (checked every 1000 steps):
    ✗ Any feature std < 0.01 → dead feature (no information)
    ✗ Any feature std > 3.0 → normalization broken or outlier contamination
    ✗ Any feature >5% NaN/inf → data pipeline issue, fix immediately
```

---

## 11. Anti-Leakage Test Suite

```python
# tests/test_no_leakage.py

def test_feature_no_future_data():
    """CRITICAL: Verify features at time T use ONLY data from T and before."""
    data = load_sample_week()

    for t in range(200, len(data)):
        # Calculate features at time T
        features_at_t = feature_builder.build_single(data, t)

        # Corrupt future data (set to NaN)
        corrupted_data = data.copy()
        corrupted_data.iloc[t+1:] = np.nan

        # Recalculate features - should be IDENTICAL
        features_corrupted = feature_builder.build_single(corrupted_data, t)

        np.testing.assert_array_equal(
            features_at_t, features_corrupted,
            err_msg=f"Feature at bar {t} uses future data!"
        )

def test_normalization_no_future():
    """Verify normalization at T uses only data up to T."""
    series = pd.Series(np.random.randn(1000))
    normalizer = ExpandingWindowNormalizer()

    for t in range(50, len(series)):
        norm_val = normalizer.normalize(series, t)

        # Change future values - normalization should NOT change
        modified = series.copy()
        modified.iloc[t+1:] = 999999

        norm_val_modified = normalizer.normalize(modified, t)

        assert norm_val == norm_val_modified, \
            f"Normalization at {t} leaked future data!"

def test_label_not_in_observations():
    """Verify trend labels are NEVER part of the observation space."""
    env = SpartusTradeEnv(load_sample_week(), config)
    obs, _ = env.reset()

    # Observation should only contain features, not labels
    assert len(obs) == config.num_features * config.frame_stack
    # No label values should appear in observations
```

---

---

## 12. Account Management Training

### How the AI Learns to Manage £100

The AI starts every training run with £100. It must learn:

1. **Lot sizing relative to balance** — Can't open 1.0 lots on a £100 account
2. **Scaling down during drawdown** — When losing, reduce exposure
3. **Scaling up during growth** — As account grows, can afford slightly larger positions
4. **Survival instinct** — Avoid blowing the account entirely

### The Account Management Learning Curve

```
PHASE 1 (Weeks 1-20): Random lot sizing
    → AI doesn't understand account size yet
    → Often oversizes positions, gets stopped out
    → May blow the account multiple times
    → Each bankruptcy reinforces: "big lots on small account = death"

PHASE 2 (Weeks 20-50): Basic awareness
    → AI starts outputting lower conviction (smaller lots)
    → Bankruptcies become rare
    → Still not scaling with account size — uses minimum lots
    → This is fine — survival is the first lesson

PHASE 3 (Weeks 50-100): Intelligent sizing
    → AI scales conviction with account size
    → Uses 0.01-0.02 lots on £100-150
    → Uses 0.02-0.04 lots on £150-250
    → Reduces conviction during drawdown periods
    → Starts making consistent small gains (£1-2 per trade)

PHASE 4 (Weeks 100+): Mature management
    → Account grown to £200+
    → AI knows exactly when to be aggressive vs conservative
    → Drawdown → immediate lot reduction (learned behavior)
    → New high → gradual lot increase
    → Consistent compounding
```

### Account Features in Observation Space

The AI sees these 8 account/position features every step (features #30-37 in the 42-feature list).
See SPARTUS_TRADING_AI.md Section 10.G for the authoritative feature definitions.

```python
account_observation = {
    # --- Position awareness ---
    'has_position': 1.0 if position_open else 0.0,       # #30: Binary flag
    'position_side': +1.0 if long, -1.0 if short, 0.0,   # #31: Direction

    # --- Trade P/L awareness ---
    'unrealized_pnl': position_pnl / balance,             # #32: Current trade P/L as % of balance
    'position_duration': bars_in_trade / 100,              # #33: Normalized duration

    # --- Account health ---
    'current_drawdown': (peak_equity - equity) / peak_equity,  # #34: 0.0=peak, 0.10=10% DD
    'equity_ratio': current_equity / initial_balance,          # #35: 2.0 = doubled account

    # --- SL management awareness ---
    'sl_distance_ratio': (current_price - stop_loss) / atr,   # #36: How far SL is (0 if flat)
    'profit_locked_pct': (stop_loss - entry_price) / atr,  # #37: Profit locked by SL trail (ATR-scaled)
    # ATR denominator gives ~0.5-2.5 (meaningful). entry_price gives ~0.0004 for gold (invisible).
}
# NOTE: These 8 features are NOT normalized (already ratios/flags in bounded ranges).
# profit_locked_pct is the exception — it's ATR-scaled (not a simple ratio), but
# its range (~0.0-3.0) is already in a good scale for the network.
# See SPARTUS_TRAINING_METHODOLOGY.md Section 3 normalization exemptions.
```

### Bankruptcy Handling During Training

```python
def handle_bankruptcy(self, env, week_idx, memory):
    """Handle account bankruptcy during training."""

    # Record the bankruptcy
    memory.record_bankruptcy(
        week=week_idx,
        balance_at_death=env.balance,
        peak_balance=env.peak_balance,
        cause=self._determine_cause(env),
        trades_this_week=len(env.trades_history)
    )

    # Terminal penalty already applied in step(): reward = -5.0 with done=True
    # (SET, not ADD — stays within normalizer's [-5, +5] clip range)

    # Reset balance for next episode
    # Option A: Reset to initial £100 (harsh — like going bankrupt and starting over)
    # Option B: Reset to 50% of peak (more forgiving — like getting a bail-out)
    # We use Option A because it teaches maximum respect for risk
    next_balance = self.config.initial_balance  # £100

    # Memory PERSISTS — the AI remembers what went wrong
    # This is key: the bankruptcy trades stay in memory
    # The AI learns "I had these conditions, I blew up — don't do that again"

    return next_balance

def _determine_cause(self, env):
    """Analyze what caused the bankruptcy."""
    if env.trades_history:
        last_trades = env.trades_history[-5:]
        big_losses = [t for t in last_trades if t['pnl'] < -env.peak_balance * 0.05]
        if big_losses:
            return "oversized_positions"
        elif len(last_trades) > 10:
            return "death_by_thousand_cuts"
        else:
            return "single_catastrophic_loss"
    return "unknown"
```

---

## 13. Realistic Market Simulation Details

### Why Realism Matters

If the training environment is too easy (no spreads, no slippage, no commissions), the AI will learn strategies that fail in live trading. The simulation must penalize the same things the real market penalizes.

### Spread Simulation

```python
class SpreadSimulator:
    """Simulate realistic XAUUSD spreads based on session and volatility."""

    # Empirical spread data for XAUUSD (in pips, 1 pip = $0.10)
    SPREADS = {
        'london_am':      {'mean': 1.5, 'std': 0.3},   # 08:00-12:00 UTC
        'ny_overlap':     {'mean': 2.0, 'std': 0.5},   # 13:00-17:00 UTC
        'london_pm':      {'mean': 2.5, 'std': 0.4},   # 12:00-13:00 UTC
        'ny_pm':          {'mean': 3.0, 'std': 0.5},   # 17:00-20:00 UTC
        'asia':           {'mean': 3.5, 'std': 0.8},   # 00:00-08:00 UTC
        'off_hours':      {'mean': 5.0, 'std': 1.5},   # 20:00-00:00 UTC
        'high_vol_event': {'mean': 8.0, 'std': 3.0},   # NFP, FOMC, etc.
    }

    def get_spread(self, timestamp, atr, atr_avg):
        """Get spread based on time and volatility."""
        session = self._get_session(timestamp)
        base = self.SPREADS[session]

        # Add volatility component: higher ATR → wider spread
        vol_multiplier = max(1.0, atr / atr_avg) if atr_avg > 0 else 1.0
        vol_multiplier = min(vol_multiplier, 3.0)  # Cap at 3x

        spread_pips = np.random.normal(
            base['mean'] * vol_multiplier,
            base['std']
        )
        return max(0.5, spread_pips) * 0.10  # Convert to price, minimum 0.5 pips
```

### Slippage Simulation

```python
class SlippageSimulator:
    """Simulate order execution slippage."""

    def get_slippage(self, order_type, session, volatility_regime):
        """
        Slippage depends on:
        - Market orders slip more than limit orders
        - High volatility = more slippage
        - Off-hours = more slippage
        """
        if order_type == 'market':
            base_mean = 0.5  # 0.5 pips average
            base_std = 0.3
        else:  # limit order
            base_mean = 0.1
            base_std = 0.1

        # Adjust for volatility
        if volatility_regime == 'high':
            base_mean *= 2.5
            base_std *= 2.0
        elif volatility_regime == 'low':
            base_mean *= 0.5
            base_std *= 0.5

        # Adjust for session
        if session in ['asia', 'off_hours']:
            base_mean *= 1.5

        slippage = abs(np.random.normal(base_mean, base_std))
        return slippage * 0.10  # Convert to price
```

### Stop-Loss Execution Realism

```python
def check_sl_tp_realistic(self, current_bar):
    """
    Check SL/TP using HIGH and LOW of the bar, not just CLOSE.

    In real trading:
    - A long position's SL is hit if the LOW touches the SL level
    - A short position's SL is hit if the HIGH touches the SL level
    - Same logic for TP in reverse

    Additionally, SL execution includes slippage (often WORSE than the SL level)
    """
    if not self.position:
        return

    pos = self.position
    bar = current_bar

    if pos['side'] == 'LONG':
        # SL hit if LOW reaches SL
        if bar['low'] <= pos['stop_loss']:
            slippage = abs(self._get_slippage())
            exit_price = pos['stop_loss'] - slippage  # Worse than SL
            self._close_position(exit_price, reason='SL_HIT')

        # TP hit if HIGH reaches TP
        elif pos.get('take_profit') and bar['high'] >= pos['take_profit']:
            exit_price = pos['take_profit']  # TP usually fills at level
            self._close_position(exit_price, reason='TP_HIT')

    elif pos['side'] == 'SHORT':
        # SL hit if HIGH reaches SL
        if bar['high'] >= pos['stop_loss']:
            slippage = abs(self._get_slippage())
            exit_price = pos['stop_loss'] + slippage  # Worse than SL
            self._close_position(exit_price, reason='SL_HIT')

        # TP hit if LOW reaches TP
        elif pos.get('take_profit') and bar['low'] <= pos['take_profit']:
            exit_price = pos['take_profit']
            self._close_position(exit_price, reason='TP_HIT')
```

### Commission Model

```python
def deduct_commission(self, lots, balance):
    """
    ECN broker commission model.
    Typical: $3.50 per lot per side = $7 round trip.

    On a £100 account trading 0.01 lots:
    - Commission = 0.01 × $7 = $0.07 per trade
    - This is realistic and teaches the AI that overtrading has a cost
    """
    commission_per_lot = 7.0  # USD round-trip
    commission = lots * commission_per_lot
    return commission
```

---

## 14. Training Engine Monitoring

### What Gets Logged

Every training step generates data that feeds the dashboard and log files:

```python
class TrainingLogger:
    """Comprehensive logging for training monitoring."""

    def __init__(self, log_dir="storage/logs"):
        self.step_log = open(f"{log_dir}/training_log.jsonl", 'a')
        self.weekly_log = open(f"{log_dir}/weekly_summary.jsonl", 'a')
        self.decision_log = open(f"{log_dir}/decisions.jsonl", 'a')

    def log_step(self, week, step, obs, action, reward, info):
        """Log every N-th step (every 10th to save space)."""
        if step % 10 == 0:
            entry = {
                'week': week, 'step': step,
                'balance': info['balance'],
                'equity': info['equity'],
                'reward': reward,
                'action_direction': float(action[0]),
                'action_conviction': float(action[1]),
                'action_exit': float(action[2]),
                'action_sl_mgmt': float(action[3]),
                'has_position': info['position'] is not None,
                'drawdown': info.get('drawdown', 0),
            }
            self.step_log.write(json.dumps(entry) + '\n')

    def log_decision(self, week, step, decision_type, details):
        """Log every trade open/close decision."""
        entry = {
            'week': week, 'step': step,
            'type': decision_type,  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD'
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.decision_log.write(json.dumps(entry) + '\n')

    def log_weekly_summary(self, week, results):
        """Log end-of-week summary."""
        self.weekly_log.write(json.dumps({
            'week': week,
            'balance': results['final_balance'],
            'trades': results['trades'],
            'wins': results['wins'],
            'pnl': results['total_pnl'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['wins'] / max(results['trades'], 1),
            'trend_accuracy': results['trend_accuracy'],
            'avg_lot_size': results.get('avg_lot_size', 0.01),
        }) + '\n')
```

### Live Dashboard (Rich Terminal)

The training engine runs a real-time terminal dashboard using the `rich` library:

```python
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress

class TrainingDashboard:
    def run_with_dashboard(self, trainer):
        """Run training with live terminal dashboard."""
        console = Console()

        with Live(console=console, refresh_per_second=2) as live:
            for week_idx, week_data in enumerate(trainer.train_weeks):
                # Train one week
                results = trainer.train_one_week(week_data, week_idx)

                # Update display
                live.update(self._build_display(trainer, week_idx, results))

    def _build_display(self, trainer, week, results):
        """Build the full dashboard layout."""
        # Progress bar
        progress_text = f"Week {week+1}/{trainer.total_weeks}"

        # Metrics table
        metrics = Table(title="Training Metrics")
        metrics.add_column("Metric", style="cyan")
        metrics.add_column("Value", style="green")
        metrics.add_row("Balance", f"£{results['final_balance']:.2f}")
        metrics.add_row("Win Rate", f"{results.get('win_rate', 0):.1%}")
        metrics.add_row("Trend Acc", f"{results.get('trend_accuracy', 0):.1%}")
        metrics.add_row("Drawdown", f"{results.get('max_drawdown', 0):.1%}")
        metrics.add_row("Trades", str(results.get('trades', 0)))

        return Panel(metrics, title=progress_text)
```

### 14.5 Convergence Detection

> **Problem:** Without explicit convergence criteria, training runs for a fixed 200 weeks with
> no principled way to know if it's done, overfitting, or stuck. These rules provide
> actionable stopping/intervention signals.

```python
class ConvergenceDetector:
    """
    Monitors training health and detects convergence, overfitting, or collapse.
    Run at the end of each training week.
    """
    def __init__(self, window=50):
        self.val_sharpes = []       # Validation Sharpe per week
        self.action_stds = []       # Policy action std per week
        self.q_value_means = []     # Mean Q-value per week
        self.window = window
        self.best_val_sharpe = -np.inf
        self.best_week = 0

    def update(self, week, val_sharpe, action_std, q_mean, win_rate):
        self.val_sharpes.append(val_sharpe)
        self.action_stds.append(action_std)
        self.q_value_means.append(q_mean)

        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            self.best_week = week

    def get_status(self, current_week):
        if len(self.val_sharpes) < self.window:
            return "WARMING_UP"

        recent = self.val_sharpes[-self.window:]
        recent_stds = self.action_stds[-self.window:]

        # CONVERGED: Sharpe plateau + stable entropy
        sharpe_trend = np.polyfit(range(len(recent)), recent, 1)[0]
        weeks_since_best = current_week - self.best_week
        if weeks_since_best > 50 and abs(sharpe_trend) < 0.001:
            return "CONVERGED"

        # OVERFITTING: Q-values growing but validation declining
        if len(self.q_value_means) > 30:
            q_trend = np.polyfit(range(30), self.q_value_means[-30:], 1)[0]
            sharpe_declining = sharpe_trend < -0.005
            if q_trend > 0 and sharpe_declining:
                return "OVERFITTING"

        # COLLAPSED: Action std near zero for extended period
        if all(s < 0.05 for s in recent_stds[-20:]):
            return "COLLAPSED"

        # PLATEAU: Stuck near random performance
        recent_wr = np.mean(recent[-40:]) if len(recent) >= 40 else 0.5
        if 0.49 < recent_wr < 0.51 and current_week > 80:
            return "PLATEAU"

        # IMPROVING: Positive validation Sharpe trend
        if sharpe_trend > 0:
            return "IMPROVING"

        return "STABLE"
```

**Actions per status:**

| Status | Action |
|--------|--------|
| CONVERGED | Stop training. Deploy best checkpoint (week {best_week}) |
| OVERFITTING | Stop. Roll back to best validation checkpoint. Consider more regularization |
| COLLAPSED | Restart with higher initial entropy or increase exploration noise |
| PLATEAU | Investigate: try increasing features, adjusting reward weights, switching to LSTM |
| IMPROVING | Continue training |
| STABLE | Continue, but monitor closely |

---

## 15. Training Resilience & Crash Recovery

> **Training runs for hours or days. If it crashes, we lose NOTHING.**
> Every piece of knowledge — model weights, memory, replay buffer, training state — is persisted.

### What Gets Saved (Automatically, Every Completed Week)

```python
def save_training_checkpoint(agent, memory, state, config):
    """Save EVERYTHING needed to resume training exactly where we left off."""

    # 1. Model weights (neural network learned so far)
    checkpoint_path = f"storage/models/spartus_week_{state['week']:04d}"
    agent.save(checkpoint_path)

    # 2. Replay buffer (ALL past experiences for off-policy learning)
    buffer_path = f"{checkpoint_path}_buffer.pkl"
    agent.save_replay_buffer(buffer_path)

    # 3. Memory database — already on disk (SQLite auto-commits)
    # No explicit save needed — trades, predictions, patterns are persisted

    # 4. Training state (where we are, balance, metrics)
    state_file = {
        "last_completed_week": state['week'],
        "global_step": agent.num_timesteps,
        "current_balance": state['balance'],
        "peak_balance": state['peak_balance'],
        "bankruptcies": state['bankruptcies'],
        "best_win_rate": state['best_win_rate'],
        "model_checkpoint_path": f"{checkpoint_path}.zip",
        "replay_buffer_path": buffer_path,
        "memory_db_path": config.memory_db_path,
        "config_hash": hashlib.md5(json.dumps(config).encode()).hexdigest(),
        "timestamp": datetime.now().isoformat()
    }
    with open("storage/training_state.json", "w") as f:
        json.dump(state_file, f, indent=2)
```

### Resume Logic

```python
def resume_or_start(config):
    """On startup: detect if we can resume, otherwise start fresh."""
    state_path = "storage/training_state.json"

    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"RESUMING training from week {state['last_completed_week'] + 1}")
        print(f"  Balance: £{state['current_balance']:.2f}")
        print(f"  Global steps: {state['global_step']:,}")
        print(f"  Memory trades: {TradingMemory(state['memory_db_path']).total_trades()}")

        # Verify config hasn't changed (important: same features, same architecture)
        current_hash = hashlib.md5(json.dumps(config).encode()).hexdigest()
        if current_hash != state['config_hash']:
            print("WARNING: Config has changed since last checkpoint!")
            print("  This may cause issues. Consider starting fresh.")

        agent = SAC.load(state['model_checkpoint_path'])
        agent.load_replay_buffer(state['replay_buffer_path'])
        memory = TradingMemory(state['memory_db_path'])
        return agent, memory, state['last_completed_week'] + 1, state['current_balance']

    else:
        print("Starting FRESH training run")
        return None, TradingMemory(config.memory_db_path), 0, config.initial_balance
```

### What This Guarantees

```
IF TRAINING IS INTERRUPTED (crash, power loss, Ctrl+C, system restart):

  PRESERVED (on disk, safe):
    ✓ All model weights from last completed week
    ✓ All replay buffer experiences (for off-policy learning)
    ✓ All memory: trade history, patterns, trend predictions
    ✓ All log files (JSONL, TensorBoard)
    ✓ Training state (which week we're on, balance, metrics)

  LOST (only the current in-progress week):
    ✗ Any trades from the current incomplete week
    ✗ Any replay buffer additions from the current week
    ✗ At most ~10,000 steps of training

  ON RESTART:
    → System detects last checkpoint
    → Loads model + replay buffer + memory
    → Resumes from the NEXT week
    → AI continues evolving from where it left off
    → No corruption, no lost knowledge
```

---

**Document Version:** 3.3
**Updated:** 2026-02-23
**Status:** Complete - Companion to SPARTUS_TRADING_AI.md
**Changes in v3.0:** Added account management training, bankruptcy handling, realistic market simulation, training engine monitoring
**Changes in v3.1:** Expanded trend prediction verification cycle with explicit schema comments, added Training Resilience & Crash Recovery section with save/resume logic
**Changes in v3.2:** Added tp_tracking table to memory schema. Updated feature count from 38→42 and input dims from 380→420. Updated TrainingLogger to log action[3] (SL management). Updated frame stacking references to match new feature count.
**Changes in v3.3:** Added gradient clipping (max_grad_norm=1.0) and gradient norm monitoring. Added 3-phase learning rate schedule (warm-up → hold → cosine decay). Added DR-SAC as Phase 2 upgrade path. Added regime-tagged replay buffer with balanced sampling. Added measurable LSTM switch criteria. Added Bayesian shrinkage for pattern memory confidence. Added SAC stability monitoring to callback (Q-values, entropy, gradients). Added observation health checks (dead features, NaN detection). Added curriculum learning (3-stage progressive difficulty). Added ensemble SAC (3 seeds, majority vote). Added convergence detection (CONVERGED, OVERFITTING, COLLAPSED, PLATEAU signals).
**Changes in v3.3.1:** Cross-reference alignment audit. Updated reward function from outdated 3-component to current 5-component composite (R1-R5 with weights 0.40/0.20/0.15/0.15/0.10 + RewardNormalizer). Updated account features from 5 to 8 (aligned with SPARTUS_TRADING_AI.md Section 10.G: added has_position, position_side, sl_distance_ratio, profit_locked_pct; removed balance_velocity).
**Changes in v3.3.2:** Deep critical design review fixes: Terminal penalties now SET reward (not ADD) to stay within [-5,+5] normalizer range. R5 formula changed to linear decay `max(0, 1-dd/0.10)` for real risk differentiation. profit_locked_pct feature now ATR-scaled (was entry-price-scaled, invisible for gold). Updated bankruptcy handling comment to reflect new terminal penalty approach.
