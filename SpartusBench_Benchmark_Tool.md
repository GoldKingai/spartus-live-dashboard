# SpartusBench -- Benchmark & Model Progression System

**Version:** 1.0.0
**Status:** Specification (pre-build)
**Parent System:** Spartus Trading AI v3.3.2
**Date:** 2026-03-02

---

## Table of Contents

1. [Purpose & Goals](#1-purpose--goals)
2. [Benchmark Suite Definition](#2-benchmark-suite-definition)
3. [Metrics List](#3-metrics-list)
4. [Detectors](#4-detectors)
5. [Scoring & Champion Protocol](#5-scoring--champion-protocol)
6. [Locked Test Protocol](#6-locked-test-protocol)
7. [Persistence Layer](#7-persistence-layer)
8. [Model Discovery & Loading](#8-model-discovery--loading)
9. [Gating Pass-Rate Diagnostics](#9-gating-pass-rate-diagnostics)
10. [Reproducibility Guarantees](#10-reproducibility-guarantees)
11. [CLI Commands](#11-cli-commands)
12. [UI Layout & Wireframes](#12-ui-layout--wireframes)
13. [Integration Points](#13-integration-points)
14. [Shared Evaluation Module (Refactoring)](#14-shared-evaluation-module-refactoring)
15. [Implementation Roadmap](#15-implementation-roadmap)

---

## 1. Purpose & Goals

### Why SpartusBench Exists

Spartus already has functional evaluation: three standalone eval scripts (`eval_validation.py`, `eval_stress_matrix.py`, `eval_checkpoints.py`), training-loop validation, convergence tracking, exporter pre-flight checks, and live dashboard analytics. But the process is **manual and scattered**. There is no persistent record, no automated champion selection, and no regression detection between models.

SpartusBench unifies all evaluation into a single tool that makes model progress **undeniable, comparable, and repeatable**.

### Success Criteria

1. **One command** replaces 3+ manual script invocations to produce a complete model assessment.
2. **Persistent history** -- every benchmark run is recorded in an append-only database, forever.
3. **Automatic regression detection** -- aggression drift, conviction collapse, and stress fragility are flagged before deployment.
4. **Composite scoring** -- SpartusScore reduces 15+ metrics to one comparable number with transparent component breakdown.
5. **Champion protocol** -- clear, deterministic promotion/demotion rules with no subjective calls.
6. **Locked test discipline** -- test set access is gated, logged permanently, and auditable.
7. **Gating diagnostics** -- shows whether a model will actually trade in production vs deadlock behind gates.
8. **Reproducibility** -- every run carries version hashes so any result can be re-derived from the same data + config + seed.
9. **Benchmark travels with the model** -- every deployed package carries its SpartusScore and benchmark metadata.

### What SpartusBench Does NOT Do

- It does not train models. Training is `scripts/train.py`.
- It does not execute live trades. Live execution is `live_dashboard/`.
- It does not modify model weights. It is read-only against model files.
- It does not delete or overwrite benchmark history. The database is append-only.

---

## 2. Benchmark Suite Definition

### 2.1 Tier Structure

| Tier | Name | Runs On | When | Required? |
|------|------|---------|------|-----------|
| **T1** | Validation Eval | Val weeks (deterministic) | Every benchmark | Yes |
| **T2** | Stress Matrix | Val weeks (7 cost scenarios) | Every benchmark | Yes |
| **T3** | Regime Segmentation | Val trades (sliced by ATR/session/day/direction/reason) | Every benchmark | Yes |
| **T4** | Churn Diagnostic | Val trades (frequency + edge + cost analysis) | Every benchmark | Yes |
| **T5** | Reward Ablation | Val steps (R1-R5 component decomposition) | Every benchmark | Yes |
| **T6** | Gating Diagnostics | Val steps (gate pass-rates per bar) | Every benchmark | Yes |
| **T7** | Locked Test | Test weeks (single-pass discipline) | On explicit request | No |

### 2.2 Tier 1 -- Validation Eval

Run the model deterministically across **all validation weeks** (no random sampling -- use the full val set for stable metrics).

**Environment configuration for all benchmark runs:**
```python
deterministic       = True      # model.predict(obs, deterministic=True)
is_validation       = True      # no writes to shared memory/SQLite
observation_noise   = 0.0       # no noise injection
spread_jitter       = 0.0       # no domain randomization
slippage_jitter     = 0.0
commission_jitter   = 0.0
start_offset_max    = 0         # always start at bar 0
seed                = 42        # fixed seed for reproducibility
initial_balance     = 100.0     # fixed starting balance
```

**Metrics computed:** See Section 3 for full list.

### 2.3 Tier 2 -- Stress Matrix

Run the model across validation weeks under 7 cost scenarios. Each scenario multiplies base config values:

| Scenario | Spread Multiplier | Slip Mean Mult | Slip Std Mult | Priority Weight |
|----------|-------------------|----------------|---------------|-----------------|
| `base` | 1.0x | 1.0x | 1.0x | -- (reference) |
| `2x_spread` | 2.0x | 1.0x | 1.0x | **0.35** (highest) |
| `combined_2x2x` | 2.0x | 2.0x | 2.0x | **0.30** |
| `3x_spread` | 3.0x | 1.0x | 1.0x | **0.20** |
| `2x_slip_mean` | 1.0x | 2.0x | 1.0x | 0.05 |
| `2x_slip_std` | 1.0x | 1.0x | 2.0x | 0.05 |
| `5x_spread` | 5.0x | 1.0x | 1.0x | 0.05 |

**Application method:** Deep-copy eval config, multiply the relevant spread/slippage attributes by the scenario multiplier. Scenario multipliers apply to ALL session spreads (london, ny, asia, off_hours).

**Base spread values (from TrainingConfig):**
- `spread_london_pips`: 1.5
- `spread_ny_pips`: 2.0
- `spread_asia_pips`: 3.0
- `spread_off_hours_pips`: 5.0
- `slippage_mean_pips`: 0.5
- `slippage_std_pips`: 0.3

**Per-scenario output:** Same metrics as T1, plus PF retention ratio vs base.

### 2.4 Tier 3 -- Regime Segmentation

Slice all T1 validation trades across 5 dimensions:

**A. ATR Quartiles (volatility regime):**
- Compute ATR(14) at each trade's entry bar from the raw feature `atr_14_raw`.
- Compute 25th, 50th, 75th percentiles across ALL validation trades.
- Bucket: Q1 (low vol, ATR < p25), Q2 (p25-p50), Q3 (p50-p75), Q4 (high vol, ATR >= p75).

**B. Trading Session:**
- London (07:00-12:00 UTC), NY Overlap (12:00-16:00 UTC), NY PM (16:00-20:00 UTC), Asia (00:00-07:00 UTC), Off-hours (20:00-24:00 UTC).
- Based on trade entry time (UTC hour).

**C. Day of Week:**
- Monday through Friday (trade entry calendar day).

**D. Trade Direction:**
- LONG vs SHORT.

**E. Close Reason:**
- TP_HIT, SL_HIT, AGENT_CLOSE, EMERGENCY_STOP, TIMEOUT.

**Per-slice metrics:** Trades, Win%, Net P/L, PF, Avg P/L, Avg Hold Bars.

### 2.5 Tier 4 -- Churn Diagnostic

Analyzes whether the model has a real edge or is just churning through costs:

**Formulas:**
```
trading_days           = len(val_weeks) * 5
trades_per_day         = total_trades / trading_days
avg_spread_pips        = mean(spread_london, spread_ny, spread_asia, spread_off_hours)
avg_slippage_pips      = slippage_mean_pips * 2  (entry + exit)
avg_cost_pips          = avg_spread_pips + avg_slippage_pips
avg_cost_points        = avg_cost_pips * pip_price
avg_lot                = mean(trade.lots for all trades)
vpp                    = tick_value / tick_size
est_cost_per_trade     = avg_cost_points * avg_lot * vpp
total_est_cost         = est_cost_per_trade * total_trades
net_pnl                = sum(trade.pnl for all trades)
est_gross_pnl          = net_pnl + total_est_cost
net_edge_per_trade     = net_pnl / total_trades
gross_edge_per_trade   = est_gross_pnl / total_trades
cost_to_edge_ratio     = total_est_cost / abs(net_pnl) if net_pnl != 0
```

**Breakdown by close reason:** Count, Win%, Avg P/L, Total P/L per reason.
**Breakdown by side:** Count, Win%, Avg P/L, Total P/L per LONG/SHORT.

### 2.6 Tier 5 -- Reward Ablation

Decomposes the reward signal by its 5 components during validation rollout:

```
R1 (Position P/L):     weight = 0.40
R2 (Trade Quality):    weight = 0.20
R3 (Drawdown Penalty): weight = 0.15
R4 (Diff Sharpe):      weight = 0.15
R5 (Risk Bonus):       weight = 0.10

Per-step: capture raw r1, r2, r3, r4, r5 from env info dict.
Weighted sums: w1 = 0.40 * sum(r1), w2 = 0.20 * sum(r2), etc.
Total weighted = w1 + w2 + w3 + w4 + w5

R5 distribution: count(r5 > 0), count(r5 < 0), count(r5 == 0)
R5 contribution: w5 / total_weighted * 100  (percentage of total reward from R5)
```

**Red flag:** R5 contribution > 40% of total reward = reward hacking risk.

### 2.7 Tier 6 -- Gating Diagnostics

See Section 9 for full specification.

### 2.8 Tier 7 -- Locked Test

See Section 6 for full specification.

---

## 3. Metrics List

### 3.1 Core Performance Metrics (Tier 1)

| Metric | Formula | Unit | Source |
|--------|---------|------|--------|
| **Sharpe (annualized)** | `(mean(weekly_returns) / std(weekly_returns, ddof=1)) * sqrt(52)` | ratio | eval_validation.py:468 |
| **Sortino (annualized)** | `(mean(weekly_returns) / std(negative_returns, ddof=1)) * sqrt(52)` | ratio | eval_validation.py:474 |
| **Profit Factor** | `sum(winning_pnls) / abs(sum(losing_pnls))` | ratio | eval_validation.py:460 |
| **Win Rate** | `count(pnl > 0) / total_trades * 100` | % | eval_validation.py:495 |
| **Max Drawdown** | `max((peak_equity - equity) / peak_equity) * 100` | % | eval_validation.py:478 |
| **Net P/L** | `sum(all_trade_pnls)` | $ | direct |
| **Avg P/L per Trade** | `net_pnl / total_trades` | $ | eval_validation.py:505 |
| **Total Trades** | `count(closed_trades)` | int | direct |
| **Time in Market** | `sum(has_position_steps) / total_steps * 100` | % | eval_validation.py:392 |
| **Trades per Day** | `total_trades / (val_weeks * 5)` | float | eval_stress_matrix.py:358 |
| **Avg Hold (bars)** | `mean(trade.hold_bars)` | bars | eval_validation.py:503 |
| **Median Hold (bars)** | `median(trade.hold_bars)` | bars | eval_validation.py:504 |

### 3.2 New Metrics (to add)

| Metric | Formula | Unit | Purpose |
|--------|---------|------|---------|
| **Calmar Ratio** | `annualized_return / max_dd_pct` where `annualized_return = mean(weekly_returns) * 52` | ratio | Return vs worst drawdown |
| **Recovery Factor** | `net_pnl / max_drawdown_dollars` | ratio | Ability to recover from DD |
| **Tail Ratio** | `abs(percentile(pnls, 95)) / abs(percentile(pnls, 5))` | ratio | P/L distribution asymmetry |
| **Expectancy** | `win_rate * avg_win - (1 - win_rate) * abs(avg_loss)` | $ | Expected $ per trade |
| **Max Consecutive Losses** | `max_run(trade.pnl < 0)` | int | Worst losing streak |
| **Max Consecutive Wins** | `max_run(trade.pnl > 0)` | int | Best winning streak |
| **Gross Profit** | `sum(pnl for pnl > 0)` | $ | Total winning amount |
| **Gross Loss** | `abs(sum(pnl for pnl <= 0))` | $ | Total losing amount |
| **Avg Win** | `mean(pnl for pnl > 0)` | $ | Average winning trade |
| **Avg Loss** | `abs(mean(pnl for pnl <= 0))` | $ | Average losing trade |
| **Win/Loss Ratio** | `avg_win / avg_loss` | ratio | Reward-to-risk per trade |
| **Flat Bar %** | `sum(not has_position) / total_steps * 100` | % | How often model does nothing |
| **Entry Timing Score** | `count(max_favorable > 0) / total_trades * 100` | % | Price moved favorably after entry |
| **SL Quality Score** | For SL-hit trades: `count(max_favorable > 0) / sl_hit_trades * 100` | % | Was SL well-placed? |
| **Conviction Distribution** | `{mean, std, p10, p50, p90}` of trade conviction values | stats | Is conviction well-calibrated? |
| **Long/Short Split** | `{long_count, short_count, long_pnl, short_pnl, long_pf, short_pf}` | mixed | Directional balance |

### 3.3 Stress-Specific Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| **PF Retention** | `scenario_pf / base_pf` | ratio |
| **Sharpe Retention** | `scenario_sharpe / base_sharpe` | ratio |
| **Trade Count Change** | `scenario_trades / base_trades` | ratio |
| **Worst-Case Retention** | `min(pf_retention across all scenarios)` | ratio |
| **Weighted Stress Score** | See Section 5.2 | 0-100 |

### 3.4 Metric Thresholds (Advisory, Not Hard-Fail)

| Metric | Good | Watch | Concern |
|--------|------|-------|---------|
| Sharpe | > 2.0 | 1.0 - 2.0 | < 1.0 |
| PF | > 1.5 | 1.0 - 1.5 | < 1.0 |
| Win Rate | > 50% | 40 - 50% | < 40% |
| Max DD | < 10% | 10 - 20% | > 20% |
| Sortino | > 3.0 | 1.5 - 3.0 | < 1.5 |
| Trades/Day | 1 - 5 | 0.5 - 8 | < 0.5 or > 8 |
| Avg Hold | 6 - 30 bars | 3 - 50 bars | < 3 or > 50 |
| Expectancy | > 0.5 | 0 - 0.5 | < 0 |
| Max Consec Loss | < 6 | 6 - 10 | > 10 |
| PF Retention (2x spread) | > 0.80 | 0.65 - 0.80 | < 0.65 |
| PF Retention (combined) | > 0.70 | 0.50 - 0.70 | < 0.50 |

---

## 4. Detectors

### 4.1 Aggression Drift Detector

**What it detects:** The model is becoming reckless -- trading more frequently, with lower quality, and taking more risk. Pattern: TIM increasing + PF decreasing + Max DD increasing + churn increasing.

**Trigger logic (requires comparison vs champion or prior benchmark):**
```python
def detect_aggression_drift(current, reference):
    tim_delta   = current.tim_pct - reference.tim_pct         # TIM% change
    pf_delta    = current.val_pf - reference.val_pf            # PF change
    dd_delta    = current.max_dd_pct - reference.max_dd_pct    # MaxDD% change
    tpd_delta   = current.trades_per_day - reference.trades_per_day  # churn change

    # Primary trigger: 3 of 4 moving in the bad direction simultaneously
    bad_signals = [
        tim_delta > 5.0,     # TIM increased by 5+ percentage points
        pf_delta < -0.2,     # PF dropped by 0.2+
        dd_delta > 2.0,      # MaxDD increased by 2+ percentage points
        tpd_delta > 1.5,     # Trades/day increased by 1.5+
    ]
    is_drifting = sum(bad_signals) >= 3

    # Severity (0-4): count of severe sub-thresholds
    severity = sum([
        tim_delta > 10.0,          # Severe TIM increase
        pf_delta < -0.5,           # Severe PF drop
        dd_delta > 5.0,            # Severe DD increase
        tpd_delta > 3.0,           # Severe churn increase
    ])

    return {
        "detected": is_drifting,
        "severity": severity,
        "tim_delta": tim_delta,
        "pf_delta": pf_delta,
        "dd_delta": dd_delta,
        "tpd_delta": tpd_delta,
        "bad_signals": sum(bad_signals),
    }
```

**Output:** `AGGRESSION_DRIFT` flag with severity 0-4.

### 4.2 Conviction Collapse / Gating Deadlock Detector

**What it detects:** The model is alive (producing varied actions) but nothing gets through the gating pipeline. The model won't trade in production. This is what we saw in live demo -- policy was active but direction/conviction/spread gates conspired to block everything.

**Trigger logic (uses gating diagnostics from Tier 6):**
```python
def detect_conviction_collapse(result):
    trades        = result.val_trades
    action_std    = result.mean_action_std       # mean std across 4 action dims
    flat_pct      = result.flat_bar_pct          # % of bars with no position
    dir_pass_rate = result.gate_direction_pass    # % passing direction threshold
    conv_pass_rate = result.gate_conviction_pass  # % passing conviction floor

    # Primary trigger: policy is active but nothing trades
    is_deadlocked = (
        action_std > 0.10 and        # policy IS producing varied actions
        trades < 5 and               # but almost no trades executed
        flat_pct > 95.0              # model was flat 95%+ of the time
    )

    # Gate bottleneck analysis
    gate_analysis = {
        "direction_pass_rate": dir_pass_rate,
        "conviction_pass_rate": conv_pass_rate,
        "spread_pass_rate": result.gate_spread_pass,
        "lot_sizing_pass_rate": result.gate_lot_pass,
        "bottleneck": _identify_bottleneck(result),  # which gate blocks the most
    }

    return {
        "detected": is_deadlocked,
        "gate_analysis": gate_analysis,
        "action_std": action_std,
        "flat_pct": flat_pct,
        "total_trades": trades,
    }

def _identify_bottleneck(result):
    """Return the gate with the lowest pass rate."""
    gates = {
        "direction": result.gate_direction_pass,
        "conviction": result.gate_conviction_pass,
        "spread": result.gate_spread_pass,
        "lot_sizing": result.gate_lot_pass,
    }
    return min(gates, key=gates.get)
```

**Output:** `CONVICTION_COLLAPSE` flag with gate bottleneck identification.

### 4.3 Stress Fragility Detector

**What it detects:** The model performs well under base conditions but collapses when costs increase. This reveals a model whose edge is too thin to survive realistic execution costs.

**Trigger logic:**
```python
def detect_stress_fragility(result):
    base_pf = result.stress_results["base"]["pf"]
    if base_pf <= 0:
        return {"detected": True, "reason": "base_pf_zero_or_negative"}

    retentions = {}
    for scenario, metrics in result.stress_results.items():
        if scenario == "base":
            continue
        retentions[scenario] = metrics["pf"] / base_pf

    # Critical scenarios (spread-dominated, highest real-world impact)
    r_2x_spread  = retentions.get("2x_spread", 0)
    r_combined   = retentions.get("combined_2x2x", 0)
    r_3x_spread  = retentions.get("3x_spread", 0)

    worst_retention = min(retentions.values()) if retentions else 0
    worst_scenario  = min(retentions, key=retentions.get) if retentions else "none"

    # Fragility detected if any critical scenario collapses
    is_fragile = (
        r_2x_spread < 0.65 or
        r_combined < 0.50 or
        r_3x_spread < 0.40
    )

    return {
        "detected": is_fragile,
        "retentions": retentions,
        "worst_retention": worst_retention,
        "worst_scenario": worst_scenario,
        "r_2x_spread": r_2x_spread,
        "r_combined": r_combined,
        "r_3x_spread": r_3x_spread,
    }
```

**Output:** `STRESS_FRAGILITY` flag with per-scenario retention breakdown.

### 4.4 Overfitting Detector

**What it detects:** Performance looks great on validation but degrades on unseen data, or training metrics improving while validation metrics stagnate/decline.

**Trigger logic (requires locked test result OR cross-checkpoint comparison):**
```python
def detect_overfitting(result, prior_results=None):
    signals = []

    # Signal 1: Val Sharpe declining across recent checkpoints
    if prior_results and len(prior_results) >= 3:
        recent_sharpes = [r.val_sharpe for r in prior_results[-3:]] + [result.val_sharpe]
        if all(recent_sharpes[i] < recent_sharpes[i-1] for i in range(1, len(recent_sharpes))):
            signals.append("val_sharpe_monotonic_decline")

    # Signal 2: Locked test vs validation gap (if test available)
    if result.test_pf is not None and result.val_pf is not None:
        pf_gap = result.val_pf - result.test_pf
        if pf_gap > 0.5:
            signals.append(f"val_test_pf_gap={pf_gap:.2f}")

    # Signal 3: MaxDD on test much worse than validation
    if result.test_max_dd_pct is not None and result.val_max_dd_pct is not None:
        dd_gap = result.test_max_dd_pct - result.val_max_dd_pct
        if dd_gap > 5.0:
            signals.append(f"val_test_dd_gap={dd_gap:.1f}%")

    return {
        "detected": len(signals) >= 2,
        "signals": signals,
    }
```

### 4.5 Reward Hacking Detector

**What it detects:** The model is exploiting the R5 risk bonus (being-in-position reward) rather than learning real trading edge. Pattern: R5 dominates total reward while R1 (actual P/L) is weak.

**Trigger logic (from Tier 5 reward ablation):**
```python
def detect_reward_hacking(result):
    r5_contribution = result.r5_pct_of_total_reward
    r1_contribution = result.r1_pct_of_total_reward

    is_hacking = (
        r5_contribution > 40.0 and      # R5 is 40%+ of total reward
        r1_contribution < 20.0           # R1 (actual P/L) is below 20%
    )

    return {
        "detected": is_hacking,
        "r5_pct": r5_contribution,
        "r1_pct": r1_contribution,
        "r5_pos_count": result.r5_positive_steps,
        "r5_neg_count": result.r5_negative_steps,
    }
```

---

## 5. Scoring & Champion Protocol

### 5.1 SpartusScore (0-100)

A composite score that compresses multiple metrics into a single comparable number. Every component has a transparent formula so the score can be audited.

```
SpartusScore = sum(component_score * weight for each component)
```

**Components and weights:**

| Component | Weight | Formula | Range |
|-----------|--------|---------|-------|
| `val_sharpe` | 0.25 | `min(sharpe / 5.0, 1.0) * 100` | 0-100 |
| `val_pf` | 0.20 | `min((pf - 1.0) / 2.0, 1.0) * 100` (0 if pf < 1.0) | 0-100 |
| `stress_robustness` | 0.25 | See Section 5.2 | 0-100 |
| `max_dd_penalty` | 0.15 | `max(0, 100 - max_dd_pct * 5)` | 0-100 |
| `trade_quality` | 0.15 | `(win_rate_score * 0.5) + (lesson_score * 0.5)` | 0-100 |

**Sub-formulas:**
```
win_rate_score   = min(win_rate / 60.0, 1.0) * 100
lesson_score     = (good_trade_pct + small_win_pct) * 100  (proportion of trades with positive lessons)
```

**Note on TIM%:** TIM is deliberately **not** a SpartusScore component. TIM varies legitimately across market regimes and model strategies. It is tracked, displayed, and used in the aggression drift detector (Section 4.1), but it does not directly affect the score. A model with 20% TIM and a model with 40% TIM should be compared on PF, Sharpe, DD, and stress robustness -- not penalized for trading frequency alone.

### 5.2 Stress Robustness Score (0-100)

Weighted by scenario importance (spread-heavy scenarios weighted highest because spread is the dominant execution cost for XAUUSD). Includes a worst-case penalty to prevent collapse from hiding behind averaging.

```python
def compute_stress_score(retentions: dict) -> float:
    """
    retentions = {scenario: pf_retention_ratio}
    where pf_retention = scenario_pf / base_pf
    """
    WEIGHTS = {
        "2x_spread":     0.35,
        "combined_2x2x": 0.30,
        "3x_spread":     0.20,
        "2x_slip_mean":  0.05,
        "2x_slip_std":   0.05,
        "5x_spread":     0.05,
    }

    # Weighted average retention
    weighted_sum = 0.0
    for scenario, weight in WEIGHTS.items():
        r = retentions.get(scenario, 0.0)
        weighted_sum += min(r, 1.0) * weight  # cap at 1.0 (can't score > base)

    weighted_avg = weighted_sum  # already 0.0-1.0 range

    # Worst-case penalty: if ANY scenario collapses, the score is dragged down
    worst = min(retentions.values()) if retentions else 0.0
    worst_penalty = max(0.0, 1.0 - worst) * 0.3  # 30% weight on worst case

    score = max(0.0, weighted_avg - worst_penalty) * 100
    return min(score, 100.0)
```

### 5.3 Hard-Fail Rules (Automatic Disqualification)

A model that triggers any hard-fail rule is marked `DISQUALIFIED` in the leaderboard and cannot become champion. Hard-fails are chosen conservatively -- only conditions that indicate a fundamentally broken model.

| Rule | Trigger Condition | Rationale |
|------|-------------------|-----------|
| `stress_2x_spread_collapse` | `PF(2x_spread) / PF(base) < 0.65` | Edge too thin to survive normal spread variation |
| `stress_combined_collapse` | `PF(combined_2x2x) / PF(base) < 0.50` | Cannot handle realistic worst-case costs |
| `max_dd_blowup` | `max_dd_pct > 25.0%` | Unacceptable risk of account destruction |
| `negative_pf` | `val_pf < 1.0` | Model loses money on validation set |
| `zero_trades` | `val_trades < 10` | Model doesn't trade (or trivially few trades) |
| `reward_hacking` | `R5 contribution > 50% AND R1 contribution < 15%` | Model exploiting position bonus, not learning edge |

**Removed from hard-fail (by design):**
- `side_bias` -- Directional skew is not inherently bad. A model that prefers LONG in an uptrend or SHORT in a downtrend is correct behavior. Instead, **side balance is tracked as a warning** in the report with the following logic:
  - If opposite-side PF < 1.0 AND opposite-side trades >= 10: flag `SIDE_WEAKNESS_WARNING` in report.
  - If opposite-side trades < 3: flag `SIDE_AVOIDANCE_WARNING` (model may not function in regime reversal).
  - Neither is a disqualification.

### 5.4 Champion Selection Protocol

```
1. Run SpartusBench on candidate model (full suite: T1-T6).
2. Check hard-fail rules:
   - Any fail -> DISQUALIFIED. Cannot promote. Record result anyway.
3. Check detectors (aggression drift, conviction collapse, stress fragility, reward hacking):
   - Any detected -> FLAG in report. Not an automatic disqualification, but
     requires human review before promotion.
4. Compute SpartusScore.
5. Compare vs current champion (most recent model with is_champion=True):
   a. score > champion_score + 2.0 -> PROMOTE (clear improvement)
   b. score within +/- 2.0         -> DRAW (keep current champion; it has more live hours)
   c. score < champion_score - 2.0 -> REGRESSION (keep current, investigate why)
6. If PROMOTE:
   a. Set is_champion = True for this run in DB.
   b. Set dethroned_at = now() for prior champion in leaderboard table.
   c. Generate delta-vs-champion report.
   d. Flag for human review (no auto-deploy -- human confirms before live).
7. If DRAW or REGRESSION:
   a. Record result in DB (append-only, never discarded).
   b. Generate delta-vs-champion report for analysis.
```

### 5.5 Delta-vs-Champion Report Format

Auto-generated for every benchmark run:

```markdown
## Delta vs Champion: {champion_id} -> {candidate_id}

| Metric             | Champion     | Candidate    | Delta    | Verdict    |
|--------------------|-------------|-------------|----------|------------|
| SpartusScore       | 74.2        | 76.8        | +2.6     | PROMOTE    |
| Sharpe             | 4.09        | 3.82        | -0.27    | REGRESS    |
| PF                 | 2.24        | 2.41        | +0.17    | IMPROVE    |
| Win Rate           | 52.0%       | 54.3%       | +2.3%    | IMPROVE    |
| Max DD             | 20.8%       | 18.2%       | -2.6%    | IMPROVE    |
| Stress Robustness  | 82.1        | 85.3        | +3.2     | IMPROVE    |
| TIM%               | 32%         | 41%         | +9%      | WATCH      |
| Trades/Day         | 2.1         | 3.4         | +1.3     | WATCH      |
| PF Retention (2x)  | 0.90        | 0.88        | -0.02    | NEUTRAL    |

Detectors:
- Aggression Drift: NOT DETECTED
- Conviction Collapse: NOT DETECTED
- Stress Fragility: NOT DETECTED
- Reward Hacking: NOT DETECTED

Hard Fails: NONE

Verdict: PROMOTE {candidate_id} as new champion (+2.6 SpartusScore)
Flags: TIM% increasing -- monitor for aggression drift over next checkpoints.
```

---

## 6. Locked Test Protocol

### 6.1 Purpose

The test set (final ~15% of weeks, after purge gap) is reserved for final go-live assessment. Unlike validation, test weeks are never seen during training's validation loop. Running locked test gives the closest approximation of true out-of-sample performance.

### 6.2 Confirmation Gating

Locked test requires an explicit flag:
```bash
python -m spartusbench run W170 --suite locked_test --confirm-test
```

Without `--confirm-test`, the command fails with:
```
ERROR: Locked test requires --confirm-test flag.
Test set evaluation is permanently recorded in the benchmark database.
Run with --confirm-test to proceed.
```

### 6.3 Multiple Runs Allowed

Unlike a one-shot exam, locked test CAN be run multiple times for the same model. This is necessary to:
- Confirm that refactors didn't change model behavior (same weights, different code path).
- Verify reproducibility (same seed should give same results).
- Compare test performance across retraining generations.

However, every run is **permanently recorded** in the append-only database with full audit metadata.

### 6.4 Audit Trail (Non-Negotiable)

Every locked test run records:

| Field | Description |
|-------|-------------|
| `run_id` | UUID |
| `timestamp` | ISO8601 UTC |
| `model_id` | Checkpoint or package identifier (e.g., "W170") |
| `model_path` | Absolute path to model file |
| `model_file_hash` | SHA256 of the model ZIP file |
| `operator` | System username (`os.getlogin()`) |
| `data_manifest_hash` | Hash of test week file paths + mtimes (Section 10.3) |
| `split_hash` | Hash of train/val/test week indices + purge gap (Section 10.4) |
| `feature_hash` | Hash of feature list + ordering + obs_dim (Section 10.5) |
| `config_hash` | SHA256 of serialized eval config (Section 10.6) |
| `seed` | RNG seed used |
| `test_weeks_used` | JSON array of test week indices |
| `result_hash` | SHA256 of results JSON (for tamper detection) |

### 6.5 Locked Test Metrics

Same as Tier 1 (full validation metrics) but computed on test weeks. Stored in `benchmark_runs` table under `test_*` columns.

---

## 7. Persistence Layer

### 7.1 Database: `storage/benchmark/spartusbench.db` (SQLite, append-only)

**WAL mode** for concurrent read access. Write operations are single-threaded (benchmark runner).

**Append-only policy:** Rows are NEVER updated or deleted. The only exception is `leaderboard.dethroned_at`, which is set when a new champion replaces the old one.

### 7.2 Schema

```sql
-- =====================================================
-- Table: benchmark_runs
-- One row per benchmark invocation.
-- =====================================================
CREATE TABLE benchmark_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT UNIQUE NOT NULL,       -- UUID v4
    timestamp           TEXT NOT NULL,              -- ISO8601 UTC
    model_id            TEXT NOT NULL,              -- "W0170", "spartus_best", etc.
    model_path          TEXT NOT NULL,
    model_file_hash     TEXT,                       -- SHA256 of model ZIP
    suite               TEXT NOT NULL,              -- "full", "validation_only", "locked_test"
    seed                INTEGER NOT NULL DEFAULT 42,
    operator            TEXT,                       -- os.getlogin()

    -- Reproducibility hashes (Section 10)
    data_manifest_hash  TEXT NOT NULL,
    split_hash          TEXT NOT NULL,
    feature_hash        TEXT NOT NULL,
    config_hash         TEXT NOT NULL,

    -- Tier 1: Validation metrics
    val_trades          INTEGER,
    val_win_pct         REAL,
    val_pf              REAL,
    val_sharpe          REAL,
    val_sortino         REAL,
    val_max_dd_pct      REAL,
    val_net_pnl         REAL,
    val_tim_pct         REAL,
    val_trades_day      REAL,
    val_avg_hold        REAL,
    val_median_hold     REAL,
    val_calmar          REAL,
    val_recovery_factor REAL,
    val_tail_ratio      REAL,
    val_expectancy      REAL,
    val_max_consec_loss INTEGER,
    val_max_consec_win  INTEGER,
    val_gross_profit    REAL,
    val_gross_loss      REAL,
    val_avg_win         REAL,
    val_avg_loss        REAL,
    val_win_loss_ratio  REAL,
    val_flat_bar_pct    REAL,
    val_entry_timing    REAL,
    val_sl_quality      REAL,
    val_long_count      INTEGER,
    val_short_count     INTEGER,
    val_long_pnl        REAL,
    val_short_pnl       REAL,
    val_long_pf         REAL,
    val_short_pf        REAL,

    -- Tier 2: Stress summary
    stress_base_pf          REAL,
    stress_2x_spread_pf     REAL,
    stress_3x_spread_pf     REAL,
    stress_5x_spread_pf     REAL,
    stress_2x_slip_mean_pf  REAL,
    stress_2x_slip_std_pf   REAL,
    stress_combined_pf      REAL,
    stress_robustness_score REAL,
    stress_worst_retention  REAL,
    stress_worst_scenario   TEXT,

    -- Tier 4: Churn diagnostic
    churn_cost_per_trade    REAL,
    churn_total_cost        REAL,
    churn_net_edge          REAL,
    churn_gross_edge        REAL,
    churn_cost_to_edge      REAL,

    -- Tier 5: Reward ablation
    r1_pct_of_total         REAL,
    r2_pct_of_total         REAL,
    r3_pct_of_total         REAL,
    r4_pct_of_total         REAL,
    r5_pct_of_total         REAL,

    -- Tier 6: Gating pass-rates (Section 9)
    gate_direction_pass     REAL,      -- % of candidate bars passing direction threshold
    gate_conviction_pass    REAL,      -- % passing conviction floor
    gate_spread_pass        REAL,      -- % passing spread gate (estimated)
    gate_lot_pass           REAL,      -- % passing lot sizing (not returning 0)
    gate_overall_pass       REAL,      -- % of candidate bars that would result in a trade

    -- Conviction stats
    conv_mean               REAL,
    conv_std                REAL,
    conv_p10                REAL,
    conv_p50                REAL,
    conv_p90                REAL,

    -- Action stats (mean across all steps)
    action_direction_mean   REAL,
    action_direction_std    REAL,
    action_conviction_mean  REAL,
    action_conviction_std   REAL,
    action_exit_mean        REAL,
    action_exit_std         REAL,
    action_sl_mean          REAL,
    action_sl_std           REAL,

    -- Composite score
    spartus_score           REAL,
    score_val_sharpe        REAL,      -- component breakdown
    score_val_pf            REAL,
    score_stress            REAL,
    score_max_dd            REAL,
    score_quality           REAL,

    -- Hard-fail / detector results
    hard_fails              TEXT,      -- JSON array: [] or ["negative_pf", ...]
    is_disqualified         INTEGER DEFAULT 0,
    detector_aggression     INTEGER DEFAULT 0,  -- 0=clear, 1=detected
    detector_collapse       INTEGER DEFAULT 0,
    detector_fragility      INTEGER DEFAULT 0,
    detector_overfitting    INTEGER DEFAULT 0,
    detector_reward_hack    INTEGER DEFAULT 0,
    detector_details        TEXT,      -- JSON blob with full detector outputs

    -- Champion flag
    is_champion             INTEGER DEFAULT 0,

    -- Locked test (NULL if not run)
    test_trades             INTEGER,
    test_win_pct            REAL,
    test_pf                 REAL,
    test_sharpe             REAL,
    test_sortino            REAL,
    test_max_dd_pct         REAL,
    test_net_pnl            REAL,
    test_tim_pct            REAL,
    test_weeks_used         TEXT       -- JSON array of week indices
);

-- =====================================================
-- Table: stress_details
-- One row per scenario per benchmark run.
-- =====================================================
CREATE TABLE stress_details (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    scenario        TEXT NOT NULL,
    set_name        TEXT NOT NULL DEFAULT 'VAL',
    trades          INTEGER,
    win_pct         REAL,
    net_pnl         REAL,
    pf              REAL,
    sharpe          REAL,
    sortino         REAL,
    max_dd_pct      REAL,
    tim_pct         REAL,
    avg_hold        REAL,
    trades_per_day  REAL,
    long_count      INTEGER,
    short_count     INTEGER,
    long_pnl        REAL,
    short_pnl       REAL,
    pf_retention    REAL       -- scenario_pf / base_pf
);

-- =====================================================
-- Table: regime_details
-- One row per regime slice per benchmark run.
-- =====================================================
CREATE TABLE regime_details (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    slice_type      TEXT NOT NULL,   -- "atr_quartile", "session", "day", "direction", "close_reason", "lesson_type"
    slice_value     TEXT NOT NULL,   -- "Q1", "London", "Monday", "LONG", "TP_HIT", "GOOD_TRADE"
    trades          INTEGER,
    win_pct         REAL,
    net_pnl         REAL,
    pf              REAL,
    avg_pnl         REAL,
    avg_hold        REAL
);

-- =====================================================
-- Table: benchmark_trades
-- Every trade from benchmark runs (for deep audit).
-- =====================================================
CREATE TABLE benchmark_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    scenario        TEXT NOT NULL DEFAULT 'base',
    trade_num       INTEGER,
    week            INTEGER,
    step            INTEGER,
    side            TEXT,
    entry_price     REAL,
    exit_price      REAL,
    lots            REAL,
    pnl             REAL,
    pnl_pct         REAL,
    hold_bars       INTEGER,
    conviction      REAL,
    close_reason    TEXT,
    lesson_type     TEXT,
    session         TEXT,
    atr_at_entry    REAL,
    max_favorable   REAL,
    initial_sl      REAL,
    initial_tp      REAL,
    final_sl        REAL
);

-- =====================================================
-- Table: leaderboard
-- Champion history (derived from benchmark_runs).
-- =====================================================
CREATE TABLE leaderboard (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    model_id        TEXT NOT NULL,
    spartus_score   REAL NOT NULL,
    crowned_at      TEXT NOT NULL,
    dethroned_at    TEXT,           -- NULL if current champion
    notes           TEXT
);

-- =====================================================
-- Table: locked_test_audit
-- Dedicated audit trail for test set access.
-- =====================================================
CREATE TABLE locked_test_audit (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES benchmark_runs(run_id),
    timestamp           TEXT NOT NULL,
    operator            TEXT NOT NULL,
    model_id            TEXT NOT NULL,
    model_file_hash     TEXT NOT NULL,
    data_manifest_hash  TEXT NOT NULL,
    split_hash          TEXT NOT NULL,
    feature_hash        TEXT NOT NULL,
    config_hash         TEXT NOT NULL,
    seed                INTEGER NOT NULL,
    test_weeks_used     TEXT NOT NULL,  -- JSON array
    result_hash         TEXT NOT NULL   -- SHA256 of results JSON
);

-- Indices
CREATE INDEX idx_runs_model     ON benchmark_runs(model_id);
CREATE INDEX idx_runs_timestamp ON benchmark_runs(timestamp);
CREATE INDEX idx_runs_champion  ON benchmark_runs(is_champion);
CREATE INDEX idx_stress_run     ON stress_details(run_id);
CREATE INDEX idx_regime_run     ON regime_details(run_id);
CREATE INDEX idx_trades_run     ON benchmark_trades(run_id);
CREATE INDEX idx_trades_scenario ON benchmark_trades(run_id, scenario);
CREATE INDEX idx_leader_score   ON leaderboard(spartus_score DESC);
CREATE INDEX idx_audit_model    ON locked_test_audit(model_id);
```

### 7.3 Run Folder Artifacts

Each benchmark run also generates file artifacts in `storage/benchmark/runs/{run_id}/`:

```
storage/benchmark/
├── spartusbench.db
└── runs/
    └── {run_id}/
        ├── report.md                  -- Human-readable summary (Section 5.5 format)
        ├── report.json                -- Machine-readable full results
        ├── delta_vs_champion.md       -- Comparison vs current champion
        ├── equity_curve.png           -- Validation equity curve chart
        ├── stress_comparison.png      -- Bar chart: PF per stress scenario
        ├── regime_heatmap.png         -- Grid: ATR quartile x Session performance
        ├── reward_decomposition.png   -- Stacked bar: R1-R5 contribution
        ├── gating_funnel.png          -- Funnel chart: bars -> direction -> conviction -> trade
        ├── conviction_distribution.png -- Histogram of trade conviction values
        └── trades.csv                 -- Full trade list (for external audit/analysis)
```

---

## 8. Model Discovery & Loading

### 8.1 Discovery Rules

SpartusBench discovers models in the following locations and order:

```
1. storage/models/spartus_week_NNNN.zip      -- Weekly checkpoints (glob pattern)
2. storage/models/spartus_best.zip            -- Best validation checkpoint
3. storage/models/spartus_champion_*.zip      -- Named champion packages
4. storage/models/spartus_latest.zip          -- Latest training checkpoint
5. Any .zip file passed via --model <path>    -- Explicit path override
```

**Naming convention for identification:**
- `spartus_week_NNNN.zip` -> model_id = `W{NNNN}` (e.g., `W0170`)
- `spartus_best.zip` -> model_id = `best`
- `spartus_champion_WNNNN.zip` -> model_id = `champion_W{NNNN}`
- Explicit path -> model_id = filename stem

### 8.2 Loading Pipeline

```python
class ModelLoader:
    def load_for_benchmark(self, model_path: Path) -> EvalBundle:
        """
        Returns EvalBundle containing everything needed for evaluation.
        """
        # 1. Load SB3 model
        model = SAC.load(str(model_path), device="cpu")  # CPU for deterministic eval

        # 2. Load companion files
        reward_state_path = model_path.with_suffix(".reward_state.json")
        meta_path = model_path.with_suffix(".meta.json")

        # 3. Load config (from package if deployment ZIP, else use TrainingConfig defaults)
        config = self._extract_config(model_path)

        # 4. Validate dimensions
        assert model.observation_space.shape[0] == config.obs_dim  # 670

        # 5. Discover data weeks
        val_weeks, test_weeks = self._discover_and_split()

        return EvalBundle(
            model=model,
            config=config,
            val_weeks=val_weeks,
            test_weeks=test_weeks,
            reward_state=reward_state,
            metadata=metadata,
        )
```

### 8.3 Config Resolution Order

1. If model path is a deployment package ZIP (contains config.json inside): use bundled config.
2. If companion `.meta.json` exists: extract week number, use stored metadata.
3. Fallback: use `TrainingConfig()` defaults from `src/config.py`.

### 8.4 Feature Loading

For each evaluation week, features are loaded from cache:

```python
def load_features(year: int, week: int) -> pd.DataFrame:
    cache_path = Path(f"storage/features/{year}/week_{week:02d}_features.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    raise FileNotFoundError(f"Feature cache missing: {cache_path}")
```

SpartusBench does NOT rebuild feature caches. If a cache is missing, the week is skipped with a warning.

---

## 9. Gating Pass-Rate Diagnostics

### 9.1 Purpose

The gating diagnostic answers: **"Will this model actually trade when deployed live?"**

During training evaluation, there's no spread gate, no circuit breaker, no weekend manager. A model can score great on T1-T5 but deadlock in production because its conviction outputs are all below 0.15, or its direction outputs are all in the [-0.3, 0.3] dead zone.

### 9.2 Gates Simulated

SpartusBench simulates the **training-environment gates** (which are present during both training and benchmark eval) and **estimates** the live-only gates:

| Gate | Threshold | Source | Simulation |
|------|-----------|--------|------------|
| **Direction Threshold** | `abs(direction) >= 0.3` | `config.direction_threshold` | **Exact** -- same gate in training env |
| **Conviction Floor** | `conviction >= 0.15` (live) / `0.30` (training) | `config.min_conviction` | **Simulated at both thresholds** |
| **Lot Sizing** | `calculate_lot_size() > 0` | Risk manager | **Exact** -- same lot sizing logic |
| **Spread Gate (estimated)** | `spread_points < 50 AND spread < 2.5 * EMA` | Live broker_constraints | **Estimated** -- uses session-based spread from config |

### 9.3 Computation

On every step of the validation rollout (not just steps where a trade was attempted), record the raw action and simulate the gating chain:

```python
# Per step (regardless of whether model is flat or has position):
raw_action = model.predict(obs, deterministic=True)
direction  = raw_action[0]
conviction = (raw_action[1] + 1.0) / 2.0  # map [-1,1] -> [0,1]

# Gate checks (simulated for all bars, not just "wants to trade" bars):
passes_direction_training = abs(direction) >= 0.3
passes_direction_loose    = abs(direction) >= 0.2   # also track a looser threshold
passes_conviction_live    = conviction >= 0.15
passes_conviction_train   = conviction >= 0.30

# For candidate bars (passes direction threshold):
if passes_direction_training:
    side = "LONG" if direction > 0 else "SHORT"
    lots = risk_manager.calculate_lot_size(balance, conviction, atr, dd, side)
    passes_lot_sizing = lots > 0

    # Spread estimate based on session
    hour = current_bar_utc_hour
    if 8 <= hour < 12:
        est_spread_pips = config.spread_london_pips
    elif 12 <= hour < 20:
        est_spread_pips = config.spread_ny_pips
    elif 0 <= hour < 8:
        est_spread_pips = config.spread_asia_pips
    else:
        est_spread_pips = config.spread_off_hours_pips
    passes_spread = est_spread_pips * 10 < 50  # convert to points, check hard max
```

### 9.4 Reported Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `gate_direction_pass` | `bars_passing_direction / total_bars * 100` | % of ALL bars where model wants to trade |
| `gate_conviction_pass_live` | `bars_passing_conv_0.15 / bars_passing_direction * 100` | % of candidate bars passing live conviction |
| `gate_conviction_pass_train` | `bars_passing_conv_0.30 / bars_passing_direction * 100` | % of candidate bars passing training conviction |
| `gate_lot_pass` | `bars_with_lots_gt_0 / bars_passing_all_prior * 100` | % surviving lot sizing |
| `gate_spread_pass` | `bars_passing_spread / bars_passing_all_prior * 100` | % surviving spread gate (estimated) |
| `gate_overall_pass` | `bars_passing_all_gates / total_bars * 100` | End-to-end: what % of bars would produce a trade |
| `gate_promote_rate` | `bars_promoted_to_min_lot / bars_attempted_lot_sizing * 100` | How often min-lot promotion fires |

### 9.5 Funnel Visualization

The gating funnel chart shows the progressive filtering:

```
Total Bars:          2,400 (100%)
  |
  v  Direction gate (|dir| >= 0.3)
Candidate Bars:      1,680 (70%)
  |
  v  Conviction gate (conv >= 0.15)
Conv-Passing:        1,344 (56%)
  |
  v  Lot sizing (lots > 0)
Lot-Passing:         1,210 (50%)
  |
  v  Spread gate (est)
Trade-Ready:         1,150 (48%)
  |
  v  Already has position / cooldown
Actual Trades:         185 (7.7%)
```

---

## 10. Reproducibility Guarantees

### 10.1 Seed Strategy

Every benchmark run uses a deterministic seed chain:

```python
MASTER_SEED = 42  # default, overridable via --seed

# Per-run seeding:
np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)
random.seed(MASTER_SEED)

# Per-week seeding within run:
week_seed = MASTER_SEED + week_idx * 100

# Environment seeding:
env = SpartusTradeEnv(..., seed=week_seed)
```

### 10.2 Deterministic Predict

All benchmark predictions use:
```python
action, _states = model.predict(obs, deterministic=True)
```

No stochastic sampling. Same observation always produces same action.

### 10.3 Data Manifest Hash

Captures exactly which data files were used and their content identity:

```python
def compute_data_manifest_hash(week_indices: list[int], data_dir: Path) -> str:
    """Hash of (file_path, file_size, mtime) for all data files used."""
    entries = []
    for week_idx in sorted(week_indices):
        year, wk = week_idx_to_year_week(week_idx)
        for tf in ["M5", "H1", "H4", "D1"]:
            path = data_dir / str(year) / f"week_{wk:02d}_{tf}.parquet"
            if path.exists():
                stat = path.stat()
                entries.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
        # Feature cache
        feat_path = Path(f"storage/features/{year}/week_{wk:02d}_features.parquet")
        if feat_path.exists():
            stat = feat_path.stat()
            entries.append(f"{feat_path}|{stat.st_size}|{stat.st_mtime_ns}")
    manifest = "\n".join(entries)
    return hashlib.sha256(manifest.encode()).hexdigest()
```

### 10.4 Split Hash

Captures the exact train/val/test partition:

```python
def compute_split_hash(train_weeks, val_weeks, test_weeks, purge_gap) -> str:
    payload = json.dumps({
        "train": sorted(train_weeks),
        "val": sorted(val_weeks),
        "test": sorted(test_weeks),
        "purge_gap": purge_gap,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

### 10.5 Feature Ordering Hash

Captures the feature set identity (protects against feature reordering or additions):

```python
def compute_feature_hash(config) -> str:
    payload = json.dumps({
        "num_features": config.num_features,           # 67
        "obs_dim": config.obs_dim,                     # 670
        "frame_stack": config.frame_stack,             # 10
        "market_feature_names": list(config.market_feature_names),
        "norm_exempt_features": list(config.norm_exempt_features),
        "account_feature_names": list(config.account_feature_names),
        "memory_feature_names": list(config.memory_feature_names),
        "norm_window": config.norm_window,             # 200
        "norm_clip": config.norm_clip,                 # 5.0
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

### 10.6 Eval Config Hash

Captures the exact evaluation environment configuration:

```python
def compute_config_hash(eval_config) -> str:
    """Hash of all config fields that affect evaluation outcomes."""
    fields = {
        "initial_balance": eval_config.initial_balance,
        "direction_threshold": eval_config.direction_threshold,
        "min_conviction": eval_config.min_conviction,
        "max_dd": eval_config.max_dd,
        "daily_trade_hard_cap": eval_config.daily_trade_hard_cap,
        "risk_per_trade": eval_config.risk_per_trade,
        "spread_london_pips": eval_config.spread_london_pips,
        "spread_ny_pips": eval_config.spread_ny_pips,
        "spread_asia_pips": eval_config.spread_asia_pips,
        "spread_off_hours_pips": eval_config.spread_off_hours_pips,
        "slippage_mean_pips": eval_config.slippage_mean_pips,
        "slippage_std_pips": eval_config.slippage_std_pips,
        "commission_per_lot": eval_config.commission_per_lot,
        "pip_price": eval_config.pip_price,
        "trade_tick_value": eval_config.trade_tick_value,
        "trade_tick_size": eval_config.trade_tick_size,
        "observation_noise_std": 0.0,  # always 0 for benchmark
        "spread_jitter": 0.0,
        "slippage_jitter": 0.0,
    }
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

---

## 11. CLI Commands

### 11.1 Entry Point

```bash
python -m spartusbench <command> [options]
```

### 11.2 Commands

#### `run` -- Execute a benchmark

```bash
# Full benchmark (T1-T6) on a weekly checkpoint
python -m spartusbench run W170

# Full benchmark on the best model
python -m spartusbench run best

# Full benchmark with explicit model path
python -m spartusbench run --model storage/models/spartus_week_0170.zip

# Locked test (T7) -- requires confirmation
python -m spartusbench run W170 --suite locked_test --confirm-test

# Validation-only (T1 only, quick check)
python -m spartusbench run W170 --suite validation_only

# Stress-only (T2 only)
python -m spartusbench run W170 --suite stress_only

# Custom seed
python -m spartusbench run W170 --seed 123

# Skip plot generation (faster, CLI-only output)
python -m spartusbench run W170 --no-plots
```

**Output:**
```
SpartusBench v1.0.0
==================

Model:    W0170 (storage/models/spartus_week_0170.zip)
Suite:    full (T1-T6)
Seed:     42
Config:   sha256:a1b2c3...

Running T1: Validation Eval ... [37 weeks]
  Sharpe: 4.09 | PF: 2.24 | Win: 52.0% | MaxDD: 20.8% | Trades: 1044
Running T2: Stress Matrix ... [7 scenarios x 37 weeks]
  base:        PF=2.24 | retention=1.00
  2x_spread:   PF=2.01 | retention=0.90
  combined:    PF=1.82 | retention=0.81
  3x_spread:   PF=1.68 | retention=0.75
  ...
Running T3: Regime Segmentation ...
Running T4: Churn Diagnostic ...
Running T5: Reward Ablation ...
Running T6: Gating Diagnostics ...
  Direction pass: 68.2% | Conviction pass: 89.1% | Lot pass: 96.3%
  Overall pass: 58.7% of bars -> trade-ready

Detectors:
  Aggression Drift:     CLEAR
  Conviction Collapse:  CLEAR
  Stress Fragility:     CLEAR
  Reward Hacking:       CLEAR

Hard Fails:  NONE
SpartusScore: 74.2

vs Champion (W160, score=71.8): +2.4 -> PROMOTE

Report: storage/benchmark/runs/abc123/report.md
```

#### `compare` -- Compare two models

```bash
python -m spartusbench compare W170 W194
python -m spartusbench compare W170 best
```

**Output:** Side-by-side metric table with deltas and verdicts (Section 5.5 format).

#### `leaderboard` -- Show champion history

```bash
python -m spartusbench leaderboard
python -m spartusbench leaderboard --top 10
python -m spartusbench leaderboard --all  # include disqualified
```

**Output:**
```
SpartusBench Leaderboard
========================

Rank  Model   Score   Sharpe  PF     MaxDD   Stress  Status
----  ------  ------  ------  -----  ------  ------  -----------
  1   W0170   74.2    4.09    2.24   20.8%   82.1    CHAMPION
  2   W0180   72.8    3.91    2.18   19.2%   80.5    (dethroned)
  3   W0160   71.8    3.85    2.12   21.1%   79.2    (dethroned)
  4   W0194   68.1    3.22    1.95   22.4%   75.8    regression
  -   W0120   --      --      --     --      --      DISQUALIFIED (negative_pf)
```

#### `show` -- Show details of a specific run

```bash
python -m spartusbench show <run_id>
python -m spartusbench show --last             # most recent run
python -m spartusbench show --model W170       # most recent run for W170
```

**Output:** Full report (same as report.md) printed to terminal.

#### `audit` -- Show locked test audit trail

```bash
python -m spartusbench audit
python -m spartusbench audit --model W170
```

**Output:** Table of all locked test runs with timestamps, operators, and hashes.

#### `discover` -- List available models

```bash
python -m spartusbench discover
```

**Output:**
```
Available Models
================

ID       Path                                        Size    Has Meta  Has Reward State
-------  ------------------------------------------  ------  --------  ----------------
W0000    storage/models/spartus_week_0000.zip        10.2MB  No        Yes
W0010    storage/models/spartus_week_0010.zip        10.2MB  No        Yes
...
W0170    storage/models/spartus_week_0170.zip        10.2MB  Yes       Yes
W0194    storage/models/spartus_week_0194.zip        10.2MB  No        Yes
best     storage/models/spartus_best.zip             10.2MB  Yes       Yes

Total: 20 checkpoints found
```

---

## 12. UI Layout & Wireframes

The UI is designed for human review and workflow efficiency. SpartusBench functions perfectly without the UI (CLI-first), but the UI makes comparing models, reviewing detectors, and tracking champion progression much faster.

### 12.1 Window Structure

```
+================================================================+
|  SpartusBench - Benchmark & Model Progression                  |
|================================================================|
| [Leaderboard] [Run Benchmark] [Compare] [Run Detail] [Locked] |
+================================================================+
|                                                                |
|              << TAB CONTENT AREA >>                            |
|                                                                |
+================================================================+
| Status: Ready | Champion: W0170 (74.2) | DB: 47 runs          |
+================================================================+
```

### 12.2 Tab 1: Leaderboard

```
+================================================================+
|  LEADERBOARD                                          [Refresh]|
|================================================================|
|                                                                |
|  +-----------------------------------------------------------+|
|  | Rank | Model | Score | Sharpe | PF   | MaxDD | Stress | St||
|  |------|-------|-------|--------|------|-------|--------|----||
|  |  1*  | W0170 | 74.2  | 4.09   | 2.24 | 20.8% | 82.1   | C||
|  |  2   | W0180 | 72.8  | 3.91   | 2.18 | 19.2% | 80.5   |  ||
|  |  3   | W0160 | 71.8  | 3.85   | 2.12 | 21.1% | 79.2   |  ||
|  |  4   | W0194 | 68.1  | 3.22   | 1.95 | 22.4% | 75.8   |  ||
|  |  -   | W0120 | DQ    | --     | 0.88 | 28.3% | --     | D||
|  +-----------------------------------------------------------+|
|                                                                |
|  Champion History (Timeline):                                  |
|  +-----------------------------------------------------------+|
|  |  Score                                                     ||
|  |  80 |                                                      ||
|  |  75 |              *** W170 ***                            ||
|  |  70 |     ** W160 **          \                            ||
|  |  65 |  * W140 *                 W194 (regression)          ||
|  |  60 |                                                      ||
|  |     +--+-----+-----+-----+-----+-----+--> Checkpoint      ||
|  |        W120  W140  W160  W170  W180  W194                  ||
|  +-----------------------------------------------------------+|
|                                                                |
|  [Double-click row to open Run Detail tab]                     |
+================================================================+
```

### 12.3 Tab 2: Run Benchmark

```
+================================================================+
|  RUN BENCHMARK                                                 |
|================================================================|
|                                                                |
|  Model Selection:                                              |
|  +---------------------------+  [Discover Models]              |
|  | W0170 (champion)      [v]|                                  |
|  +---------------------------+                                  |
|  | W0194                    |                                  |
|  | W0180                    |                                  |
|  | spartus_best             |                                  |
|  | [Browse file...]         |                                  |
|  +---------------------------+                                  |
|                                                                |
|  Suite:                                                        |
|  (o) Full (T1-T6)      Recommended for champion evaluation    |
|  ( ) Validation Only    Quick check (T1 only, ~2 min)         |
|  ( ) Stress Only        Cost robustness (T2 only, ~10 min)    |
|                                                                |
|  Seed: [42          ]                                          |
|  [x] Generate plots                                            |
|  [ ] Compare vs champion after completion                      |
|                                                                |
|  +-----------------------------------------------------------+|
|  |                    [RUN BENCHMARK]                         ||
|  +-----------------------------------------------------------+|
|                                                                |
|  Progress:                                                     |
|  +-----------------------------------------------------------+|
|  | T1: Validation Eval    [==============>     ] 78%  25/37w ||
|  | T2: Stress Matrix      [                    ] --   pending||
|  | T3: Regime Segment      [                    ] --   pending||
|  | T4: Churn Diagnostic    [                    ] --   pending||
|  | T5: Reward Ablation     [                    ] --   pending||
|  | T6: Gating Diagnostics  [                    ] --   pending||
|  +-----------------------------------------------------------+|
|                                                                |
|  Live Metrics (updating as T1 runs):                           |
|  Trades: 847 | Win%: 51.2 | PF: 2.18 | MaxDD: 19.3%          |
+================================================================+
```

### 12.4 Tab 3: Compare

```
+================================================================+
|  COMPARE MODELS                                                |
|================================================================|
|                                                                |
|  Model A: [W0170 (champion) [v]]    Model B: [W0194        [v]|
|                                                                |
|  +-----------------------------------------------------------+|
|  | Metric             | W0170   | W0194   | Delta  | Verdict ||
|  |--------------------|---------|---------|--------|---------|
|  | SpartusScore       | 74.2    | 68.1    | -6.1   | REGRESS ||
|  | Sharpe             | 4.09    | 3.22    | -0.87  | REGRESS ||
|  | PF                 | 2.24    | 1.95    | -0.29  | REGRESS ||
|  | Win Rate           | 52.0%   | 49.8%   | -2.2%  | REGRESS ||
|  | Max DD             | 20.8%   | 22.4%   | +1.6%  | REGRESS ||
|  | Stress (2x spread) | 0.90    | 0.82    | -0.08  | WATCH   ||
|  | TIM%               | 32%     | 38%     | +6%    | NEUTRAL ||
|  | Trades/Day         | 2.1     | 2.8     | +0.7   | NEUTRAL ||
|  | Expectancy         | 0.70    | 0.52    | -0.18  | REGRESS ||
|  | Max Consec Loss    | 5       | 7       | +2     | WATCH   ||
|  +-----------------------------------------------------------+|
|                                                                |
|  Detectors:                                                    |
|  +-----------------------------------------------------------+|
|  | Aggression Drift:    NOT DETECTED                          ||
|  | Conviction Collapse: NOT DETECTED                          ||
|  | Stress Fragility:    NOT DETECTED                          ||
|  | Overfitting:         DETECTED (val_sharpe declining x3)    ||
|  +-----------------------------------------------------------+|
|                                                                |
|  Equity Curves (overlaid):                                     |
|  +-----------------------------------------------------------+|
|  |    $                                                       ||
|  |  120|         /W0170/                                      ||
|  |  115|        /      \     /\                               ||
|  |  110|      /         \   /  \  /--                         ||
|  |  105|    /            \ /    \/                             ||
|  |  100|  /               X W0194                             ||
|  |   95|/                                                     ||
|  |     +-----+-----+-----+-----+-----+----> Val Weeks        ||
|  +-----------------------------------------------------------+|
+================================================================+
```

### 12.5 Tab 4: Run Detail

```
+================================================================+
|  RUN DETAIL: abc123 (W0170, 2026-03-02 14:30:00 UTC)          |
|================================================================|
|                                                                |
|  [Summary] [Stress] [Regime] [Churn] [Reward] [Gates] [Trades]|
|                                                                |
|  --- SUMMARY sub-tab ---                                       |
|                                                                |
|  SpartusScore: 74.2    [=======================] CHAMPION      |
|  Score Breakdown:                                              |
|    val_sharpe:  81.8 x 0.25 = 20.5                            |
|    val_pf:      62.0 x 0.20 = 12.4                            |
|    stress:      82.1 x 0.25 = 20.5                            |
|    max_dd:      0.0  x 0.15 = 0.0                             |
|    quality:     70.8 x 0.15 = 10.6                            |
|                                                                |
|  +--Quick Stats----+  +--Stress Retention--+  +--Detectors---+|
|  | Sharpe:  4.09   |  | 2x spread:  0.90  |  | Aggression: -||
|  | PF:      2.24   |  | Combined:   0.81  |  | Collapse:   -||
|  | Win%:    52.0%  |  | 3x spread:  0.75  |  | Fragility:  -||
|  | MaxDD:   20.8%  |  | Worst:      0.62  |  | Overfitting:-||
|  | Sortino: 32.2   |  | Score:      82.1  |  | Rew Hack:   -||
|  | Trades:  1044   |  +-------------------+  +--------------+|
|  | TIM%:    32%    |                                          |
|  | Expect:  0.70   |  +--Gates Funnel-----+                  |
|  | ConsecL: 5      |  | Direction: 68.2%  |                  |
|  +-----------------+  | Conviction: 89.1% |                  |
|                       | Lot Sizing: 96.3% |                  |
|  Hard Fails: NONE     | Overall:    58.7% |                  |
|                       +-------------------+                   |
|                                                                |
|  Equity Curve:                                                 |
|  [chart: balance over validation weeks]                        |
|                                                                |
|  Hashes:                                                       |
|  data_manifest: sha256:a1b2c3d4...                             |
|  split:         sha256:e5f6a7b8...                             |
|  features:      sha256:c9d0e1f2...                             |
|  config:        sha256:34567890...                             |
+================================================================+
```

### 12.6 Tab 5: Locked Test

```
+================================================================+
|  LOCKED TEST                                                   |
|================================================================|
|                                                                |
|  +-----------------------------------------------------------+|
|  |  WARNING: Test set evaluation is permanently recorded.     ||
|  |                                                            ||
|  |  The test set (weeks {N}-{M}) is reserved for final       ||
|  |  go-live assessment. Every run is logged in the audit      ||
|  |  trail with operator, timestamp, and result hashes.        ||
|  |                                                            ||
|  |  This action cannot be undone or hidden.                   ||
|  +-----------------------------------------------------------+|
|                                                                |
|  Model: [W0170 (champion) [v]]                                 |
|  Seed:  [42          ]                                         |
|                                                                |
|  [x] I understand this is permanently recorded                 |
|  [ ] Also run full benchmark (T1-T6) for comparison            |
|                                                                |
|  +-----------------------------------------------------------+|
|  |              [RUN LOCKED TEST]                             ||
|  +-----------------------------------------------------------+|
|                                                                |
|  Audit Trail (most recent):                                    |
|  +-----------------------------------------------------------+|
|  | Date       | Operator | Model | Sharpe | PF   | MaxDD    ||
|  |------------|----------|-------|--------|------|----------|
|  | 2026-03-01 | cjohn   | W0170 | 3.82   | 2.11 | 18.9%   ||
|  | 2026-02-28 | cjohn   | W0160 | 3.55   | 1.98 | 21.2%   ||
|  +-----------------------------------------------------------+|
|                                                                |
|  Test vs Validation Comparison (last run):                     |
|  +-----------------------------------------------------------+|
|  | Metric  | Validation | Test   | Gap    | Status           ||
|  |---------|------------|--------|--------|------------------|
|  | Sharpe  | 4.09       | 3.82   | -0.27  | OK (< 1.0 gap)  ||
|  | PF      | 2.24       | 2.11   | -0.13  | OK (< 0.5 gap)  ||
|  | MaxDD   | 20.8%      | 18.9%  | -1.9%  | OK               ||
|  | Win%    | 52.0%      | 50.3%  | -1.7%  | OK               ||
|  +-----------------------------------------------------------+|
+================================================================+
```

---

## 13. Integration Points

### 13.1 Exporter: Bundle Benchmark Metadata

When `exporter.package_model()` creates a deployment ZIP, if a benchmark run exists for that checkpoint, include it:

```python
# In src/training/exporter.py, extend package_model():
def package_model(self, week, benchmark_run_id=None):
    # ... existing packaging logic ...

    if benchmark_run_id:
        bench_db = Path("storage/benchmark/spartusbench.db")
        if bench_db.exists():
            run_data = query_benchmark_run(bench_db, benchmark_run_id)
            if run_data:
                package_contents["benchmark.json"] = {
                    "run_id": run_data.run_id,
                    "spartus_score": run_data.spartus_score,
                    "val_sharpe": run_data.val_sharpe,
                    "val_pf": run_data.val_pf,
                    "stress_robustness": run_data.stress_robustness_score,
                    "max_dd_pct": run_data.val_max_dd_pct,
                    "hard_fails": json.loads(run_data.hard_fails),
                    "is_champion": bool(run_data.is_champion),
                    "is_disqualified": bool(run_data.is_disqualified),
                    "benchmark_timestamp": run_data.timestamp,
                    "detector_flags": {
                        "aggression": bool(run_data.detector_aggression),
                        "collapse": bool(run_data.detector_collapse),
                        "fragility": bool(run_data.detector_fragility),
                        "overfitting": bool(run_data.detector_overfitting),
                        "reward_hacking": bool(run_data.detector_reward_hack),
                    },
                    "gate_overall_pass": run_data.gate_overall_pass,
                }
```

### 13.2 Live Dashboard: Display Benchmark Metadata on Model Load

When `ModelLoader` loads a deployment package, display benchmark info in the dashboard startup log and status panel:

```
[INFO] Model loaded: W0170
[INFO] Benchmark: SpartusScore=74.2 | PF=2.24 | Stress=82.1 | CHAMPION since 2026-02-15
[INFO] Gate pass rate: 58.7% of bars -> trade-ready
```

### 13.3 Training Callback: Auto-Benchmark on New Best

In the training callback, when a new `best_val_sharpe` is found and a checkpoint is saved, optionally trigger a SpartusBench run:

```python
# In src/training/callback.py, after saving best checkpoint:
if auto_benchmark and new_best_saved:
    subprocess.Popen([
        sys.executable, "-m", "spartusbench", "run",
        f"W{week_idx:04d}", "--suite", "validation_only", "--no-plots",
    ])
    # Runs in background, doesn't block training
```

This is optional (controlled by a config flag `auto_benchmark: true`). The full suite is too expensive to run every checkpoint -- `validation_only` is a quick T1-only check that takes ~2 minutes.

### 13.4 Training Dashboard: Benchmark Tab (Tab 7)

Add a seventh tab to `src/training/qt_dashboard.py`:

```
Tab 7: BENCHMARK
+---------+---------+---------+---------+---------+---------+----------+
| OVERVIEW| METRICS | AI INT  | DB VIEW | JOURNAL | EXPORT  | BENCHMARK|
+---------+---------+---------+---------+---------+---------+----------+
```

Contents: Leaderboard table (top 10) + "Benchmark Current" button + last run summary. Minimal -- most detailed review happens in the standalone SpartusBench UI.

---

## 14. Shared Evaluation Module (Refactoring)

### 14.1 Current Duplication

The same metric computations and evaluation logic exist in multiple files:

| Function | Duplicated In | Copies |
|----------|---------------|--------|
| Sharpe ratio | trainer.py, callback.py, tab_performance.py, eval_validation.py, eval_stress_matrix.py, eval_checkpoints.py | 6 |
| Profit factor | trade_env.py, tab_analytics.py, eval_validation.py, eval_stress_matrix.py, eval_checkpoints.py | 5 |
| Max drawdown | trainer.py, risk_manager.py (x2), tab_performance.py | 4 |
| Week splitting | trainer.py, eval_validation.py, eval_stress_matrix.py, eval_checkpoints.py | 4 |
| Deterministic rollout | eval_validation.py, eval_stress_matrix.py, eval_checkpoints.py | 3 |
| Stress scenarios | eval_validation.py, eval_stress_matrix.py | 2 |
| Trade classification | src/memory/trade_analyzer.py, live_dashboard/memory/trade_analyzer.py | 2 |
| Session binning | src/memory/trading_memory.py, live_dashboard/memory/trading_memory.py | 2 |

### 14.2 Proposed Shared Module

```
src/evaluation/
├── __init__.py
├── metrics.py           -- Pure functions: sharpe(), sortino(), pf(), max_dd(), etc.
├── rollout.py           -- rollout_week(), rollout_weeks(), collect_trades()
├── stress.py            -- STRESS_SCENARIOS dict, apply_stress(), compute_retentions()
├── splits.py            -- split_weeks(), discover_weeks(), compute_split_hash()
├── regime.py            -- atr_quartiles(), session_bucketing(), day_bucketing()
├── churn.py             -- edge_analysis(), cost_estimation()
├── trade_classifier.py  -- classify_trade() (single source of truth)
├── hashing.py           -- data_manifest_hash(), split_hash(), feature_hash(), config_hash()
└── types.py             -- BenchmarkResult, TradeRecord, StressResult dataclasses
```

**After refactoring:**
- `trainer.py` imports from `src/evaluation/metrics` and `src/evaluation/splits`
- `eval_validation.py`, `eval_stress_matrix.py`, `eval_checkpoints.py` import from `src/evaluation/`
- `spartusbench/` imports from `src/evaluation/`
- `live_dashboard/` imports from `src/evaluation/metrics` and `src/evaluation/trade_classifier`

### 14.3 Refactoring Rules

1. **No behavior change.** Extracted functions must produce identical results to current implementations.
2. **No new dependencies.** Shared module uses only numpy, pandas, and stdlib.
3. **Keep existing scripts working.** After refactoring, `python scripts/eval_validation.py` must produce identical output.
4. **Test with checksums.** Run eval scripts before and after refactoring; diff outputs to confirm no change.

---

## 15. Implementation Roadmap

### Phase 1: Shared Evaluation Module (Foundation)

**Scope:** Extract `src/evaluation/` from existing code.
**Risk:** Low -- pure refactoring, no behavior change.
**Verification:** Run existing eval scripts before/after; diff outputs.
**Files created:** `src/evaluation/*.py` (8 files)
**Files modified:** `trainer.py`, `callback.py`, `eval_*.py` (3 scripts), `trade_analyzer.py` (2 copies)

### Phase 2: Core SpartusBench Engine

**Scope:** Build the benchmark runner, discovery, scoring, and storage.
**Risk:** Low -- new code, doesn't touch training or live.
**Files created:** `spartusbench/*.py` (8 files)
**Verification:** Benchmark W170 (known champion), verify metrics match existing eval scripts.
**Deliverable:** `python -m spartusbench run W170` produces full results + SQLite record.

### Phase 3: Detectors + Hard-Fail Rules

**Scope:** Implement all 5 detectors, hard-fail rules, champion protocol.
**Risk:** Low -- new code, advisory only.
**Files created:** `spartusbench/detectors/*.py` (5 files), `spartusbench/scoring.py`, `spartusbench/regression.py`
**Verification:** Fabricate regression scenarios (modify metrics manually), verify detectors trigger.
**Deliverable:** `python -m spartusbench compare W170 W194` with detector output.

### Phase 4: Gating Diagnostics

**Scope:** Add Tier 6 gating pass-rate simulation.
**Risk:** Low -- extends rollout with action logging.
**Files created:** Extensions to `spartusbench/runner.py` and `spartusbench/suite.py`
**Verification:** Run on a model known to deadlock in live; verify low gate pass-rate.
**Deliverable:** Gate funnel data in benchmark results.

### Phase 5: Reports + Plots

**Scope:** Markdown report generator, JSON export, matplotlib charts.
**Risk:** None -- output only.
**Files created:** `spartusbench/reports/*.py` (3 files)
**Verification:** Generate reports, human review.
**Deliverable:** `storage/benchmark/runs/{run_id}/report.md` + `.png` charts.

### Phase 6: Reproducibility Hashing

**Scope:** Implement all 4 hash functions, embed in every run record.
**Risk:** None -- computation only, no side effects.
**Files created:** `src/evaluation/hashing.py`
**Verification:** Run same benchmark twice; verify identical hashes. Change one feature cache file; verify data_manifest_hash changes.

### Phase 7: Integration

**Scope:** Exporter bundling, live dashboard display, training callback hook.
**Risk:** Medium -- touches existing code paths, but minimal changes.
**Files modified:** `exporter.py`, `model_loader.py` (live), `callback.py`
**Verification:** Package a model with benchmark metadata; load in live dashboard; verify metadata displays.

### Phase 8: UI (Optional, Can Be Built Anytime)

**Scope:** PyQt6 standalone benchmark window with 5 tabs.
**Risk:** Low -- UI only, no logic changes.
**Files created:** `spartusbench/ui/*.py`
**Verification:** Visual inspection + functional testing of all tabs.
**Note:** UI can be built at any phase after Phase 2. It reads from the same SQLite database the CLI writes to.

---

## Appendix A: Architecture Recommendations

Based on full review of the existing Spartus codebase:

### A.1 Separate Benchmark DB from Training/Live DBs

Training uses `storage/memory/spartus_memory.db`. Live uses `live_dashboard/storage/memory/spartus_live.db`. SpartusBench should use its own `storage/benchmark/spartusbench.db`. This keeps concerns separated and prevents any write contention.

### A.2 CPU-Only Evaluation

Benchmark runs should use `device="cpu"` for model loading. GPU adds non-determinism (floating-point order) and isn't needed -- inference is fast enough on CPU for evaluation. This also means SpartusBench can run on machines without a GPU.

### A.3 No Parallel Environments for Benchmarking

Training uses `DummyVecEnv(n_envs=4)` for throughput. Benchmarking should use a single environment (`n_envs=1`) for simplicity and determinism. The slight speed cost is acceptable for reproducibility.

### A.4 Store Raw Trade Lists, Not Just Aggregates

The `benchmark_trades` table stores every individual trade. This is critical for:
- Post-hoc regime analysis (re-slice trades without re-running the benchmark)
- Forensic debugging (inspect specific losing streaks)
- External audit (export to CSV for third-party review)

### A.5 Feature Cache Validation

Before running a benchmark, SpartusBench should verify that feature caches exist for all required weeks and that their column count matches the expected 54 precomputed features. Missing or incompatible caches should be flagged immediately, not discovered mid-run.

### A.6 Conviction Threshold Mismatch Warning

The benchmark should prominently note that the training conviction threshold (0.30) differs from the live threshold (0.15). Gating diagnostics should report pass rates at BOTH thresholds so the user can assess the impact.

---

## Appendix B: Future Extensions (Not in v1.0)

These are explicitly out of scope for the initial build but documented for future consideration:

1. **Walk-Forward Validation** -- Sliding window: train on weeks 0-100, val on 101-115, then shift forward. Measures stability across time.
2. **Adversarial Stress Tests** -- Flash crashes, liquidity droughts, random gap opens. Requires extending the environment simulator.
3. **Multi-Symbol Benchmarking** -- When Spartus expands beyond XAUUSD, benchmark per-symbol and portfolio-level.
4. **A/B Testing Framework** -- Run two models side-by-side on the same market data, compare decision-by-decision.
5. **Automatic Retraining Trigger** -- When SpartusScore drops below threshold on new data, automatically queue a retraining job.
6. **REST API** -- Expose benchmark results via HTTP for external dashboards or CI/CD integration.
7. **Benchmark-as-CI-Gate** -- Integrate with GitHub Actions: PR that changes model code must pass benchmark regression check.

---

*End of SpartusBench Specification v1.0.0*
