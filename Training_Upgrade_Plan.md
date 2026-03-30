# Spartus AI — Training Upgrade Plan (V2)

> **Created:** 2026-03-06
> **Last Updated:** 2026-03-06
> **Collaborators:** User + Claude (implementation) + ChatGPT (architecture/strategy)
> **Status:** COMPLETE — All 6 changes implemented, tested, and verified. Ready for training.

---

## Purpose

This document tracks the design and refinement of the next training upgrade for Spartus AI. It is the single source of truth for what changes will be made before the next model is trained.

**Process:**
1. ChatGPT proposes upgrades based on Week 1 live results
2. Claude reviews for technical feasibility and implementation impact
3. We refine until the plan is locked
4. When approved: "execute the upgrade plan" → Claude implements from this document

---

## Current System Baseline

- **Model:** W170 (SAC, 67 features, 670 obs dims, frame stack x10)
- **Training:** 170 weeks on M5 XAUUSD data (2015-present), Dukascopy
- **Architecture:** SAC with [256,256] pi/qf networks, lr=3e-4, buffer=200K, batch=1024
- **Reward:** 5-component (R1=0.40 P/L, R2=0.20 quality, R3=0.15 DD, R4=0.15 Sharpe, R5=0.10 risk bonus)
- **Live normalization:** Frozen mode (static baseline from training)
- **SL logic:** Initial SL = `max(2.5 - conviction, 1.0) * ATR`. Trailing via agent's `sl_adjustment` action [0,1]. Can only tighten, never loosen. Min trail distance = 0.5 ATR.
- **No R-based stop management.** No breakeven logic. No automatic profit lock. Agent must learn when to trail via reward signals.
- **Reward has no "giveback" penalty.** Trades that go +2R then reverse to SL hit are only penalized via negative P/L in R1.

---

## Week 1 Live Demo Results (March 2-6, 2026)

Full report: `live_dashboard/reports/week1_demo_performance.md`

### Key Metrics
| Metric | Value |
|---|---|
| Trades | 152 |
| P/L | +44.18 GBP (+4.8%) |
| Win Rate | 46.7% |
| Profit Factor | 1.09 |
| Direction Accuracy | 47.1% |
| Max Drawdown | 115.04 GBP (12.5%) |
| Avg Hold | 2.4 bars (12 min) |
| Time in Market | 32.7% |

### Critical Issues Identified

1. **LONG bias:** 63% of trades are LONG, net P/L = -82.79. SHORT = +126.97 (51.8% WR)
2. **Low conviction trades lose money:** Conv 0.3-0.5 = -83.32 (worst bucket). Conv > 0.5 = +124.44 (71% WR)
3. **Cannot detect regime shifts:** March 3 NY, 5 consecutive LONGs into sell-off = -73.79 in 1 hour
4. **Exits too early:** 76% of trades exit at min_hold_bars (3 bars / 15 min)
5. **NY session catastrophic:** -62.29 GBP, 41% WR. 14:00 UTC alone = -59.78
6. **Trailing SL barely used:** 24 losses had >2pt favorable excursion before reversing to SL
7. **Direction accuracy near random:** 47.1% overall, 44.3% for LONGs
8. **Conviction collapse under extreme moves:** Normalizer contamination required frozen mode fix

### What Worked
- SHORT trades profitable (51.8% WR, +126.97)
- High conviction trades (>0.5) strongly profitable (71% WR, +124.44)
- Safety systems (circuit breaker, weekend manager, emergency stop) all worked correctly
- Asia session best performer (+43.84, 49% WR)
- Frozen normalization mode eliminated conviction collapse

---

## ChatGPT's Proposal (2026-03-06) — Claude's Analysis

### Proposal 1: Profit Protection System (Staged R-Based SL)

**What ChatGPT proposes:**
- Stage 1: At +1.0R → move SL to breakeven + buffer
- Stage 2: At +1.5R → lock SL at +0.5R
- Stage 3: At +2.0R → activate ATR trailing stop (1.0 ATR)
- Stage 4: AI exit_urgency can still close, but can't override locked profit
- Implement as execution-layer rules (not AI-learned), then later teach AI to internalize

**Claude's assessment: AGREE with modifications**

This is the RIGHT approach for the current stage. The live data proves the agent hasn't learned to trail effectively (76% exit at min hold, 24 trades with >2pt MFE that ended as losses). A rule-based safety net is correct before trying to teach the AI.

**Technical feasibility:** HIGH. The existing `adjust_stop_loss()` in both training (`src/risk/risk_manager.py:143`) and live (`live_dashboard/core/trade_executor.py:777`) already tracks `_initial_sl`, `_max_favorable`, and can only tighten. Adding R-based stage logic is straightforward.

**Modifications needed:**
1. **R calculation must use initial risk distance, not ATR.** Currently the system normalizes everything to ATR. R = `abs(entry_price - initial_sl)`. This is already implicitly computed but not stored as a named variable.
2. **Buffer for breakeven must account for spread + broker stop level.** Live system already has `broker_constraints.stops_level` and `spread_current_points`. Training must simulate this.
3. **Stage tracking per trade.** Add `protection_stage: int` to position dict (0=initial, 1=BE, 2=locked, 3=trailing).
4. **Training environment must mirror these rules exactly** so the agent learns to work WITH the protection layer, not fight it.
5. **The +1R / +1.5R / +2R thresholds should be configurable** in `config.py`, not hardcoded.

**Where this gets applied:**
- Training: `src/risk/risk_manager.py` → new `apply_profit_protection()` method called BEFORE `adjust_stop_loss()`
- Live: `live_dashboard/core/trade_executor.py` → `_handle_in_position()` calls protection before AI trail
- Protection overrides AI's sl_adjustment if protection floor is higher

**Edge cases to handle:**
- Trade that gaps through +1R to +2R in one bar → skip to Stage 3 directly
- Spread spike after BE move → buffer must be large enough (use `max(spread, avg_spread * 1.5)`)
- Very small R distance (high conviction = 1.0 ATR SL) → absolute minimum buffer in points

**Verdict: LOCK IN** with the modifications above.

---

### Proposal 2: Trend Direction Improvements

**What ChatGPT proposes:**
- Add trend context features (momentum, alignment, persistence, directional volatility expansion)
- Add regime detection features (trending up/down, ranging, high vol)
- Add re-entry penalty after SL (penalize same-direction re-entry after stop)

**Claude's assessment: PARTIALLY AGREE**

**Already implemented (no action needed):**
- Multi-timeframe alignment: `mtf_alignment`, `htf_momentum` (features exist)
- Regime detection: `corr_gold_usd_100`, `corr_gold_spx_100` (Upgrade 4, already in)
- Session momentum: `session_momentum` (Upgrade 5, already in)
- Trend direction: `h1_trend_dir`, `h4_trend_dir`, `d1_trend_dir` (Group E)

**What's actually missing (root cause of LONG bias):**
The features EXIST but the agent hasn't LEARNED to use them. 170 training weeks may not be enough, or the reward signal is too weak on direction. The March 3 massacre shows the agent ignoring strong bearish multi-timeframe signals.

**Re-entry penalty: AGREE — this is the key missing piece.**
Currently there is NO penalty for: LONG → SL hit → immediate LONG again → SL hit again. The agent just sees two independent negative R1 signals. A specific penalty for same-direction re-entry within N bars of a stop loss would directly address the March 3 behavior.

**Implementation approach for re-entry penalty:**
- Track `last_close_reason` and `last_close_side` and `bars_since_last_close` in trade_env
- If new trade opens same direction within 6 bars of an SL_HIT: apply penalty multiplier to R1 (e.g., 1.5x negative)
- This teaches "if you just got stopped out LONG, think twice before going LONG again immediately"
- Config params: `reentry_penalty_bars: int = 6`, `reentry_penalty_mult: float = 1.5`

**New features to consider adding:**
- `bars_since_last_sl_hit`: How many bars since the last stop loss (normalized to [0,1] over 50 bars). Gives the agent explicit memory of recent failures. Currently the 5 memory features don't capture this specific signal.
- `last_trade_was_sl_long` / `last_trade_was_sl_short`: Binary flags for what just happened. Decays to 0 after N bars.

**Verdict:** Re-entry penalty = LOCK IN. New direction features = DISCUSS FURTHER (may not be needed if re-entry penalty + longer training is sufficient).

---

### Proposal 3: Session Awareness (NY Handling)

**What ChatGPT proposes:**
- Require stronger conviction during high-volatility windows
- Reduce position size during NY
- Require multi-timeframe confirmation

**Claude's assessment: PARTIALLY AGREE — but as live config, not training change**

The agent already has session features (`hour_sin`, `hour_cos`, `session_quality`, `london_ny_overlap`). It SHOULD learn to avoid bad NY entries through negative R1 signal from repeated NY losses.

**Better approach for immediate improvement:**
- **Live config change (not training):** Raise `normal_conviction_threshold` from 0.15 to 0.30 globally. The Week 1 data proves conv < 0.3 is net zero or negative.
- **Optional: session-specific conviction threshold.** E.g., NY session requires conv > 0.40. This is a live execution rule, not a training change.
- **Training approach:** The reward signals should handle this naturally over more training weeks. The agent will learn "NY trades lose more" from R1.

**What should NOT be done:**
- Hardcoded position size reduction by session. The agent's conviction already controls lot sizing. If conviction is low, lots are small.
- Multi-timeframe confirmation as a hard gate. The agent should learn this, not have it forced.

**Verdict:** Raise global conviction threshold to 0.30 in live config = LOCK IN. Session-specific thresholds = DEFER (let training data tell us if needed). No training changes needed for this.

---

### Proposal 4: Trade Hold Improvements

**What ChatGPT proposes:**
- Adjust reward to penalize premature exits
- Reward holding during valid trends

**Claude's assessment: AGREE**

R2 already has `hold_quality = min(hold_bars / 10, 1.0)` which ramps up over 10 bars. But the current min_hold_bars is only 3, and 76% of trades exit at exactly 3 bars.

**Root cause analysis:**
- R2 only fires on trade CLOSE (not per-step). A trade that closes at 3 bars gets `hold_quality = 0.3` but the R:R might be positive, so R2 is still positive.
- R1 fires every step with P/L. If price dips briefly, R1 goes negative, and the agent panics and exits at the first allowed bar.
- The agent hasn't learned that holding through minor dips is better than cutting early.

**Proposed changes:**
1. **Increase min_hold_bars from 3 to 6** (30 min). Simple config change. Forces longer holds, gives more time for trades to develop.
2. **Steepen R2 hold_quality curve.** Change from `min(hold_bars / 10, 1.0)` to `min(hold_bars / 20, 1.0)`. This means full R2 credit only at 20 bars (100 min), making early exits less rewarded.
3. **Add R2 early-exit penalty.** If trade closes profitably but held < 6 bars: `R2 *= 0.5`. Teaches "yes you made money, but you left more on the table."

**Verdict:** Increase min_hold_bars to 6 = LOCK IN. R2 curve steepening = LOCK IN. Early-exit penalty = DISCUSS (might be too aggressive).

---

### Proposal 5: Training System Updates

**What ChatGPT proposes:**
- Add reward for profit protection success
- Penalty for repeat entries after losses
- Trend alignment reward

**Claude's assessment: SELECTIVE AGREE**

The profit protection layer (Proposal 1) is execution-level rules. It doesn't need its own reward component — R1 will naturally be higher when protection prevents givebacks.

Re-entry penalty: Already covered in Proposal 2 analysis. LOCK IN.

Trend alignment reward: This risks reward hacking. The agent could learn to only trade when trends are "aligned" and miss valid counter-trend setups. Better to let R1 (P/L) naturally teach direction.

**What IS needed in training:**
- The profit protection rules must be SIMULATED in the training environment so the agent learns to work with them
- The re-entry penalty must be implemented in the reward function

**Verdict:** Simulate protection in training env = LOCK IN. Re-entry penalty in reward = LOCK IN. Separate trend alignment reward = REJECT.

---

### Proposal 6: Benchmark System (SpartusBench) Updates

**What ChatGPT proposes:**
New metrics: positive excursion retention, giveback ratio, winner-to-loser failure rate, protection activation stats, protected profit contribution.

**Claude's assessment: AGREE — but this is a SEPARATE workstream**

These metrics are valuable for evaluating the profit protection system. They should be added to SpartusBench but don't block training upgrades.

**Key metrics to add:**
1. **Giveback ratio:** `(MFE - final_pnl) / MFE` — how much profit was surrendered
2. **+1R retention rate:** % of trades that reached +1R and closed positive
3. **Protection stage distribution:** How many trades reached each stage
4. **Protected P/L contribution:** Total P/L from trades closed by protection vs agent exit

**Verdict:** AGREE, implement alongside or after training changes. Not a blocker.

---

### Proposal 7: Live Dashboard Updates

**What ChatGPT proposes:**
- Dynamic SL modification logging (PROTECT_BE, PROTECT_LOCK, PROTECT_TRAIL events)
- Per-trade display of protection stage, current R, locked R

**Claude's assessment: AGREE**

This flows naturally from Proposal 1. The live `trade_executor.py` already has the infrastructure:
- `_initial_sl` tracked at entry
- `_max_favorable` updated each bar
- `_bridge.modify_position()` for MT5 SL changes
- Logging to trades.jsonl and alerts.jsonl

**Additional logging needed:**
- Protection stage transitions logged to actions.jsonl
- Per-trade: `initial_r`, `current_r`, `max_r`, `locked_r`, `protection_stage`

**Verdict:** LOCK IN as part of Proposal 1 implementation.

---

## Locked-In Changes

### 1. Profit Protection Layer (Rule-Based Staged SL)

**Rationale:** 24 losing trades had >2pt favorable excursion. Agent hasn't learned to trail effectively. Rule-based safety net needed now.

**Specification:**
```
Stage 0 — Initial Entry
  SL = max(2.5 - conviction, 1.0) * ATR  (existing logic, unchanged)
  R = abs(entry_price - initial_sl)  (reference for all stages)

Stage 1 — Breakeven (+1.0R)
  When MFE >= 1.0 * R:
    new_sl = entry_price + buffer
    buffer = max(spread_points * point, 0.5 * point)  # cover spread + tiny lock

Stage 2 — Profit Lock (+1.5R)
  When MFE >= 1.5 * R:
    new_sl = entry_price + 0.5 * R  (locks +0.5R guaranteed)

Stage 3 — ATR Trail (+2.0R)
  When MFE >= 2.0 * R:
    trail_distance = max(1.0 * ATR, broker_min_sl_distance)
    new_sl = current_price - trail_distance  (for LONG)
    SL can only tighten from this point

AI Interaction:
  - AI's sl_adjustment action still operates but cannot move SL below protection floor
  - AI's exit_urgency can close at any time (but locked profit is preserved)
  - Protection floor = max(protection_sl, ai_proposed_sl)
```

**Config params (new in config.py):**
```python
# Profit Protection
protection_be_trigger_r: float = 1.0
protection_be_buffer_pips: float = 0.5
protection_lock_trigger_r: float = 1.5
protection_lock_amount_r: float = 0.5
protection_trail_trigger_r: float = 2.0
protection_trail_atr_mult: float = 1.0
```

**Files to modify:**
- `src/risk/risk_manager.py` — Add `apply_profit_protection()` method
- `src/environment/trade_env.py` — Call protection in step(), track stage in position dict
- `src/config.py` — Add protection config params
- `live_dashboard/core/trade_executor.py` — Mirror protection in `_handle_in_position()`
- `live_dashboard/core/risk_manager.py` — Add same method
- `live_dashboard/config/live_config.py` — Add protection config

### 2. Re-Entry Penalty (Same Direction After SL Hit)

**Rationale:** March 3 NY massacre: 5 consecutive LONGs into a sell-off. No existing penalty for this behavior.

**Specification:**
```
Track in trade_env:
  last_sl_side: str = None    # "LONG" or "SHORT"
  last_sl_step: int = 0       # step when last SL hit occurred

On SL_HIT:
  last_sl_side = position["side"]
  last_sl_step = current_step

On new trade open:
  bars_since_sl = current_step - last_sl_step
  if (new_side == last_sl_side) and (bars_since_sl < reentry_penalty_bars):
    Apply R1 penalty multiplier: R1 *= reentry_penalty_mult (1.5x negative amplification)
    Clear: last_sl_side = None  (only penalize once)
```

**Config params:**
```python
reentry_penalty_bars: int = 6       # Window: 30 min on M5
reentry_penalty_mult: float = 1.5   # 50% amplified loss signal
```

**Files to modify:**
- `src/environment/trade_env.py` — Track last SL, apply penalty in reward calc
- `src/config.py` — Add reentry config params

### 3. Increased Minimum Hold Time

**Rationale:** 76% of trades exit at min_hold_bars (3 bars / 15 min). Agent hasn't learned patience.

**Change:** `min_hold_bars: 3 → 6` (15 min → 30 min)

**Files to modify:**
- `src/config.py` — Change default
- `live_dashboard/config/live_config.py` — Change default
- `live_dashboard/config/default_config.yaml` — Change default

### 4. Steeper R2 Hold Quality Curve

**Rationale:** Current `hold_quality = min(hold_bars / 10, 1.0)` gives 30% credit at 3 bars. Need stronger incentive to hold.

**Change:** `hold_quality = min(hold_bars / 20, 1.0)`
- 3 bars → 15% credit (was 30%)
- 6 bars → 30% credit (was 60%)
- 10 bars → 50% credit (was 100%)
- 20 bars → 100% credit (was 100% at 10)

**Files to modify:**
- `src/environment/reward.py` — Change denominator in R2 calc

### 5. Global Conviction Threshold Increase (Live Only)

**Rationale:** Week 1 data proves conv < 0.3 is net zero or negative. Conv 0.3-0.5 is the worst bucket (-83.32).

**Change:** `normal_conviction_threshold: 0.15 → 0.30`

**Files to modify:**
- `live_dashboard/config/live_config.py` — Change default
- `live_dashboard/config/default_config.yaml` — Change default

### 6. Protection Logging (Live Dashboard)

**Rationale:** Transparency for monitoring protection system effectiveness.

**Additions:**
- Log protection stage transitions to alerts.jsonl
- Add to trades.jsonl: `initial_r`, `max_r_reached`, `protection_stage_max`, `locked_r`
- Log format: `PROTECT_BE trade=143 stage=1 locked=0.0R` etc.

**Files to modify:**
- `live_dashboard/core/trade_executor.py` — Add protection logging
- `live_dashboard/utils/logger.py` — No changes needed (uses existing log_alert)

---

## Rejected / Deferred Ideas

### Rejected: Separate Trend Alignment Reward Component
**Reason:** Risks reward hacking. Agent could learn to only trade "aligned" trends and miss valid counter-trend setups. R1 (P/L) already naturally teaches direction through profit/loss.

### Rejected: Hardcoded Position Size Reduction by Session
**Reason:** Agent's conviction already controls lot sizing. If conviction is low, lots are small. Adding a separate session multiplier adds complexity without clear benefit.

### Rejected: Multi-Timeframe Confirmation as Hard Gate
**Reason:** The agent should LEARN this relationship, not have it forced. Hardcoded gates reduce the agent's ability to find novel patterns.

### Deferred: Session-Specific Conviction Thresholds
**Reason:** Global threshold increase to 0.30 addresses the immediate problem. Session-specific thresholds may be needed later based on Week 2 data but add complexity now.

### Deferred: New Direction/Regime Features
**Reason:** The agent already has 67 features including multi-timeframe trends, regime correlations, and session microstructure. The problem is the agent hasn't LEARNED to use them, not that the features are missing. Re-entry penalty + longer training should help first. If direction accuracy stays poor after 50+ more training weeks, revisit.

### Deferred: Additional Memory Feature (`bars_since_last_sl_hit`)
**Reason:** The re-entry penalty in the reward function achieves the same goal without adding observation dimensions. If the penalty isn't sufficient, this feature can be added later.

### Deferred: SpartusBench Metrics Update
**Reason:** Valuable but doesn't block training. Can be implemented alongside or after the training changes.

---

## Implementation Checklist

> When the plan is approved, this becomes the build checklist.

- [x] All changes documented and approved
- [x] **Training code:**
  - [x] Profit protection in `src/risk/risk_manager.py` — `apply_profit_protection()` method
  - [x] Protection simulation in `src/environment/trade_env.py` — called before `adjust_stop_loss()`
  - [x] Re-entry penalty in `src/environment/trade_env.py` + `reward.py` — tracked at open, amplified in R1
  - [x] R2 hold quality curve steepened in `reward.py` — denominator 10→20
  - [x] Config changes in `src/config.py` — protection params + reentry params + min_hold=6
  - [x] min_hold_bars 3→6
- [x] **Live dashboard:**
  - [x] Profit protection in `live_dashboard/core/trade_executor.py` — `_handle_in_position()` calls protection
  - [x] Protection in `live_dashboard/core/risk_manager.py` — `apply_profit_protection()` method
  - [x] Config changes in live_config.py + default_config.yaml
  - [x] Conviction threshold 0.15→0.30
  - [x] Protection logging — stage transitions to alerts.jsonl + fields in trades.jsonl
  - [x] min_hold_bars 3→6
- [x] **Testing:**
  - [x] Verify protection stages trigger correctly in training env
  - [x] Verify re-entry penalty fires on consecutive same-direction SL hits
  - [x] Verify protection SL floor overrides AI sl_adjustment
  - [x] Verify live MT5 SL modification works with protection (code path verified: protection→AI trail→modify_position)
  - [x] Anti-leakage test still passes
- [x] **Delete old caches + checkpoints before training** (features/, models/, training_state.json, logs/, memory DB all cleared)

---

## Notes / Discussion Log

### 2026-03-06 — Initial Setup
- Created this document based on Week 1 live demo results
- Previous Feature_Upgrade.md (Part A: 5 upgrades) is FULLY IMPLEMENTED
- Previous Part B (progressive training roadmap) is OUTDATED and superseded by this plan

### 2026-03-06 — ChatGPT's First Proposal + Claude's Analysis
- ChatGPT proposed 7 upgrade areas
- Claude analyzed each against the existing codebase
- **Locked in 6 changes** (protection layer, re-entry penalty, min hold, R2 curve, conviction threshold, protection logging)
- **Rejected 3 ideas** (trend alignment reward, session position sizing, MTF hard gate)
- **Deferred 4 ideas** (session-specific thresholds, new direction features, bars_since_sl feature, SpartusBench metrics)
- Key insight: Most proposed "new features" already exist in the system — the problem is the agent hasn't learned to use them in 170 weeks. Re-entry penalty and profit protection address the two biggest behavioral failures directly.
- **ChatGPT approved all 6 changes, confirmed 3 rejections.**

### 2026-03-06 — ChatGPT Final Approval
- All 6 locked-in changes confirmed by ChatGPT
- 3 rejections confirmed (trend alignment reward, session position sizing, MTF hard gate)
- **Optional proposal from ChatGPT (loss-streak directional dampening):** REJECTED by Claude, confirmed by ChatGPT. Reason: re-entry penalty already solves the same problem through reward signal (proper RL channel), dampening would create hidden state the agent can't observe, and it reduces lot size instead of preventing the trade.
- **Plan is now APPROVED and ready for implementation. All 6 changes locked, no open items.**
