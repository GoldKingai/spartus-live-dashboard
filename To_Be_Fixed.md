# To Be Fixed — Training System Issues

**Generated:** 2026-03-08 (Fresh Training, Week 11)
**Status:** ALL IMPLEMENTED — restart training to apply

---

## FIX-1: Dead Zone — Model Wastes Time in Untradeable Episodes

**Severity:** HIGH
**Evidence:**
- At Week 10, balance dropped to £49.81 (below 50% bankruptcy threshold)
- The training appeared "stuck" for several minutes — timesteps advanced but no trades executed
- Balance was unchanged across 80+ consecutive log entries (ts=109920 to ts=110000)
- Training log: 33 trades, balance frozen at 49.81, no new trades being opened

**Root Cause:**
When balance drops significantly mid-week, drawdown exceeds `max_dd` (10% of peak).
The risk manager returns 0.0 lots for every trade attempt (line 66-67 in `risk_manager.py`):
```python
dd = (peak_balance - balance) / peak_balance
if dd >= self.cfg.max_dd:  # 10%
    return 0.0
```

The env's emergency stop at line 298 (`trade_env.py`) sets `done = True` when DD >= max_dd,
but only when the env is CHECKED — which requires the step to evaluate the DD condition.
With DummyVecEnv, the env auto-resets on `done=True`, and the bankruptcy check in
`env.reset()` (line 163) resets balance to initial if below 50%.

However, between the emergency stop and the reset, there's a period where:
1. The env resets balance to £100 (bankruptcy reset)
2. But peak_balance stays at old high (£100.57) via `max(self.peak_balance, self.balance)`
3. DD = (100.57 - 100) / 100.57 = 0.57% — this is actually fine
4. The model can trade again after reset

The REAL waste happens in the trainer level: `trainer._train_week()` calls
`model.learn(total_timesteps=steps_per_week)`. During those steps, if the env
hits bankruptcy repeatedly, it resets and continues — but the outer trainer state
(balance, peak_balance) only syncs AFTER `model.learn()` completes (line 315).

The visible "stuck" behavior is actually the env cycling through:
reset → trade → lose → emergency stop → reset → repeat
...all within one `model.learn()` call. The trainer balance appears frozen because
it only updates between weeks.

**What actually happened:** Training WAS working — it just looked stuck in the
monitoring output. Week 10 completed, bankruptcy handler fired, balance reset to £100,
and Week 11 started with 87+ trades and healthy activity.

**Optimization (not a bug fix):**
Add early episode truncation when the env detects it's in an untradeable state.
Instead of stepping through hundreds of bars where DD > max_dd and no trades can open,
truncate the episode immediately:

```python
# In trade_env.py step(), after emergency stop checks:
# If no position and DD > max_dd, truncate immediately
# (the env will just cycle reset → emergency stop → reset otherwise)
if not self.position and current_dd >= self.cfg.max_dd:
    truncated = True  # End episode, auto-reset will fix balance
```

This saves wasted computation — instead of processing 500+ bars doing nothing,
the env immediately resets and starts a productive episode.

**Files to modify:**
- `src/environment/trade_env.py`: Add early truncation when flat + DD > max_dd

**Impact:** Faster training (no wasted steps), cleaner logs, no more "appears stuck" behavior.
Does NOT change learning outcomes — the model gets the same -4.0 penalty and reset either way.

---

## FIX-2: Validation Runs on ALL 87 Weeks Every 5 Training Weeks (~2 Hours Each)

**Severity:** CRITICAL (training speed)
**Evidence:**
- Validation runs every 5 training weeks (`validation_interval = 5`)
- FIX-8 from previous run changed validation to use ALL val weeks (87 weeks)
- Each validation run: 87 weeks × ~1,400 M5 bars = ~121,800 inference steps
- Measured from alerts.log: each validation takes **~1.1-1.3 hours**
- Over a full 408-week training run: ~82 validation runs × 1.1 hours = **~90 hours of validation**
- Pure training time at 15.8 sps: ~9 hours for 50 weeks
- After 11.4 hours wall time, only ~5 hours was actual training (rest was validation)
- **Actual total time: ~120+ hours (5+ days), not the 30 hours the ETA shows**
- The ETA calculation does NOT account for validation time — it only counts training steps

**Root Cause:**
The previous fix (FIX-8) correctly changed validation from 3 random weeks to ALL val weeks
for stable best-model selection. But the interaction with `validation_interval = 5` was not
considered. The old 3-week validation took ~4 minutes — fine every 5 weeks. The new 87-week
validation takes ~2 hours — catastrophic every 5 weeks.

**User-visible symptom:**
Training appears to "freeze" for ~2 hours every 5 training weeks. Step counter stops
advancing, balance doesn't change, no trades executed. Looks like a crash but it's just
validation running in the foreground with no progress indication.

**Fix:**
```python
# In src/config.py:
validation_interval: int = 20    # Was 5, validate every 20 weeks instead

# Also consider capping val weeks to reduce per-run cost:
# In src/training/trainer.py _validate():
max_val_weeks: int = 30    # Use 30 random val weeks instead of all 87
```

**Option A (simple — recommended):** Change `validation_interval` from 5 to 20.
- Validations drop from ~82 to ~20 over a full run
- Total val time: ~20 × 2.1 = ~42 hours (saves ~130 hours)
- Still enough validation points for convergence tracking
- Training total: ~72 hours (3 days) instead of ~200 hours (8+ days)

**Option B (aggressive):** Also cap val weeks at 30 (randomly sampled).
- Each validation: 30 × 1,400 / 16 = ~44 minutes (instead of 2.1 hours)
- With interval=20: ~20 × 0.73 = ~15 hours total validation
- Training total: ~45 hours (2 days)
- Slightly noisier best-model selection, but still far better than old 3-week random

**Option C (both):** Interval=20 + cap at 30 val weeks = best of both worlds.

**Files to modify:**
- `src/config.py`: `validation_interval: 5 → 20`
- `src/training/trainer.py` `_validate()`: Optionally cap val weeks with
  `val_subset = rng.choice(self._val_weeks, size=min(30, len(self._val_weeks)), replace=False)`

**Impact:** Reduces training time from ~8 days to ~2-3 days with no meaningful loss in
model selection quality. The model is validated less frequently but on the same data,
so the best checkpoint is still selected on a meaningful metric.

---

## IMPLEMENTATION NOTES

- These fixes require stopping training and restarting
- Delete feature caches if feature_builder.py changes (not needed for these fixes)
- Old checkpoints remain compatible (no obs_dim change)

---

## SUMMARY TABLE

| Fix | Issue | Severity | Type | Status |
|-----|-------|----------|------|--------|
| FIX-1 | Dead zone — wasted steps when untradeable | HIGH | Performance optimization | DONE |
| FIX-2 | Validation too frequent + too many weeks (~2hr each) | CRITICAL | Training speed | DONE |
