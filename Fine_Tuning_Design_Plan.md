# Spartus Training Engine v3.4 — Dashboard Upgrade Plan

**Created:** 2026-03-07
**Status:** READY TO IMPLEMENT
**Spec Version:** v3.4 (extends v3.3.2)

---

## Part A: Problems to Fix

1. **Training auto-starts.** When you launch the dashboard, training begins immediately
   in a background thread (`train.py` lines 91-96). There is no "Start" button — only
   Pause and Quit. You cannot choose what to do before training runs.

2. **No model selection.** The only way to load a model is `--resume` CLI flag, which
   always loads `spartus_latest.zip`. You cannot pick a specific checkpoint to resume
   from. If you forget `--resume`, all progress is lost.

3. **No Start/Resume distinction in UI.** The dashboard has no way to choose between
   starting fresh or resuming. This is a CLI-only flag that's easy to forget.

---

## Part B: Asymmetric Loss Penalty

The one reward improvement to apply alongside the dashboard upgrade.

**What it does:** Punish losing trades harder than rewarding winning trades. When the
model loses money, multiply the R1 penalty by 1.5x. This teaches the model to avoid
bad trades more aggressively than it seeks good ones.

**Validated against our system:**
- 3-line change in `reward.py` `_calc_r1()` method (line 238)
- No conflict with existing R2-R5 components, reward normalizer, or entropy floor
- Compatible with existing re-entry penalty (they stack — re-entry losses get both
  the 1.5x loss penalty AND the 1.5x re-entry penalty)
- Asymmetric rewards are standard practice in financial RL (supported by multiple
  papers: Risk-Adjusted DRL, Self-Rewarding DRL, Risk-Aware RL)

**Expected impact:** +3-5% win rate improvement. The model learns to be more
selective about entries, reducing the WRONG_DIRECTION rate (currently 47.8% of trades).

**Implementation:**

```python
# In RewardCalculator._calc_r1() — src/environment/reward.py line 238:
def _calc_r1(self, equity_return, position, trade_result=None):
    if position is None and trade_result is None:
        return 0.0
    r1 = equity_return * 500.0
    # Asymmetric loss penalty: punish losses harder than rewarding wins
    if r1 < 0:
        r1 *= self.cfg.loss_penalty_mult  # Default 1.5
    # Re-entry penalty (existing)
    if trade_result and trade_result.get("is_reentry", False):
        if r1 < 0:
            r1 *= self.cfg.reentry_penalty_mult
    return r1
```

```python
# In TrainingConfig — src/config.py:
loss_penalty_mult: float = 1.5   # Asymmetric loss penalty multiplier
```

---

## Part C: Dashboard Changes

### C1. New Training Flow

**Current flow** (`train.py` lines 91-96):
```
Launch → Training starts immediately → Dashboard shows progress
```

**New flow:**
```
Launch → Dashboard shows IDLE state → User clicks Start or Resume → Training begins
```

The dashboard launches in IDLE state. The header shows "IDLE — Select action to begin"
instead of week/step info. The user clicks either Start (fresh) or Resume (from
checkpoint). Only then does the training thread start.

### C2. Header Layout Changes

**Current header** (`qt_dashboard.py` lines 204-245):
```
[Title "SPARTUS TRAINING ENGINE v3.3"] [Info label (week/step/stage)] [Health badge] [Pause btn] [Quit btn]
```

**New header:**
```
[Title "SPARTUS TRAINING ENGINE v3.4"] [Info label] [Health badge] [Start btn] [Resume btn] [Pause btn] [Quit btn]
```

- **Start button** (green): Creates a fresh model and begins training from week 0.
  Disabled while training is running. Shows confirmation dialog: "This will create
  a new model from scratch. Continue?"
- **Resume button** (blue): Opens a dropdown of available checkpoints in
  `storage/models/`. User selects one, training resumes from that point. Disabled
  while training is running.
- **Pause button** (existing): Unchanged. Disabled in IDLE state.
- **Quit button** (existing): Unchanged. Works in all states.

When training starts, Start and Resume disable. Pause enables. Info label switches
from "IDLE" to showing week/step/convergence as it does now.

### C3. Resume Model Selection

When the user clicks Resume, a dialog appears listing available models:

```
+------------------------------------------+
|  Select Model to Resume                  |
|  ----------------------------------------|
|  spartus_latest.zip    (Week 67, 10 MB)  |
|  spartus_best.zip      (Week 60, 10 MB)  |
|  spartus_week_0060.zip (Week 60, 10 MB)  |
|  spartus_week_0050.zip (Week 50, 10 MB)  |
|  spartus_week_0040.zip (Week 40, 10 MB)  |
|  ... (all .zip files in storage/models/) |
|  ----------------------------------------|
|  [Cancel]                      [Resume]  |
+------------------------------------------+
```

The dialog reads the model directory and lists all `.zip` files with their file size
and modification date. The user picks one and clicks Resume. The trainer loads that
model via `SAC.load()` and continues training.

### C4. State Machine

The dashboard operates in three states:

```
IDLE  →(Start)→  RUNNING  →(Pause)→  PAUSED
  ^                  |                   |
  |    (Finished)    v     (Resume)      v
  +←←←←←←←←←←←←←←←←+←←←←←←←←←←←←←←←←←+
```

| State | Start | Resume | Pause | Quit | Info Label |
|---|---|---|---|---|---|
| IDLE | Enabled | Enabled | Disabled | Enabled | "IDLE — Ready" |
| RUNNING | Disabled | Disabled | Enabled | Enabled | "W067 | Step 12345 | IMPROVING" |
| PAUSED | Disabled | Disabled | Shows "Resume" | Enabled | "PAUSED at W067" |

When training finishes (all weeks complete) or stops due to error, state returns to
IDLE. Start and Resume re-enable so the user can begin another run.

---

## Part D: Code Changes

### D1. `scripts/train.py`

**Remove** the auto-start training thread (lines 91-96). Instead, pass the training
configuration to the dashboard and let it start training on user command.

```python
# New flow — dashboard starts first, training waits for user
if has_qt:
    app = QApplication(sys.argv)
    from PyQt6.QtGui import QFont
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    dashboard = SpartusQtDashboard(cfg, shared_metrics)

    # Give dashboard the ability to start training
    dashboard.set_training_launcher(
        train_fn=_run_training_thread,
        cfg=cfg,
        shared_metrics=shared_metrics,
        max_weeks=args.weeks,
        seed=args.seed,
    )

    dashboard.show()
    app.exec()
```

The `--resume` CLI flag still works for headless mode (`--no-dashboard`). When using
the dashboard, the UI controls replace the CLI flag.

### D2. `src/training/qt_dashboard.py`

Changes to existing code:

1. **Header**: Add Start and Resume buttons alongside existing Pause and Quit.
2. **State management**: Track IDLE/RUNNING/PAUSED state. Enable/disable buttons
   based on state.
3. **Resume dialog**: QDialog with QListWidget showing available model files.
4. **Training launcher**: Method that creates and starts the training thread.
   Called when user clicks Start or Resume.

```python
def _start_fresh(self):
    """User clicked Start — begin fresh training."""
    reply = QMessageBox.question(
        self, "Start Fresh Training",
        "This will create a new model from scratch.\nContinue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    if reply != QMessageBox.StandardButton.Yes:
        return
    self._launch_training(resume=False, model_path=None)

def _start_resume(self):
    """User clicked Resume — show model picker and resume."""
    model_path = self._show_model_picker()
    if model_path is None:
        return  # User cancelled
    self._launch_training(resume=True, model_path=model_path)

def _launch_training(self, resume: bool, model_path: Optional[str]):
    """Start the training thread."""
    if model_path:
        self.shared_metrics["_resume_model_path"] = model_path
    self._train_thread = threading.Thread(
        target=self._train_fn,
        args=(self._cfg, self._shared_metrics,
              self._max_weeks, resume, self._seed),
        daemon=False,
    )
    self._train_thread.start()
    self._set_state("RUNNING")
```

Estimated new code: ~150-200 lines (buttons, dialog, state management).

### D3. `src/training/trainer.py`

One small change: support loading a specific model path instead of always loading
`spartus_latest.zip`.

```python
def _load_training_state(self):
    """Resume from saved training state."""
    # Allow override of model path (from dashboard model picker)
    model_path = self.shared_metrics.get(
        "_resume_model_path",
        self.cfg.model_dir / "spartus_latest.zip"
    )
    # ... rest of existing load logic unchanged
```

This is a ~5 line change to the existing `_load_training_state()` method.

### D4. `src/environment/reward.py`

Add asymmetric loss penalty to `_calc_r1()`:

```python
if r1 < 0:
    r1 *= self.cfg.loss_penalty_mult
```

3 lines.

### D5. `src/config.py`

Add one field:

```python
loss_penalty_mult: float = 1.5
```

1 line.

---

## Part E: File Changes Summary

| File | Change | Lines |
|---|---|---|
| `scripts/train.py` | Remove auto-start, pass launcher to dashboard | ~20 modified |
| `src/training/qt_dashboard.py` | Add Start/Resume buttons, model picker dialog, state machine | ~200 new |
| `src/training/trainer.py` | Support custom model path in `_load_training_state()` | ~5 modified |
| `src/environment/reward.py` | Asymmetric loss penalty in `_calc_r1()` | ~3 new |
| `src/config.py` | Add `loss_penalty_mult` field | ~1 new |

**Total: ~230 lines of new/modified code.**

---

## Part F: Implementation Order

1. Add `loss_penalty_mult` to config
2. Add asymmetric loss penalty to reward.py
3. Modify train.py to not auto-start when dashboard is active
4. Add Start/Resume buttons to dashboard header
5. Add model picker dialog
6. Add state machine (IDLE/RUNNING/PAUSED)
7. Test: launch dashboard, verify IDLE state, click Start, verify training runs
8. Test: launch dashboard, click Resume, pick checkpoint, verify resume works

---

## Part G: After Implementation

1. Stop current training (Quit button)
2. Launch upgraded dashboard
3. Click Resume → select `spartus_latest.zip` (Week 67)
4. Training continues from W67 with asymmetric loss penalty active
5. Let it run through remaining ~133 weeks
6. Evaluate results — if single model hits 52-55% WR consistently, consider
   ensemble as a future v3.5 upgrade

---

## Sources

- [Self-Rewarding DRL for Trading](https://www.mdpi.com/2227-7390/12/24/4020) — MDPI Mathematics 2024
- [Risk-Adjusted DRL: Multi-Reward Approach](https://link.springer.com/article/10.1007/s44196-025-00875-8) — Springer 2025
- [Risk-Aware RL Reward for Trading](https://arxiv.org/html/2506.04358v1) — arXiv 2025
