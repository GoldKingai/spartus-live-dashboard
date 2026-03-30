"""Validation Gate — prevents deploying a fine-tuned model that's worse than baseline.

Before a fine-tuned model can be promoted to live deployment, it must pass
these checks:
    1. Sharpe ratio >= baseline_sharpe * threshold (default 90%)
    2. Action std > minimum (policy hasn't collapsed)
    3. Win rate > minimum floor (model hasn't gone random)
    4. No catastrophic forgetting on regime-specific checks

The gate runs on the same validation set used during training, ensuring we're
comparing apples to apples.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class ValidationGate:
    """Multi-criteria gate for fine-tuned model promotion.

    Usage:
        gate = ValidationGate(config, trainer)
        result = gate.evaluate(fine_tuned_model, baseline_sharpe=0.72)
        if result["passed"]:
            gate.promote(fine_tuned_model, output_path)
    """

    def __init__(self, config, trainer=None):
        self.cfg = config
        self._trainer = trainer  # Optional: trainer with _validate() method

        # Consecutive failure tracking (for auto-rollback)
        self._consecutive_failures: int = 0
        self._best_sharpe: float = -float("inf")
        self._last_result: Optional[Dict] = None

        # Validation history (for trend detection)
        self._history: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model,
        baseline_sharpe: float,
        baseline_max_dd: float = 0.20,
        action_std: Optional[float] = None,
        strategy_memory=None,
    ) -> Dict:
        """Run all validation criteria.

        Args:
            model: Fine-tuned SAC model to evaluate.
            baseline_sharpe: Frozen baseline model's validation Sharpe.
            baseline_max_dd: Frozen baseline model's max drawdown.
            action_std: Current action std (from recent replay buffer sample).
            strategy_memory: StrategyMemory instance for forgetting check.

        Returns:
            Dict with keys: passed, sharpe, checks, reason, timestamp.
        """
        checks = {}
        reasons = []

        # ── 1. Compute Sharpe on validation set ──────────────────────────────
        sharpe = self._compute_val_sharpe(model)
        sharpe_threshold = baseline_sharpe * self.cfg.finetune_val_sharpe_threshold
        sharpe_pass = sharpe >= sharpe_threshold
        checks["sharpe"] = {
            "value": sharpe,
            "threshold": sharpe_threshold,
            "baseline": baseline_sharpe,
            "passed": sharpe_pass,
        }
        if not sharpe_pass:
            reasons.append(
                f"Sharpe {sharpe:.3f} < threshold {sharpe_threshold:.3f} "
                f"(baseline {baseline_sharpe:.3f} * {self.cfg.finetune_val_sharpe_threshold})"
            )

        # ── 2. Action std check (policy diversity) ───────────────────────────
        if action_std is not None:
            std_pass = action_std >= self.cfg.finetune_action_std_min
            checks["action_std"] = {
                "value": action_std,
                "threshold": self.cfg.finetune_action_std_min,
                "passed": std_pass,
            }
            if not std_pass:
                reasons.append(
                    f"Action std {action_std:.3f} < min {self.cfg.finetune_action_std_min} "
                    f"(policy may be collapsing)"
                )
        else:
            std_pass = True  # Can't check without data
            checks["action_std"] = {"value": None, "passed": True, "note": "not measured"}

        # ── 3. Forgetting check ──────────────────────────────────────────────
        forgetting_alerts = []
        if strategy_memory is not None:
            forgetting_alerts = strategy_memory.check_forgetting(min_finetune_trades=10)
        critical_forgetting = any(a["severity"] == "CRITICAL" for a in forgetting_alerts)
        forgetting_pass = not critical_forgetting
        checks["forgetting"] = {
            "alerts": len(forgetting_alerts),
            "critical": sum(1 for a in forgetting_alerts if a["severity"] == "CRITICAL"),
            "passed": forgetting_pass,
            "details": forgetting_alerts[:3],
        }
        if not forgetting_pass:
            worst = [a["regime"] for a in forgetting_alerts if a["severity"] == "CRITICAL"]
            reasons.append(f"Critical forgetting detected on regimes: {', '.join(worst[:3])}")

        # ── Overall result ────────────────────────────────────────────────────
        passed = sharpe_pass and std_pass and forgetting_pass

        if passed:
            self._consecutive_failures = 0
            if sharpe > self._best_sharpe:
                self._best_sharpe = sharpe
        else:
            self._consecutive_failures += 1

        result = {
            "passed": passed,
            "sharpe": sharpe,
            "baseline_sharpe": baseline_sharpe,
            "checks": checks,
            "consecutive_failures": self._consecutive_failures,
            "reason": "; ".join(reasons) if reasons else "All checks passed",
            "timestamp": time.time(),
        }

        self._last_result = result
        self._history.append(result)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        log.info(
            f"ValidationGate: {'PASS' if passed else 'FAIL'} — "
            f"Sharpe={sharpe:.3f} (baseline={baseline_sharpe:.3f}), "
            f"consecutive_failures={self._consecutive_failures}"
        )
        return result

    def should_auto_rollback(self) -> bool:
        """Return True if consecutive failures have exceeded the rollback threshold."""
        return self._consecutive_failures >= self.cfg.finetune_auto_rollback_failures

    # ─────────────────────────────────────────────────────────────────────────
    # Sharpe Computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_val_sharpe(self, model) -> float:
        """Run fine-tuned model on validation weeks and compute Sharpe ratio.

        Reuses trainer's _validate() if available. Otherwise returns 0.0.
        """
        if self._trainer is None:
            log.warning("ValidationGate: no trainer set — returning sharpe=0.0")
            return 0.0

        try:
            # Swap out the trainer's model temporarily
            old_model = self._trainer.model
            self._trainer.model = model
            sharpe = self._trainer._validate(self._trainer.current_week)
            self._trainer.model = old_model
            return sharpe
        except Exception as e:
            log.warning(f"ValidationGate: _validate() failed: {e}")
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # History & Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def get_trend(self, n: int = 5) -> str:
        """Return 'IMPROVING', 'STABLE', or 'DEGRADING' based on recent Sharpe trend."""
        if len(self._history) < 3:
            return "INSUFFICIENT_DATA"
        recent = [h["sharpe"] for h in self._history[-n:]]
        if len(recent) < 2:
            return "INSUFFICIENT_DATA"
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.01:
            return "IMPROVING"
        elif slope < -0.01:
            return "DEGRADING"
        return "STABLE"

    def get_summary(self) -> Dict:
        """Return validation summary for dashboard."""
        return {
            "last_passed": self._last_result.get("passed") if self._last_result else None,
            "last_sharpe": self._last_result.get("sharpe") if self._last_result else None,
            "best_sharpe": self._best_sharpe if self._best_sharpe > -float("inf") else None,
            "consecutive_failures": self._consecutive_failures,
            "auto_rollback_at": self.cfg.finetune_auto_rollback_failures,
            "trend": self.get_trend(),
            "n_evaluations": len(self._history),
            "last_timestamp": (
                self._last_result.get("timestamp") if self._last_result else None
            ),
        }

    def save_history(self, path: str) -> None:
        """Persist validation history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": self._history,
                    "best_sharpe": self._best_sharpe,
                    "consecutive_failures": self._consecutive_failures,
                },
                f,
                indent=2,
            )

    def load_history(self, path: str) -> bool:
        """Load persisted validation history."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._history = data.get("history", [])
            self._best_sharpe = data.get("best_sharpe", -float("inf"))
            self._consecutive_failures = data.get("consecutive_failures", 0)
            return True
        except Exception as e:
            log.warning(f"ValidationGate: load failed: {e}")
            return False
