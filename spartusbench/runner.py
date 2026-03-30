"""Main benchmark orchestrator for SpartusBench.

Coordinates all tiers, detectors, scoring, persistence, and reporting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .database import BenchmarkDB
from .detectors.aggression import detect_aggression_drift
from .detectors.collapse import detect_conviction_collapse
from .detectors.fragility import detect_stress_fragility
from .detectors.overfitting import detect_overfitting
from .detectors.reward_hacking import detect_reward_hacking
from .discovery import (
    discover_and_split_weeks, load_model_for_benchmark,
    resolve_model_path, validate_feature_caches,
)
from .hashing import (
    compute_config_hash, compute_data_manifest_hash,
    compute_feature_hash, compute_model_file_hash,
    compute_result_hash, compute_split_hash,
)
from .scoring import (
    check_hard_fails, check_side_warnings,
    compute_spartus_score, evaluate_champion_candidacy,
)
from .tiers.t1_validation import run_t1
from .tiers.t2_stress import run_t2
from .tiers.t3_regime import run_t3
from .tiers.t4_churn import run_t4
from .tiers.t5_reward import run_t5
from .tiers.t6_gating import run_t6, compute_action_stats
from .tiers.t7_locked import run_t7
from .types import BenchmarkResult, EvalBundle

log = logging.getLogger("spartusbench.runner")

SUITES = {
    "full": ["T1", "T2", "T3", "T4", "T5", "T6"],
    "validation_only": ["T1"],
    "stress_only": ["T2"],
    "locked_test": ["T7"],
}


class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db = BenchmarkDB(db_path)

    def run(
        self,
        model_ref: str,
        suite: str = "full",
        seed: int = 42,
        confirm_test: bool = False,
        generate_plots: bool = True,
        compare_vs_champion: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark.

        Args:
            model_ref: Model reference (e.g., "W0170", "best", or path)
            suite: Suite name ("full", "validation_only", "stress_only", "locked_test")
            seed: RNG seed for reproducibility
            confirm_test: Required for locked_test suite
            generate_plots: Whether to generate charts
            compare_vs_champion: Auto-compare vs champion after run
            progress_callback: Optional (tier, step, total) -> None

        Returns:
            Complete BenchmarkResult
        """
        if suite not in SUITES:
            raise ValueError(f"Unknown suite: {suite}. Choose from {list(SUITES.keys())}")

        if suite == "locked_test" and not confirm_test:
            raise ValueError(
                "Locked test requires confirm_test=True. "
                "Test set evaluation is permanently recorded."
            )

        # Load model
        log.info("=" * 60)
        log.info("SpartusBench v1.0.0")
        log.info("=" * 60)

        bundle = load_model_for_benchmark(model_ref)
        log.info("Model:  %s (%s)", bundle.model_id, bundle.model_path)
        log.info("Suite:  %s (%s)", suite, ", ".join(SUITES[suite]))
        log.info("Seed:   %d", seed)

        # Validate feature caches
        target_weeks = bundle.test_weeks if suite == "locked_test" else bundle.val_weeks
        valid_weeks, warnings = validate_feature_caches(target_weeks)
        for w in warnings:
            log.warning("Feature cache: %s", w)
        if not valid_weeks:
            raise RuntimeError("No valid feature caches found for evaluation weeks")

        if suite == "locked_test":
            bundle.test_weeks = valid_weeks
        else:
            bundle.val_weeks = valid_weeks

        # Compute hashes
        all_weeks = bundle.val_weeks + bundle.test_weeks
        train_weeks_estimate = list(range(0, min(bundle.val_weeks) - 2)) if bundle.val_weeks else []

        data_hash = compute_data_manifest_hash(all_weeks)
        split_hash = compute_split_hash(
            train_weeks_estimate, bundle.val_weeks, bundle.test_weeks
        )
        feature_hash = compute_feature_hash(bundle.config)
        config_hash = compute_config_hash(bundle.config)
        model_hash = compute_model_file_hash(bundle.model_path)

        log.info("Config: sha256:%s...", config_hash[:12])

        # Initialize result
        result = BenchmarkResult(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=bundle.model_id,
            model_path=str(bundle.model_path),
            model_file_hash=model_hash,
            suite=suite,
            seed=seed,
            operator=_get_operator(),
            data_manifest_hash=data_hash,
            split_hash=split_hash,
            feature_hash=feature_hash,
            config_hash=config_hash,
        )

        tiers = SUITES[suite]

        # ---- T1: Validation Eval ----
        if "T1" in tiers:
            self._log_tier("T1", "Validation Eval", len(bundle.val_weeks))
            t1_result = run_t1(
                model=bundle.model,
                config=bundle.config,
                val_weeks=bundle.val_weeks,
                seed=seed,
                progress_callback=progress_callback,
            )
            # Copy T1 metrics to result
            _copy_t1_to_result(t1_result, result)

        # ---- T2: Stress Matrix ----
        if "T2" in tiers:
            self._log_tier("T2", "Stress Matrix", 7)
            stress_results = run_t2(
                model=bundle.model,
                config=bundle.config,
                val_weeks=bundle.val_weeks,
                seed=seed,
                progress_callback=progress_callback,
            )
            result.stress_results = stress_results

            # Compute stress score
            retentions = {
                s: sr.pf_retention for s, sr in stress_results.items()
                if s != "base"
            }
            from .scoring import compute_stress_score
            result.stress_robustness_score = compute_stress_score(retentions)
            if retentions:
                result.stress_worst_retention = min(retentions.values())
                result.stress_worst_scenario = min(retentions, key=retentions.get)

        # ---- T3: Regime Segmentation ----
        if "T3" in tiers:
            self._log_tier("T3", "Regime Segmentation")
            result.regime_slices = run_t3(result.base_trades)

        # ---- T4: Churn Diagnostic ----
        if "T4" in tiers:
            self._log_tier("T4", "Churn Diagnostic")
            result.churn = run_t4(
                trades=result.base_trades,
                val_weeks_count=len(bundle.val_weeks),
                config=bundle.config,
            )

        # ---- T5: Reward Ablation ----
        if "T5" in tiers:
            self._log_tier("T5", "Reward Ablation")
            rollout = getattr(result, '_rollout', None)
            if rollout and rollout.r1_values:
                result.reward_ablation = run_t5(
                    rollout.r1_values, rollout.r2_values,
                    rollout.r3_values, rollout.r4_values,
                    rollout.r5_values,
                )
            else:
                log.warning("T5 skipped: no rollout reward data (T1 must run first)")

        # ---- T6: Gating Diagnostics ----
        if "T6" in tiers:
            self._log_tier("T6", "Gating Diagnostics")
            rollout = getattr(result, '_rollout', None)
            if rollout and rollout.raw_actions:
                result.gating = run_t6(rollout.raw_actions, bundle.config)
                result.action_stats = compute_action_stats(rollout.raw_actions)
            else:
                log.warning("T6 skipped: no rollout action data (T1 must run first)")

        # ---- T7: Locked Test ----
        if "T7" in tiers:
            self._log_tier("T7", "Locked Test", len(bundle.test_weeks))
            test_metrics = run_t7(
                model=bundle.model,
                config=bundle.config,
                test_weeks=bundle.test_weeks,
                seed=seed,
                progress_callback=progress_callback,
            )
            for k, v in test_metrics.items():
                setattr(result, k, v)

        # ---- Detectors ----
        log.info("")
        log.info("Running detectors...")

        champion = self.db.get_current_champion()
        prior_results = self.db.get_prior_results(result.model_id)

        # Build flat metrics dict for detectors
        metrics_dict = result.to_dict()
        if result.action_stats:
            metrics_dict["action_direction_std"] = result.action_stats.direction_std
            metrics_dict["action_conviction_std"] = result.action_stats.conviction_std
        if result.gating:
            metrics_dict["gate_direction_pass"] = result.gating.direction_pass_pct
            metrics_dict["gate_conviction_pass"] = result.gating.conviction_pass_live_pct
            metrics_dict["gate_spread_pass"] = result.gating.spread_pass_pct
            metrics_dict["gate_lot_pass"] = result.gating.lot_pass_pct

        result.detectors = [
            detect_aggression_drift(metrics_dict, champion),
            detect_conviction_collapse(metrics_dict),
            detect_stress_fragility(result.stress_results) if result.stress_results else
                _no_detector("stress_fragility"),
            detect_overfitting(metrics_dict, prior_results),
            detect_reward_hacking(result.reward_ablation) if result.reward_ablation else
                _no_detector("reward_hacking"),
        ]

        for d in result.detectors:
            status = "DETECTED" if d.detected else "CLEAR"
            log.info("  %-25s %s", d.name + ":", status)

        # ---- Hard-fail rules ----
        result.hard_fails = check_hard_fails(result)
        result.is_disqualified = len(result.hard_fails) > 0

        if result.hard_fails:
            log.warning("Hard Fails: %s", ", ".join(result.hard_fails))
        else:
            log.info("Hard Fails: NONE")

        # ---- Side warnings ----
        side_warnings = check_side_warnings(result)
        for w in side_warnings:
            log.warning(w)

        # ---- Scoring ----
        result.score = compute_spartus_score(result)
        log.info("")
        log.info("SpartusScore: %.1f", result.score.spartus_score)

        # ---- Champion protocol ----
        if compare_vs_champion and suite in ("full", "validation_only"):
            verdict, delta = evaluate_champion_candidacy(result, champion)
            if champion:
                champ_score = float(champion.get("spartus_score", 0) or 0)
                champ_id = champion.get("model_id", "?")
                log.info(
                    "vs Champion (%s, score=%.1f): %+.1f -> %s",
                    champ_id, champ_score, delta, verdict,
                )
            else:
                log.info("No existing champion. Verdict: %s", verdict)

        # ---- Persist ----
        self.db.save_run(result)

        # ---- Champion promotion (after save_run so FK exists) ----
        if compare_vs_champion and suite in ("full", "validation_only"):
            if verdict == "PROMOTE":
                result.is_champion = True
                self.db.promote_champion(
                    result.run_id, result.model_id,
                    result.score.spartus_score, result.timestamp,
                )
                log.info("*** %s promoted to CHAMPION ***", result.model_id)

        if suite == "locked_test":
            result_hash = compute_result_hash(result.to_json())
            self.db.save_locked_test_audit(result, result_hash)

        # ---- Reports ----
        run_dir = _ensure_run_dir(result.run_id)
        _save_report_json(result, run_dir)
        _save_report_md(result, run_dir, champion)
        _save_trades_csv(result, run_dir)

        if generate_plots:
            try:
                from .reports.plots import generate_all_plots
                generate_all_plots(result, run_dir)
            except ImportError:
                log.debug("Plot generation skipped (matplotlib not available)")

        log.info("")
        log.info("Report: %s", run_dir / "report.md")

        # Clean up transient data
        if hasattr(result, '_rollout'):
            del result._rollout

        return result

    def _log_tier(self, tier: str, name: str, count: int = 0):
        suffix = f" [{count} weeks]" if count else ""
        log.info("")
        log.info("Running %s: %s ...%s", tier, name, suffix)


def _no_detector(name: str):
    from .types import DetectorResult
    return DetectorResult(name=name, detected=False, details={"reason": "no_data"})


def _copy_t1_to_result(t1: BenchmarkResult, result: BenchmarkResult):
    """Copy all T1 metrics from t1_result to the main result."""
    for field_name in [
        "val_trades", "val_win_pct", "val_pf", "val_sharpe", "val_sortino",
        "val_max_dd_pct", "val_net_pnl", "val_tim_pct", "val_trades_day",
        "val_avg_hold", "val_median_hold", "val_calmar", "val_recovery_factor",
        "val_tail_ratio", "val_expectancy", "val_max_consec_loss", "val_max_consec_win",
        "val_gross_profit", "val_gross_loss", "val_avg_win", "val_avg_loss",
        "val_win_loss_ratio", "val_flat_bar_pct", "val_entry_timing", "val_sl_quality",
        "val_long_count", "val_short_count", "val_long_pnl", "val_short_pnl",
        "val_long_pf", "val_short_pf",
        "base_trades", "weekly_returns", "total_steps", "in_position_steps",
        "conviction_stats",
    ]:
        setattr(result, field_name, getattr(t1, field_name))
    # Copy transient rollout data
    if hasattr(t1, '_rollout'):
        result._rollout = t1._rollout


def _get_operator() -> str:
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get("USERNAME", os.environ.get("USER", "unknown"))


def _ensure_run_dir(run_id: str) -> Path:
    run_dir = Path("storage/benchmark/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_report_json(result: BenchmarkResult, run_dir: Path):
    path = run_dir / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.to_json())


def _save_report_md(
    result: BenchmarkResult,
    run_dir: Path,
    champion: Optional[Dict] = None,
):
    """Generate markdown report."""
    from .reports.markdown import generate_report_md
    md = generate_report_md(result, champion)
    path = run_dir / "report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


def _save_trades_csv(result: BenchmarkResult, run_dir: Path):
    """Export trades to CSV for external audit."""
    if not result.base_trades:
        return
    import csv
    path = run_dir / "trades.csv"
    fields = [
        "trade_num", "week", "step", "side", "entry_price", "exit_price",
        "lots", "pnl", "pnl_pct", "hold_bars", "conviction", "close_reason",
        "lesson_type", "session", "atr_at_entry", "max_favorable",
        "initial_sl", "initial_tp", "final_sl",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in result.base_trades:
            row = {k: getattr(t, k, "") for k in fields}
            writer.writerow(row)
