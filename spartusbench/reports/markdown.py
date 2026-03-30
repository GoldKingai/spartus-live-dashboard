"""Markdown report generation for SpartusBench."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import BenchmarkResult


def generate_report_md(
    result: BenchmarkResult,
    champion: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a human-readable markdown report."""
    lines = []

    lines.append(f"# SpartusBench Report: {result.model_id}")
    lines.append(f"**Run ID:** {result.run_id}")
    lines.append(f"**Timestamp:** {result.timestamp}")
    lines.append(f"**Suite:** {result.suite}")
    lines.append(f"**Seed:** {result.seed}")
    lines.append(f"**Operator:** {result.operator}")
    lines.append("")

    # Score summary
    lines.append("## SpartusScore")
    lines.append("")
    score = result.score
    lines.append(f"**SpartusScore: {score.spartus_score:.1f}**")
    lines.append("")
    lines.append("| Component | Raw Score | Weight | Contribution |")
    lines.append("|-----------|-----------|--------|--------------|")
    lines.append(f"| val_sharpe | {score.val_sharpe_component:.1f} | 0.25 | {score.val_sharpe_component * 0.25:.1f} |")
    lines.append(f"| val_pf | {score.val_pf_component:.1f} | 0.20 | {score.val_pf_component * 0.20:.1f} |")
    lines.append(f"| stress | {score.stress_component:.1f} | 0.25 | {score.stress_component * 0.25:.1f} |")
    lines.append(f"| max_dd | {score.max_dd_component:.1f} | 0.15 | {score.max_dd_component * 0.15:.1f} |")
    lines.append(f"| quality | {score.quality_component:.1f} | 0.15 | {score.quality_component * 0.15:.1f} |")
    lines.append("")

    # Hard fails
    if result.hard_fails:
        lines.append(f"**HARD FAILS:** {', '.join(result.hard_fails)}")
        lines.append(f"**STATUS: DISQUALIFIED**")
    else:
        lines.append("**Hard Fails:** NONE")
        if result.is_champion:
            lines.append("**STATUS: CHAMPION**")
    lines.append("")

    # T1 Validation metrics
    lines.append("## T1: Validation Eval")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Sharpe (ann.) | {result.val_sharpe:.2f} |")
    lines.append(f"| Sortino (ann.) | {result.val_sortino:.2f} |")
    lines.append(f"| Profit Factor | {result.val_pf:.2f} |")
    lines.append(f"| Win Rate | {result.val_win_pct:.1f}% |")
    lines.append(f"| Max Drawdown | {result.val_max_dd_pct:.1f}% |")
    lines.append(f"| Net P/L | ${result.val_net_pnl:.2f} |")
    lines.append(f"| Trades | {result.val_trades} |")
    lines.append(f"| Trades/Day | {result.val_trades_day:.1f} |")
    lines.append(f"| TIM% | {result.val_tim_pct:.1f}% |")
    lines.append(f"| Avg Hold (bars) | {result.val_avg_hold:.1f} |")
    lines.append(f"| Median Hold (bars) | {result.val_median_hold:.1f} |")
    lines.append(f"| Calmar Ratio | {result.val_calmar:.2f} |")
    lines.append(f"| Recovery Factor | {result.val_recovery_factor:.2f} |")
    lines.append(f"| Tail Ratio | {result.val_tail_ratio:.2f} |")
    lines.append(f"| Expectancy | ${result.val_expectancy:.3f} |")
    lines.append(f"| Max Consec Losses | {result.val_max_consec_loss} |")
    lines.append(f"| Max Consec Wins | {result.val_max_consec_win} |")
    lines.append(f"| Gross Profit | ${result.val_gross_profit:.2f} |")
    lines.append(f"| Gross Loss | ${result.val_gross_loss:.2f} |")
    lines.append(f"| Avg Win | ${result.val_avg_win:.3f} |")
    lines.append(f"| Avg Loss | ${result.val_avg_loss:.3f} |")
    lines.append(f"| Win/Loss Ratio | {result.val_win_loss_ratio:.2f} |")
    lines.append(f"| Flat Bar % | {result.val_flat_bar_pct:.1f}% |")
    lines.append(f"| Entry Timing | {result.val_entry_timing:.1f}% |")
    lines.append(f"| SL Quality | {result.val_sl_quality:.1f}% |")
    lines.append("")

    # Long/Short split
    lines.append("### Long/Short Split")
    lines.append("")
    lines.append("| Side | Trades | P/L | PF |")
    lines.append("|------|--------|-----|-----|")
    lines.append(f"| LONG | {result.val_long_count} | ${result.val_long_pnl:.2f} | {result.val_long_pf:.2f} |")
    lines.append(f"| SHORT | {result.val_short_count} | ${result.val_short_pnl:.2f} | {result.val_short_pf:.2f} |")
    lines.append("")

    # T2 Stress
    if result.stress_results:
        lines.append("## T2: Stress Matrix")
        lines.append("")
        lines.append(f"**Stress Robustness Score:** {result.stress_robustness_score:.1f}")
        lines.append(f"**Worst Retention:** {result.stress_worst_retention:.3f} ({result.stress_worst_scenario})")
        lines.append("")
        lines.append("| Scenario | PF | Retention | Trades | MaxDD |")
        lines.append("|----------|------|-----------|--------|-------|")
        for scenario in ["base", "2x_spread", "combined_2x2x", "3x_spread",
                         "2x_slip_mean", "2x_slip_std", "5x_spread"]:
            sr = result.stress_results.get(scenario)
            if sr:
                lines.append(
                    f"| {scenario} | {sr.pf:.2f} | {sr.pf_retention:.2f} "
                    f"| {sr.trades} | {sr.max_dd_pct:.1f}% |"
                )
        lines.append("")

    # T3 Regime
    if result.regime_slices:
        lines.append("## T3: Regime Segmentation")
        lines.append("")
        current_type = ""
        for rs in result.regime_slices:
            if rs.slice_type != current_type:
                current_type = rs.slice_type
                lines.append(f"### {current_type.replace('_', ' ').title()}")
                lines.append("")
                lines.append("| Slice | Trades | Win% | Net P/L | PF | Avg P/L | Avg Hold |")
                lines.append("|-------|--------|------|---------|----|---------|----------|")
            lines.append(
                f"| {rs.slice_value} | {rs.trades} | {rs.win_pct:.1f}% | "
                f"${rs.net_pnl:.2f} | {rs.pf:.2f} | ${rs.avg_pnl:.3f} | "
                f"{rs.avg_hold:.1f} |"
            )
        lines.append("")

    # T4 Churn
    if result.churn and result.churn.trading_days > 0:
        ch = result.churn
        lines.append("## T4: Churn Diagnostic")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Trades/Day | {ch.trades_per_day:.1f} |")
        lines.append(f"| Avg Spread (pips) | {ch.avg_spread_pips:.1f} |")
        lines.append(f"| Avg Slippage (pips) | {ch.avg_slippage_pips:.1f} |")
        lines.append(f"| Est. Cost/Trade | ${ch.est_cost_per_trade:.4f} |")
        lines.append(f"| Total Est. Cost | ${ch.total_est_cost:.2f} |")
        lines.append(f"| Net Edge/Trade | ${ch.net_edge_per_trade:.4f} |")
        lines.append(f"| Gross Edge/Trade | ${ch.gross_edge_per_trade:.4f} |")
        lines.append(f"| Cost-to-Edge Ratio | {ch.cost_to_edge_ratio:.2f} |")
        lines.append("")

    # T5 Reward Ablation
    if result.reward_ablation and result.reward_ablation.total_weighted != 0:
        ra = result.reward_ablation
        lines.append("## T5: Reward Ablation")
        lines.append("")
        lines.append("| Component | Weight | Sum | Weighted | % of Total |")
        lines.append("|-----------|--------|-----|----------|------------|")
        for i, name in enumerate(["R1 P/L", "R2 Quality", "R3 DD", "R4 Sharpe", "R5 Risk"]):
            rsum = [ra.r1_sum, ra.r2_sum, ra.r3_sum, ra.r4_sum, ra.r5_sum][i]
            rw = [ra.r1_weighted, ra.r2_weighted, ra.r3_weighted, ra.r4_weighted, ra.r5_weighted][i]
            rpct = [ra.r1_pct, ra.r2_pct, ra.r3_pct, ra.r4_pct, ra.r5_pct][i]
            weight = [0.40, 0.20, 0.15, 0.15, 0.10][i]
            lines.append(f"| {name} | {weight} | {rsum:.2f} | {rw:.2f} | {rpct:.1f}% |")
        lines.append("")

        if ra.r5_pct > 40.0:
            lines.append(f"**WARNING:** R5 contribution {ra.r5_pct:.1f}% > 40% threshold")
            lines.append("")

    # T6 Gating
    if result.gating and result.gating.total_bars > 0:
        g = result.gating
        lines.append("## T6: Gating Diagnostics")
        lines.append("")
        lines.append("| Gate | Pass Count | Pass Rate |")
        lines.append("|------|------------|-----------|")
        lines.append(f"| Direction (|dir| >= 0.3) | {g.direction_pass_count} | {g.direction_pass_pct:.1f}% |")
        lines.append(f"| Conviction (>= 0.15 live) | {g.conviction_pass_live_count} | {g.conviction_pass_live_pct:.1f}% |")
        lines.append(f"| Conviction (>= 0.30 train) | {g.conviction_pass_train_count} | {g.conviction_pass_train_pct:.1f}% |")
        lines.append(f"| Lot Sizing | {g.lot_pass_count} | {g.lot_pass_pct:.1f}% |")
        lines.append(f"| Spread | {g.spread_pass_count} | {g.spread_pass_pct:.1f}% |")
        lines.append(f"| **Overall** | **{g.overall_pass_count}** | **{g.overall_pass_pct:.1f}%** |")
        lines.append("")

    # Locked test
    if result.test_trades is not None:
        lines.append("## T7: Locked Test")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Trades | {result.test_trades} |")
        lines.append(f"| Win Rate | {result.test_win_pct:.1f}% |")
        lines.append(f"| PF | {result.test_pf:.2f} |")
        lines.append(f"| Sharpe | {result.test_sharpe:.2f} |")
        lines.append(f"| MaxDD | {result.test_max_dd_pct:.1f}% |")
        lines.append(f"| Net P/L | ${result.test_net_pnl:.2f} |")
        lines.append("")

    # Detectors
    lines.append("## Detectors")
    lines.append("")
    for d in result.detectors:
        status = "**DETECTED**" if d.detected else "CLEAR"
        sev = f" (severity={d.severity})" if d.detected else ""
        lines.append(f"- **{d.name}:** {status}{sev}")
    lines.append("")

    # Delta vs Champion
    if champion:
        lines.append("## Delta vs Champion")
        lines.append("")
        lines.append(_generate_delta_table(result, champion))
        lines.append("")

    # Conviction stats
    if result.conviction_stats:
        cs = result.conviction_stats
        lines.append("## Conviction Distribution")
        lines.append("")
        lines.append(f"Mean={cs.mean:.3f} | Std={cs.std:.3f} | "
                     f"P10={cs.p10:.3f} | P50={cs.p50:.3f} | P90={cs.p90:.3f}")
        lines.append("")

    # Hashes
    lines.append("## Reproducibility Hashes")
    lines.append("")
    lines.append(f"- **data_manifest:** sha256:{result.data_manifest_hash[:16]}...")
    lines.append(f"- **split:** sha256:{result.split_hash[:16]}...")
    lines.append(f"- **features:** sha256:{result.feature_hash[:16]}...")
    lines.append(f"- **config:** sha256:{result.config_hash[:16]}...")
    lines.append(f"- **model_file:** sha256:{result.model_file_hash[:16]}...")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by SpartusBench v1.0.0*")

    return "\n".join(lines)


def _generate_delta_table(result: BenchmarkResult, champion: Dict) -> str:
    """Generate delta-vs-champion comparison table."""
    lines = []
    champ_id = champion.get("model_id", "?")
    cand_id = result.model_id

    lines.append(f"| Metric | {champ_id} | {cand_id} | Delta | Verdict |")
    lines.append("|--------|---------|---------|-------|---------|")

    metrics = [
        ("SpartusScore", "spartus_score", False),
        ("Sharpe", "val_sharpe", False),
        ("PF", "val_pf", False),
        ("Win Rate", "val_win_pct", False),
        ("Max DD", "val_max_dd_pct", True),
        ("Stress Score", "stress_robustness_score", False),
        ("TIM%", "val_tim_pct", None),
        ("Trades/Day", "val_trades_day", None),
        ("Expectancy", "val_expectancy", False),
        ("Max Consec Loss", "val_max_consec_loss", True),
    ]

    for name, key, invert in metrics:
        champ_val = float(champion.get(key, 0) or 0)
        # SpartusScore is nested in result.score
        if key == "spartus_score":
            cand_val = float(result.score.spartus_score) if result.score else 0.0
        else:
            cand_val = float(getattr(result, key, 0) or 0)
        delta = cand_val - champ_val

        if invert is None:
            verdict = "NEUTRAL"
        elif invert:
            verdict = "IMPROVE" if delta < 0 else ("REGRESS" if delta > 0.5 else "NEUTRAL")
        else:
            verdict = "IMPROVE" if delta > 0.01 else ("REGRESS" if delta < -0.01 else "NEUTRAL")

        if abs(delta) < 0.005:
            verdict = "NEUTRAL"

        lines.append(
            f"| {name} | {champ_val:.2f} | {cand_val:.2f} | {delta:+.2f} | {verdict} |"
        )

    return "\n".join(lines)


def generate_comparison_md(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
) -> str:
    """Generate a side-by-side comparison report for two benchmark runs."""
    id_a = result_a.get("model_id", "Model A")
    id_b = result_b.get("model_id", "Model B")

    lines = []
    lines.append(f"# Model Comparison: {id_a} vs {id_b}")
    lines.append("")
    lines.append(f"| Metric | {id_a} | {id_b} | Delta | Verdict |")
    lines.append("|--------|---------|---------|-------|---------|")

    metrics = [
        ("SpartusScore", "spartus_score"),
        ("Sharpe", "val_sharpe"),
        ("PF", "val_pf"),
        ("Win Rate", "val_win_pct"),
        ("Max DD", "val_max_dd_pct"),
        ("Stress Score", "stress_robustness_score"),
        ("TIM%", "val_tim_pct"),
        ("Trades/Day", "val_trades_day"),
        ("Expectancy", "val_expectancy"),
        ("Max Consec Loss", "val_max_consec_loss"),
    ]

    for name, key in metrics:
        val_a = float(result_a.get(key, 0) or 0)
        val_b = float(result_b.get(key, 0) or 0)
        delta = val_b - val_a

        if key == "val_max_dd_pct":
            verdict = "IMPROVE" if delta < 0 else "REGRESS" if delta > 0.5 else "NEUTRAL"
        elif key in ("val_tim_pct", "val_trades_day"):
            verdict = "NEUTRAL"
        else:
            verdict = "IMPROVE" if delta > 0.01 else "REGRESS" if delta < -0.01 else "NEUTRAL"

        lines.append(f"| {name} | {val_a:.2f} | {val_b:.2f} | {delta:+.2f} | {verdict} |")

    lines.append("")

    # Detector comparison
    lines.append("## Detectors")
    lines.append("")
    det_keys = [
        ("Aggression Drift", "detector_aggression"),
        ("Conviction Collapse", "detector_collapse"),
        ("Stress Fragility", "detector_fragility"),
        ("Overfitting", "detector_overfitting"),
        ("Reward Hacking", "detector_reward_hack"),
    ]
    for name, key in det_keys:
        a = "DETECTED" if result_a.get(key) else "CLEAR"
        b = "DETECTED" if result_b.get(key) else "CLEAR"
        lines.append(f"- **{name}:** {id_a}={a} | {id_b}={b}")

    return "\n".join(lines)
