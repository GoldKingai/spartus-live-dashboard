"""Plot generation for SpartusBench benchmark reports.

Generates matplotlib charts saved as PNG files.
All functions are optional -- if matplotlib is not installed,
the benchmark still works (CLI-only output).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..types import BenchmarkResult, StressResult, TradeRecord

log = logging.getLogger("spartusbench.plots")


def generate_all_plots(result: BenchmarkResult, run_dir: Path) -> None:
    """Generate all benchmark charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plot generation")
        return

    try:
        _plot_equity_curve(result, run_dir)
    except Exception as e:
        log.warning("Failed to generate equity curve: %s", e)

    try:
        _plot_stress_comparison(result, run_dir)
    except Exception as e:
        log.warning("Failed to generate stress comparison: %s", e)

    try:
        _plot_reward_decomposition(result, run_dir)
    except Exception as e:
        log.warning("Failed to generate reward decomposition: %s", e)

    try:
        _plot_gating_funnel(result, run_dir)
    except Exception as e:
        log.warning("Failed to generate gating funnel: %s", e)

    try:
        _plot_conviction_distribution(result, run_dir)
    except Exception as e:
        log.warning("Failed to generate conviction distribution: %s", e)


def _plot_equity_curve(result: BenchmarkResult, run_dir: Path) -> None:
    """Plot validation equity curve."""
    import matplotlib.pyplot as plt

    if not result.base_trades:
        return

    equity = [100.0]
    for t in result.base_trades:
        equity.append(equity[-1] + t.pnl)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=1.0, color="#2196F3")
    ax.axhline(y=100.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(range(len(equity)), equity, 100.0,
                    where=[e >= 100 for e in equity], alpha=0.1, color="green")
    ax.fill_between(range(len(equity)), equity, 100.0,
                    where=[e < 100 for e in equity], alpha=0.1, color="red")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Equity ($)")
    ax.set_title(f"Equity Curve: {result.model_id} (SpartusScore={result.score.spartus_score:.1f})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "equity_curve.png", dpi=150)
    plt.close(fig)


def _plot_stress_comparison(result: BenchmarkResult, run_dir: Path) -> None:
    """Plot stress scenario PF comparison."""
    import matplotlib.pyplot as plt

    if not result.stress_results:
        return

    scenarios = ["base", "2x_spread", "combined_2x2x", "3x_spread",
                 "2x_slip_mean", "2x_slip_std", "5x_spread"]
    pfs = []
    labels = []
    colors = []

    for s in scenarios:
        sr = result.stress_results.get(s)
        if sr:
            pfs.append(sr.pf)
            labels.append(s.replace("_", "\n"))
            colors.append("#4CAF50" if sr.pf >= 1.0 else "#F44336")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, pfs, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_ylabel("Profit Factor")
    ax.set_title(f"Stress Matrix: {result.model_id}")
    ax.legend()

    for bar, pf in zip(bars, pfs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{pf:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(run_dir / "stress_comparison.png", dpi=150)
    plt.close(fig)


def _plot_reward_decomposition(result: BenchmarkResult, run_dir: Path) -> None:
    """Plot reward component stacked bar."""
    import matplotlib.pyplot as plt

    ra = result.reward_ablation
    if not ra or ra.total_weighted == 0:
        return

    components = ["R1\nP/L", "R2\nQuality", "R3\nDD", "R4\nSharpe", "R5\nRisk"]
    pcts = [ra.r1_pct, ra.r2_pct, ra.r3_pct, ra.r4_pct, ra.r5_pct]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(components, pcts, color=colors, alpha=0.8)
    ax.axhline(y=40.0, color="red", linestyle="--", alpha=0.5, label="R5 warning (40%)")
    ax.set_ylabel("% of Total Reward")
    ax.set_title(f"Reward Decomposition: {result.model_id}")
    ax.legend()

    for i, (comp, pct) in enumerate(zip(components, pcts)):
        ax.text(i, pct + 0.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(run_dir / "reward_decomposition.png", dpi=150)
    plt.close(fig)


def _plot_gating_funnel(result: BenchmarkResult, run_dir: Path) -> None:
    """Plot gating funnel chart."""
    import matplotlib.pyplot as plt

    g = result.gating
    if not g or g.total_bars == 0:
        return

    stages = ["Total Bars", "Direction", "Conviction\n(live)", "Lot Sizing", "Overall"]
    counts = [
        g.total_bars, g.direction_pass_count,
        g.conviction_pass_live_count, g.lot_pass_count,
        g.overall_pass_count,
    ]
    pcts = [c / g.total_bars * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(stages[::-1], pcts[::-1], color="#2196F3", alpha=0.8)
    ax.set_xlabel("% of Total Bars")
    ax.set_title(f"Gating Funnel: {result.model_id}")

    for bar, pct, count in zip(bars, pcts[::-1], counts[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% ({count:,})", va="center", fontsize=9)

    ax.set_xlim(0, 110)
    fig.tight_layout()
    fig.savefig(run_dir / "gating_funnel.png", dpi=150)
    plt.close(fig)


def _plot_conviction_distribution(result: BenchmarkResult, run_dir: Path) -> None:
    """Plot histogram of trade conviction values."""
    import matplotlib.pyplot as plt

    if not result.base_trades:
        return

    convictions = [t.conviction for t in result.base_trades]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(convictions, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=0.15, color="orange", linestyle="--", alpha=0.7, label="Live threshold (0.15)")
    ax.axvline(x=0.30, color="red", linestyle="--", alpha=0.7, label="Training threshold (0.30)")
    ax.set_xlabel("Conviction")
    ax.set_ylabel("Trade Count")
    ax.set_title(f"Conviction Distribution: {result.model_id}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(run_dir / "conviction_distribution.png", dpi=150)
    plt.close(fig)
