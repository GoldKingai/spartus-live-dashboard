"""SpartusBench CLI entry point.

Usage:
    python -m spartusbench <command> [options]

Commands:
    run          Run a benchmark
    compare      Compare two models
    leaderboard  Show champion history
    show         Show details of a specific run
    audit        Show locked test audit trail
    discover     List available models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="spartusbench",
        description="SpartusBench -- Benchmark & Model Progression System",
    )
    parser.add_argument("--version", action="version", version=f"SpartusBench v{__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- run ----
    p_run = subparsers.add_parser("run", help="Run a benchmark")
    p_run.add_argument("model", nargs="?", help="Model reference (e.g., W0170, best, or path)")
    p_run.add_argument("--model", dest="model_path", help="Explicit model path")
    p_run.add_argument("--suite", default="full",
                       choices=["full", "validation_only", "stress_only", "locked_test"])
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--confirm-test", action="store_true",
                       help="Required for locked_test suite")
    p_run.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p_run.add_argument("--no-compare", action="store_true", help="Skip champion comparison")

    # ---- compare ----
    p_compare = subparsers.add_parser("compare", help="Compare two models")
    p_compare.add_argument("model_a", help="First model reference")
    p_compare.add_argument("model_b", help="Second model reference")

    # ---- leaderboard ----
    p_leader = subparsers.add_parser("leaderboard", help="Show leaderboard")
    p_leader.add_argument("--top", type=int, default=10, help="Number of entries")
    p_leader.add_argument("--all", action="store_true", help="Include disqualified")

    # ---- show ----
    p_show = subparsers.add_parser("show", help="Show run details")
    p_show.add_argument("run_id", nargs="?", help="Run ID")
    p_show.add_argument("--last", action="store_true", help="Most recent run")
    p_show.add_argument("--model", dest="show_model", help="Most recent run for model")

    # ---- audit ----
    p_audit = subparsers.add_parser("audit", help="Show locked test audit")
    p_audit.add_argument("--model", dest="audit_model", help="Filter by model")

    # ---- discover ----
    subparsers.add_parser("discover", help="List available models")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy loggers
    logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "run":
            return cmd_run(args)
        elif args.command == "compare":
            return cmd_compare(args)
        elif args.command == "leaderboard":
            return cmd_leaderboard(args)
        elif args.command == "show":
            return cmd_show(args)
        elif args.command == "audit":
            return cmd_audit(args)
        elif args.command == "discover":
            return cmd_discover(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as e:
        logging.error("Error: %s", e)
        if args.verbose:
            logging.exception("Full traceback:")
        return 1

    return 0


def cmd_run(args) -> int:
    """Execute a benchmark run."""
    from .runner import BenchmarkRunner

    model_ref = args.model or args.model_path
    if not model_ref:
        print("Error: model reference required. E.g.: python -m spartusbench run W0170")
        return 1

    runner = BenchmarkRunner()
    result = runner.run(
        model_ref=model_ref,
        suite=args.suite,
        seed=args.seed,
        confirm_test=args.confirm_test,
        generate_plots=not args.no_plots,
        compare_vs_champion=not args.no_compare,
    )

    return 0


def cmd_compare(args) -> int:
    """Compare two benchmark runs."""
    from .database import BenchmarkDB
    from .reports.markdown import generate_comparison_md

    db = BenchmarkDB()

    # Get most recent run for each model
    run_a = db.get_latest_run(args.model_a)
    run_b = db.get_latest_run(args.model_b)

    if not run_a:
        print(f"No benchmark run found for {args.model_a}")
        return 1
    if not run_b:
        print(f"No benchmark run found for {args.model_b}")
        return 1

    md = generate_comparison_md(run_a, run_b)
    print(md)
    return 0


def cmd_leaderboard(args) -> int:
    """Show the leaderboard."""
    from .database import BenchmarkDB

    db = BenchmarkDB()
    entries = db.get_leaderboard(top_n=args.top, include_dq=args.all)

    if not entries:
        print("No benchmark runs found. Run a benchmark first:")
        print("  python -m spartusbench run W0170")
        return 0

    print("")
    print("SpartusBench Leaderboard")
    print("=" * 80)
    print("")
    print(f"{'Rank':<6} {'Model':<10} {'Score':<8} {'Sharpe':<8} {'PF':<7} "
          f"{'MaxDD':<8} {'Stress':<8} {'Status'}")
    print(f"{'----':<6} {'------':<10} {'------':<8} {'------':<8} {'-----':<7} "
          f"{'------':<8} {'------':<8} {'-----------'}")

    for i, entry in enumerate(entries, 1):
        model_id = entry.get("model_id", "?")
        score = entry.get("spartus_score", 0) or 0
        sharpe = entry.get("val_sharpe", 0) or 0
        pf = entry.get("val_pf", 0) or 0
        dd = entry.get("val_max_dd_pct", 0) or 0
        stress = entry.get("stress_robustness_score", 0) or 0
        is_dq = entry.get("is_disqualified", 0)
        is_champ = entry.get("is_champion", 0)

        if is_dq:
            status = "DISQUALIFIED"
            rank_str = "-"
        elif is_champ:
            status = "CHAMPION"
            rank_str = f"{i}*"
        elif entry.get("dethroned_at"):
            status = "(dethroned)"
            rank_str = str(i)
        else:
            status = ""
            rank_str = str(i)

        if is_dq:
            print(f"{rank_str:<6} {model_id:<10} {'DQ':<8} {'--':<8} "
                  f"{pf:<7.2f} {dd:<8.1f} {'--':<8} {status}")
        else:
            print(f"{rank_str:<6} {model_id:<10} {score:<8.1f} {sharpe:<8.2f} "
                  f"{pf:<7.2f} {dd:<7.1f}% {stress:<8.1f} {status}")

    print("")
    return 0


def cmd_show(args) -> int:
    """Show details of a specific run."""
    from .database import BenchmarkDB

    db = BenchmarkDB()

    if args.last:
        run = db.get_latest_run()
    elif args.show_model:
        run = db.get_latest_run(args.show_model)
    elif args.run_id:
        run = db.get_run(args.run_id)
    else:
        print("Specify --last, --model <id>, or a run_id")
        return 1

    if not run:
        print("Run not found.")
        return 1

    # Print key metrics
    print("")
    print(f"SpartusBench Run: {run.get('run_id', '?')}")
    print(f"Model: {run.get('model_id', '?')}")
    print(f"Timestamp: {run.get('timestamp', '?')}")
    print(f"Suite: {run.get('suite', '?')}")
    print(f"SpartusScore: {run.get('spartus_score', 0):.1f}")
    print("")

    print(f"  Sharpe:    {run.get('val_sharpe', 0):.2f}")
    print(f"  PF:        {run.get('val_pf', 0):.2f}")
    print(f"  Win%:      {run.get('val_win_pct', 0):.1f}%")
    print(f"  MaxDD:     {run.get('val_max_dd_pct', 0):.1f}%")
    print(f"  Trades:    {run.get('val_trades', 0)}")
    print(f"  Net P/L:   ${run.get('val_net_pnl', 0):.2f}")
    print(f"  TIM%:      {run.get('val_tim_pct', 0):.1f}%")
    print(f"  Stress:    {run.get('stress_robustness_score', 0):.1f}")
    print("")

    # Hard fails
    hard_fails = run.get("hard_fails", "[]")
    if hard_fails and hard_fails != "[]":
        print(f"  Hard Fails: {hard_fails}")
    else:
        print("  Hard Fails: NONE")

    # Champion status
    if run.get("is_champion"):
        print("  Status: CHAMPION")
    elif run.get("is_disqualified"):
        print("  Status: DISQUALIFIED")

    print("")
    return 0


def cmd_audit(args) -> int:
    """Show locked test audit trail."""
    from .database import BenchmarkDB

    db = BenchmarkDB()
    entries = db.get_locked_test_audit(args.audit_model)

    if not entries:
        print("No locked test runs found.")
        return 0

    print("")
    print("Locked Test Audit Trail")
    print("=" * 80)
    print("")
    print(f"{'Date':<22} {'Operator':<10} {'Model':<8} {'Seed':<6} {'Result Hash'}")
    print(f"{'----':<22} {'--------':<10} {'-----':<8} {'----':<6} {'-----------'}")

    for entry in entries:
        print(
            f"{entry.get('timestamp', '?'):<22} "
            f"{entry.get('operator', '?'):<10} "
            f"{entry.get('model_id', '?'):<8} "
            f"{entry.get('seed', 42):<6} "
            f"{entry.get('result_hash', '?')[:16]}..."
        )

    print("")
    return 0


def cmd_discover(args) -> int:
    """List available models."""
    from .discovery import discover_models

    models = discover_models()

    if not models:
        print("No models found in storage/models/")
        return 0

    print("")
    print("Available Models")
    print("=" * 80)
    print("")
    print(f"{'ID':<12} {'Path':<50} {'Size':<8} {'Meta':<6} {'Reward'}")
    print(f"{'--':<12} {'----':<50} {'----':<8} {'----':<6} {'------'}")

    for m in models:
        print(
            f"{m['model_id']:<12} "
            f"{m['path']:<50} "
            f"{m['size_mb']:<7.1f}MB "
            f"{'Yes' if m['has_meta'] else 'No':<6} "
            f"{'Yes' if m['has_reward_state'] else 'No'}"
        )

    print(f"\nTotal: {len(models)} checkpoints found")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
