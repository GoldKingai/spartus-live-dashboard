"""Spartus Training Entry Point.

Usage:
    python scripts/train.py [--resume] [--weeks N] [--seed S] [--no-dashboard]
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig


def _create_and_run(cfg, shared_metrics, max_weeks, resume, seed):
    """Create trainer and run — all in the SAME thread to avoid SQLite issues."""
    from src.training.trainer import SpartusTrainer

    trainer = SpartusTrainer(config=cfg, shared_metrics=shared_metrics, seed=seed)
    shared_metrics["_trainer"] = trainer  # expose for summary
    trainer.run(max_weeks=max_weeks, resume=resume)


def _run_training_thread(cfg, shared_metrics, max_weeks, resume, seed):
    """Wrapper for background thread with error handling."""
    try:
        _create_and_run(cfg, shared_metrics, max_weeks, resume, seed)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        shared_metrics["_error"] = str(e)
        traceback.print_exc()
    finally:
        shared_metrics["_training_done"] = True


def _print_summary(shared_metrics):
    """Print final training summary."""
    trainer = shared_metrics.get("_trainer")
    quit_reason = "quit requested" if shared_metrics.get("_quit_requested") else "finished"
    if shared_metrics.get("_error"):
        quit_reason = f"error: {shared_metrics['_error']}"

    print(f"\nTraining {quit_reason}.")
    if trainer:
        print(f"Final balance: \u00a3{trainer.balance:.2f}")
        print(f"Peak balance: \u00a3{trainer.peak_balance:.2f}")
        print(f"Weeks completed: {trainer.current_week}")
        print(f"Convergence: {trainer.convergence.state}")


def main():
    parser = argparse.ArgumentParser(description="Train the Spartus Trading AI")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--weeks", type=int, default=None, help="Override total training weeks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable live dashboard")
    args = parser.parse_args()

    cfg = TrainingConfig()
    shared_metrics = {"_start_time": time.time()}

    if args.no_dashboard:
        # No dashboard — run training directly in main thread
        try:
            _create_and_run(cfg, shared_metrics, args.weeks, args.resume, args.seed)
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        except Exception as e:
            shared_metrics["_error"] = str(e)
            traceback.print_exc()
        _print_summary(shared_metrics)
        return

    # Check PyQt6 before proceeding
    try:
        from PyQt6.QtWidgets import QApplication
        from src.training.qt_dashboard import SpartusQtDashboard
        has_qt = True
    except ImportError:
        has_qt = False
        print("PyQt6 not installed — running without dashboard.")
        print("Install with: pip install PyQt6 pyqtgraph")

    if has_qt:
        app = QApplication(sys.argv)
        # Set default font to prevent "QFont::setPointSize: Point size <= 0 (-1)"
        # warning from pyqtgraph's axis labels on Windows
        from PyQt6.QtGui import QFont
        default_font = QFont("Segoe UI", 10)
        app.setFont(default_font)

        dashboard = SpartusQtDashboard(cfg, shared_metrics)

        # Give dashboard the ability to launch training on user command
        dashboard.set_training_launcher(
            train_fn=_run_training_thread,
            max_weeks=args.weeks,
            seed=args.seed,
        )

        # If --resume passed via CLI, auto-start in resume mode
        if args.resume:
            dashboard.auto_resume_on_show()

        dashboard.show()
        app.exec()  # Blocks until window is closed

        # Dashboard closed — signal training to stop
        shared_metrics["_quit_requested"] = True

        # Wait for training thread to finish cleanly (if one was started)
        train_thread = shared_metrics.get("_train_thread")
        if train_thread and train_thread.is_alive():
            train_thread.join(timeout=60)

        _print_summary(shared_metrics)
    else:
        # No Qt — run with --resume flag from CLI directly
        try:
            _create_and_run(cfg, shared_metrics, args.weeks, args.resume, args.seed)
        except KeyboardInterrupt:
            shared_metrics["_quit_requested"] = True
            print("\nTraining interrupted.")
        except Exception as e:
            shared_metrics["_error"] = str(e)
            traceback.print_exc()
        _print_summary(shared_metrics)


if __name__ == "__main__":
    main()
