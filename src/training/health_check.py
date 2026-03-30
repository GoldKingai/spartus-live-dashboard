"""Quick health check for the training system.

Usage:
    python -m src.training.health_check
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import TrainingConfig


def check_health():
    """Run a quick health check on all system components."""
    cfg = TrainingConfig()
    issues = []
    ok = []

    # 1. Check data directory
    data_dir = cfg.data_dir
    if data_dir.exists():
        m5_files = list(data_dir.glob("**/*_M5.parquet"))
        ok.append(f"Data: {len(m5_files)} M5 weekly files found")
    else:
        issues.append(f"Data directory missing: {data_dir}")

    # 2. Check feature cache
    feat_dir = cfg.feature_dir
    if feat_dir.exists():
        cached = list(feat_dir.glob("**/*_features.parquet"))
        ok.append(f"Feature cache: {len(cached)} files")
    else:
        ok.append("Feature cache: not yet created (will be built on first run)")

    # 3. Check model directory
    model_dir = cfg.model_dir
    if model_dir.exists():
        models = list(model_dir.glob("*.zip"))
        ok.append(f"Models: {len(models)} checkpoints")
        if (model_dir / "spartus_latest.zip").exists():
            ok.append("  Latest model: exists")
    else:
        ok.append("Models: no checkpoints yet")

    # 4. Check training state
    state_path = cfg.training_state_path
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        ok.append(f"Training state: week {state.get('current_week', '?')}, "
                   f"balance=£{state.get('balance', 0):.2f}")
    else:
        ok.append("Training state: not started")

    # 5. Check memory database
    db_path = cfg.memory_db_path
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        ok.append(f"Memory DB: {trades} trades, {preds} predictions")
    else:
        ok.append("Memory DB: not yet created")

    # 6. Check logs
    log_dir = cfg.log_dir
    if log_dir.exists():
        log_files = list(log_dir.glob("*.jsonl")) + list(log_dir.glob("*.log"))
        ok.append(f"Logs: {len(log_files)} files")
    else:
        ok.append("Logs: not yet created")

    # 7. Check dependencies
    try:
        import torch
        gpu = "CUDA" if torch.cuda.is_available() else "CPU only"
        ok.append(f"PyTorch: {torch.__version__} ({gpu})")
    except ImportError:
        issues.append("PyTorch not installed")

    try:
        import stable_baselines3
        ok.append(f"SB3: {stable_baselines3.__version__}")
    except ImportError:
        issues.append("stable-baselines3 not installed")

    # Print results
    print("\n=== Spartus Training System Health Check ===\n")
    for item in ok:
        print(f"  [OK] {item}")
    for item in issues:
        print(f"  [!!] {item}")

    print(f"\n  Status: {'HEALTHY' if not issues else f'{len(issues)} issues found'}")
    return len(issues) == 0


if __name__ == "__main__":
    healthy = check_health()
    sys.exit(0 if healthy else 1)
