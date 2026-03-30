"""Training monitor -- read-only inspection of live training state.

Reads from SQLite memory DB, training_state.json, and JSONL logs without
interfering with the running training process. Safe to run from a separate
terminal or from Claude in any conversation.

Usage:
    python scripts/monitor.py                  # Full status report
    python scripts/monitor.py --trades 10      # Last N trades
    python scripts/monitor.py --journal 10     # Last N journal entries
    python scripts/monitor.py --predictions    # Prediction accuracy stats
    python scripts/monitor.py --checkpoints    # List saved checkpoints
    python scripts/monitor.py --internals      # AI internals from JSONL logs
    python scripts/monitor.py --hardware       # GPU/CPU/RAM utilization
    python scripts/monitor.py --alerts 20      # Last N alerts
    python scripts/monitor.py --health         # Learning health diagnostic
    python scripts/monitor.py --evolution      # AI growth / evolution over time
    python scripts/monitor.py --deep           # Full deep report (everything)
    python scripts/monitor.py --watch          # Auto-refresh every 10s
"""

import argparse
import json
import os
import sqlite3
import subprocess
import time
import sys
from pathlib import Path

# Force UTF-8 on Windows terminals to avoid cp1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass  # Fallback: ASCII-safe chars used throughout anyway

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import TrainingConfig


def get_db_connection(cfg: TrainingConfig):
    """Open read-only connection to memory DB."""
    db_path = cfg.memory_db_path
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _read_last_jsonl_lines(path: Path, n: int = 1):
    """Read last N lines from a JSONL file efficiently."""
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            # Seek from end to find last N newlines
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []
            # Read up to 100KB from the end (enough for N recent entries)
            read_size = min(size, 100_000 * n)
            f.seek(size - read_size)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.strip().split("\n")
        results = []
        for line in lines[-n:]:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return results
    except Exception:
        return []


def show_training_state(cfg: TrainingConfig):
    """Show current training state from JSON file."""
    state_path = cfg.training_state_path
    if not state_path.exists():
        print("No training state file found -- training may not have started.")
        return

    with open(state_path) as f:
        state = json.load(f)

    print_header("TRAINING STATE")
    print(f"  Current Week:    {state.get('current_week', '?')}")
    print(f"  Balance:         GBP {state.get('balance', 0):.2f}")
    print(f"  Peak Balance:    GBP {state.get('peak_balance', 0):.2f}")
    drawdown = 0
    peak = state.get('peak_balance', 0)
    bal = state.get('balance', 0)
    if peak > 0:
        drawdown = (peak - bal) / peak
    print(f"  Drawdown:        {drawdown:.1%}")
    print(f"  Seed:            {state.get('seed', '?')}")

    conv = state.get('convergence', {})
    print(f"\n  Convergence:     {conv.get('state', '?')}")
    bvs = conv.get('best_val_sharpe', '?')
    print(f"  Best Val Sharpe: {bvs}")
    print(f"  Weeks Since Best:{conv.get('weeks_since_best', '?')}")
    print(f"  Val Points:      {conv.get('n_val_points', 0)}")

    ts = state.get('timestamp', 0)
    if ts:
        elapsed = time.time() - ts
        if elapsed < 60:
            ago = f"{elapsed:.0f}s ago"
        elif elapsed < 3600:
            ago = f"{elapsed/60:.0f}m ago"
        else:
            ago = f"{elapsed/3600:.1f}h ago"
        print(f"\n  Last Updated:    {ago}")


def show_trade_summary(conn):
    """Show trade statistics."""
    total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if total == 0:
        print("  No trades recorded yet.")
        return

    wins = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0]
    losses = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl <= 0").fetchone()[0]
    total_pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()[0]
    avg_pnl = conn.execute("SELECT COALESCE(AVG(pnl), 0) FROM trades").fetchone()[0]
    avg_hold = conn.execute("SELECT COALESCE(AVG(hold_bars), 0) FROM trades").fetchone()[0]
    gross_profit = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE pnl > 0").fetchone()[0]
    gross_loss = conn.execute("SELECT COALESCE(SUM(ABS(pnl)), 0) FROM trades WHERE pnl < 0").fetchone()[0]

    win_rate = wins / total if total > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    print_header("TRADE SUMMARY")
    print(f"  Total Trades:    {total}")
    print(f"  Win Rate:        {win_rate:.1%} ({wins}W / {losses}L)")
    print(f"  Total P/L:       GBP {total_pnl:+.2f}")
    print(f"  Avg Trade P/L:   GBP {avg_pnl:+.4f}")
    print(f"  Profit Factor:   {profit_factor:.2f}")
    print(f"  Avg Hold Bars:   {avg_hold:.1f}")

    # Per-week breakdown
    weeks = conn.execute("""
        SELECT week, COUNT(*) as cnt, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as pnl
        FROM trades GROUP BY week ORDER BY week
    """).fetchall()
    if weeks:
        print(f"\n  Per-Week Breakdown:")
        print(f"  {'Week':>6} {'Trades':>8} {'Win%':>8} {'P/L':>12}")
        for w, cnt, ww, wpnl in weeks:
            wr = ww / cnt if cnt > 0 else 0
            print(f"  {w:>6} {cnt:>8} {wr:>7.0%}  GBP {wpnl:>+10.2f}")


def show_recent_trades(conn, n: int = 10):
    """Show the most recent N trades."""
    rows = conn.execute("""
        SELECT id, week, side, entry_price, exit_price, pnl, hold_bars,
               close_reason, conviction, lot_size
        FROM trades ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()

    print_header(f"LAST {n} TRADES")
    if not rows:
        print("  No trades yet.")
        return

    print(f"  {'ID':>5} {'Wk':>3} {'Side':>5} {'Entry':>10} {'Exit':>10} {'P/L':>12} {'Bars':>5} {'Reason':>15} {'Conv':>5} {'Lots':>6}")
    print(f"  {'-'*87}")
    for r in rows:
        tid, wk, side, entry, exit_p, pnl, hold, reason, conv, lots = r
        pnl_str = f"GBP {pnl:+.4f}"
        print(f"  {tid:>5} {wk:>3} {side:>5} {entry:>10.2f} {exit_p:>10.2f} {pnl_str:>12} {hold:>5} {reason:>15} {conv:>5.2f} {lots:>6.2f}")


def show_journal(conn, n: int = 10):
    """Show recent journal entries."""
    rows = conn.execute("""
        SELECT j.trade_id, j.week, j.lesson_type, j.sl_quality,
               j.direction_correct, j.summary, t.side, t.pnl, t.hold_bars
        FROM trade_journal j
        LEFT JOIN trades t ON j.trade_id = t.id
        ORDER BY j.id DESC LIMIT ?
    """, (n,)).fetchall()

    print_header(f"LAST {n} JOURNAL ENTRIES")
    if not rows:
        print("  No journal entries yet.")
        return

    for r in rows:
        tid, wk, lesson, sl_q, dir_c, summary, side, pnl, hold = r
        dir_icon = "Y" if dir_c == 1 else "N" if dir_c == 0 else "?"
        pnl_str = f"GBP {pnl:+.4f}" if pnl is not None else "?"
        print(f"  Trade #{tid} | Week {wk} | {side} {pnl_str} | {hold} bars")
        print(f"    Lesson: {lesson} | Direction: {dir_icon} | SL Quality: {sl_q}")
        print(f"    {summary}")
        print()


def show_predictions(conn):
    """Show prediction accuracy statistics."""
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    verified = conn.execute("SELECT COUNT(*) FROM predictions WHERE verified_at_step IS NOT NULL").fetchone()[0]
    pending = total - verified

    print_header("PREDICTION STATS")
    print(f"  Total Predictions: {total}")
    print(f"  Verified:          {verified}")
    print(f"  Pending:           {pending}")

    if verified > 0:
        correct = conn.execute("SELECT SUM(correct) FROM predictions WHERE verified_at_step IS NOT NULL").fetchone()[0]
        accuracy = correct / verified if verified > 0 else 0
        print(f"  Overall Accuracy:  {accuracy:.1%}")

        # UP/DOWN split
        up = conn.execute("""
            SELECT COUNT(*), SUM(correct) FROM predictions
            WHERE verified_at_step IS NOT NULL AND predicted_direction > 0
        """).fetchone()
        down = conn.execute("""
            SELECT COUNT(*), SUM(correct) FROM predictions
            WHERE verified_at_step IS NOT NULL AND predicted_direction < 0
        """).fetchone()

        if up[0] > 0:
            print(f"  UP Accuracy:       {up[1]/up[0]:.1%} ({up[0]} predictions)")
        if down[0] > 0:
            print(f"  DOWN Accuracy:     {down[1]/down[0]:.1%} ({down[0]} predictions)")

    # TP/SL stats
    tp_total = conn.execute("SELECT COUNT(*) FROM tp_tracking").fetchone()[0]
    if tp_total > 0:
        tp_hits = conn.execute("SELECT COUNT(*) FROM tp_tracking WHERE tp_hit=1").fetchone()[0]
        sl_hits = conn.execute("SELECT COUNT(*) FROM tp_tracking WHERE sl_hit=1").fetchone()[0]
        print(f"\n  TP Hit Rate:       {tp_hits/tp_total:.1%}")
        print(f"  SL Hit Rate:       {sl_hits/tp_total:.1%}")


def show_checkpoints(cfg: TrainingConfig):
    """List saved model checkpoints."""
    model_dir = cfg.model_dir
    if not model_dir.exists():
        print("  No model directory found.")
        return

    print_header("MODEL CHECKPOINTS")
    files = sorted(model_dir.glob("spartus_*.zip"))
    if not files:
        print("  No checkpoints found.")
        return

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
        print(f"  {f.name:35s}  {size_mb:6.1f} MB  {mtime}")

    print(f"\n  Total: {len(files)} files, "
          f"{sum(f.stat().st_size for f in files) / (1024*1024):.0f} MB")


def show_lesson_breakdown(conn):
    """Show lesson type distribution from journal."""
    rows = conn.execute("""
        SELECT lesson_type, COUNT(*) as cnt,
               SUM(CASE WHEN direction_correct=1 THEN 1 ELSE 0 END) as dir_ok
        FROM trade_journal GROUP BY lesson_type ORDER BY cnt DESC
    """).fetchall()

    if not rows:
        return

    print_header("LESSON TYPE BREAKDOWN")
    total = sum(r[1] for r in rows)
    for lesson, cnt, dir_ok in rows:
        pct = cnt / total if total > 0 else 0
        print(f"  {lesson:30s}  {cnt:>4}  ({pct:>5.1%})  DirOK: {dir_ok}/{cnt}")


# ============================================================
# NEW: AI Internals from JSONL logs
# ============================================================

def show_ai_internals(cfg: TrainingConfig):
    """Show AI internals from the latest training_log.jsonl entry."""
    log_path = cfg.log_dir / "training_log.jsonl"
    entries = _read_last_jsonl_lines(log_path, n=1)
    if not entries:
        print("  No training log data found.")
        return

    d = entries[0]
    print_header("AI INTERNALS (from training log)")

    # SAC Network State
    print("  -- SAC Network --")
    print(f"  Entropy Alpha:     {d.get('entropy_alpha', '?'):.6f}")
    print(f"  Policy Entropy %:  {d.get('policy_entropy_pct', '?'):.1f}%")
    print(f"  Actor Loss:        {d.get('actor_loss', '?'):.4f}")
    print(f"  Critic Loss:       {d.get('critic_loss', '?'):.4f}")
    print(f"  Q-Value Mean:      {d.get('q_value_mean', '?'):.4f}")
    print(f"  Q-Value Max:       {d.get('q_value_max', '?'):.4f}")
    print(f"  Actor Grad Norm:   {d.get('actor_grad_norm', '?'):.4f}")
    print(f"  Critic Grad Norm:  {d.get('critic_grad_norm', '?'):.4f}")
    print(f"  Grad Clip %:       {d.get('grad_clip_pct', 0):.1f}%")
    print(f"  Learning Rate:     {d.get('learning_rate', '?'):.2e}")
    print(f"  Replay Buffer:     {d.get('buffer_pct', 0):.1f}% full")

    # Action Distribution
    print("\n  -- Action Distribution --")
    print(f"  Action Mean:       {d.get('action_mean', '?'):.4f}")
    print(f"  Action Std:        {d.get('action_std', '?'):.4f}")

    # Reward Components
    print("\n  -- Reward Breakdown --")
    print(f"  Raw Reward:        {d.get('raw_reward', 0):.6f}")
    print(f"  Normalized Reward: {d.get('reward', 0):.6f}")
    print(f"    R1 Position PnL: {d.get('r1_position_pnl', 0):.6f}")
    print(f"    R2 Trade Quality:{d.get('r2_trade_quality', 0):.6f}")
    print(f"    R3 Drawdown:     {d.get('r3_drawdown', 0):.6f}")
    print(f"    R4 Sharpe:       {d.get('r4_sharpe', 0):.6f}")
    print(f"    R5 Risk Bonus:   {d.get('r5_risk_bonus', 0):.6f}")
    print(f"  Reward Clip %:     {d.get('reward_clip_pct', 0):.1f}%")
    print(f"  Reward Run Mean:   {d.get('reward_running_mean', 0):.6f}")
    print(f"  Reward Run Std:    {d.get('reward_running_std', 0):.6f}")

    # Journal Metrics
    print("\n  -- Journal Self-Reflection --")
    print(f"  Direction Accuracy:{d.get('journal_direction_accuracy', 0):.1%}")
    print(f"  SL Quality Score:  {d.get('journal_sl_quality_score', 0):.1%}")
    print(f"  Early Close Rate:  {d.get('journal_early_close_rate', 0):.1%}")
    print(f"  Wrong Dir Rate:    {d.get('journal_wrong_direction_rate', 0):.1%}")
    print(f"  Good Trade Rate:   {d.get('journal_good_trade_rate', 0):.1%}")

    # Safety/Anti-hack
    print("\n  -- Safety Counters --")
    print(f"  Trade Cap Hits:    {d.get('trade_cap_hits', 0)}")
    print(f"  Hold Blocks:       {d.get('hold_blocks', 0)}")
    print(f"  Conviction Blocks: {d.get('conviction_blocks', 0)}")

    # Throughput
    print("\n  -- Performance --")
    print(f"  Steps/sec:         {d.get('steps_per_sec', 0):.1f}")
    print(f"  Timestep:          {d.get('timestep', '?')}")
    print(f"  Week:              {d.get('week', '?')}")

    ts = d.get('timestamp', 0)
    if ts:
        elapsed = time.time() - ts
        if elapsed < 60:
            ago = f"{elapsed:.0f}s ago"
        elif elapsed < 3600:
            ago = f"{elapsed/60:.0f}m ago"
        else:
            ago = f"{elapsed/3600:.1f}h ago"
        print(f"  Log Age:           {ago}")


# ============================================================
# NEW: Hardware Utilization
# ============================================================

def show_hardware():
    """Show GPU, CPU, and RAM utilization."""
    print_header("HARDWARE UTILIZATION")

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            print(f"  GPU Utilization:   {parts[0]}%")
            print(f"  GPU Memory:        {parts[1]} / {parts[2]} MiB")
            print(f"  GPU Temperature:   {parts[3]}C")
        else:
            print("  GPU: nvidia-smi not available")
    except Exception:
        print("  GPU: nvidia-smi not available")

    # CPU
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "LoadPercentage,NumberOfCores,NumberOfLogicalProcessors",
             "/format:csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if len(lines) >= 2:
                parts = lines[-1].split(",")
                if len(parts) >= 4:
                    print(f"  CPU Load:          {parts[1]}%")
                    print(f"  CPU Cores:         {parts[2]} cores / {parts[3]} threads")
    except Exception:
        print("  CPU: wmic not available")

    # RAM
    try:
        result = subprocess.run(
            ["wmic", "os", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/format:csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if len(lines) >= 2:
                parts = lines[-1].split(",")
                if len(parts) >= 3:
                    free_mb = int(parts[1]) / 1024
                    total_mb = int(parts[2]) / 1024
                    used_mb = total_mb - free_mb
                    pct = used_mb / total_mb * 100 if total_mb > 0 else 0
                    print(f"  RAM Used:          {used_mb:.0f} / {total_mb:.0f} MB ({pct:.0f}%)")
                    print(f"  RAM Free:          {free_mb:.0f} MB")
    except Exception:
        print("  RAM: wmic not available")


# ============================================================
# NEW: Recent Alerts
# ============================================================

def show_alerts(cfg: TrainingConfig, n: int = 20):
    """Show last N alerts from alerts.log."""
    alerts_path = cfg.log_dir / "alerts.log"
    if not alerts_path.exists():
        print("  No alerts log found.")
        return

    print_header(f"LAST {n} ALERTS")
    try:
        with open(alerts_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 50_000)
            f.seek(size - read_size)
            data = f.read().decode("utf-8", errors="replace")
        lines = [l for l in data.strip().split("\n") if l.strip()]
        for line in lines[-n:]:
            print(f"  {line}")
    except Exception as e:
        print(f"  Error reading alerts: {e}")


# ============================================================
# NEW: Learning Health Diagnostic
# ============================================================

def show_health(cfg: TrainingConfig, conn):
    """Comprehensive learning health diagnostic.

    Reads from weekly summaries and training log to assess whether the AI
    is learning properly or drifting.
    """
    print_header("LEARNING HEALTH DIAGNOSTIC")

    # --- Weekly Progression ---
    weekly_path = cfg.log_dir / "weekly_summary.jsonl"
    weeks = _read_last_jsonl_lines(weekly_path, n=50)

    if len(weeks) < 2:
        print("  Not enough weekly data for health analysis (need >= 2 weeks).")
        print(f"  Weeks available: {len(weeks)}")
        return

    print(f"  Weeks analyzed: {len(weeks)}")

    # Balance trend
    balances = [w.get("balance", 0) for w in weeks]
    first_half = balances[:len(balances)//2]
    second_half = balances[len(balances)//2:]
    avg_first = sum(first_half) / len(first_half) if first_half else 0
    avg_second = sum(second_half) / len(second_half) if second_half else 0

    if avg_second > avg_first * 1.05:
        bal_status = "IMPROVING"
    elif avg_second < avg_first * 0.8:
        bal_status = "DEGRADING"
    else:
        bal_status = "FLAT"
    print(f"\n  Balance Trend:     {bal_status}")
    print(f"    First half avg:  GBP {avg_first:.2f}")
    print(f"    Second half avg: GBP {avg_second:.2f}")

    # Action std trend (collapse detection)
    stds = [w.get("action_std", 0) for w in weeks if w.get("action_std")]
    if stds:
        latest_std = stds[-1]
        avg_std = sum(stds) / len(stds)
        if latest_std < 0.05:
            std_status = "!! COLLAPSED !!"
        elif latest_std < 0.2:
            std_status = "LOW (risk of collapse)"
        elif latest_std > 0.9:
            std_status = "HIGH (still exploring)"
        else:
            std_status = "HEALTHY"
        print(f"\n  Action Std:        {std_status}")
        print(f"    Latest:          {latest_std:.4f}")
        print(f"    Average:         {avg_std:.4f}")

    # Sharpe trend (filter out extreme outliers from early training)
    sharpes = [w.get("sharpe", 0) for w in weeks if "sharpe" in w]
    sharpes = [s for s in sharpes if abs(s) < 100]  # Filter garbage values
    if len(sharpes) >= 4:
        early = sum(sharpes[:len(sharpes)//2]) / (len(sharpes)//2)
        late = sum(sharpes[len(sharpes)//2:]) / (len(sharpes) - len(sharpes)//2)
        if late > early + 0.05:
            sharpe_status = "IMPROVING"
        elif late < early - 0.05:
            sharpe_status = "DEGRADING"
        else:
            sharpe_status = "FLAT"
        print(f"\n  Sharpe Trend:      {sharpe_status}")
        print(f"    Early avg:       {early:.4f}")
        print(f"    Late avg:        {late:.4f}")

    # Training speed
    speeds = [w.get("train_time_s", 0) for w in weeks if w.get("train_time_s")]
    if speeds:
        avg_speed = sum(speeds) / len(speeds)
        print(f"\n  Avg Week Duration: {avg_speed:.0f}s ({avg_speed/60:.1f}m)")

    # --- Trade quality from DB ---
    if conn:
        # Win rate per 3-week window to see progression
        week_stats = conn.execute("""
            SELECT week, COUNT(*) as cnt,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl) as avg_pnl,
                   AVG(hold_bars) as avg_hold
            FROM trades GROUP BY week ORDER BY week
        """).fetchall()

        if len(week_stats) >= 4:
            print(f"\n  -- Win Rate Progression --")
            print(f"  {'Week':>6} {'WinRate':>8} {'AvgPnL':>10} {'AvgHold':>8}")
            for w, cnt, wins, avg_pnl, avg_hold in week_stats:
                wr = wins / cnt if cnt > 0 else 0
                print(f"  {w:>6} {wr:>7.0%}  GBP {avg_pnl:>+7.4f} {avg_hold:>7.1f}")

            # Overall trend
            early_wr = sum(r[2] for r in week_stats[:len(week_stats)//2]) / \
                       max(1, sum(r[1] for r in week_stats[:len(week_stats)//2]))
            late_wr = sum(r[2] for r in week_stats[len(week_stats)//2:]) / \
                      max(1, sum(r[1] for r in week_stats[len(week_stats)//2:]))

            if late_wr > early_wr + 0.03:
                wr_status = "IMPROVING"
            elif late_wr < early_wr - 0.03:
                wr_status = "DEGRADING"
            else:
                wr_status = "FLAT"
            print(f"\n  Win Rate Trend:    {wr_status} ({early_wr:.0%} -> {late_wr:.0%})")

        # Circuit breaker frequency (should decrease over time)
        es_stats = conn.execute("""
            SELECT week, COUNT(*) as total,
                   SUM(CASE WHEN close_reason IN ('EMERGENCY_STOP','CIRCUIT_BREAKER') THEN 1 ELSE 0 END) as es
            FROM trades GROUP BY week ORDER BY week
        """).fetchall()

        if len(es_stats) >= 4:
            early_es = sum(r[2] for r in es_stats[:len(es_stats)//2]) / \
                       max(1, sum(r[1] for r in es_stats[:len(es_stats)//2]))
            late_es = sum(r[2] for r in es_stats[len(es_stats)//2:]) / \
                      max(1, sum(r[1] for r in es_stats[len(es_stats)//2:]))

            if late_es < early_es - 0.05:
                es_status = "IMPROVING (fewer emergency stops)"
            elif late_es > early_es + 0.05:
                es_status = "DEGRADING (more emergency stops)"
            else:
                es_status = "FLAT"
            print(f"  Emergency Stop:    {es_status} ({early_es:.0%} -> {late_es:.0%})")

    # --- Latest log entry for real-time metrics ---
    log_path = cfg.log_dir / "training_log.jsonl"
    entries = _read_last_jsonl_lines(log_path, n=1)
    if entries:
        d = entries[0]
        entropy_alpha = d.get("entropy_alpha", 0)
        action_std = d.get("action_std", 0)

        print(f"\n  -- Real-Time Health Flags --")
        flags = []
        if entropy_alpha < 0.01:
            flags.append("  [WARN] Entropy alpha very low ({:.6f}) -- exploration may be dying".format(entropy_alpha))
        if action_std < 0.1:
            flags.append("  [CRIT] Action std collapsed ({:.4f}) -- model may be stuck".format(action_std))
        if d.get("grad_clip_pct", 0) > 50:
            flags.append("  [WARN] Gradient clipping at {:.0f}% -- training may be unstable".format(d["grad_clip_pct"]))
        if d.get("critic_grad_norm", 0) > 100:
            flags.append("  [WARN] Critic gradient norm high ({:.1f}) -- potential instability".format(d["critic_grad_norm"]))
        if d.get("buffer_pct", 0) < 10:
            flags.append("  [INFO] Replay buffer only {:.0f}% full -- early training".format(d["buffer_pct"]))
        if d.get("journal_wrong_direction_rate", 0) > 0.6:
            flags.append("  [WARN] Wrong direction rate {:.0%} -- model not learning direction".format(d["journal_wrong_direction_rate"]))

        if flags:
            for f in flags:
                print(f)
        else:
            print("  No health flags -- all metrics within normal range.")


# ============================================================
# NEW: AI Evolution Report — week-by-week growth tracking
# ============================================================

def show_evolution(cfg: TrainingConfig, conn):
    """Week-by-week AI evolution report.

    Tracks how the AI's 'education level' progresses over time across
    multiple skill dimensions. Like a school report card for every week.
    """
    print_header("AI EVOLUTION REPORT")

    if not conn:
        print("  No database connection -- cannot generate evolution report.")
        return

    # === Gather per-week trade metrics from DB ===
    trade_weeks = conn.execute("""
        SELECT week, COUNT(*) as trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               AVG(pnl) as avg_pnl,
               SUM(pnl) as total_pnl,
               AVG(hold_bars) as avg_hold,
               SUM(CASE WHEN close_reason IN ('EMERGENCY_STOP','CIRCUIT_BREAKER') THEN 1 ELSE 0 END) as es_count,
               SUM(CASE WHEN close_reason='TP_HIT' THEN 1 ELSE 0 END) as tp_count,
               SUM(CASE WHEN close_reason='AGENT_CLOSE' THEN 1 ELSE 0 END) as agent_close,
               AVG(conviction) as avg_conviction
        FROM trades GROUP BY week ORDER BY week
    """).fetchall()

    if len(trade_weeks) < 2:
        print("  Not enough weeks to show evolution (need >= 2).")
        return

    # === Gather per-week journal metrics ===
    journal_weeks = {}
    rows = conn.execute("""
        SELECT week, COUNT(*) as cnt,
               SUM(CASE WHEN direction_correct=1 THEN 1 ELSE 0 END) as dir_ok,
               SUM(CASE WHEN lesson_type='GOOD_TRADE' THEN 1 ELSE 0 END) as good,
               SUM(CASE WHEN lesson_type='WRONG_DIRECTION' THEN 1 ELSE 0 END) as wrong_dir,
               SUM(CASE WHEN lesson_type='SCALP_WIN' THEN 1 ELSE 0 END) as scalp,
               SUM(CASE WHEN lesson_type IN ('EMERGENCY_STOP','CIRCUIT_BREAKER') THEN 1 ELSE 0 END) as es
        FROM trade_journal GROUP BY week ORDER BY week
    """).fetchall()
    for r in rows:
        journal_weeks[r[0]] = r

    # === Gather per-week prediction accuracy ===
    pred_weeks = {}
    rows = conn.execute("""
        SELECT week, COUNT(*) as total,
               SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN verified_at_step IS NOT NULL THEN 1 ELSE 0 END) as verified
        FROM predictions GROUP BY week ORDER BY week
    """).fetchall()
    for r in rows:
        pred_weeks[r[0]] = r

    # === Gather weekly summary data (action_std, sharpe) ===
    weekly_path = cfg.log_dir / "weekly_summary.jsonl"
    weekly_data = {}
    all_weeks_jsonl = _read_last_jsonl_lines(weekly_path, n=200)
    for w in all_weeks_jsonl:
        wk = w.get("week")
        if wk is not None:
            weekly_data[wk] = w

    # === Print Week-by-Week Report Card ===
    print("\n  WEEK-BY-WEEK SCORECARD")
    print(f"  {'Wk':>4} {'WinR':>6} {'AvgPnL':>8} {'DirAcc':>7} {'PredAcc':>8} "
          f"{'ES%':>5} {'TP%':>5} {'Hold':>5} {'ActStd':>7} {'Grade':>7}")
    print(f"  {'-'*72}")

    grades = []
    for tw in trade_weeks:
        wk, trades, wins, avg_pnl, total_pnl, avg_hold, es, tp, agent, avg_conv = tw

        win_rate = wins / trades if trades > 0 else 0

        # Direction accuracy from journal
        jw = journal_weeks.get(wk)
        dir_acc = jw[2] / jw[1] if jw and jw[1] > 0 else 0

        # Prediction accuracy
        pw = pred_weeks.get(wk)
        pred_acc = pw[2] / pw[3] if pw and pw[3] > 0 else 0

        # Emergency stop and TP rates
        es_rate = es / trades if trades > 0 else 0
        tp_rate = tp / trades if trades > 0 else 0

        # Action std from weekly summary
        wd = weekly_data.get(wk, {})
        action_std = wd.get("action_std", 0)

        # === Calculate composite grade (0-100) ===
        # Weighted score across key learning dimensions
        # Each dimension scored 0-100, then weighted

        # 1. Win rate (0-100): 0% = 0, 30% = 50, 60% = 100
        wr_score = min(100, win_rate / 0.6 * 100)

        # 2. Direction accuracy (0-100): random = 0, perfect = 100
        dir_score = max(0, (dir_acc - 0.5) * 200)  # 50% = 0, 100% = 100

        # 3. Prediction accuracy (0-100): 50% random = 0, 70% = 100
        pred_score = max(0, (pred_acc - 0.5) / 0.2 * 100)

        # 4. Risk management (0-100): fewer emergency stops = better
        risk_score = max(0, (1 - es_rate) * 100) if trades > 0 else 50

        # 5. Trade quality (0-100): TP hits and good trades
        good_trades = jw[3] if jw else 0
        scalps = jw[5] if jw else 0
        quality_score = min(100, (good_trades + scalps + tp) / max(1, trades) * 300)

        # 6. Exploration health (0-100): action_std 0.3-0.8 is ideal
        if action_std > 0:
            if 0.3 <= action_std <= 0.8:
                explore_score = 100
            elif action_std < 0.05:
                explore_score = 0
            elif action_std < 0.3:
                explore_score = action_std / 0.3 * 100
            else:
                explore_score = max(0, 100 - (action_std - 0.8) * 200)
        else:
            explore_score = 0

        # Composite grade
        grade = (
            wr_score * 0.25 +       # Win rate matters most
            dir_score * 0.20 +       # Getting direction right
            risk_score * 0.20 +      # Not blowing up
            pred_score * 0.15 +      # Prediction ability
            quality_score * 0.10 +   # Quality of winning trades
            explore_score * 0.10     # Still exploring / not collapsed
        )
        grades.append((wk, grade))

        # Grade letter
        if grade >= 80:
            letter = "A"
        elif grade >= 65:
            letter = "B"
        elif grade >= 50:
            letter = "C"
        elif grade >= 35:
            letter = "D"
        elif grade >= 20:
            letter = "E"
        else:
            letter = "F"

        print(f"  {wk:>4} {win_rate:>5.0%} {avg_pnl:>+7.3f} {dir_acc:>6.0%} "
              f"{pred_acc:>7.0%}  {es_rate:>4.0%} {tp_rate:>4.0%} {avg_hold:>5.1f} "
              f"{action_std:>6.3f}  {grade:>4.0f} {letter}")

    # === Evolution Summary ===
    print(f"\n  EVOLUTION SUMMARY")
    print(f"  {'-'*50}")

    if len(grades) >= 4:
        first_q = [g for _, g in grades[:len(grades)//4]] or [0]
        last_q = [g for _, g in grades[-max(1, len(grades)//4):]] or [0]
        first_avg = sum(first_q) / len(first_q)
        last_avg = sum(last_q) / len(last_q)
        overall_avg = sum(g for _, g in grades) / len(grades)
        best_wk, best_grade = max(grades, key=lambda x: x[1])
        worst_wk, worst_grade = min(grades, key=lambda x: x[1])

        print(f"  Overall Grade Avg:   {overall_avg:.0f}/100")
        print(f"  First Quarter Avg:   {first_avg:.0f}/100")
        print(f"  Latest Quarter Avg:  {last_avg:.0f}/100")
        print(f"  Best Week:           Week {best_wk} ({best_grade:.0f}/100)")
        print(f"  Worst Week:          Week {worst_wk} ({worst_grade:.0f}/100)")

        # Trend assessment
        delta = last_avg - first_avg
        if delta > 5:
            trend = "GROWING -- AI is learning and improving"
        elif delta > 0:
            trend = "SLIGHT GROWTH -- marginal improvement"
        elif delta > -5:
            trend = "STAGNANT -- not improving, not degrading"
        elif delta > -15:
            trend = "DECLINING -- performance getting worse"
        else:
            trend = "REGRESSING -- significant degradation"
        print(f"\n  Growth Trend:        {trend}")
        print(f"  Grade Change:        {delta:+.0f} points (first quarter -> latest quarter)")
    elif len(grades) >= 2:
        first_g = grades[0][1]
        last_g = grades[-1][1]
        print(f"  First Week Grade:    {first_g:.0f}/100")
        print(f"  Latest Week Grade:   {last_g:.0f}/100")
        print(f"  Change:              {last_g - first_g:+.0f} points")

    # === Skill Dimension Breakdown (latest vs earliest) ===
    if len(trade_weeks) >= 4:
        print(f"\n  SKILL BREAKDOWN (Early vs Recent)")
        print(f"  {'-'*50}")

        # Calculate per-dimension for early weeks and late weeks
        n_compare = max(1, len(trade_weeks) // 4)
        early_w = trade_weeks[:n_compare]
        late_w = trade_weeks[-n_compare:]

        def _calc_rates(week_list):
            t_trades = sum(w[1] for w in week_list)
            t_wins = sum(w[2] for w in week_list)
            t_es = sum(w[6] for w in week_list)
            t_tp = sum(w[7] for w in week_list)
            wr = t_wins / t_trades if t_trades > 0 else 0
            esr = t_es / t_trades if t_trades > 0 else 0
            tpr = t_tp / t_trades if t_trades > 0 else 0
            avg_pnl = sum(w[3] * w[1] for w in week_list) / t_trades if t_trades > 0 else 0
            avg_hold = sum(w[5] * w[1] for w in week_list) / t_trades if t_trades > 0 else 0
            return wr, esr, tpr, avg_pnl, avg_hold

        e_wr, e_es, e_tp, e_pnl, e_hold = _calc_rates(early_w)
        l_wr, l_es, l_tp, l_pnl, l_hold = _calc_rates(late_w)

        # Direction accuracy early vs late
        e_dir_total = sum(journal_weeks.get(w[0], (0,0,0))[1] for w in early_w if w[0] in journal_weeks)
        e_dir_ok = sum(journal_weeks.get(w[0], (0,0,0))[2] for w in early_w if w[0] in journal_weeks)
        l_dir_total = sum(journal_weeks.get(w[0], (0,0,0))[1] for w in late_w if w[0] in journal_weeks)
        l_dir_ok = sum(journal_weeks.get(w[0], (0,0,0))[2] for w in late_w if w[0] in journal_weeks)
        e_dir = e_dir_ok / e_dir_total if e_dir_total > 0 else 0
        l_dir = l_dir_ok / l_dir_total if l_dir_total > 0 else 0

        def _arrow(early, late, higher_is_better=True):
            diff = late - early
            if higher_is_better:
                if diff > 0.03: return ">>>"
                if diff > 0.01: return ">>"
                if diff > -0.01: return "=="
                if diff > -0.03: return "<<"
                return "<<<"
            else:
                if diff < -0.03: return ">>>"
                if diff < -0.01: return ">>"
                if diff < 0.01: return "=="
                if diff < 0.03: return "<<"
                return "<<<"

        print(f"  {'Skill':>20}  {'Early':>8}  {'Recent':>8}  {'Trend':>5}")
        print(f"  {'Win Rate':>20}  {e_wr:>7.0%}  {l_wr:>7.0%}  {_arrow(e_wr, l_wr):>5}")
        print(f"  {'Direction Accuracy':>20}  {e_dir:>7.0%}  {l_dir:>7.0%}  {_arrow(e_dir, l_dir):>5}")
        print(f"  {'Emergency Stop %':>20}  {e_es:>7.0%}  {l_es:>7.0%}  {_arrow(e_es, l_es, False):>5}")
        print(f"  {'TP Hit Rate':>20}  {e_tp:>7.0%}  {l_tp:>7.0%}  {_arrow(e_tp, l_tp):>5}")
        print(f"  {'Avg P/L per Trade':>20}  {e_pnl:>+7.3f}  {l_pnl:>+7.3f}  {_arrow(e_pnl, l_pnl):>5}")
        print(f"  {'Avg Hold Duration':>20}  {e_hold:>7.1f}  {l_hold:>7.1f}  {_arrow(e_hold, l_hold):>5}")

    # === Milestones Check ===
    print(f"\n  LEARNING MILESTONES")
    print(f"  {'-'*50}")

    total_trades = sum(w[1] for w in trade_weeks)
    total_wins = sum(w[2] for w in trade_weeks)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    any_tp = any(w[7] > 0 for w in trade_weeks)
    any_good_week = any(w[4] > 0 for w in trade_weeks)  # positive total P/L week
    latest_es_rate = trade_weeks[-1][6] / trade_weeks[-1][1] if trade_weeks[-1][1] > 0 else 1
    latest_wr = trade_weeks[-1][2] / trade_weeks[-1][1] if trade_weeks[-1][1] > 0 else 0

    milestones = [
        ("First trade executed",            total_trades > 0),
        ("First winning trade",             total_wins > 0),
        ("First TP hit",                    any_tp),
        ("Win rate above 15%",              overall_wr > 0.15),
        ("Win rate above 25%",              overall_wr > 0.25),
        ("Win rate above 40%",              overall_wr > 0.40),
        ("Profitable week (net positive)",  any_good_week),
        ("Emergency stop rate below 20%",   latest_es_rate < 0.20),
        ("Emergency stop rate below 10%",   latest_es_rate < 0.10),
        ("Prediction accuracy above 55%",   False),  # Will check below
        ("Direction accuracy above 50%",    False),  # Will check below
        ("Profit factor above 0.5",         False),  # Will check below
        ("Profit factor above 1.0",         False),  # Will check below
    ]

    # Check prediction milestone
    total_pred = conn.execute("SELECT COUNT(*) FROM predictions WHERE verified_at_step IS NOT NULL").fetchone()[0]
    correct_pred = conn.execute("SELECT SUM(correct) FROM predictions WHERE verified_at_step IS NOT NULL").fetchone()[0] or 0
    overall_pred_acc = correct_pred / total_pred if total_pred > 0 else 0
    milestones[9] = ("Prediction accuracy above 55%", overall_pred_acc > 0.55)

    # Direction accuracy milestone
    total_jdir = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
    ok_jdir = conn.execute("SELECT SUM(direction_correct) FROM trade_journal").fetchone()[0] or 0
    overall_dir_acc = ok_jdir / total_jdir if total_jdir > 0 else 0
    milestones[10] = ("Direction accuracy above 50%", overall_dir_acc > 0.50)

    # Profit factor milestones
    gp = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE pnl > 0").fetchone()[0]
    gl = conn.execute("SELECT COALESCE(SUM(ABS(pnl)), 0) FROM trades WHERE pnl < 0").fetchone()[0]
    pf = gp / gl if gl > 0 else 0
    milestones[11] = ("Profit factor above 0.5", pf > 0.5)
    milestones[12] = ("Profit factor above 1.0", pf > 1.0)

    achieved = 0
    for label, done in milestones:
        icon = "[X]" if done else "[ ]"
        print(f"  {icon} {label}")
        if done:
            achieved += 1

    print(f"\n  Milestones: {achieved}/{len(milestones)} achieved")

    # === Phase Readiness Check ===
    print(f"\n  PHASE READINESS CHECK")
    print(f"  {'-'*50}")

    # Calculate 10-week rolling averages from the last 10 weeks
    recent_n = min(10, len(trade_weeks))
    recent_weeks = trade_weeks[-recent_n:]

    r_trades = sum(w[1] for w in recent_weeks)
    r_wins = sum(w[2] for w in recent_weeks)
    r_wr = r_wins / r_trades if r_trades > 0 else 0

    r_es = sum(w[6] for w in recent_weeks)
    r_es_rate = r_es / r_trades if r_trades > 0 else 1

    r_dir_total = sum(journal_weeks.get(w[0], (0,0,0))[1] for w in recent_weeks if w[0] in journal_weeks)
    r_dir_ok = sum(journal_weeks.get(w[0], (0,0,0))[2] for w in recent_weeks if w[0] in journal_weeks)
    r_dir_acc = r_dir_ok / r_dir_total if r_dir_total > 0 else 0

    r_gp = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM trades
        WHERE pnl > 0 AND week >= ?
    """, (recent_weeks[0][0],)).fetchone()[0]
    r_gl = conn.execute("""
        SELECT COALESCE(SUM(ABS(pnl)), 0) FROM trades
        WHERE pnl < 0 AND week >= ?
    """, (recent_weeks[0][0],)).fetchone()[0]
    r_pf = r_gp / r_gl if r_gl > 0 else 0

    r_avg_hold = sum(w[5] * w[1] for w in recent_weeks) / r_trades if r_trades > 0 else 0

    # Get action_std from latest weekly summary
    latest_std = 0
    for w in reversed(recent_weeks):
        wd = weekly_data.get(w[0], {})
        if wd.get("action_std", 0) > 0:
            latest_std = wd["action_std"]
            break

    # Get convergence state
    conv_state = "?"
    try:
        with open(cfg.training_state_path) as f:
            ts = json.load(f)
        conv_state = ts.get("convergence", {}).get("state", "?")
    except Exception:
        pass

    # Phase 1 graduation criteria
    p1_criteria = [
        ("Win rate >= 25%",         r_wr >= 0.25,          f"{r_wr:.0%}"),
        ("Emergency stop < 15%",    r_es_rate < 0.15,      f"{r_es_rate:.0%}"),
        ("Direction accuracy >= 35%", r_dir_acc >= 0.35,    f"{r_dir_acc:.0%}"),
        ("Profit factor >= 0.30",   r_pf >= 0.30,          f"{r_pf:.2f}"),
        ("Avg hold >= 2.0 bars",    r_avg_hold >= 2.0,     f"{r_avg_hold:.1f}"),
        ("Action std > 0.2",        latest_std > 0.2,      f"{latest_std:.3f}"),
        ("Not COLLAPSED/WARMING_UP", conv_state not in ("COLLAPSED", "WARMING_UP"), conv_state),
    ]

    p1_passed = sum(1 for _, ok, _ in p1_criteria if ok)
    p1_ready = p1_passed == len(p1_criteria)

    print(f"\n  Phase 1 -> Phase 2 (Pattern Recognition)")
    print(f"  Based on {recent_n}-week rolling average:")
    for label, ok, val in p1_criteria:
        icon = "[X]" if ok else "[ ]"
        print(f"    {icon} {label:35s} (current: {val})")

    if p1_ready:
        print(f"\n  >>> PHASE 1 COMPLETE -- Ready for Phase 2! <<<")
    else:
        remaining = len(p1_criteria) - p1_passed
        print(f"\n  Phase 1: {p1_passed}/{len(p1_criteria)} criteria met ({remaining} remaining)")

    # === Overall Assessment ===
    print(f"\n  OVERALL ASSESSMENT")
    print(f"  {'-'*50}")

    if len(grades) >= 4:
        latest_grade = grades[-1][1]
        if latest_grade >= 60 and delta > 5:
            assessment = "AI is learning well and progressing. On track."
        elif latest_grade >= 40 and delta > 0:
            assessment = "AI is making slow progress. Still in early learning phase."
        elif latest_grade >= 30 and abs(delta) < 5:
            assessment = "AI is stuck in a plateau. May need hyperparameter adjustment."
        elif delta < -10:
            assessment = "AI is regressing. Training may need to be reset or investigated."
        elif latest_grade < 20:
            assessment = "AI is performing very poorly. Review reward function and environment."
        else:
            assessment = "AI is in early training. Give it more time before drawing conclusions."

        print(f"  {assessment}")
    else:
        print("  Too early to assess -- need more training weeks.")


# ============================================================
# Report Modes
# ============================================================

def snapshot_json(cfg: TrainingConfig):
    """Output comprehensive structured JSON snapshot for machine consumption.

    Designed for Claude to parse and analyze training state in a single call.
    Combines all data sources: training_state.json, training_log.jsonl,
    weekly_summary.jsonl, decisions.jsonl, alerts.log, and SQLite DB.
    """
    import math

    snap = {
        "snapshot_time": time.time(),
        "training_running": False,
        "state": {},
        "convergence": {},
        "latest_metrics": {},
        "per_week_summary": [],
        "trade_stats": {},
        "recent_trades": [],
        "recent_alerts": [],
        "action_dimensions": {},
        "obs_health": {},
        "reward_analysis": {},
        "health_flags": [],
        "milestones": {},
    }

    # --- Training state ---
    state_path = cfg.training_state_path
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        elapsed = time.time() - state.get("timestamp", 0)
        snap["training_running"] = elapsed < 600
        snap["state"] = {
            "current_week": state.get("current_week", 0),
            "balance": state.get("balance", 0),
            "peak_balance": state.get("peak_balance", 0),
            "drawdown_pct": ((state.get("peak_balance", 1) - state.get("balance", 0))
                             / max(state.get("peak_balance", 1), 0.01) * 100),
            "bankruptcies": state.get("bankruptcies", 0),
            "best_checkpoint_week": state.get("best_checkpoint_week", 0),
            "seconds_since_update": elapsed,
        }
        conv = state.get("convergence", {})
        # Filter out Inf/NaN from sharpes
        train_sharpes = [s for s in conv.get("train_sharpes", [])
                         if isinstance(s, (int, float)) and math.isfinite(s)]
        snap["convergence"] = {
            "state": conv.get("state", "UNKNOWN"),
            "best_val_sharpe": conv.get("best_val_sharpe") if isinstance(conv.get("best_val_sharpe"), (int, float)) and math.isfinite(conv.get("best_val_sharpe", 0)) else None,
            "weeks_since_best": conv.get("weeks_since_best", 0),
            "n_val_points": conv.get("n_val_points", 0),
            "train_sharpes": train_sharpes,
            "action_stds": conv.get("action_stds", []),
        }

    # --- Latest training log entry ---
    log_path = cfg.log_dir / "training_log.jsonl"
    entries = _read_last_jsonl_lines(log_path, n=1)
    if entries:
        d = entries[0]
        snap["latest_metrics"] = {
            "timestep": d.get("timestep", 0),
            "week": d.get("week", 0),
            "balance": d.get("balance", 0),
            "entropy_alpha": d.get("entropy_alpha", 0),
            "actor_loss": d.get("actor_loss", 0),
            "critic_loss": d.get("critic_loss", 0),
            "q_value_mean": d.get("q_value_mean", 0),
            "q_value_max": d.get("q_value_max", 0),
            "actor_grad_norm": d.get("actor_grad_norm", 0),
            "critic_grad_norm": d.get("critic_grad_norm", 0),
            "grad_clip_pct": d.get("grad_clip_pct", 0),
            "action_mean": d.get("action_mean", 0),
            "action_std": d.get("action_std", 0),
            "reward": d.get("reward", 0),
            "raw_reward": d.get("raw_reward", 0),
            "reward_running_mean": d.get("reward_running_mean", 0),
            "reward_running_std": d.get("reward_running_std", 0),
            "reward_clip_pct": d.get("reward_clip_pct", 0),
            "steps_per_sec": d.get("steps_per_sec", 0),
            "buffer_pct": d.get("buffer_pct", 0),
            "learning_rate": d.get("learning_rate", 0),
            "total_trades": d.get("total_trades", 0),
            "win_rate": d.get("win_rate", 0),
            "profit_factor": d.get("profit_factor", 0),
        }
        # Per-action dimensions (new metrics)
        for dim in ["direction", "conviction", "exit", "sl_mgmt"]:
            m_key = f"act_{dim}_mean"
            s_key = f"act_{dim}_std"
            if m_key in d:
                snap["action_dimensions"][dim] = {
                    "mean": d[m_key], "std": d[s_key],
                }
        # Obs health
        for hk in ("dead_features", "exploding_features", "nan_features"):
            if hk in d:
                snap["obs_health"][hk] = d[hk]
        # Reward components
        snap["reward_analysis"] = {
            "r1_position_pnl": d.get("r1_position_pnl", 0),
            "r2_trade_quality": d.get("r2_trade_quality", 0),
            "r3_drawdown": d.get("r3_drawdown", 0),
            "r4_sharpe": d.get("r4_sharpe", 0),
            "r5_risk_bonus": d.get("r5_risk_bonus", 0),
        }
        # Journal reflection
        snap["journal"] = {
            "direction_accuracy": d.get("journal_direction_accuracy", 0),
            "sl_quality_score": d.get("journal_sl_quality_score", 0),
            "early_close_rate": d.get("journal_early_close_rate", 0),
            "wrong_direction_rate": d.get("journal_wrong_direction_rate", 0),
            "good_trade_rate": d.get("journal_good_trade_rate", 0),
        }

    # --- Per-week summary ---
    weekly_path = cfg.log_dir / "weekly_summary.jsonl"
    all_weeks = _read_last_jsonl_lines(weekly_path, n=200)
    for w in all_weeks:
        snap["per_week_summary"].append({
            "week": w.get("week"),
            "balance": w.get("balance", 0),
            "sharpe": w.get("sharpe", 0) if isinstance(w.get("sharpe"), (int, float)) and math.isfinite(w.get("sharpe", 0)) else 0,
            "action_std": w.get("action_std", 0),
            "episode_trades": w.get("episode_trades", 0),
            "train_time_s": w.get("train_time_s", 0),
            "data_year": w.get("data_year"),
            "data_week": w.get("data_week"),
        })

    # --- Trade stats from DB ---
    conn = get_db_connection(cfg)
    if conn:
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        if total > 0:
            wins = conn.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0]
            total_pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()[0]
            avg_pnl = conn.execute("SELECT COALESCE(AVG(pnl), 0) FROM trades").fetchone()[0]
            avg_hold = conn.execute("SELECT COALESCE(AVG(hold_bars), 0) FROM trades").fetchone()[0]
            gp = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE pnl > 0").fetchone()[0]
            gl = conn.execute("SELECT COALESCE(SUM(ABS(pnl)), 0) FROM trades WHERE pnl < 0").fetchone()[0]

            # Exit reason breakdown
            reasons = conn.execute("""
                SELECT close_reason, COUNT(*) FROM trades GROUP BY close_reason
            """).fetchall()
            reason_map = {r: c for r, c in reasons}

            snap["trade_stats"] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "avg_hold_bars": avg_hold,
                "profit_factor": gp / gl if gl > 0 else 0,
                "exit_reasons": reason_map,
            }

            # Per-week trade breakdown
            week_rows = conn.execute("""
                SELECT week, COUNT(*) as cnt,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as pnl, AVG(pnl) as avg_pnl, AVG(hold_bars) as avg_hold
                FROM trades GROUP BY week ORDER BY week
            """).fetchall()
            snap["per_week_trades"] = [
                {"week": w, "trades": c, "wins": wn, "pnl": p, "avg_pnl": ap, "avg_hold": ah}
                for w, c, wn, p, ap, ah in week_rows
            ]

        # Recent trades
        recent = conn.execute("""
            SELECT week, side, entry_price, exit_price, pnl, hold_bars,
                   close_reason, conviction, lot_size
            FROM trades ORDER BY id DESC LIMIT 20
        """).fetchall()
        snap["recent_trades"] = [
            {"week": r[0], "side": r[1], "entry": r[2], "exit": r[3],
             "pnl": r[4], "hold_bars": r[5], "reason": r[6],
             "conviction": r[7], "lots": r[8]}
            for r in recent
        ]
        conn.close()

    # --- Recent alerts ---
    alerts_path = cfg.log_dir / "alerts.log"
    if alerts_path.exists():
        try:
            with open(alerts_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_size = min(size, 20_000)
                f.seek(size - read_size)
                data = f.read().decode("utf-8", errors="replace")
            lines = [l.strip() for l in data.strip().split("\n") if l.strip()]
            snap["recent_alerts"] = lines[-15:]
        except Exception:
            pass

    # --- Health flags ---
    flags = []
    lm = snap.get("latest_metrics", {})
    if lm.get("entropy_alpha", 1) <= 0.01:
        flags.append("WARN: Entropy alpha at floor (0.01) -- exploration clamped")
    if lm.get("action_std", 1) < 0.1:
        flags.append("CRITICAL: Action std collapsed -- policy may be stuck")
    elif lm.get("action_std", 1) < 0.2:
        flags.append("WARN: Action std low -- risk of collapse")
    if lm.get("grad_clip_pct", 0) > 50:
        flags.append("WARN: Gradient clipping at %.0f%%" % lm["grad_clip_pct"])
    if lm.get("critic_grad_norm", 0) > 100:
        flags.append("WARN: Critic gradient norm high (%.1f)" % lm["critic_grad_norm"])
    journal = snap.get("journal", {})
    if journal.get("wrong_direction_rate", 0) > 0.6:
        flags.append("WARN: Wrong direction rate %.0f%% -- model not learning direction" %
                      (journal["wrong_direction_rate"] * 100))
    ts = snap.get("trade_stats", {})
    if ts.get("total", 0) > 50 and ts.get("win_rate", 0) < 0.20:
        flags.append("CRITICAL: Win rate below 20%% after %d trades" % ts["total"])
    # Check reward hacking (R5 dominance)
    ra = snap.get("reward_analysis", {})
    if (abs(ra.get("r5_risk_bonus", 0)) > 0 and
        ra.get("r1_position_pnl", 0) == 0 and
        ra.get("r2_trade_quality", 0) == 0 and
        ra.get("r4_sharpe", 0) == 0 and
        lm.get("total_trades", 0) < 5):
        flags.append("WARN: Only R5 active -- possible reward hacking (sitting idle)")
    # Convergence stalled
    conv = snap.get("convergence", {})
    if conv.get("n_val_points", 0) == 0 and snap["state"].get("current_week", 0) >= 20:
        flags.append("WARN: No validation points after %d weeks" % snap["state"]["current_week"])
    snap["health_flags"] = flags

    # --- Milestones ---
    snap["milestones"] = {
        "first_trade": ts.get("total", 0) > 0,
        "first_win": ts.get("wins", 0) > 0,
        "win_rate_above_25": ts.get("win_rate", 0) > 0.25,
        "win_rate_above_40": ts.get("win_rate", 0) > 0.40,
        "profit_factor_above_0.5": ts.get("profit_factor", 0) > 0.5,
        "profit_factor_above_1.0": ts.get("profit_factor", 0) > 1.0,
        "profitable_week": any(w.get("pnl", 0) > 0 for w in snap.get("per_week_trades", [])),
    }

    # Clean any non-finite values
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    print(json.dumps(_clean(snap), indent=2))


def full_report(cfg: TrainingConfig):
    """Print standard status report."""
    show_training_state(cfg)

    conn = get_db_connection(cfg)
    if conn:
        show_trade_summary(conn)
        show_lesson_breakdown(conn)
        show_predictions(conn)
        show_recent_trades(conn, n=5)
        conn.close()

    show_checkpoints(cfg)


def deep_report(cfg: TrainingConfig):
    """Print comprehensive deep report -- everything visible."""
    show_training_state(cfg)
    show_hardware()

    conn = get_db_connection(cfg)
    if conn:
        show_trade_summary(conn)
        show_lesson_breakdown(conn)
        show_predictions(conn)
        show_recent_trades(conn, n=10)
    else:
        conn = None

    show_ai_internals(cfg)
    show_health(cfg, conn)
    show_evolution(cfg, conn)
    show_alerts(cfg, n=10)
    show_checkpoints(cfg)

    if conn:
        conn.close()


def watch_mode(cfg: TrainingConfig, interval: int = 10):
    """Auto-refresh full report."""
    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")
            print(f"[Spartus Monitor -- auto-refresh every {interval}s -- Ctrl+C to stop]")
            full_report(cfg)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor Spartus Training")
    parser.add_argument("--trades", type=int, default=0, help="Show last N trades")
    parser.add_argument("--journal", type=int, default=0, help="Show last N journal entries")
    parser.add_argument("--predictions", action="store_true", help="Show prediction stats")
    parser.add_argument("--checkpoints", action="store_true", help="List checkpoints")
    parser.add_argument("--internals", action="store_true", help="Show AI internals")
    parser.add_argument("--hardware", action="store_true", help="Show hardware utilization")
    parser.add_argument("--alerts", type=int, default=0, help="Show last N alerts")
    parser.add_argument("--health", action="store_true", help="Learning health diagnostic")
    parser.add_argument("--evolution", action="store_true", help="AI evolution / growth report")
    parser.add_argument("--deep", action="store_true", help="Full deep report (everything)")
    parser.add_argument("--snapshot", action="store_true", help="JSON snapshot for machine consumption")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 10s")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")
    args = parser.parse_args()

    cfg = TrainingConfig()

    if args.snapshot:
        snapshot_json(cfg)
        return

    if args.watch:
        watch_mode(cfg, args.interval)
        return

    if args.deep:
        deep_report(cfg)
        return

    if args.trades > 0:
        conn = get_db_connection(cfg)
        if conn:
            show_recent_trades(conn, args.trades)
            conn.close()
        return

    if args.journal > 0:
        conn = get_db_connection(cfg)
        if conn:
            show_journal(conn, args.journal)
            conn.close()
        return

    if args.predictions:
        conn = get_db_connection(cfg)
        if conn:
            show_predictions(conn)
            conn.close()
        return

    if args.internals:
        show_ai_internals(cfg)
        return

    if args.hardware:
        show_hardware()
        return

    if args.alerts > 0:
        show_alerts(cfg, args.alerts)
        return

    if args.evolution:
        conn = get_db_connection(cfg)
        show_evolution(cfg, conn)
        if conn:
            conn.close()
        return

    if args.health:
        conn = get_db_connection(cfg)
        show_health(cfg, conn)
        if conn:
            conn.close()
        return

    if args.checkpoints:
        show_checkpoints(cfg)
        return

    # Default: full report
    full_report(cfg)


if __name__ == "__main__":
    main()
