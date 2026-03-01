"""Live trading monitor -- read-only inspection of live trading state.

Reads from SQLite memory DB, JSONL logs, and state files without
interfering with the running live trading system.  Concurrent-safe
via SQLite WAL mode.  Safe to run from a separate terminal.

Usage:
    python live_dashboard/scripts/live_monitor.py                 # Quick status
    python live_dashboard/scripts/live_monitor.py --deep          # Full diagnostics
    python live_dashboard/scripts/live_monitor.py --compare       # Training vs live
    python live_dashboard/scripts/live_monitor.py --session       # Per-session breakdown
    python live_dashboard/scripts/live_monitor.py --health        # Feature health + drift
    python live_dashboard/scripts/live_monitor.py --weekly        # Weekly report summary
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows terminals to avoid cp1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Resolve base directory (live_dashboard/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent

# Data source paths (all relative to live_dashboard/)
_DB_PATH = _BASE_DIR / "storage" / "memory" / "live_trading.db"
_LOGS_DIR = _BASE_DIR / "storage" / "logs"
_STATE_DIR = _BASE_DIR / "storage" / "state"
_NORMALIZER_STATE = _STATE_DIR / "normalizer_state.json"

# Training DB for comparison mode
_TRAINING_DB_PATH = _BASE_DIR.parent / "storage" / "memory" / "spartus_memory.db"
_TRAINING_STATE_PATH = _BASE_DIR.parent / "storage" / "training_state.json"

# ---------------------------------------------------------------------------
# Currency symbol -- read from state file written by the dashboard
# ---------------------------------------------------------------------------
_CURRENCY_SYMBOLS = {
    "USD": "$", "GBP": "\u00a3", "EUR": "\u20ac", "JPY": "\u00a5",
    "AUD": "A$", "CAD": "C$", "CHF": "Fr", "NZD": "NZ$",
    "SGD": "S$", "HKD": "HK$", "ZAR": "R", "PLN": "z\u0142",
}
_CS = "$"  # Will be overwritten by _load_currency_symbol()


def _load_currency_symbol() -> str:
    """Read the account currency symbol from the dashboard state file."""
    global _CS
    ccy_path = _STATE_DIR / "account_currency.json"
    if ccy_path.exists():
        try:
            data = json.loads(ccy_path.read_text(encoding="utf-8"))
            _CS = data.get("symbol", "$")
            return _CS
        except Exception:
            pass
    return _CS


# ============================================================================
# Box-drawing helpers
# ============================================================================

# Box characters
_H = "\u2500"  # horizontal
_V = "\u2502"  # vertical
_TL = "\u250C"  # top-left
_TR = "\u2510"  # top-right
_BL = "\u2514"  # bottom-left
_BR = "\u2518"  # bottom-right
_ML = "\u251C"  # middle-left
_MR = "\u2524"  # middle-right
_TM = "\u252C"  # top-middle
_BM = "\u2534"  # bottom-middle
_CR = "\u253C"  # cross


def _box_header(title: str, width: int = 62):
    """Print a boxed header."""
    inner = width - 2
    print(f"  {_TL}{_H * inner}{_TR}")
    print(f"  {_V} {title:<{inner - 1}}{_V}")
    print(f"  {_BL}{_H * inner}{_BR}")


def _box_row(label: str, value: str, label_w: int = 22):
    """Print a label-value row with box alignment."""
    print(f"    {label:<{label_w}} {value}")


def _box_table(headers: list, rows: list, col_widths: list = None):
    """Print a formatted table with box-drawing characters.

    Args:
        headers: list of column header strings.
        rows: list of tuples/lists of cell values.
        col_widths: optional list of column widths.  Auto-calculated if None.
    """
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for r in rows:
                if i < len(r):
                    max_w = max(max_w, len(str(r[i])))
            col_widths.append(max_w + 1)

    # Top border
    top = _TL + _TM.join(_H * (w + 1) for w in col_widths) + _TR
    mid = _ML + _CR.join(_H * (w + 1) for w in col_widths) + _MR
    bot = _BL + _BM.join(_H * (w + 1) for w in col_widths) + _BR

    print(f"    {top}")

    # Header row
    hdr_cells = []
    for i, h in enumerate(headers):
        hdr_cells.append(f" {str(h):>{col_widths[i]}}")
    print(f"    {_V}{''.join(hdr_cells)} {_V}")
    print(f"    {mid}")

    # Data rows
    for r in rows:
        cells = []
        for i in range(len(headers)):
            val = str(r[i]) if i < len(r) else ""
            cells.append(f" {val:>{col_widths[i]}}")
        print(f"    {_V}{''.join(cells)} {_V}")

    print(f"    {bot}")


# ============================================================================
# Data access helpers
# ============================================================================

def _get_db(db_path: Path = None):
    """Open a read-only WAL connection to the live trading DB."""
    path = db_path or _DB_PATH
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Exception:
        return None


def _read_jsonl_tail(path: Path, n: int) -> list:
    """Read last N entries from a JSONL file (efficient tail)."""
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []
            read_size = min(size, 100_000 * max(1, n))
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.strip().split("\n")
        results = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results
    except Exception:
        return []


def _read_json(path: Path) -> dict:
    """Read a JSON file safely."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _ago(ts_val) -> str:
    """Convert a timestamp (epoch float or ISO string) to a human-readable 'ago' string."""
    if not ts_val:
        return "?"
    try:
        if isinstance(ts_val, str):
            dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
        else:
            elapsed = time.time() - float(ts_val)
    except Exception:
        return "?"

    if elapsed < 0:
        return "in the future"
    if elapsed < 60:
        return f"{elapsed:.0f}s ago"
    if elapsed < 3600:
        return f"{elapsed / 60:.0f}m ago"
    if elapsed < 86400:
        return f"{elapsed / 3600:.1f}h ago"
    return f"{elapsed / 86400:.1f}d ago"


# ============================================================================
# Quick mode sections
# ============================================================================

def show_connection_status():
    """Show whether the DB and key files exist."""
    _box_header("CONNECTION & FILE STATUS")

    files = [
        ("Live DB", _DB_PATH),
        ("Trades log", _LOGS_DIR / "trades.jsonl"),
        ("Actions log", _LOGS_DIR / "actions.jsonl"),
        ("Feature stats", _LOGS_DIR / "feature_stats.jsonl"),
        ("Weekly summary", _LOGS_DIR / "weekly_summary.jsonl"),
        ("Normalizer state", _NORMALIZER_STATE),
    ]
    for label, path in files:
        if path.exists():
            sz = path.stat().st_size
            mod = _ago(path.stat().st_mtime)
            if sz < 1024:
                sz_str = f"{sz} B"
            elif sz < 1024 * 1024:
                sz_str = f"{sz / 1024:.1f} KB"
            else:
                sz_str = f"{sz / (1024 * 1024):.1f} MB"
            _box_row(label, f"OK  ({sz_str}, modified {mod})")
        else:
            _box_row(label, "MISSING")


def show_account_summary(conn):
    """Show account-level summary from trades."""
    _box_header("ACCOUNT SUMMARY")

    total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if total == 0:
        _box_row("Status", "No trades recorded yet")
        return

    row = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
               SUM(pnl),
               AVG(pnl),
               AVG(hold_bars),
               SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END),
               SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END),
               MAX(pnl),
               MIN(pnl),
               SUM(CASE WHEN paper_trade = 1 THEN 1 ELSE 0 END)
        FROM trades
    """).fetchone()

    cnt, wins, net_pnl, avg_pnl, avg_hold, gross_profit, gross_loss, best, worst, paper = row
    losses = cnt - wins
    win_rate = wins / cnt if cnt > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    _box_row("Total Trades", f"{cnt}  ({paper} paper)")
    _box_row("Win Rate", f"{win_rate:.1%}  ({wins}W / {losses}L)")
    _box_row("Net P/L", f"{_CS}{net_pnl:+.2f}")
    _box_row("Avg Trade P/L", f"{_CS}{avg_pnl:+.4f}")
    _box_row("Profit Factor", f"{profit_factor:.2f}")
    _box_row("Avg Hold Bars", f"{avg_hold:.1f}")
    _box_row("Best / Worst", f"{_CS}{best:+.2f} / {_CS}{worst:+.2f}")


def show_position_status(conn):
    """Show current position status from the most recent actions log."""
    _box_header("CURRENT POSITION")

    actions = _read_jsonl_tail(_LOGS_DIR / "actions.jsonl", 1)
    if not actions:
        _box_row("Status", "No action data available")
        return

    a = actions[0]
    has_pos = a.get("has_position", False)
    ts = a.get("timestamp", "?")

    if has_pos:
        _box_row("Position", "OPEN")
        _box_row("Direction", str(a.get("direction", "?")))
        _box_row("Conviction", f"{a.get('conviction', 0):.3f}")
        _box_row("Exit Urgency", f"{a.get('exit_urgency', 0):.3f}")
        _box_row("SL Adjustment", f"{a.get('sl_adjustment', 0):.3f}")
    else:
        _box_row("Position", "FLAT")
        _box_row("Last Decision", str(a.get("decision", "?")))

    _box_row("Balance", f"{_CS}{a.get('balance', 0):.2f}")
    _box_row("Equity", f"{_CS}{a.get('equity', 0):.2f}")
    _box_row("Last Update", _ago(ts))

    if a.get("trade_rejected"):
        _box_row("Rejected", str(a.get("reject_reason", "unknown")))


def show_today_summary(conn):
    """Show today's trade summary."""
    _box_header("TODAY'S SUMMARY")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
               SUM(pnl),
               AVG(pnl)
        FROM trades
        WHERE timestamp LIKE ?
    """, (f"{today}%",)).fetchone()

    cnt, wins, net_pnl, avg_pnl = row
    if cnt is None or cnt == 0:
        _box_row("Today", "No trades today")
        return

    losses = cnt - wins
    win_rate = wins / cnt if cnt > 0 else 0

    _box_row("Trades Today", f"{cnt}")
    _box_row("Win Rate", f"{win_rate:.0%}  ({wins}W / {losses}L)")
    _box_row("Net P/L Today", f"{_CS}{net_pnl:+.2f}")
    _box_row("Avg P/L", f"{_CS}{avg_pnl:+.4f}")


def show_recent_trades(conn, n: int = 5):
    """Show the most recent N trades."""
    _box_header(f"LAST {n} TRADES")

    rows = conn.execute("""
        SELECT id, timestamp, side, entry_price, exit_price, pnl,
               hold_bars, close_reason, conviction, lot_size, paper_trade
        FROM trades ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()

    if not rows:
        _box_row("Status", "No trades yet")
        return

    headers = ["ID", "Time", "Side", "Entry", "Exit", "P/L", "Bars", "Reason", "Conv", "Lots", "Paper"]
    table_rows = []
    for r in rows:
        tid, ts, side, entry, exit_p, pnl, hold, reason, conv, lots, paper = r
        # Truncate timestamp to HH:MM
        ts_short = ts[11:16] if ts and len(ts) > 16 else str(ts)[:5]
        pnl_str = f"{_CS}{pnl:+.2f}"
        paper_str = "Y" if paper else "N"
        table_rows.append((
            str(tid), ts_short, side or "?",
            f"{entry:.2f}", f"{exit_p:.2f}",
            pnl_str, str(hold or 0), (reason or "?")[:12],
            f"{conv:.2f}" if conv else "0.00",
            f"{lots:.3f}" if lots else "0.000", paper_str,
        ))

    _box_table(headers, table_rows, [5, 5, 5, 9, 9, 8, 4, 12, 4, 5, 5])


def show_feature_status():
    """Show feature pipeline health summary."""
    _box_header("FEATURE PIPELINE STATUS")

    # Normalizer state
    norm_state = _read_json(_NORMALIZER_STATE)
    if norm_state:
        _box_row("Normalizer", "LOADED")
        _box_row("  Bars processed", str(norm_state.get("bars_processed", "?")))
        ts = norm_state.get("timestamp", norm_state.get("updated_at"))
        if ts:
            _box_row("  Last updated", _ago(ts))
    else:
        _box_row("Normalizer", "NOT FOUND (cold start)")

    # Feature stats from last log entry
    stats = _read_jsonl_tail(_LOGS_DIR / "feature_stats.jsonl", 1)
    if stats:
        s = stats[0]
        ts = s.get("timestamp")
        _box_row("Feature Stats", f"Available (updated {_ago(ts)})")

        # Check for NaN/Inf warnings
        nan_count = s.get("nan_count", 0)
        inf_count = s.get("inf_count", 0)
        if nan_count > 0 or inf_count > 0:
            _box_row("  WARNINGS", f"NaN={nan_count}  Inf={inf_count}")
        else:
            _box_row("  Data Quality", "OK (no NaN/Inf)")

        # Drift count
        drift_count = s.get("drift_count", 0)
        if drift_count > 0:
            _box_row("  Feature Drift", f"{drift_count} features drifting")
        else:
            _box_row("  Feature Drift", "None detected")
    else:
        _box_row("Feature Stats", "No data yet")


def show_alerts_summary():
    """Show recent alerts summary."""
    _box_header("RECENT ALERTS")

    alerts = _read_jsonl_tail(_LOGS_DIR / "alerts.jsonl", 5)
    if not alerts:
        _box_row("Alerts", "None")
        return

    for a in alerts:
        level = a.get("level", "INFO")
        msg = a.get("message", "?")
        ts = a.get("timestamp", "")
        ts_short = ts[11:19] if ts and len(ts) > 19 else str(ts)[:8]
        icon = "[!]" if level in ("WARNING", "CRITICAL", "ERROR") else "[i]"
        print(f"    {icon} {ts_short} [{level}] {msg}")


# ============================================================================
# Deep mode sections
# ============================================================================

def show_action_distribution():
    """Show action distribution statistics from recent actions."""
    _box_header("ACTION DISTRIBUTION")

    actions = _read_jsonl_tail(_LOGS_DIR / "actions.jsonl", 200)
    if not actions:
        _box_row("Status", "No action data")
        return

    dirs = [a.get("direction", 0) for a in actions if "direction" in a]
    convs = [a.get("conviction", 0) for a in actions if "conviction" in a]
    exits = [a.get("exit_urgency", 0) for a in actions if "exit_urgency" in a]
    sls = [a.get("sl_adjustment", 0) for a in actions if "sl_adjustment" in a]

    def _stats(vals, name):
        if not vals:
            _box_row(name, "no data")
            return
        import statistics
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0
        mn, mx = min(vals), max(vals)
        _box_row(name, f"mean={mean:+.3f}  std={std:.3f}  [{mn:+.3f}, {mx:+.3f}]")

    _stats(dirs, "Direction")
    _stats(convs, "Conviction")
    _stats(exits, "Exit Urgency")
    _stats(sls, "SL Adjustment")

    # Decision distribution
    decisions = [a.get("decision", "?") for a in actions]
    from collections import Counter
    dec_counts = Counter(decisions)
    total = len(decisions)
    print()
    _box_row("Decisions (last 200)", "")
    for dec, cnt in dec_counts.most_common(10):
        pct = cnt / total if total > 0 else 0
        _box_row(f"  {dec}", f"{cnt} ({pct:.0%})")

    # Action std collapse warning
    if dirs:
        import statistics
        d_std = statistics.stdev(dirs) if len(dirs) > 1 else 0
        if d_std < 0.10:
            print(f"\n    [WARN] Direction std = {d_std:.4f} -- possible policy collapse!")
        if d_std < 0.05:
            print(f"    [CRIT] Direction std = {d_std:.4f} -- policy COLLAPSED!")

    # Flat rate warning
    if decisions:
        flat_count = sum(1 for d in decisions if d in ("HOLD", "FLAT", "NO_TRADE"))
        flat_rate = flat_count / len(decisions)
        if flat_rate > 0.90:
            print(f"    [WARN] Flat rate = {flat_rate:.0%} -- model may be too passive")


def show_trade_quality(conn):
    """Show trade quality metrics from the journal table."""
    _box_header("TRADE QUALITY (JOURNAL)")

    # Check if journal table exists
    try:
        rows = conn.execute("""
            SELECT lesson_type, COUNT(*) as cnt
            FROM journal
            GROUP BY lesson_type
            ORDER BY cnt DESC
        """).fetchall()
    except sqlite3.OperationalError:
        _box_row("Journal", "Table not found")
        return

    if not rows:
        _box_row("Journal", "No entries yet")
        return

    total = sum(r[1] for r in rows)
    headers = ["Lesson Type", "Count", "Pct"]
    table_rows = []
    for lesson, cnt in rows:
        pct = f"{cnt / total:.0%}" if total > 0 else "0%"
        table_rows.append((lesson or "?", str(cnt), pct))

    _box_table(headers, table_rows, [20, 6, 6])


def show_session_performance(conn):
    """Show per-session breakdown."""
    _box_header("SESSION PERFORMANCE")

    try:
        rows = conn.execute("""
            SELECT session_at_entry,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as net_pnl,
                   AVG(pnl) as avg_pnl,
                   AVG(hold_bars) as avg_hold
            FROM trades
            WHERE session_at_entry IS NOT NULL
            GROUP BY session_at_entry
            ORDER BY net_pnl DESC
        """).fetchall()
    except sqlite3.OperationalError:
        _box_row("Sessions", "No session data")
        return

    if not rows:
        _box_row("Sessions", "No trade data")
        return

    headers = ["Session", "Trades", "WinRate", "Net P/L", "Avg P/L", "AvgHold"]
    table_rows = []
    for session, cnt, wins, net_pnl, avg_pnl, avg_hold in rows:
        wr = f"{wins / cnt:.0%}" if cnt > 0 else "0%"
        table_rows.append((
            session or "?", str(cnt), wr,
            f"{_CS}{net_pnl:+.2f}", f"{_CS}{avg_pnl:+.3f}", f"{avg_hold:.1f}",
        ))

    _box_table(headers, table_rows, [8, 6, 7, 10, 9, 7])


def show_day_of_week(conn):
    """Show day-of-week P/L breakdown."""
    _box_header("DAY OF WEEK PERFORMANCE")

    try:
        rows = conn.execute("""
            SELECT timestamp, pnl FROM trades
            WHERE timestamp IS NOT NULL
        """).fetchall()
    except sqlite3.OperationalError:
        return

    if not rows:
        _box_row("Day breakdown", "No trades")
        return

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_data = {d: {"count": 0, "pnl": 0.0, "wins": 0} for d in day_names}

    for ts_str, pnl in rows:
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            day = day_names[dt.weekday()]
            day_data[day]["count"] += 1
            day_data[day]["pnl"] += pnl or 0.0
            if pnl and pnl > 0:
                day_data[day]["wins"] += 1
        except (ValueError, TypeError):
            continue

    headers = ["Day", "Trades", "WinRate", "Net P/L"]
    table_rows = []
    for d in day_names:
        dd = day_data[d]
        cnt = dd["count"]
        wr = f"{dd['wins'] / cnt:.0%}" if cnt > 0 else "-"
        table_rows.append((d, str(cnt), wr, f"{_CS}{dd['pnl']:+.2f}"))

    _box_table(headers, table_rows, [5, 6, 7, 10])


def show_feature_drift_detail():
    """Show detailed feature health and drift information."""
    _box_header("FEATURE HEALTH & DRIFT DETAIL")

    stats_list = _read_jsonl_tail(_LOGS_DIR / "feature_stats.jsonl", 5)
    if not stats_list:
        _box_row("Feature stats", "No data available")
        return

    latest = stats_list[-1]
    ts = latest.get("timestamp", "?")
    _box_row("Latest snapshot", _ago(ts))

    # Per-feature drift info
    drifted = latest.get("drifted_features", [])
    if drifted:
        print(f"\n    Drifted features ({len(drifted)}):")
        for feat in drifted:
            if isinstance(feat, dict):
                name = feat.get("name", "?")
                drift = feat.get("drift_sigma", 0)
                print(f"      - {name}: {drift:+.2f} sigma")
            else:
                print(f"      - {feat}")
    else:
        _box_row("Drift", "No features drifting")

    # Correlation drift
    corr_drift = latest.get("correlation_drift", {})
    if corr_drift:
        print(f"\n    Correlation drift:")
        for pair, drift_val in corr_drift.items():
            if isinstance(drift_val, (int, float)):
                level = "RED" if abs(drift_val) > 0.25 else "YELLOW" if abs(drift_val) > 0.15 else "OK"
                print(f"      {pair}: {drift_val:+.3f} [{level}]")

    # Feature summary stats
    feat_means = latest.get("feature_means", {})
    feat_stds = latest.get("feature_stds", {})
    if feat_means and feat_stds:
        print(f"\n    Feature summary (top 10 by |mean|):")
        combined = []
        for name in feat_means:
            m = feat_means[name]
            s = feat_stds.get(name, 0)
            combined.append((name, m, s))
        combined.sort(key=lambda x: abs(x[1]), reverse=True)
        headers = ["Feature", "Mean", "Std"]
        rows_data = [(name[:25], f"{m:.4f}", f"{s:.4f}") for name, m, s in combined[:10]]
        _box_table(headers, rows_data, [25, 8, 8])


def show_hardware():
    """Show GPU, CPU, and RAM utilization."""
    _box_header("HARDWARE UTILIZATION")

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            _box_row("GPU Utilization", f"{parts[0]}%")
            _box_row("GPU Memory", f"{parts[1]} / {parts[2]} MiB")
            _box_row("GPU Temperature", f"{parts[3]}C")
        else:
            _box_row("GPU", "nvidia-smi not available")
    except Exception:
        _box_row("GPU", "nvidia-smi not available")

    # CPU
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "LoadPercentage,NumberOfCores,NumberOfLogicalProcessors",
             "/format:csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
            if len(lines) >= 2:
                parts = lines[-1].split(",")
                if len(parts) >= 4:
                    _box_row("CPU Load", f"{parts[1]}%")
                    _box_row("CPU Cores", f"{parts[2]} cores / {parts[3]} threads")
    except Exception:
        _box_row("CPU", "wmic not available")

    # RAM
    try:
        result = subprocess.run(
            ["wmic", "os", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/format:csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
            if len(lines) >= 2:
                parts = lines[-1].split(",")
                if len(parts) >= 3:
                    free_mb = int(parts[1]) / 1024
                    total_mb = int(parts[2]) / 1024
                    used_mb = total_mb - free_mb
                    pct = used_mb / total_mb * 100 if total_mb > 0 else 0
                    _box_row("RAM Used", f"{used_mb:.0f} / {total_mb:.0f} MB ({pct:.0f}%)")
                    _box_row("RAM Free", f"{free_mb:.0f} MB")
    except Exception:
        _box_row("RAM", "wmic not available")


# ============================================================================
# Compare mode: Training vs Live
# ============================================================================

def show_compare():
    """Compare training vs live metrics side by side."""
    _box_header("TRAINING vs LIVE COMPARISON")

    live_conn = _get_db(_DB_PATH)
    train_conn = _get_db(_TRAINING_DB_PATH)

    if not live_conn:
        _box_row("Live DB", "NOT FOUND")
    if not train_conn:
        _box_row("Training DB", "NOT FOUND")

    if not live_conn and not train_conn:
        return

    def _get_metrics(conn, label):
        """Extract comparable metrics from a trades table."""
        if not conn:
            return {"label": label, "trades": 0, "win_rate": "-",
                    "avg_pnl": "-", "pf": "-", "avg_hold": "-"}
        try:
            row = conn.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                       AVG(pnl),
                       AVG(hold_bars),
                       SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END),
                       SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END)
                FROM trades
            """).fetchone()
        except sqlite3.OperationalError:
            return {"label": label, "trades": 0, "win_rate": "-",
                    "avg_pnl": "-", "pf": "-", "avg_hold": "-"}

        cnt, wins, avg_pnl, avg_hold, gp, gl = row
        if not cnt or cnt == 0:
            return {"label": label, "trades": 0, "win_rate": "-",
                    "avg_pnl": "-", "pf": "-", "avg_hold": "-"}

        win_rate = f"{wins / cnt:.1%}" if cnt > 0 else "-"
        pf = f"{gp / gl:.2f}" if gl and gl > 0 else "999"
        return {
            "label": label,
            "trades": cnt,
            "win_rate": win_rate,
            "avg_pnl": f"{_CS}{avg_pnl:+.4f}" if avg_pnl else "-",
            "pf": pf,
            "avg_hold": f"{avg_hold:.1f}" if avg_hold else "-",
        }

    train_m = _get_metrics(train_conn, "Training")
    live_m = _get_metrics(live_conn, "Live")

    headers = ["Metric", "Training", "Live"]
    rows_data = [
        ("Total Trades", str(train_m["trades"]), str(live_m["trades"])),
        ("Win Rate", train_m["win_rate"], live_m["win_rate"]),
        ("Avg Trade P/L", train_m["avg_pnl"], live_m["avg_pnl"]),
        ("Profit Factor", train_m["pf"], live_m["pf"]),
        ("Avg Hold Bars", train_m["avg_hold"], live_m["avg_hold"]),
    ]

    _box_table(headers, rows_data, [15, 12, 12])

    # Training state info
    train_state = _read_json(_TRAINING_STATE_PATH)
    if train_state:
        print()
        _box_row("Training Week", str(train_state.get("current_week", "?")))
        _box_row("Training Balance", f"{_CS}{train_state.get('balance', 0):.2f}")
        conv = train_state.get("convergence", {})
        _box_row("Convergence", str(conv.get("state", "?")))
        _box_row("Best Val Sharpe", str(conv.get("best_val_sharpe", "?")))

    if live_conn:
        live_conn.close()
    if train_conn:
        train_conn.close()


# ============================================================================
# Session mode
# ============================================================================

def show_session_breakdown():
    """Per-session breakdown table."""
    conn = _get_db()
    if not conn:
        print("  No live trading database found.")
        return

    show_session_performance(conn)
    show_day_of_week(conn)
    conn.close()


# ============================================================================
# Health mode
# ============================================================================

def show_health_report():
    """Feature health + drift details."""
    conn = _get_db()

    show_feature_status()
    show_feature_drift_detail()

    if conn:
        # Check for anomalous patterns
        _box_header("TRADE ANOMALY CHECK")

        # Zero-lot trades
        try:
            zero_lots = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE lot_size <= 0"
            ).fetchone()[0]
            _box_row("Zero-lot trades", str(zero_lots))
        except sqlite3.OperationalError:
            pass

        # Zero-PnL trades (possible no-fill)
        try:
            zero_pnl = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE pnl = 0"
            ).fetchone()[0]
            _box_row("Zero-PnL trades", str(zero_pnl))
        except sqlite3.OperationalError:
            pass

        # Very short trades (< 1 bar)
        try:
            short = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE hold_bars < 1"
            ).fetchone()[0]
            _box_row("Ultra-short (<1 bar)", str(short))
        except sqlite3.OperationalError:
            pass

        # Emergency stops
        try:
            es = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE close_reason IN ('EMERGENCY_STOP', 'CIRCUIT_BREAKER')"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            es_pct = f"{es / total:.0%}" if total > 0 else "0%"
            _box_row("Emergency stops", f"{es} ({es_pct})")
        except sqlite3.OperationalError:
            pass

        conn.close()

    # Action health
    _box_header("MODEL BEHAVIOR HEALTH")
    actions = _read_jsonl_tail(_LOGS_DIR / "actions.jsonl", 100)
    if actions:
        import statistics
        dirs = [a.get("direction", 0) for a in actions if "direction" in a]
        if dirs:
            d_std = statistics.stdev(dirs) if len(dirs) > 1 else 0
            d_mean = statistics.mean(dirs)
            _box_row("Direction std", f"{d_std:.4f}")
            _box_row("Direction mean", f"{d_mean:+.4f}")

            if d_std < 0.05:
                print("    [CRIT] Policy COLLAPSED -- direction std < 0.05")
            elif d_std < 0.10:
                print("    [WARN] Policy at risk -- direction std < 0.10")
            elif d_std < 0.20:
                print("    [INFO] Policy narrowing -- direction std < 0.20")
            else:
                print("    [OK] Policy exploration healthy")

        decisions = [a.get("decision", "") for a in actions]
        flat_count = sum(1 for d in decisions if d in ("HOLD", "FLAT", "NO_TRADE"))
        if decisions:
            flat_rate = flat_count / len(decisions)
            _box_row("Flat rate", f"{flat_rate:.0%}")
            if flat_rate > 0.95:
                print("    [CRIT] Model almost never trades (flat rate > 95%)")
            elif flat_rate > 0.90:
                print("    [WARN] Model very passive (flat rate > 90%)")
    else:
        _box_row("Actions", "No data")


# ============================================================================
# Weekly mode
# ============================================================================

def show_weekly_report():
    """Weekly summary table."""
    _box_header("WEEKLY PERFORMANCE REPORTS")

    summaries = _read_jsonl_tail(_LOGS_DIR / "weekly_summary.jsonl", 52)
    if not summaries:
        _box_row("Weeklies", "No weekly summaries yet")
        return

    headers = ["Week", "Trades", "WinRate", "Net P/L", "Avg P/L", "PF", "Best", "Worst"]
    table_rows = []

    for s in summaries:
        week_label = s.get("week_label", s.get("week", "?"))
        trades = s.get("trades", 0)
        wins = s.get("wins", 0)
        net_pnl = s.get("net_pnl", 0)
        avg_pnl = s.get("avg_pnl", 0)
        pf = s.get("profit_factor", 0)
        best = s.get("best_trade", 0)
        worst = s.get("worst_trade", 0)

        wr = f"{wins / trades:.0%}" if trades > 0 else "-"
        pf_str = f"{pf:.2f}" if pf else "-"

        table_rows.append((
            str(week_label)[:10], str(trades), wr,
            f"{_CS}{net_pnl:+.2f}", f"{_CS}{avg_pnl:+.3f}",
            pf_str, f"{_CS}{best:+.2f}", f"{_CS}{worst:+.2f}",
        ))

    _box_table(headers, table_rows, [10, 6, 7, 10, 9, 6, 8, 8])

    # Totals
    total_trades = sum(s.get("trades", 0) for s in summaries)
    total_pnl = sum(s.get("net_pnl", 0) for s in summaries)
    total_wins = sum(s.get("wins", 0) for s in summaries)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0

    print()
    _box_row("Total Weeks", str(len(summaries)))
    _box_row("Total Trades", str(total_trades))
    _box_row("Overall Win Rate", f"{overall_wr:.1%}")
    _box_row("Cumulative P/L", f"{_CS}{total_pnl:+.2f}")

    # Trend
    if len(summaries) >= 4:
        first_half = summaries[:len(summaries) // 2]
        second_half = summaries[len(summaries) // 2:]
        first_pnl = sum(s.get("net_pnl", 0) for s in first_half)
        second_pnl = sum(s.get("net_pnl", 0) for s in second_half)
        if second_pnl > first_pnl:
            _box_row("P/L Trend", "IMPROVING")
        elif second_pnl < first_pnl * 0.5:
            _box_row("P/L Trend", "DEGRADING")
        else:
            _box_row("P/L Trend", "FLAT")


# ============================================================================
# Report orchestrators
# ============================================================================

def quick_report():
    """Default quick status report."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n  Spartus Live Monitor -- {now_str}")
    print(f"  {'=' * 50}")

    show_connection_status()

    conn = _get_db()
    if conn:
        show_account_summary(conn)
        show_position_status(conn)
        show_today_summary(conn)
        show_recent_trades(conn, 5)
        conn.close()
    else:
        print("\n  No live trading database found at:")
        print(f"    {_DB_PATH}")
        print("  The live trading system may not have started yet.")

    show_feature_status()
    show_alerts_summary()


def deep_report():
    """Full diagnostics report."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n  Spartus Live Monitor (DEEP) -- {now_str}")
    print(f"  {'=' * 50}")

    show_connection_status()

    conn = _get_db()
    if conn:
        show_account_summary(conn)
        show_position_status(conn)
        show_today_summary(conn)
        show_recent_trades(conn, 10)
        show_trade_quality(conn)
        show_session_performance(conn)
        show_day_of_week(conn)
        conn.close()
    else:
        print("\n  No live trading database found.")

    show_action_distribution()
    show_feature_status()
    show_feature_drift_detail()
    show_hardware()
    show_alerts_summary()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Spartus Live Trading Monitor (read-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_dashboard/scripts/live_monitor.py                # Quick status
  python live_dashboard/scripts/live_monitor.py --deep         # Full diagnostics
  python live_dashboard/scripts/live_monitor.py --compare      # Training vs live
  python live_dashboard/scripts/live_monitor.py --session      # Per-session breakdown
  python live_dashboard/scripts/live_monitor.py --health       # Feature health + drift
  python live_dashboard/scripts/live_monitor.py --weekly       # Weekly report summary
""",
    )
    parser.add_argument("--deep", action="store_true",
                        help="Full diagnostics (action distribution, trade quality, "
                             "session performance, feature drift, hardware)")
    parser.add_argument("--compare", action="store_true",
                        help="Training vs live comparison table")
    parser.add_argument("--session", action="store_true",
                        help="Per-session breakdown table")
    parser.add_argument("--health", action="store_true",
                        help="Feature health + drift details")
    parser.add_argument("--weekly", action="store_true",
                        help="Weekly report summary")

    args = parser.parse_args()

    # Load the account currency symbol from state file
    _load_currency_symbol()

    if args.deep:
        deep_report()
    elif args.compare:
        show_compare()
    elif args.session:
        show_session_breakdown()
    elif args.health:
        show_health_report()
    elif args.weekly:
        show_weekly_report()
    else:
        quick_report()


if __name__ == "__main__":
    main()
