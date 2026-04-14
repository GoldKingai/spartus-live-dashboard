"""Live trade logging system -- writes JSONL log files for analysis."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class LiveLogger:
    """Manages all live trading log files.

    Log files:
    - trades.jsonl: Full trade details (per trade close)
    - actions.jsonl: All 4 action values + decision (every bar)
    - alerts.jsonl: Warnings and errors (on event)
    - observations.jsonl: 670-dim obs + action (configurable interval)
    - feature_stats.jsonl: Per-feature mean/std/min/max (per session boundary)
    - weekly_summary.jsonl: Weekly aggregated metrics (Sunday 23:59 UTC)
    """

    def __init__(self, log_dir: str = "storage/logs"):
        """Initialize log directory and file handles."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._bar_count = 0

    def log_trade(self, trade_data: dict):
        """Log completed trade to trades.jsonl."""
        trade_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl('trades.jsonl', trade_data)

    def log_action(self, action_data: dict):
        """Log every action decision to actions.jsonl.

        action_data should contain:
        - timestamp, bar_time
        - action_raw: [4 values]
        - direction, conviction, exit_urgency, sl_adjustment
        - decision: str
        - has_position: bool
        - balance, equity
        - trade_rejected: bool
        - reject_reason: str or None
        """
        action_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl('actions.jsonl', action_data)

    def log_alert(self, level: str, message: str, details: dict = None):
        """Log alert event to alerts.jsonl."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message,
        }
        if details:
            entry['details'] = details
        self._append_jsonl('alerts.jsonl', entry)

    def log_observation(self, obs_data: dict, every_n_bars: int = 12):
        """Log observation every N bars to observations.jsonl."""
        self._bar_count += 1
        if self._bar_count % every_n_bars == 0:
            obs_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            obs_data['bar_idx'] = self._bar_count
            # Convert numpy arrays to lists for JSON (key may vary by model obs_dim)
            import numpy as np
            for key in list(obs_data.keys()):
                if key.startswith('observation_') and isinstance(obs_data[key], np.ndarray):
                    obs_data[key] = obs_data[key].tolist()
            self._append_jsonl('observations.jsonl', obs_data)

    def log_feature_stats(self, stats_data: dict):
        """Log feature distribution stats at session boundaries."""
        stats_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl('feature_stats.jsonl', stats_data)

    def log_weekly_summary(self, summary: dict):
        """Log weekly performance summary."""
        summary['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl('weekly_summary.jsonl', summary)

    def get_recent_alerts(self, limit: int = 50) -> list:
        """Read recent alerts from alerts.jsonl."""
        return self._read_jsonl_tail('alerts.jsonl', limit)

    def get_recent_actions(self, limit: int = 100) -> list:
        """Read recent actions from actions.jsonl."""
        return self._read_jsonl_tail('actions.jsonl', limit)

    def get_all_trades(self) -> list:
        """Read all trades from trades.jsonl."""
        return self._read_jsonl('trades.jsonl')

    def get_weekly_summaries(self) -> list:
        """Read all weekly summaries."""
        return self._read_jsonl('weekly_summary.jsonl')

    def _append_jsonl(self, filename: str, data: dict):
        """Append a JSON line to file."""
        path = self.log_dir / filename
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, default=str) + '\n')
        except Exception as e:
            log.error(f"Failed to write to {filename}: {e}")

    def _read_jsonl(self, filename: str) -> list:
        """Read all lines from JSONL file."""
        path = self.log_dir / filename
        if not path.exists():
            return []
        results = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        except Exception as e:
            log.error(f"Failed to read {filename}: {e}")
        return results

    def generate_weekly_summary(self) -> dict:
        """Generate and log a weekly performance summary from trades.jsonl.

        Call this at the end of each trading week (e.g. Friday 22:00 UTC)
        or on manual request. Reads all trades from the current week
        and computes aggregate metrics for post-analysis.

        Returns:
            The summary dict that was logged.
        """
        from datetime import timedelta
        from collections import Counter

        all_trades = self.get_all_trades()
        if not all_trades:
            return {}

        # Determine current week boundary (Monday 00:00 UTC)
        now = datetime.now(timezone.utc)
        days_since_monday = now.weekday()
        week_start = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Filter trades to this week
        week_trades = []
        for t in all_trades:
            ts = t.get("timestamp", "")
            if ts and ts >= week_start.isoformat():
                week_trades.append(t)

        if not week_trades:
            return {}

        pnls = [t.get("pnl", 0) or 0 for t in week_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        lessons = Counter(t.get("lesson_type", "UNKNOWN") for t in week_trades)
        sessions = Counter(t.get("session", "UNKNOWN") for t in week_trades)
        sides = Counter(t.get("side", "?") for t in week_trades)
        convictions = [t.get("conviction", 0) or 0 for t in week_trades]
        holds = [t.get("hold_bars", 0) or 0 for t in week_trades]
        close_reasons = Counter(t.get("close_reason", "?") for t in week_trades)

        # Protection stage stats
        protection_stages = [t.get("protection_stage_max", 0) or 0 for t in week_trades]
        sl_mod_counts = [t.get("sl_modification_count", 0) or 0 for t in week_trades]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0

        summary = {
            "week_start": week_start.isoformat(),
            "week_end": now.isoformat(),
            "total_trades": len(week_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / max(len(week_trades), 1) * 100, 1),
            "net_pnl": round(sum(pnls), 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(gross_profit / max(gross_loss, 0.01), 2),
            "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 4),
            "avg_win": round(sum(wins) / max(len(wins), 1), 4) if wins else 0,
            "avg_loss": round(sum(losses) / max(len(losses), 1), 4) if losses else 0,
            "best_trade": round(max(pnls), 4),
            "worst_trade": round(min(pnls), 4),
            "avg_conviction": round(sum(convictions) / max(len(convictions), 1), 3),
            "avg_hold_bars": round(sum(holds) / max(len(holds), 1), 1),
            "lesson_breakdown": dict(lessons),
            "session_breakdown": dict(sessions),
            "side_breakdown": dict(sides),
            "close_reason_breakdown": dict(close_reasons),
            # V2 protection analytics
            "protection_activations": sum(1 for s in protection_stages if s > 0),
            "protection_stage_avg": round(sum(protection_stages) / max(len(protection_stages), 1), 2),
            "avg_sl_modifications": round(sum(sl_mod_counts) / max(len(sl_mod_counts), 1), 1),
            # Model traceability
            "model_version": week_trades[-1].get("model_version", "unknown") if week_trades else "unknown",
        }

        self.log_weekly_summary(summary)
        return summary

    def _read_jsonl_tail(self, filename: str, limit: int) -> list:
        """Read last N lines from JSONL file (efficient tail)."""
        path = self.log_dir / filename
        if not path.exists():
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            results = []
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results
        except Exception as e:
            log.error(f"Failed to read tail of {filename}: {e}")
            return []
