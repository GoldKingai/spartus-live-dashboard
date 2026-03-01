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
            # Convert numpy arrays to lists for JSON
            if 'observation_670' in obs_data:
                import numpy as np
                if isinstance(obs_data['observation_670'], np.ndarray):
                    obs_data['observation_670'] = obs_data['observation_670'].tolist()
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
