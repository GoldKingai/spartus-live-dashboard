"""Training logger with 4 output streams.

- training_log.jsonl:    Step-level metrics (every N steps)
- weekly_summary.jsonl:  Per-week aggregated stats
- decisions.jsonl:       Individual trade decisions
- alerts.log:            Human-readable alert messages
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import TrainingConfig


class TrainingLogger:
    """Multi-stream training logger."""

    def __init__(self, config: TrainingConfig = None, log_dir: Optional[Path] = None):
        cfg = config or TrainingConfig()
        self.log_dir = log_dir or cfg.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._training_log = self.log_dir / "training_log.jsonl"
        self._weekly_log = self.log_dir / "weekly_summary.jsonl"
        self._decisions_log = self.log_dir / "decisions.jsonl"
        self._alerts_log = self.log_dir / "alerts.log"

        # In-memory buffers for current week
        self._step_metrics: List[Dict] = []
        self._trade_decisions: List[Dict] = []
        self._alerts: List[str] = []
        self._current_week = 0

        # Alert cooldown: {alert_key: last_fire_time}
        self._alert_cooldowns: Dict[str, float] = {}
        self._alert_cooldown_secs = 60  # Same alert suppressed for 60s

    def set_week(self, week: int):
        self._current_week = week

    # === Step-level logging =================================================

    def log_step(self, metrics: Dict[str, Any]):
        """Log step-level metrics."""
        metrics["week"] = self._current_week
        metrics["timestamp"] = time.time()
        self._step_metrics.append(metrics)
        self._append_jsonl(self._training_log, metrics)

    # === Trade decision logging =============================================

    def log_trade(self, trade: Dict[str, Any]):
        """Log an individual trade decision."""
        trade["week"] = self._current_week
        trade["timestamp"] = time.time()
        self._trade_decisions.append(trade)
        self._append_jsonl(self._decisions_log, trade)

    # === Weekly summary =====================================================

    def log_weekly_summary(self, summary: Dict[str, Any]):
        """Log end-of-week summary."""
        summary["week"] = self._current_week
        summary["timestamp"] = time.time()
        self._append_jsonl(self._weekly_log, summary)

    # === Alerts =============================================================

    def log_alert(self, level: str, message: str, details: Optional[Dict] = None):
        """Log an alert to the alerts file and in-memory buffer.

        Uses a 60-second cooldown per alert key to prevent spam.
        The alert key is the first 30 chars of the message (strips numbers).
        """
        # Deduplicate: extract a stable key from the message (strip numbers)
        import re
        alert_key = re.sub(r"[\d.%]+", "#", message)[:40]
        now = time.time()
        last_time = self._alert_cooldowns.get(alert_key, 0)
        if now - last_time < self._alert_cooldown_secs:
            return  # Suppressed — same alert fired recently
        self._alert_cooldowns[alert_key] = now

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level.upper():8s}] W{self._current_week:03d} | {message}"
        if details:
            line += f" | {json.dumps(details)}"
        self._alerts.append(line)
        if len(self._alerts) > 200:
            self._alerts = self._alerts[-200:]

        with open(self._alerts_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # === Accessors ==========================================================

    def get_recent_alerts(self, n: int = 20) -> List[str]:
        return self._alerts[-n:]

    def get_step_metrics(self) -> List[Dict]:
        return self._step_metrics

    def get_trade_decisions(self) -> List[Dict]:
        return self._trade_decisions

    def clear_week_buffers(self):
        """Clear in-memory buffers at end of week."""
        self._step_metrics.clear()
        self._trade_decisions.clear()

    # === Internal ===========================================================

    @staticmethod
    def _append_jsonl(path: Path, data: Dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
