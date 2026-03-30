"""Generate markdown training reports for LLM analysis.

Produces structured reports summarizing training progress,
performance metrics, and areas of concern.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.config import TrainingConfig


class ReportGenerator:
    """Generates periodic markdown training reports."""

    def __init__(self, config: TrainingConfig = None):
        cfg = config or TrainingConfig()
        self.report_dir = cfg.report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        week: int,
        weekly_summaries: List[Dict],
        convergence_state: str,
        best_checkpoint: Optional[Dict] = None,
        recent_alerts: Optional[List[str]] = None,
    ) -> str:
        """Generate a markdown report and save to file.

        Returns the file path of the generated report.
        """
        report = []
        report.append(f"# Spartus Training Report — Week {week}")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overview
        report.append("## Overview")
        report.append(f"- **Training week:** {week}")
        report.append(f"- **Convergence state:** {convergence_state}")

        if weekly_summaries:
            latest = weekly_summaries[-1]
            report.append(f"- **Current balance:** £{latest.get('balance', 0):.2f}")
            report.append(f"- **Peak balance:** £{latest.get('peak_balance', 0):.2f}")
            report.append(f"- **Total trades:** {sum(s.get('episode_trades', 0) for s in weekly_summaries)}")
        report.append("")

        # Performance trend
        report.append("## Performance Trend (last 10 weeks)")
        report.append("| Week | Balance | Trades | Sharpe | Time (s) |")
        report.append("|------|---------|--------|--------|----------|")
        for s in weekly_summaries[-10:]:
            report.append(
                f"| {s.get('week', '?'):>4} "
                f"| £{s.get('balance', 0):>7.2f} "
                f"| {s.get('episode_trades', 0):>6} "
                f"| {s.get('sharpe', 0):>6.3f} "
                f"| {s.get('train_time_s', 0):>8.1f} |"
            )
        report.append("")

        # Best checkpoint
        if best_checkpoint:
            report.append("## Best Checkpoint")
            report.append(f"- **Week:** {best_checkpoint.get('week', '?')}")
            report.append(f"- **Val Sharpe:** {best_checkpoint.get('val_sharpe', 0):.3f}")
            report.append(f"- **Val Return:** {best_checkpoint.get('val_return', 0):.2f}%")
            report.append(f"- **Model:** {best_checkpoint.get('model_path', '?')}")
            report.append("")

        # Alerts
        if recent_alerts:
            report.append("## Recent Alerts")
            for alert in recent_alerts[-10:]:
                report.append(f"- {alert}")
            report.append("")

        # Save
        content = "\n".join(report)
        filepath = self.report_dir / f"report_week_{week:04d}.md"
        with open(filepath, "w") as f:
            f.write(content)

        return str(filepath)
