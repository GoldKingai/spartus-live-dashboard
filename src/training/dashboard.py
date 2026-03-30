"""Rich terminal dashboard for live training monitoring.

17 panels matching the SPARTUS_TRAINING_DASHBOARD spec.
Uses console.clear() + console.print() per frame (no Rich Live — eliminates
Windows flickering that Live mode causes).

Keyboard controls:
    P - Pause/Resume training
    Q - Quit training (graceful stop)
    Ctrl+C - Emergency stop
"""

import os
import sys
import time
import threading
from typing import Dict, List, Optional

# Enable Windows VT100 terminal sequences BEFORE importing Rich.
if sys.platform == "win32":
    os.system("")  # Triggers VT100 mode on Windows 10+

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.config import TrainingConfig
from src.training.ascii_chart import sparkline


def _color(value: float, green_above: float = None, yellow_above: float = None,
           green_below: float = None, yellow_below: float = None,
           green_min: float = None, green_max: float = None) -> str:
    """Return rich color markup based on thresholds."""
    if green_min is not None and green_max is not None:
        if green_min <= value <= green_max:
            return "green"
        return "red"
    if green_above is not None:
        if value >= green_above:
            return "green"
        if yellow_above is not None and value >= yellow_above:
            return "yellow"
        return "red"
    if green_below is not None:
        if value <= green_below:
            return "green"
        if yellow_below is not None and value <= yellow_below:
            return "yellow"
        return "red"
    return "white"


class TrainingDashboard:
    """Rich terminal UI for monitoring Spartus training.

    Panels (covering all 17 spec sections):
    1. Header + Progress (4.1, 4.10)
    2. Account & P/L (4.2)
    3. Performance Metrics (4.3)
    4. SAC Internals + Learning (4.11, 4.12)
    5. Reward Breakdown (4.14)
    6. This Week + Trade Stats (4.4)
    7. Trend Predictions + TP Accuracy (4.5, 4.6)
    8. Balance Chart (4.7)
    9. Curriculum + Convergence (4.13, 4.15)
    10. Anti-Hack + Safety + Obs Health (4.17)
    11. Alerts (4.9)
    12. Footer with controls (4.10)

    Rendering: clear + print per frame (no Rich Live).
    Keyboard: P=pause, Q=quit, Ctrl+C=emergency stop.
    """

    def __init__(self, config: TrainingConfig = None, shared_metrics: Optional[Dict] = None):
        self.cfg = config or TrainingConfig()
        self.metrics = shared_metrics if shared_metrics is not None else {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._key_thread: Optional[threading.Thread] = None
        self._console = Console(force_terminal=True, highlight=False)
        self._balance_history: List[float] = []
        self._reward_history: List[float] = []

    def start(self):
        """Start the dashboard and keyboard listener."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._key_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._key_thread.start()

    def stop(self):
        """Stop the dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _keyboard_loop(self):
        """Listen for keyboard input (P=pause, Q=quit).

        On Windows, arrow keys and function keys produce two-byte sequences
        (e.g. Down Arrow = 0xE0 + 0x50 = 'P'). We must consume both bytes
        to avoid interpreting the second byte as a real keypress.
        """
        try:
            if sys.platform == "win32":
                import msvcrt
                while self._running:
                    if msvcrt.kbhit():
                        raw = msvcrt.getch()
                        # Extended key prefix: consume the second byte and ignore
                        if raw in (b'\x00', b'\xe0'):
                            if msvcrt.kbhit():
                                msvcrt.getch()  # discard second byte
                            continue
                        key = raw.decode("utf-8", errors="ignore").lower()
                        if key:
                            self._handle_key(key)
                    time.sleep(0.1)
            else:
                import select
                while self._running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        self._handle_key(key)
        except Exception:
            pass

    def _handle_key(self, key: str):
        """Process a keypress."""
        if key == "p":
            paused = self.metrics.get("_paused", False)
            self.metrics["_paused"] = not paused
        elif key == "q":
            self.metrics["_quit_requested"] = True

    def _run_loop(self):
        """Main rendering loop — cursor-home + overwrite each frame.

        Uses ANSI cursor positioning (\033[H) to jump to top-left and
        overwrite in place. No clearing, no scrolling, no flickering.
        Renders to a string buffer first, then writes in one shot.
        """
        import io

        # Initial: clear screen, hide cursor
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()

        try:
            while self._running:
                try:
                    # Track history
                    bal = self.metrics.get("balance", 0)
                    if bal > 0:
                        self._balance_history.append(bal)
                    rew = self.metrics.get("raw_reward", 0)
                    self._reward_history.append(rew)

                    if len(self._balance_history) > 500:
                        self._balance_history = self._balance_history[-500:]
                    if len(self._reward_history) > 500:
                        self._reward_history = self._reward_history[-500:]

                    # Render layout to string buffer
                    width = self._console.width or 160
                    height = self._console.height or 45
                    buf = io.StringIO()
                    buf_console = Console(
                        file=buf,
                        force_terminal=True,
                        width=width,
                        highlight=False,
                        color_system="truecolor",
                    )
                    buf_console.print(self._build_layout())
                    frame = buf.getvalue()

                    # Move cursor home, write frame, clear below
                    sys.stdout.write("\033[H")
                    sys.stdout.write(frame)
                    sys.stdout.write("\033[J")
                    sys.stdout.flush()
                except Exception:
                    pass  # Never crash training because of dashboard
                time.sleep(1.0)
        finally:
            # Show cursor on exit
            sys.stdout.write("\033[?25h\033[J")
            sys.stdout.flush()

    def _build_layout(self) -> Layout:
        """Build the complete dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_column(
            Layout(name="row1", size=9),
            Layout(name="row2", size=9),
            Layout(name="row3", size=8),
            Layout(name="row4", size=7),
        )

        layout["row1"].split_row(
            Layout(name="account", ratio=1),
            Layout(name="performance", ratio=1),
            Layout(name="sac", ratio=1),
        )

        layout["row2"].split_row(
            Layout(name="rewards", ratio=1),
            Layout(name="this_week", ratio=1),
            Layout(name="predictions", ratio=1),
        )

        layout["row3"].split_row(
            Layout(name="chart", ratio=2),
            Layout(name="curriculum", ratio=1),
        )

        layout["row4"].split_row(
            Layout(name="safety", ratio=1),
            Layout(name="alerts", ratio=2),
        )

        # Render all panels
        layout["header"].update(self._header_panel())
        layout["account"].update(self._account_panel())
        layout["performance"].update(self._performance_panel())
        layout["sac"].update(self._sac_panel())
        layout["rewards"].update(self._reward_panel())
        layout["this_week"].update(self._this_week_panel())
        layout["predictions"].update(self._predictions_panel())
        layout["chart"].update(self._chart_panel())
        layout["curriculum"].update(self._curriculum_panel())
        layout["safety"].update(self._safety_panel())
        layout["alerts"].update(self._alerts_panel())
        layout["footer"].update(self._footer_panel())

        return layout

    # === Panel Renderers ====================================================

    def _header_panel(self) -> Panel:
        """Panel 1: Header + Progress (spec 4.1)."""
        m = self.metrics
        week = m.get("current_week", 0)
        total = m.get("total_weeks", 0)
        ts = m.get("timestep", 0)
        state = m.get("convergence_state", "WARMING_UP")
        stage = self._get_stage(week)
        stage_names = {1: "Easy", 2: "Mixed", 3: "Full"}
        paused = m.get("_paused", False)

        pct = (week / total * 100) if total > 0 else 0

        header = Text()
        header.append("  SPARTUS TRADING AI  ", style="bold white on blue")
        if paused:
            header.append("  PAUSED  ", style="bold white on red")
        header.append(f"  Week {week}/{total} ({pct:.0f}%)  ", style="bold")
        header.append(f"  Step {ts:,}  ", style="dim")
        header.append(f"  Stage {stage}/{stage_names.get(stage, '?')}  ", style="bold cyan")
        header.append(f"  [{state}]", style=self._conv_style(state))

        return Panel(header, box=box.DOUBLE)

    def _account_panel(self) -> Panel:
        """Panel 2: Account & P/L (spec 4.2)."""
        m = self.metrics
        balance = m.get("balance", 0)
        equity = m.get("equity", 0)
        peak = m.get("peak_balance", 0)
        dd = m.get("drawdown", 0)
        initial = self.cfg.initial_balance

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=10)
        table.add_column("Value", justify="right")

        pnl = balance - initial
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0
        dd_color = _color(dd, green_below=0.03, yellow_below=0.07)

        table.add_row("Balance", f"£{balance:.2f}")
        table.add_row("Equity", f"£{equity:.2f}")
        table.add_row("Peak", f"£{peak:.2f}")
        table.add_row("P/L", Text(f"£{pnl:+.2f} ({pnl_pct:+.1f}%)",
                                   style="green" if pnl >= 0 else "red"))
        table.add_row("Drawdown", Text(f"{dd:.1%}", style=dd_color))
        table.add_row("DD Limit", f"{self.cfg.max_dd:.0%}")

        border = "green" if dd < 0.05 else ("yellow" if dd < 0.08 else "red")
        return Panel(table, title="[bold]Account[/bold]", border_style=border)

    def _performance_panel(self) -> Panel:
        """Panel 3: Performance Metrics (spec 4.3)."""
        m = self.metrics
        wr = m.get("win_rate", 0)
        trades = m.get("total_trades", 0)
        sharpe = m.get("sharpe", 0)
        mean_rew = m.get("mean_ep_reward", 0)
        trend_acc = m.get("trend_accuracy", 0)
        profit_factor = m.get("profit_factor", 0)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", justify="right")

        wr_color = _color(wr, green_above=0.52, yellow_above=0.48)
        sharpe_color = _color(sharpe, green_above=0.8, yellow_above=0.3)
        trend_color = _color(trend_acc, green_above=0.55, yellow_above=0.50)

        table.add_row("Win Rate", Text(f"{wr:.1%}", style=wr_color))
        table.add_row("Total Trades", str(trades))
        table.add_row("Sharpe", Text(f"{sharpe:.3f}", style=sharpe_color))
        table.add_row("Trend Acc", Text(f"{trend_acc:.1%}", style=trend_color))
        table.add_row("Avg Reward", f"{mean_rew:.3f}")
        if profit_factor > 0:
            pf_color = _color(profit_factor, green_above=1.2, yellow_above=1.0)
            table.add_row("Profit Fac", Text(f"{profit_factor:.2f}", style=pf_color))

        return Panel(table, title="[bold]Performance[/bold]")

    def _sac_panel(self) -> Panel:
        """Panel 4: SAC Internals + Learning Metrics (spec 4.11, 4.12)."""
        m = self.metrics
        alpha = m.get("entropy_alpha")
        actor_loss = m.get("actor_loss")
        critic_loss = m.get("critic_loss")
        actor_grad = m.get("actor_grad_norm")
        lr = m.get("learning_rate")
        action_std = m.get("action_std")
        grad_clip_pct = m.get("grad_clip_pct", 0)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", justify="right")

        if alpha is not None:
            a_color = _color(alpha, green_min=0.01, green_max=1.0)
            table.add_row("Entropy a", Text(f"{alpha:.4f}", style=a_color))
        if actor_loss is not None:
            table.add_row("Actor Loss", f"{actor_loss:.4f}")
        if critic_loss is not None:
            table.add_row("Critic Loss", f"{critic_loss:.4f}")
        if actor_grad is not None:
            g_color = _color(actor_grad, green_below=10, yellow_below=50)
            table.add_row("Actor Grad", Text(f"{actor_grad:.2f}", style=g_color))
        if action_std is not None:
            s_color = _color(action_std, green_above=0.10, yellow_above=0.05)
            table.add_row("Action Std", Text(f"{action_std:.4f}", style=s_color))
        if lr is not None:
            table.add_row("LR", f"{lr:.2e}")
        if grad_clip_pct > 0:
            gc_color = _color(grad_clip_pct, green_below=0.05, yellow_below=0.30)
            table.add_row("Grad Clip %", Text(f"{grad_clip_pct:.1%}", style=gc_color))

        return Panel(table, title="[bold]SAC Internals[/bold]", border_style="cyan")

    def _reward_panel(self) -> Panel:
        """Panel 5: Reward Breakdown (spec 4.14)."""
        m = self.metrics

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Component", style="dim", width=10)
        table.add_column("Weight", justify="center", style="dim", width=5)
        table.add_column("Value", justify="right")

        components = [
            ("R1 P/L", self.cfg.r1_weight, "r1_position_pnl"),
            ("R2 Quality", self.cfg.r2_weight, "r2_trade_quality"),
            ("R3 DD", self.cfg.r3_weight, "r3_drawdown"),
            ("R4 Sharpe", self.cfg.r4_weight, "r4_sharpe"),
            ("R5 Risk", self.cfg.r5_weight, "r5_risk_bonus"),
        ]

        for label, weight, key in components:
            val = m.get(key, 0)
            table.add_row(label, f"{weight:.0%}", f"{val:.4f}")

        raw = m.get("raw_reward", 0)
        norm = m.get("reward", 0)
        clip_pct = m.get("reward_clip_pct", 0)
        table.add_row("", "", "")
        table.add_row("Raw", "", f"{raw:.4f}")
        table.add_row("Norm", "", f"{norm:.4f}")
        if clip_pct > 0:
            c_color = _color(clip_pct, green_below=0.05, yellow_below=0.15)
            table.add_row("Clip %", "", Text(f"{clip_pct:.1%}", style=c_color))

        return Panel(table, title="[bold]Reward Breakdown[/bold]")

    def _this_week_panel(self) -> Panel:
        """Panel 6: This Week + Trade Stats (spec 4.4)."""
        m = self.metrics
        week_trades = m.get("episode_trades", 0)
        daily_trades = m.get("daily_trades", 0)
        action_mean = m.get("action_mean", 0)
        action_std = m.get("action_std", 0)
        has_pos = m.get("has_position", False)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", justify="right")

        table.add_row("Week Trades", str(week_trades))
        table.add_row("Daily Trades", str(daily_trades))
        table.add_row("Position", Text("OPEN" if has_pos else "FLAT",
                                        style="cyan" if has_pos else "dim"))
        table.add_row("Action Mean", f"{action_mean:.4f}")
        table.add_row("Action Std", f"{action_std:.4f}")

        cap_hits = m.get("trade_cap_hits", 0)
        hold_blocks = m.get("hold_blocks", 0)
        if cap_hits > 0:
            table.add_row("Cap Hits", Text(str(cap_hits), style="yellow"))
        if hold_blocks > 0:
            table.add_row("Hold Blocks", Text(str(hold_blocks), style="yellow"))

        return Panel(table, title="[bold]This Week[/bold]")

    def _predictions_panel(self) -> Panel:
        """Panel 7: Trend Predictions + TP Accuracy (spec 4.5, 4.6)."""
        m = self.metrics
        trend_acc = m.get("trend_accuracy", 0)
        pending = m.get("pending_predictions", 0)
        verified = m.get("verified_predictions", 0)
        tp_hit = m.get("tp_hit_rate", 0)
        tp_reach = m.get("tp_reachable_rate", 0)
        sl_hit = m.get("sl_hit_rate", 0)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", justify="right")

        t_color = _color(trend_acc, green_above=0.55, yellow_above=0.50)
        table.add_row("Trend Acc", Text(f"{trend_acc:.1%}", style=t_color))
        table.add_row("Pending", str(pending))
        table.add_row("Verified", str(verified))
        table.add_row("", "")
        table.add_row("TP Hit", f"{tp_hit:.1%}")
        table.add_row("TP Reachable", f"{tp_reach:.1%}")
        table.add_row("SL Hit", f"{sl_hit:.1%}")

        return Panel(table, title="[bold]Predictions & TP[/bold]")

    def _chart_panel(self) -> Panel:
        """Panel 8: Balance Chart (spec 4.7)."""
        if len(self._balance_history) < 2:
            return Panel("  Collecting data...", title="[bold]Balance Chart[/bold]")

        spark = sparkline(self._balance_history, width=50)
        mn = min(self._balance_history)
        mx = max(self._balance_history)
        curr = self._balance_history[-1]

        lines = [
            f"  Balance: £{curr:.2f}  (min: £{mn:.2f}, max: £{mx:.2f})",
            f"  {spark}",
            "",
            f"  Reward: {sparkline(self._reward_history[-50:], width=50)}",
        ]

        return Panel("\n".join(lines), title="[bold]Charts[/bold]")

    def _curriculum_panel(self) -> Panel:
        """Panel 9: Curriculum + Convergence (spec 4.13, 4.15)."""
        m = self.metrics
        week = m.get("current_week", 0)
        total = m.get("total_weeks", 0)
        state = m.get("convergence_state", "WARMING_UP")
        difficulty = m.get("week_difficulty", 0)

        text = Text()

        text.append("  State: ", style="dim")
        text.append(state, style=self._conv_style(state))

        stage = self._get_stage(week)
        stage_names = {1: "Easy", 2: "Mixed", 3: "Full"}
        text.append(f"\n  Stage: {stage} ({stage_names.get(stage, '?')})")

        if difficulty > 0:
            d_color = _color(difficulty, green_below=0.4, yellow_below=0.7)
            text.append("\n  Difficulty: ")
            text.append(f"{difficulty:.2f}", style=d_color)

        if total > 0:
            pct = week / total * 100
            text.append(f"\n  Progress: {pct:.1f}%")

        best = m.get("best_val_sharpe", 0)
        if best > -999:
            text.append(f"\n  Best Sharpe: {best:.3f}")

        weeks_since = m.get("weeks_since_best", 0)
        if weeks_since > 0:
            wsb_color = _color(weeks_since, green_below=20, yellow_below=40)
            text.append("\n  Since Best: ")
            text.append(f"{weeks_since}w", style=wsb_color)

        return Panel(text, title="[bold]Curriculum & Conv[/bold]")

    def _safety_panel(self) -> Panel:
        """Panel 10: Anti-Hack + Safety + Obs Health (spec 4.17)."""
        m = self.metrics

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=14)
        table.add_column("Value", justify="right")

        cap_hits = m.get("trade_cap_hits", 0)
        hold_blocks = m.get("hold_blocks", 0)
        conv_blocks = m.get("conviction_blocks", 0)

        table.add_row("Trade Cap Hits",
                       Text(str(cap_hits), style="green" if cap_hits == 0 else "yellow"))
        table.add_row("Hold Blocks",
                       Text(str(hold_blocks), style="green" if hold_blocks < 5 else "yellow"))
        table.add_row("Conv Blocks",
                       Text(str(conv_blocks), style="green" if conv_blocks < 3 else "yellow"))

        dead = m.get("dead_features", 0)
        exploding = m.get("exploding_features", 0)
        nan_count = m.get("nan_features", 0)

        obs_ok = dead == 0 and exploding == 0 and nan_count == 0
        table.add_row("", "")
        table.add_row("Dead Features",
                       Text(str(dead), style="green" if dead == 0 else "red"))
        table.add_row("Exploding",
                       Text(str(exploding), style="green" if exploding == 0 else "red"))
        table.add_row("NaN Features",
                       Text(str(nan_count), style="green" if nan_count == 0 else "red"))

        border = "green" if obs_ok and cap_hits < 5 else "yellow"
        return Panel(table, title="[bold]Safety & Health[/bold]", border_style=border)

    def _alerts_panel(self) -> Panel:
        """Panel 11: Alerts (spec 4.9)."""
        alerts = self.metrics.get("_alerts", [])
        if not alerts:
            return Panel("  No alerts", title="[bold]Alerts[/bold]", border_style="green")

        lines = []
        for alert in alerts[-8:]:
            if "CRITICAL" in alert:
                lines.append(f"[bold red]{alert}[/bold red]")
            elif "WARNING" in alert:
                lines.append(f"[yellow]{alert}[/yellow]")
            elif "POSITIVE" in alert:
                lines.append(f"[green]{alert}[/green]")
            else:
                lines.append(f"[dim]{alert}[/dim]")

        has_critical = any("CRITICAL" in a for a in alerts[-8:])
        border = "red" if has_critical else "green"
        return Panel("\n".join(lines), title="[bold]Alerts[/bold]", border_style=border)

    def _footer_panel(self) -> Panel:
        """Panel 12: Footer with controls (spec 4.10)."""
        elapsed = time.time() - self.metrics.get("_start_time", time.time())
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)

        sys_info = ""
        try:
            import psutil
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            sys_info = f"  |  CPU: {cpu:.0f}%  RAM: {mem:.0f}%"
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    sys_info += f"  GPU: {gpu_mem:.1f}GB"
            except Exception:
                pass
        except Exception:
            pass

        text = Text()
        text.append(f"  Elapsed: {hours:02d}:{mins:02d}:{secs:02d}", style="dim")
        text.append(sys_info, style="dim")
        text.append("  |  P=pause  Q=quit  Ctrl+C=stop", style="dim")

        return Panel(text, box=box.SIMPLE)

    # === Helpers ==============================================================

    def _get_stage(self, week: int) -> int:
        if week < self.cfg.stage1_end_week:
            return 1
        elif week < self.cfg.stage2_end_week:
            return 2
        return 3

    @staticmethod
    def _conv_style(state: str) -> str:
        return {
            "WARMING_UP": "dim",
            "IMPROVING": "green",
            "CONVERGED": "bold green",
            "OVERFITTING": "red",
            "COLLAPSED": "bold red",
            "PLATEAU": "yellow",
            "STABLE": "bold green",
        }.get(state, "white")
