"""Tab 7 — LIVE FINE-TUNE dashboard tab.

PyQt6 widget showing the full live fine-tuning monitoring interface.
Embedded in the training dashboard as Tab 7.

Layout (from design plan §6.3):
  ┌── MODE SELECTOR ─────────────────────────────────────────────────┐
  │  [Historical Fine-Tune]  [Live Fine-Tune (MT5)]                  │
  │  Model: [Select Checkpoint ▼]                                    │
  │  [START]  [STOP]  [VALIDATE NOW]                                 │
  └──────────────────────────────────────────────────────────────────┘
  ┌── CONNECTION ────┐  ┌── TRAINING STATUS ──────────────────────────┐
  │  MT5 status      │  │  State, episodes, grad steps                │
  └─────────────────┘  └────────────────────────────────────────────┘
  ┌── MEMORY LAYERS ─────────────────────────────────────────────────┐
  │  L1 Slow Adaptation: KL divergence gauge                         │
  │  L2 Experience Replay: buffer tier composition                   │
  │  L3 EWC Protection: weight divergence + Fisher                   │
  │  L4 Strategy Memory: known regimes + forgetting alerts           │
  └──────────────────────────────────────────────────────────────────┘
  ┌── VALIDATION GATE ──┐  ┌── CHECKPOINTS ─────────────────────────┐
  │  Last result        │  │  Checkpoint list + Promote/Rollback     │
  └────────────────────┘  └────────────────────────────────────────┘
  ┌── LAST EPISODE ──────────────────────────────────────────────────┐
  │  Trades, WR, PF, P/L, hold, conviction                          │
  └──────────────────────────────────────────────────────────────────┘

All data is read from shared_metrics["_ft_*"] keys populated by LiveFineTuner.
"""

import time
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QComboBox, QTextEdit,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    QMessageBox, QButtonGroup, QRadioButton, QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# Import color palette from parent dashboard module
try:
    from src.training.qt_dashboard import C
except ImportError:
    C = {
        "bg": "#0d1117", "surface": "#161b22", "surface2": "#1c2333",
        "border": "#30363d", "text": "#e6edf3", "subtext": "#b1bac4",
        "label": "#8b949e", "dim": "#656d76",
        "green": "#2dcc2d", "red": "#ff3333", "yellow": "#ffcc00",
        "blue": "#58a6ff", "cyan": "#39d5ff", "mauve": "#bc8cff",
        "peach": "#ffa657",
    }

from src.training.live_fine_tuner import (
    FT_IDLE, FT_INITIALIZING, FT_COLLECTING, FT_TRAINING,
    FT_VALIDATING, FT_PROMOTING, FT_STOPPED, FT_ERROR,
)


# ── State → badge color mapping ──────────────────────────────────────────────
STATE_COLORS = {
    FT_IDLE:         "dim",
    FT_INITIALIZING: "yellow",
    FT_COLLECTING:   "blue",
    FT_TRAINING:     "green",
    FT_VALIDATING:   "cyan",
    FT_PROMOTING:    "peach",
    FT_STOPPED:      "subtext",
    FT_ERROR:        "red",
}


class _VLabel(QLabel):
    """Compact value label with color support."""

    def set(self, text: str, color: str = "text", size: int = 14):
        self.setText(str(text))
        self.setStyleSheet(
            f"color: {C.get(color, C['text'])}; font-size: {size}px; font-weight: bold;"
        )


class TabLiveFinetune(QWidget):
    """Live Fine-Tune tab widget for the training dashboard.

    Args:
        config: TrainingConfig instance.
        shared_metrics: Shared metrics dict (same as training dashboard).
        trainer: Trainer instance (passed to LiveFineTuner for val access).
    """

    def __init__(self, config, shared_metrics: Dict, trainer=None):
        super().__init__()
        self.cfg = config
        self.metrics = shared_metrics
        self._trainer = trainer
        self._tuner = None       # LiveFineTuner instance (created on Start)
        self._mode = "live"      # "live" or "historical"
        self._selected_model: Optional[str] = None

        self._v: Dict[str, _VLabel] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 6, 8, 6)

        layout.addWidget(self._build_mode_selector())

        # Middle row: connection + training status
        mid = QHBoxLayout()
        mid.addWidget(self._build_connection_panel(), stretch=1)
        mid.addWidget(self._build_training_status_panel(), stretch=2)
        layout.addLayout(mid)

        layout.addWidget(self._build_memory_layers_panel())

        # Bottom row: validation gate + checkpoints
        bot = QHBoxLayout()
        bot.addWidget(self._build_validation_panel(), stretch=1)
        bot.addWidget(self._build_checkpoints_panel(), stretch=1)
        layout.addLayout(bot)

        layout.addWidget(self._build_episode_summary_panel())

        # Refresh timer (1 Hz)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_mode_selector(self) -> QGroupBox:
        box = QGroupBox("Mode & Controls")
        h = QHBoxLayout(box)
        h.setSpacing(12)

        # Mode radio buttons
        self._radio_live = QRadioButton("Live Fine-Tune (MT5)")
        self._radio_live.setChecked(True)
        self._radio_hist = QRadioButton("Historical Fine-Tune")
        for rb in (self._radio_live, self._radio_hist):
            rb.setStyleSheet(f"color: {C['text']}; font-size: 13px;")
        self._radio_live.toggled.connect(self._on_mode_changed)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self._radio_live)
        mode_group.addButton(self._radio_hist)

        h.addWidget(self._radio_live)
        h.addWidget(self._radio_hist)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {C['border']};")
        h.addWidget(sep)

        # Model picker
        lbl = QLabel("Model:")
        lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
        h.addWidget(lbl)

        self._model_label = QLabel("(none selected)")
        self._model_label.setStyleSheet(f"color: {C['yellow']}; font-size: 13px;")
        h.addWidget(self._model_label, stretch=1)

        self._pick_btn = QPushButton("Select Model")
        self._pick_btn.setFixedWidth(130)
        self._pick_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['blue']}; "
            f"font-weight: bold; font-size: 13px;"
        )
        self._pick_btn.clicked.connect(self._pick_model)
        h.addWidget(self._pick_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"color: {C['border']};")
        h.addWidget(sep2)

        # Action buttons
        self._start_btn = QPushButton("Start Fine-Tuning")
        self._start_btn.setFixedWidth(160)
        self._start_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['green']}; "
            f"font-weight: bold; font-size: 13px;"
        )
        self._start_btn.clicked.connect(self._on_start)
        h.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['red']}; "
            f"font-weight: bold; font-size: 13px;"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        h.addWidget(self._stop_btn)

        self._validate_btn = QPushButton("Validate Now")
        self._validate_btn.setFixedWidth(130)
        self._validate_btn.setEnabled(False)
        self._validate_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['cyan']}; "
            f"font-weight: bold; font-size: 13px;"
        )
        self._validate_btn.clicked.connect(self._on_validate_now)
        h.addWidget(self._validate_btn)

        return box

    def _build_connection_panel(self) -> QGroupBox:
        box = QGroupBox("Connection")
        g = QGridLayout(box)
        g.setSpacing(4)

        rows = [
            ("MT5 Status:", "_ft_mt5_connected"),
            ("Bars collected:", "_ft_bars"),
            ("Buffer:", "_ft_buffer_pct"),
            ("Ready:", "_ft_collector_ready"),
        ]
        for i, (label, key) in enumerate(rows):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            val = _VLabel("—")
            self._v[key] = val
            g.addWidget(lbl, i, 0)
            g.addWidget(val, i, 1)

        return box

    def _build_training_status_panel(self) -> QGroupBox:
        box = QGroupBox("Training Status")
        g = QGridLayout(box)
        g.setSpacing(4)

        rows = [
            ("State:", "_ft_state"),
            ("Episodes:", "_ft_episode_count"),
            ("Last episode:", "_ft_last_episode"),
            ("Total grad steps:", "_ft_grad_steps"),
            ("KL divergence:", "_ft_kl_divergence"),
            ("EWC penalty:", "_ft_ewc_penalty"),
        ]
        for i, (label, key) in enumerate(rows):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            lbl.setFixedWidth(150)
            val = _VLabel("—")
            self._v[key] = val
            g.addWidget(lbl, i, 0)
            g.addWidget(val, i, 1)

        return box

    def _build_memory_layers_panel(self) -> QGroupBox:
        box = QGroupBox("Memory Protection Layers")
        outer = QHBoxLayout(box)

        # L1: Slow Adaptation
        l1 = QGroupBox("L1  Slow Adaptation")
        l1.setStyleSheet(f"QGroupBox {{ color: {C['blue']}; }}")
        l1g = QGridLayout(l1)
        l1_rows = [
            ("Learning rate:", "_ft_lr"),
            ("KL divergence:", "_ft_kl_gauge"),
            ("Status:", "_ft_l1_status"),
        ]
        for i, (lbl_text, key) in enumerate(l1_rows):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
            val = _VLabel("—")
            val.set("—", "label", 12)
            self._v[key] = val
            l1g.addWidget(lbl, i, 0)
            l1g.addWidget(val, i, 1)
        # KL progress bar
        self._kl_bar = QProgressBar()
        self._kl_bar.setMaximum(100)
        self._kl_bar.setFixedHeight(12)
        self._kl_bar.setTextVisible(False)
        l1g.addWidget(self._kl_bar, 3, 0, 1, 2)
        outer.addWidget(l1)

        # L2: Experience Replay
        l2 = QGroupBox("L2  Experience Replay")
        l2.setStyleSheet(f"QGroupBox {{ color: {C['cyan']}; }}")
        l2g = QGridLayout(l2)
        l2_rows = [
            ("Core memories:", "_ft_buf_core"),
            ("Supporting:", "_ft_buf_supporting"),
            ("Live/recent:", "_ft_buf_recent"),
            ("Buffer health:", "_ft_buf_health"),
        ]
        for i, (lbl_text, key) in enumerate(l2_rows):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
            val = _VLabel("—")
            val.set("—", "label", 12)
            self._v[key] = val
            l2g.addWidget(lbl, i, 0)
            l2g.addWidget(val, i, 1)
        outer.addWidget(l2)

        # L3: EWC Protection
        l3 = QGroupBox("L3  EWC Protection")
        l3.setStyleSheet(f"QGroupBox {{ color: {C['peach']}; }}")
        l3g = QGridLayout(l3)
        l3_rows = [
            ("Fisher computed:", "_ft_ewc_enabled"),
            ("Weight divergence:", "_ft_weight_divergence"),
            ("λ (penalty):", "_ft_ewc_lambda"),
            ("Status:", "_ft_l3_status"),
        ]
        for i, (lbl_text, key) in enumerate(l3_rows):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
            val = _VLabel("—")
            val.set("—", "label", 12)
            self._v[key] = val
            l3g.addWidget(lbl, i, 0)
            l3g.addWidget(val, i, 1)
        outer.addWidget(l3)

        # L4: Strategy Memory
        l4 = QGroupBox("L4  Strategy Memory")
        l4.setStyleSheet(f"QGroupBox {{ color: {C['mauve']}; }}")
        l4g = QGridLayout(l4)
        l4_rows = [
            ("Known regimes:", "_ft_strat_regimes"),
            ("Active regime:", "_ft_strat_active"),
            ("Forgetting alerts:", "_ft_strat_alerts"),
            ("Profitable:", "_ft_strat_profitable"),
        ]
        for i, (lbl_text, key) in enumerate(l4_rows):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
            val = _VLabel("—")
            val.set("—", "label", 12)
            self._v[key] = val
            l4g.addWidget(lbl, i, 0)
            l4g.addWidget(val, i, 1)
        outer.addWidget(l4)

        return box

    def _build_validation_panel(self) -> QGroupBox:
        box = QGroupBox("Validation Gate")
        g = QGridLayout(box)
        g.setSpacing(4)

        rows = [
            ("Last check:", "_ft_val_last"),
            ("Result:", "_ft_val_result"),
            ("Sharpe:", "_ft_val_sharpe"),
            ("Baseline:", "_ft_val_baseline"),
            ("Trend:", "_ft_val_trend"),
            ("Failures:", "_ft_val_failures"),
        ]
        for i, (lbl_text, key) in enumerate(rows):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            lbl.setFixedWidth(100)
            val = _VLabel("—")
            self._v[key] = val
            g.addWidget(lbl, i, 0)
            g.addWidget(val, i, 1)

        return box

    def _build_checkpoints_panel(self) -> QGroupBox:
        box = QGroupBox("Checkpoints")
        v = QVBoxLayout(box)

        self._checkpoint_list = QListWidget()
        self._checkpoint_list.setStyleSheet(
            f"background-color: {C['surface']}; color: {C['text']}; "
            f"font-size: 12px; border: 1px solid {C['border']}; "
            f"font-family: 'Cascadia Code', Consolas, monospace;"
        )
        self._checkpoint_list.setMaximumHeight(140)
        v.addWidget(self._checkpoint_list)

        btn_row = QHBoxLayout()
        self._promote_btn = QPushButton("Promote")
        self._promote_btn.setEnabled(False)
        self._promote_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['green']}; "
            f"font-weight: bold; font-size: 12px;"
        )
        self._promote_btn.clicked.connect(self._on_promote)

        self._rollback_btn = QPushButton("Rollback")
        self._rollback_btn.setEnabled(False)
        self._rollback_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['yellow']}; "
            f"font-weight: bold; font-size: 12px;"
        )
        self._rollback_btn.clicked.connect(self._on_rollback)

        btn_row.addWidget(self._promote_btn)
        btn_row.addWidget(self._rollback_btn)
        v.addLayout(btn_row)

        return box

    def _build_episode_summary_panel(self) -> QGroupBox:
        box = QGroupBox("Last Episode Summary")
        h = QHBoxLayout(box)

        ep_fields = [
            ("Trades:", "_ft_ep_trades"),
            ("Win Rate:", "_ft_ep_wr"),
            ("P/L:", "_ft_ep_pnl"),
            ("Balance:", "_ft_ep_balance"),
            ("KL drift:", "_ft_ep_kl"),
            ("EWC penalty:", "_ft_ep_ewc"),
        ]
        for lbl_text, key in ep_fields:
            cell = QVBoxLayout()
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = _VLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._v[key] = val
            cell.addWidget(lbl)
            cell.addWidget(val)
            h.addLayout(cell)

            # Separator (not after last)
            if key != "_ft_ep_ewc":
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color: {C['border']};")
                h.addWidget(sep)

        return box

    # ─────────────────────────────────────────────────────────────────────────
    # Refresh (called at 1 Hz by QTimer)
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        m = self.metrics

        # ── State ────────────────────────────────────────────────────────────
        state = m.get("_ft_state", FT_IDLE)
        state_color = STATE_COLORS.get(state, "subtext")
        self._v["_ft_state"].set(state, state_color, 14)

        # ── Connection ───────────────────────────────────────────────────────
        mt5_ok = m.get("_ft_mt5_connected", False)
        self._v["_ft_mt5_connected"].set(
            "Connected ●" if mt5_ok else "Disconnected ○",
            "green" if mt5_ok else "red", 13
        )
        bars = m.get("_ft_bars", 0)
        buf_pct = m.get("_ft_buffer_pct", 0.0)
        self._v["_ft_bars"].set(f"{bars:,}", "text", 13)
        self._v["_ft_buffer_pct"].set(
            f"{bars:,} / {self.cfg.finetune_bar_buffer_size:,}  ({buf_pct:.0f}%)",
            "text", 13
        )
        ready = m.get("_ft_collector_ready", False)
        self._v["_ft_collector_ready"].set(
            "YES — ready to train" if ready else "Collecting...",
            "green" if ready else "yellow", 13
        )

        # ── Training status ──────────────────────────────────────────────────
        ep_count = m.get("_ft_episode_count", 0)
        self._v["_ft_episode_count"].set(str(ep_count), "text", 13)

        last_ep = m.get("_ft_last_episode", 0)
        if last_ep:
            elapsed = (time.time() - last_ep) / 3600
            self._v["_ft_last_episode"].set(f"{elapsed:.1f}h ago", "subtext", 13)
        else:
            self._v["_ft_last_episode"].set("Not run yet", "dim", 13)

        grad_steps = m.get("_ft_grad_steps", 0)
        self._v["_ft_grad_steps"].set(f"{grad_steps:,}", "text", 13)

        kl = m.get("_ft_kl_divergence", 0.0)
        kl_color = "green" if kl < 0.2 else ("yellow" if kl < 0.4 else "red")
        self._v["_ft_kl_divergence"].set(f"{kl:.3f}", kl_color, 13)

        ewc_penalty = m.get("_ft_ewc_penalty", 0.0)
        self._v["_ft_ewc_penalty"].set(f"{ewc_penalty:,.0f}", "subtext", 13)

        # ── L1: Slow Adaptation ──────────────────────────────────────────────
        self._v["_ft_lr"].set(
            f"{self.cfg.finetune_lr:.0e}  (3x slower)", "subtext", 12
        )
        self._v["_ft_kl_gauge"].set(
            f"{kl:.3f} / {self.cfg.finetune_max_kl_divergence:.1f} max",
            kl_color, 12
        )
        kl_pct = int(min(100, kl / self.cfg.finetune_max_kl_divergence * 100))
        self._kl_bar.setValue(kl_pct)
        self._kl_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: "
            f"{'#2dcc2d' if kl_pct < 40 else ('#ffcc00' if kl_pct < 80 else '#ff3333')}; "
            f"border-radius: 4px; }}"
        )
        l1_status = (
            "HEALTHY" if kl < 0.2 else
            ("ELEVATED" if kl < self.cfg.finetune_kl_emergency_threshold else
             ("EMERGENCY LR" if kl < self.cfg.finetune_max_kl_divergence else "PAUSED"))
        )
        self._v["_ft_l1_status"].set(l1_status, kl_color, 12)

        # ── L2: Replay Buffer ────────────────────────────────────────────────
        tiers = m.get("_ft_buffer_tiers", {})
        if tiers:
            total = max(1, tiers.get("total", 1))
            core = tiers.get("core", 0)
            supp = tiers.get("supporting", 0)
            recent = tiers.get("recent", 0)
            self._v["_ft_buf_core"].set(
                f"{core:,}  ({core/total*100:.0f}%) — LOCKED", "green", 12
            )
            self._v["_ft_buf_supporting"].set(
                f"{supp:,}  ({supp/total*100:.0f}%)", "blue", 12
            )
            live_added = tiers.get("live_added", 0)
            self._v["_ft_buf_recent"].set(
                f"{recent:,}  ({live_added:,} live added)", "peach", 12
            )
            fill_pct = tiers.get("buffer_fill_pct", 0)
            health = "FILLING" if fill_pct < 50 else ("BALANCED" if fill_pct < 90 else "FULL")
            self._v["_ft_buf_health"].set(health, "green" if fill_pct >= 50 else "yellow", 12)
        else:
            for key in ("_ft_buf_core", "_ft_buf_supporting", "_ft_buf_recent", "_ft_buf_health"):
                self._v[key].set("—", "dim", 12)

        # ── L3: EWC ─────────────────────────────────────────────────────────
        ewc_enabled = m.get("_ft_ewc_enabled", False)
        self._v["_ft_ewc_enabled"].set(
            "YES ✓" if ewc_enabled else "NO (computing...)",
            "green" if ewc_enabled else "yellow", 12
        )
        weight_div = m.get("_ft_weight_divergence", 0.0)
        div_color = "green" if weight_div < 0.05 else ("yellow" if weight_div < 0.2 else "red")
        self._v["_ft_weight_divergence"].set(f"{weight_div:.4f}", div_color, 12)
        self._v["_ft_ewc_lambda"].set(
            f"{self.cfg.finetune_ewc_lambda:,.0f}", "subtext", 12
        )
        self._v["_ft_l3_status"].set(
            "PROTECTING" if ewc_enabled else "INACTIVE",
            "green" if ewc_enabled else "dim", 12
        )

        # ── L4: Strategy Memory ──────────────────────────────────────────────
        strat = m.get("_ft_strategy_summary", {})
        if strat:
            n_regimes = strat.get("known_regimes", 0)
            n_profitable = strat.get("profitable_regimes", 0)
            n_alerts = strat.get("forgetting_alerts", 0)
            active = strat.get("active_regime", "unknown") or "unknown"
            short_active = active[:20] if len(active) > 20 else active

            self._v["_ft_strat_regimes"].set(str(n_regimes), "text", 12)
            self._v["_ft_strat_active"].set(short_active, "blue", 12)
            self._v["_ft_strat_alerts"].set(
                str(n_alerts), "red" if n_alerts > 0 else "green", 12
            )
            self._v["_ft_strat_profitable"].set(
                f"{n_profitable}/{n_regimes}", "green" if n_profitable == n_regimes else "yellow", 12
            )

        # ── Validation Gate ──────────────────────────────────────────────────
        val_summary = m.get("_ft_val_summary", {})
        if val_summary:
            last_passed = val_summary.get("last_passed")
            last_sharpe = val_summary.get("last_sharpe")
            best_sharpe = val_summary.get("best_sharpe")
            cons_fail = val_summary.get("consecutive_failures", 0)
            trend = val_summary.get("trend", "—")
            n_evals = val_summary.get("n_evaluations", 0)
            last_ts = val_summary.get("last_timestamp")

            if last_ts:
                ago = (time.time() - last_ts) / 3600
                self._v["_ft_val_last"].set(f"{ago:.1f}h ago", "subtext", 13)
            else:
                self._v["_ft_val_last"].set("Not run yet", "dim", 13)

            if last_passed is not None:
                self._v["_ft_val_result"].set(
                    "PASS ✓" if last_passed else "FAIL ✗",
                    "green" if last_passed else "red", 14
                )
            self._v["_ft_val_sharpe"].set(
                f"{last_sharpe:.3f}" if last_sharpe is not None else "—",
                "text", 13
            )
            self._v["_ft_val_baseline"].set(
                f"{m.get('_ft_baseline_sharpe', 0):.3f}", "subtext", 13
            )
            trend_color = "green" if trend == "IMPROVING" else ("yellow" if trend == "STABLE" else "red")
            self._v["_ft_val_trend"].set(trend, trend_color, 13)
            self._v["_ft_val_failures"].set(
                f"{cons_fail} / {self.cfg.finetune_auto_rollback_failures}",
                "red" if cons_fail >= self.cfg.finetune_auto_rollback_failures - 1 else "text", 13
            )

        # ── Checkpoints list ─────────────────────────────────────────────────
        checkpoints = m.get("_ft_checkpoints", [])
        if self._checkpoint_list.count() != len(checkpoints):
            self._checkpoint_list.clear()
            for name in checkpoints:
                item = QListWidgetItem(name)
                self._checkpoint_list.addItem(item)

        have_checkpoints = len(checkpoints) > 0
        self._rollback_btn.setEnabled(have_checkpoints)

        # Enable promote only if last validation passed
        val_passed = val_summary.get("last_passed", False) if val_summary else False
        self._promote_btn.setEnabled(have_checkpoints and val_passed)

        # ── Episode summary ──────────────────────────────────────────────────
        ep_stats = m.get("_ft_last_episode_stats", {})
        if ep_stats:
            trades = ep_stats.get("trades", 0)
            wins = ep_stats.get("wins", 0)
            pnl = ep_stats.get("pnl", 0.0)
            wr = ep_stats.get("win_rate", 0.0)
            balance = ep_stats.get("balance", 0.0)
            ep_kl = ep_stats.get("kl_divergence", 0.0)
            ep_ewc = ep_stats.get("ewc_penalty", 0.0)

            self._v["_ft_ep_trades"].set(str(trades), "text", 14)
            self._v["_ft_ep_wr"].set(f"{wr:.1%}", "green" if wr >= 0.50 else "red", 14)
            pnl_color = "green" if pnl >= 0 else "red"
            self._v["_ft_ep_pnl"].set(
                f"+£{pnl:.2f}" if pnl >= 0 else f"-£{abs(pnl):.2f}",
                pnl_color, 14
            )
            self._v["_ft_ep_balance"].set(f"£{balance:,.2f}", "text", 14)
            self._v["_ft_ep_kl"].set(f"{ep_kl:.3f}", kl_color, 14)
            self._v["_ft_ep_ewc"].set(f"{ep_ewc:,.0f}", "subtext", 14)

        # ── Button states ────────────────────────────────────────────────────
        # INITIALIZING counts as running so Stop is available during MT5 prefill
        is_running = state in (FT_INITIALIZING, FT_COLLECTING, FT_TRAINING, FT_VALIDATING)
        is_idle = state in (FT_IDLE, FT_STOPPED, FT_ERROR)
        self._start_btn.setEnabled(is_idle and self._selected_model is not None)
        self._stop_btn.setEnabled(is_running)
        # Validate only makes sense once at least one episode has run
        can_validate = state in (FT_COLLECTING, FT_TRAINING, FT_VALIDATING)
        self._validate_btn.setEnabled(can_validate)

    # ─────────────────────────────────────────────────────────────────────────
    # Control handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, checked: bool) -> None:
        self._mode = "live" if self._radio_live.isChecked() else "historical"

    def _pick_model(self) -> None:
        """Show model picker dialog."""
        model_dir = self.cfg.model_dir
        if not model_dir.exists():
            QMessageBox.warning(self, "No Models", f"Model directory not found:\n{model_dir}")
            return

        zip_files = sorted(
            model_dir.glob("*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not zip_files:
            QMessageBox.warning(self, "No Models", "No model checkpoints found in storage/models/")
            return

        import datetime
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model to Fine-Tune")
        dialog.setMinimumSize(500, 300)
        dialog.setStyleSheet(
            f"background-color: {C['bg']}; color: {C['text']};"
        )
        layout = QVBoxLayout(dialog)

        lbl = QLabel("Select a checkpoint to fine-tune:")
        lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 14px; padding: 6px;")
        layout.addWidget(lbl)

        list_widget = QListWidget()
        list_widget.setStyleSheet(
            f"background-color: {C['surface']}; color: {C['text']}; "
            f"font-size: 13px; border: 1px solid {C['border']}; padding: 4px;"
        )
        for zf in zip_files:
            size_mb = zf.stat().st_size / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(zf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            item = QListWidgetItem(f"{zf.name}    ({size_mb:.1f} MB, {mtime})")
            item.setData(Qt.ItemDataRole.UserRole, str(zf))
            list_widget.addItem(item)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet(
            f"QPushButton {{ background-color: {C['surface2']}; color: {C['text']}; "
            f"font-weight: bold; padding: 6px 20px; }}"
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected = list_widget.currentItem()
        if selected:
            path = selected.data(Qt.ItemDataRole.UserRole)
            self._selected_model = path
            self._model_label.setText(Path(path).name)
            self._model_label.setStyleSheet(f"color: {C['green']}; font-size: 13px;")

    def _on_start(self) -> None:
        """Start fine-tuning."""
        if not self._selected_model:
            QMessageBox.warning(
                self, "No Model Selected",
                "Please select a model to fine-tune first."
            )
            return

        if self._mode == "live":
            reply = QMessageBox.question(
                self, "Start Live Fine-Tuning",
                f"Start live fine-tuning on:\n{Path(self._selected_model).name}\n\n"
                f"This will connect to MT5 and begin accumulating live bars.\n"
                f"Training episodes will run every "
                f"{self.cfg.finetune_episode_interval_hours:.0f} hours.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        else:
            reply = QMessageBox.question(
                self, "Start Historical Fine-Tuning",
                f"Re-train on historical data using:\n{Path(self._selected_model).name}\n\n"
                "Week counter and balance will reset.\n"
                "EWC protection will be active.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Disable start immediately — prevents double-click while tuner initializes.
        # The 1Hz refresh will take over button states once the tuner reports its state.
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        # Create and start tuner.
        # Trainer ref: explicit injection takes priority, then shared_metrics (set by Trainer.run())
        trainer = self._trainer or self.metrics.get("_trainer_ref")
        from src.training.live_fine_tuner import LiveFineTuner
        self._tuner = LiveFineTuner(self.cfg, self.metrics)
        self._tuner.start(
            model_path=self._selected_model,
            mode=self._mode,
            trainer=trainer,
        )

    def _on_stop(self) -> None:
        """Stop fine-tuning."""
        if self._tuner is not None:
            self._tuner.stop()

    def _on_validate_now(self) -> None:
        """Trigger immediate validation."""
        if self._tuner is not None:
            self._tuner.validate_now()

    def _on_promote(self) -> None:
        """Promote fine-tuned model to live."""
        if self._tuner is None:
            return

        reply = QMessageBox.question(
            self, "Promote Model",
            "Export the current fine-tuned model as a live deployment package?\n\n"
            "The package will be saved to storage/finetune/ and can be copied\n"
            "to live_dashboard/model/ for deployment.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        output_path = self._tuner.promote()
        if output_path:
            QMessageBox.information(
                self, "Model Promoted",
                f"Fine-tuned model exported to:\n{output_path}\n\n"
                "Copy this file to live_dashboard/model/ to deploy."
            )
        else:
            QMessageBox.warning(self, "Promotion Failed", "Model promotion failed. Check logs.")

    def _on_rollback(self) -> None:
        """Roll back to selected checkpoint."""
        if self._tuner is None:
            return

        selected = self._checkpoint_list.currentRow()
        if selected < 0:
            return

        reply = QMessageBox.question(
            self, "Rollback",
            f"Roll back to checkpoint #{selected}?\n"
            "Current fine-tuned weights will be replaced.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self._tuner.rollback(selected):
            QMessageBox.information(self, "Rollback Complete", "Model rolled back successfully.")
        else:
            QMessageBox.warning(self, "Rollback Failed", "Rollback failed. Check logs.")

    def set_trainer(self, trainer) -> None:
        """Inject trainer reference for validation gate access."""
        self._trainer = trainer
