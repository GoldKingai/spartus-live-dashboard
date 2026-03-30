"""PyQt6 tabbed dashboard for Spartus Trading AI training.

Seven-tab layout for clean, readable display:
  Tab 1 — OVERVIEW:        Account, Progress, Balance Chart (large), Decisions, Alerts
  Tab 2 — METRICS:         Learning, This Week, Predictions, Reward Breakdown
  Tab 3 — INTERNALS:       SAC, Convergence, Curriculum, Safety
  Tab 4 — DB VIEWER:       Browse SQLite memory database (trades, patterns, predictions, etc.)
  Tab 5 — TRADE JOURNAL:   AI reasoning, lesson classification, self-reflection
  Tab 6 — MODEL EXPORT:    Package trained model for live dashboard deployment
  Tab 7 — LIVE FINE-TUNE:  Adapt model to live market data (anti-forgetting system)

Design: dark theme, bright green/red for profit/loss, large readable fonts.
Reads from shared_metrics dict via QTimer at 1Hz.
"""

import datetime
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QGroupBox,
    QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QProgressBar, QFrame, QTabWidget, QTableWidget, QTableWidgetItem,
    QComboBox, QHeaderView, QAbstractItemView, QSplitter,
    QDialog, QListWidget, QListWidgetItem, QMessageBox, QDialogButtonBox,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QFont

import pyqtgraph as pg

from src.config import TrainingConfig
from src.training.exporter import ModelExporter
from src.training.tab_live_finetune import TabLiveFinetune

# ── Color Palette ────────────────────────────────────────────────────────────
# Dark background with BRIGHT signal colors
C = {
    "bg":       "#0d1117",      # deep dark (GitHub-dark style)
    "surface":  "#161b22",      # panel background
    "surface2": "#1c2333",      # slightly lighter panel
    "border":   "#30363d",      # panel borders
    "text":     "#e6edf3",      # bright white text
    "subtext":  "#b1bac4",      # lighter gray (readable!)
    "label":    "#8b949e",      # label gray (still readable)
    "dim":      "#656d76",      # only for truly unimportant items
    "green":    "#2dcc2d",      # BRIGHT profit green
    "red":      "#ff3333",      # BRIGHT loss red
    "yellow":   "#ffcc00",      # bright warning yellow
    "blue":     "#58a6ff",      # bright link/accent blue
    "cyan":     "#39d5ff",      # bright accent cyan
    "mauve":    "#bc8cff",      # purple accent
    "peach":    "#ffa657",      # orange accent
}

DARK_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']}; color: {C['text']};
    font-family: 'Segoe UI', 'Cascadia Code', Consolas, monospace;
}}
QGroupBox {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    border-radius: 8px; margin-top: 18px; padding: 14px 10px 10px 10px;
    font-size: 13px; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 8px;
    color: {C['cyan']}; font-size: 13px;
}}
QLabel {{ font-size: 14px; }}
QPushButton {{
    background-color: {C['surface2']}; border: 1px solid {C['border']};
    border-radius: 6px; padding: 8px 24px; font-size: 13px;
    color: {C['text']}; font-weight: bold;
}}
QPushButton:hover {{ background-color: {C['border']}; }}
QTextEdit {{
    background-color: {C['bg']}; border: 1px solid {C['border']};
    border-radius: 6px; font-family: 'Cascadia Code', Consolas, monospace;
    font-size: 13px; color: {C['text']}; padding: 6px;
}}
QProgressBar {{
    background-color: {C['border']}; border: 1px solid {C['dim']};
    border-radius: 4px; text-align: center; color: {C['text']};
    font-size: 12px; font-weight: bold;
}}
QProgressBar::chunk {{ background-color: {C['green']}; border-radius: 4px; }}
QTabWidget::pane {{
    border: 1px solid {C['border']}; border-radius: 8px;
    background-color: {C['bg']};
}}
QTabBar::tab {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px;
    padding: 10px 28px; margin-right: 4px;
    font-size: 14px; font-weight: bold; color: {C['label']};
}}
QTabBar::tab:selected {{
    background-color: {C['bg']}; color: {C['cyan']};
    border-bottom: 2px solid {C['cyan']};
}}
QTabBar::tab:hover {{ color: {C['text']}; }}
"""


def _tc(value, green_above=None, yellow_above=None,
        green_below=None, yellow_below=None,
        green_min=None, green_max=None):
    """Threshold -> color key."""
    if value is None:
        return "subtext"
    if green_min is not None and green_max is not None:
        return "green" if green_min <= value <= green_max else "red"
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
    return "text"


class VLabel(QLabel):
    """Value label with color and size support."""

    def set(self, text: str, color: str = "text", size: int = 15):
        self.setText(str(text))
        self.setStyleSheet(
            f"color: {C.get(color, C['text'])}; font-size: {size}px; font-weight: bold;")


class SpartusQtDashboard(QMainWindow):
    """PyQt6 tabbed dashboard — clean, readable, bright colors."""

    def __init__(self, config: TrainingConfig, shared_metrics: Dict):
        super().__init__()
        self.cfg = config
        self.metrics = shared_metrics
        self._v: Dict[str, VLabel] = {}
        self._bal_data: List[float] = []
        self._prev_bal: Optional[float] = None
        self._exporter = ModelExporter(config)
        self._export_combo_paths: Dict[str, str] = {}

        # Training launcher state (set via set_training_launcher)
        self._train_fn = None
        self._train_max_weeks = None
        self._train_seed = 42
        self._train_thread: Optional[threading.Thread] = None
        self._dashboard_state = "IDLE"  # IDLE, RUNNING, PAUSED
        self._auto_resume = False  # Set by auto_resume_on_show()

        self.setWindowTitle("SPARTUS TRADING AI")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # ── Header ─────────────────────────────────────────────────────────
        root.addWidget(self._build_header())

        # ── Tabs ───────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_overview_tab(), "OVERVIEW")
        self._tabs.addTab(self._build_metrics_tab(), "METRICS")
        self._tabs.addTab(self._build_internals_tab(), "AI INTERNALS")
        self._tabs.addTab(self._build_db_viewer_tab(), "DB VIEWER")
        self._tabs.addTab(self._build_journal_tab(), "TRADE JOURNAL")
        self._tabs.addTab(self._build_export_tab(), "MODEL EXPORT")
        self._tab_live_ft = TabLiveFinetune(config, shared_metrics)
        self._tabs.addTab(self._tab_live_ft, "LIVE FINE-TUNE")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, stretch=1)

        # ── Footer ─────────────────────────────────────────────────────────
        root.addWidget(self._build_footer())

        # ── Timer ──────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(1000)

    # ══ Helper: build a label-value panel ════════════════════════════════════

    def _panel(self, title: str, rows: list, label_width: int = 140) -> QGroupBox:
        box = QGroupBox(title)
        g = QGridLayout(box)
        g.setSpacing(6)
        g.setContentsMargins(12, 8, 12, 8)
        for i, (name, key) in enumerate(rows):
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            lbl.setFixedWidth(label_width)
            val = VLabel("-")
            val.set("-")
            self._v[key] = val
            g.addWidget(lbl, i, 0)
            g.addWidget(val, i, 1)
        return box

    # ══ Header ═══════════════════════════════════════════════════════════════

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(44)
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 4, 12, 4)

        title = QLabel("SPARTUS TRAINING ENGINE v3.4")
        title.setStyleSheet(
            f"color: {C['blue']}; font-size: 18px; font-weight: bold;")
        h.addWidget(title)

        h.addSpacing(24)
        self._hdr_info = QLabel("IDLE — Ready")
        self._hdr_info.setStyleSheet(f"color: {C['subtext']}; font-size: 14px;")
        h.addWidget(self._hdr_info, stretch=1)

        self._health_badge = QLabel("IDLE")
        self._health_badge.setMinimumWidth(130)
        self._health_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._health_badge.setStyleSheet(
            f"color: {C['cyan']}; font-size: 15px; font-weight: bold; "
            f"padding: 4px 16px; border: 2px solid {C['cyan']}; border-radius: 6px;")
        h.addWidget(self._health_badge)

        h.addSpacing(12)

        # Start button — fresh training
        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedWidth(100)
        self._start_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['green']}; "
            f"font-weight: bold; font-size: 13px;")
        self._start_btn.clicked.connect(self._on_start_clicked)
        h.addWidget(self._start_btn)

        # Resume button — load checkpoint and continue from saved week
        self._resume_btn = QPushButton("Resume")
        self._resume_btn.setFixedWidth(110)
        self._resume_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['blue']}; "
            f"font-weight: bold; font-size: 13px;")
        self._resume_btn.clicked.connect(self._on_resume_clicked)
        h.addWidget(self._resume_btn)

        # Pause button
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedWidth(100)
        self._pause_btn.setEnabled(False)  # Disabled in IDLE
        self._pause_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['yellow']}; "
            f"font-weight: bold; font-size: 13px;")
        self._pause_btn.clicked.connect(self._toggle_pause)
        h.addWidget(self._pause_btn)

        quit_btn = QPushButton("Quit")
        quit_btn.setFixedWidth(90)
        quit_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['red']}; "
            f"font-weight: bold; font-size: 13px;")
        quit_btn.clicked.connect(self._request_quit)
        h.addWidget(quit_btn)

        return w

    # ══ Tab 1: OVERVIEW ═════════════════════════════════════════════════════

    def _build_overview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # Row 1: Progress (full width)
        layout.addWidget(self._build_progress())

        # Row 2: Account | quick stats
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(self._panel("Account", [
            ("Starting", "acct_start"), ("Current", "acct_cur"),
            ("Peak", "acct_peak"), ("Drawdown", "acct_dd"),
            ("Return", "acct_ret"), ("Bankruptcies", "acct_bank"),
        ], label_width=110))
        row2.addWidget(self._panel("Quick Stats", [
            ("Win Rate", "qs_wr"), ("Trades", "qs_trades"),
            ("Profit Factor", "qs_pf"), ("Sharpe", "qs_sh"),
            ("Week P/L", "qs_wpnl"), ("Speed", "qs_speed"),
        ], label_width=110))
        layout.addLayout(row2)

        # Row 3: Balance Chart (LARGE — the star of the overview)
        layout.addWidget(self._build_chart(), stretch=1)

        # Row 4: Decision Log | Alerts
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(self._build_decision_log())
        row4.addWidget(self._build_alerts())
        layout.addLayout(row4)

        return tab

    # ══ Tab 2: METRICS ═══════════════════════════════════════════════════════

    def _build_metrics_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # Row 1: Learning Metrics | This Week
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(self._panel("Learning Metrics", [
            ("Win Rate", "lm_wr"), ("Profit Factor", "lm_pf"),
            ("Avg Trade P/L", "lm_avg"), ("Sharpe Ratio", "lm_sh"),
            ("Memory Trades", "lm_mem"), ("Patterns", "lm_pat"),
            ("Avg Lot Size", "lm_lot"),
        ]))
        row1.addWidget(self._panel("This Week", [
            ("Trades", "tw_trades"), ("Wins", "tw_wins"),
            ("P/L", "tw_pnl"), ("Best Trade", "tw_best"),
            ("Worst Trade", "tw_worst"), ("Avg Hold", "tw_hold"),
            ("Commission", "tw_comm"),
        ]))
        layout.addLayout(row1)

        # Row 2: Predictions & TP | Reward Breakdown
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(self._panel("Predictions && TP Accuracy", [
            ("Trend Accuracy", "pr_acc"), ("UP Accuracy", "pr_up"),
            ("DOWN Accuracy", "pr_dn"),
            ("Verified", "pr_ver"), ("Pending", "pr_pend"),
            ("TP Hit Rate", "tp_hit"), ("TP Reachable", "tp_reach"),
            ("SL Hit Rate", "tp_sl"),
        ]))
        row2.addWidget(self._panel("Reward Breakdown", [
            ("R1 Pos P/L (0.40)", "rw_r1"), ("R2 Quality (0.20)", "rw_r2"),
            ("R3 DD Pen (0.15)", "rw_r3"), ("R4 D.Sharpe (0.15)", "rw_r4"),
            ("R5 Risk-Adj (0.10)", "rw_r5"),
            ("Raw", "rw_raw"), ("Normalised", "rw_norm"),
            ("\u03bc (mean)", "rw_mu"), ("\u03c3 (std)", "rw_sig"),
            ("Clip %", "rw_clip"),
        ]))
        layout.addLayout(row2)

        layout.addStretch(1)
        return tab

    # ══ Tab 3: AI INTERNALS ══════════════════════════════════════════════════

    def _build_internals_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # Row 1: SAC Internals | Convergence
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(self._panel("SAC Internals", [
            ("\u03b1 (Entropy Coeff)", "sac_a"),
            ("Policy Entropy", "sac_ent"),
            ("Q\u0304 Mean Value", "sac_qm"),
            ("Q\u2191 Max Value", "sac_qx"),
            ("\u2207\u03c0 Actor Grad", "sac_ag"),
            ("\u2207Q Critic Grad", "sac_cg"),
            ("L\u03c0 Actor Loss", "sac_al"),
            ("LQ Critic Loss", "sac_cl"),
            ("Learning Rate", "sac_lr"),
            ("Buffer Fill", "sac_buf"),
        ], label_width=160))
        row1.addWidget(self._panel("Convergence", [
            ("Status", "cv_st"),
            ("Val Sharpe (50w)", "cv_sh"),
            ("Best Checkpoint", "cv_best"),
            ("Since Best", "cv_since"),
            ("Action Std", "cv_astd"),
            ("Entropy Trend", "cv_etrend"),
        ], label_width=160))
        layout.addLayout(row1)

        # Row 2: Curriculum & Regime | Safety & Anti-Hack
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(self._panel("Curriculum && Regime", [
            ("Stage", "cur_stg"), ("Difficulty", "cur_diff"),
            ("Regime", "cur_reg"),
            ("\u25b2 Trending Up", "cur_bup"),
            ("\u25bc Trending Down", "cur_bdn"),
            ("\u2550 Ranging", "cur_brng"),
            ("\u26a1 Volatile", "cur_bvol"),
            ("Pattern Conf", "cur_pat"),
        ], label_width=160))
        row2.addWidget(self._panel("Safety && Anti-Hack", [
            ("Daily Trades", "sf_daily"), ("Conviction", "sf_conv"),
            ("Hold Blocks", "sf_hold"), ("Obs Health", "sf_obs"),
            ("Dead Features", "sf_dead"), ("NaN Features", "sf_nan"),
            ("Grad Clip %", "sf_gclip"), ("LR Phase", "sf_lr"),
            ("Domain Noise", "sf_noise"), ("LSTM Status", "sf_lstm"),
        ], label_width=160))
        layout.addLayout(row2)

        layout.addStretch(1)
        return tab

    # ══ Panel Builders ═══════════════════════════════════════════════════════

    def _build_progress(self) -> QGroupBox:
        box = QGroupBox("Progress")
        g = QGridLayout(box)
        g.setSpacing(6)
        g.setContentsMargins(12, 8, 12, 8)

        fields_r0 = [("Week", "pg_week"), ("Stage", "pg_stage")]
        for i, (label, key) in enumerate(fields_r0):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            val = VLabel("-")
            self._v[key] = val
            g.addWidget(lbl, 0, i * 3)
            g.addWidget(val, 0, i * 3 + 1)

        self._prog_bar = QProgressBar()
        self._prog_bar.setFixedHeight(18)
        g.addWidget(self._prog_bar, 0, 6, 1, 2)

        fields_r1 = [("Steps", "pg_steps"), ("This Week", "pg_wsteps"),
                      ("Speed", "pg_speed")]
        for i, (label, key) in enumerate(fields_r1):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            val = VLabel("-")
            self._v[key] = val
            g.addWidget(lbl, 1, i * 3)
            g.addWidget(val, 1, i * 3 + 1)

        fields_r2 = [("Time", "pg_time"), ("ETA", "pg_eta"), ("Session", "pg_session")]
        for i, (label, key) in enumerate(fields_r2):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
            val = VLabel("-")
            self._v[key] = val
            g.addWidget(lbl, 2, i * 3)
            g.addWidget(val, 2, i * 3 + 1)

        return box

    def _build_chart(self) -> QGroupBox:
        box = QGroupBox("Balance Curve")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        pg.setConfigOptions(antialias=True)

        self._bplot = pg.PlotWidget()
        self._bplot.setBackground(C['bg'])
        self._bplot.setLabel('left', 'Balance (\u00a3)',
                             color=C['subtext'], **{'font-size': '13px'})
        self._bplot.setLabel('bottom', 'Time (updates)',
                             color=C['subtext'], **{'font-size': '13px'})
        self._bplot.showGrid(x=True, y=True, alpha=0.15)
        self._bplot.getAxis('left').setTextPen(C['subtext'])
        self._bplot.getAxis('bottom').setTextPen(C['subtext'])
        self._bplot.setMinimumHeight(200)

        # Starting balance reference line
        ini = self.cfg.initial_balance
        ref_line = pg.InfiniteLine(
            pos=ini, angle=0,
            pen=pg.mkPen(C['dim'], width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._bplot.addItem(ref_line)

        # Green and red segments for up/down
        self._bcurve_green = self._bplot.plot(
            pen=pg.mkPen(C['green'], width=2.5))
        self._bcurve_red = self._bplot.plot(
            pen=pg.mkPen(C['red'], width=2.5))

        lay.addWidget(self._bplot)
        return box

    def _build_decision_log(self) -> QGroupBox:
        box = QGroupBox("AI Decision Log (latest 8)")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        self._decisions_edit = QTextEdit()
        self._decisions_edit.setReadOnly(True)
        self._decisions_edit.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self._decisions_edit.setMinimumHeight(120)
        self._decisions_edit.setMaximumHeight(180)
        self._decisions_html = ""
        lay.addWidget(self._decisions_edit)
        return box

    def _build_alerts(self) -> QGroupBox:
        box = QGroupBox("Alerts && Warnings")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        self._alerts_edit = QTextEdit()
        self._alerts_edit.setReadOnly(True)
        self._alerts_edit.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self._alerts_edit.setMinimumHeight(120)
        self._alerts_edit.setMaximumHeight(180)
        self._alerts_html = ""
        lay.addWidget(self._alerts_edit)
        return box

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(32)
        w.setStyleSheet(
            f"background-color: {C['surface']}; "
            f"border-top: 1px solid {C['border']};")
        h = QHBoxLayout(w)
        h.setContentsMargins(14, 2, 14, 2)
        h.setSpacing(24)

        style = f"color: {C['subtext']}; font-size: 12px; font-family: Consolas, monospace;"
        self._foot_gpu = QLabel("GPU: ...")
        self._foot_gpu.setStyleSheet(style)
        h.addWidget(self._foot_gpu)

        self._foot_ram = QLabel("RAM: ...")
        self._foot_ram.setStyleSheet(style)
        h.addWidget(self._foot_ram)

        h.addStretch(1)

        self._foot_chk = QLabel("Checkpoint: -")
        self._foot_chk.setStyleSheet(style)
        h.addWidget(self._foot_chk)

        return w

    # ══ Tab 4: DB VIEWER ═══════════════════════════════════════════════════

    def _build_db_viewer_tab(self) -> QWidget:
        """SQLite memory database viewer — browse all 5 tables live."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Control bar ──────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        lbl = QLabel("Table:")
        lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 14px; font-weight: bold;")
        ctrl.addWidget(lbl)

        self._db_table_combo = QComboBox()
        self._db_table_combo.addItems([
            "trades", "patterns", "predictions", "tp_tracking", "checkpoints"
        ])
        self._db_table_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {C['surface2']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 6px 16px; font-size: 14px; font-weight: bold;
                min-width: 160px;
            }}
            QComboBox::drop-down {{
                border: none; width: 30px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {C['surface']}; color: {C['text']};
                border: 1px solid {C['border']}; selection-background-color: {C['border']};
                font-size: 14px;
            }}
        """)
        self._db_table_combo.currentTextChanged.connect(self._on_db_table_changed)
        ctrl.addWidget(self._db_table_combo)

        ctrl.addSpacing(10)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['cyan']}; "
            f"font-weight: bold; font-size: 13px;")
        refresh_btn.clicked.connect(self._refresh_db_view)
        ctrl.addWidget(refresh_btn)

        ctrl.addSpacing(10)

        self._db_auto_refresh = QPushButton("Auto: OFF")
        self._db_auto_refresh.setFixedWidth(110)
        self._db_auto_refresh.setCheckable(True)
        self._db_auto_refresh.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['dim']}; "
            f"font-weight: bold; font-size: 13px;")
        self._db_auto_refresh.clicked.connect(self._toggle_db_auto_refresh)
        ctrl.addWidget(self._db_auto_refresh)

        ctrl.addStretch(1)

        self._db_row_count = QLabel("Rows: -")
        self._db_row_count.setStyleSheet(
            f"color: {C['cyan']}; font-size: 14px; font-weight: bold;")
        ctrl.addWidget(self._db_row_count)

        ctrl.addSpacing(20)

        self._db_status = QLabel("")
        self._db_status.setStyleSheet(
            f"color: {C['subtext']}; font-size: 12px;")
        ctrl.addWidget(self._db_status)

        layout.addLayout(ctrl)

        # ── Table summary cards ──────────────────────────────────────────
        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)
        self._db_summary_labels = {}
        for tbl_name, icon in [("trades", "T"), ("patterns", "P"),
                                 ("predictions", "Pr"), ("tp_tracking", "TP"),
                                 ("checkpoints", "Ck")]:
            card = QLabel(f"{icon}: -")
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setStyleSheet(
                f"background-color: {C['surface2']}; color: {C['subtext']}; "
                f"border: 1px solid {C['border']}; border-radius: 6px; "
                f"padding: 4px 12px; font-size: 12px; font-weight: bold;")
            card.setFixedHeight(28)
            self._db_summary_labels[tbl_name] = card
            summary_row.addWidget(card)
        layout.addLayout(summary_row)

        # ── Data grid ────────────────────────────────────────────────────
        self._db_grid = QTableWidget()
        self._db_grid.setAlternatingRowColors(True)
        self._db_grid.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._db_grid.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._db_grid.horizontalHeader().setStretchLastSection(True)
        self._db_grid.verticalHeader().setDefaultSectionSize(26)
        self._db_grid.setStyleSheet(f"""
            QTableWidget {{
                background-color: {C['bg']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                gridline-color: {C['border']}; font-size: 13px;
                font-family: 'Cascadia Code', Consolas, monospace;
            }}
            QTableWidget::item {{
                padding: 2px 8px;
            }}
            QTableWidget::item:alternate {{
                background-color: {C['surface']};
            }}
            QTableWidget::item:selected {{
                background-color: {C['border']}; color: {C['cyan']};
            }}
            QHeaderView::section {{
                background-color: {C['surface2']}; color: {C['cyan']};
                border: 1px solid {C['border']}; padding: 4px 8px;
                font-size: 12px; font-weight: bold;
            }}
        """)
        layout.addWidget(self._db_grid, stretch=1)

        # ── DB info footer ───────────────────────────────────────────────
        info_row = QHBoxLayout()
        self._db_path_label = QLabel(f"DB: {self.cfg.memory_db_path}")
        self._db_path_label.setStyleSheet(
            f"color: {C['dim']}; font-size: 11px;")
        info_row.addWidget(self._db_path_label)

        info_row.addStretch(1)

        self._db_size_label = QLabel("Size: -")
        self._db_size_label.setStyleSheet(
            f"color: {C['dim']}; font-size: 11px;")
        info_row.addWidget(self._db_size_label)

        layout.addLayout(info_row)

        # State
        self._db_auto_on = False
        self._db_last_refresh = 0

        return tab

    def _on_db_table_changed(self, table_name: str):
        """User selected a different table from the combo box."""
        self._refresh_db_view()

    def _toggle_db_auto_refresh(self):
        """Toggle auto-refresh (every 5 seconds when DB tab is active)."""
        self._db_auto_on = not self._db_auto_on
        if self._db_auto_on:
            self._db_auto_refresh.setText("Auto: ON")
            self._db_auto_refresh.setStyleSheet(
                f"background-color: {C['surface2']}; color: {C['green']}; "
                f"font-weight: bold; font-size: 13px;")
        else:
            self._db_auto_refresh.setText("Auto: OFF")
            self._db_auto_refresh.setStyleSheet(
                f"background-color: {C['surface2']}; color: {C['dim']}; "
                f"font-weight: bold; font-size: 13px;")

    def _refresh_db_view(self):
        """Load selected table from SQLite and display in the grid."""
        db_path = Path(self.cfg.memory_db_path)
        if not db_path.exists():
            self._db_status.setText("DB file not found")
            self._db_status.setStyleSheet(f"color: {C['red']}; font-size: 12px;")
            return

        table_name = self._db_table_combo.currentText()

        try:
            # Open a separate read-only connection (safe for concurrent access)
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA journal_mode=WAL")

            # Update summary cards (row counts for all tables)
            for tbl in ["trades", "patterns", "predictions", "tp_tracking", "checkpoints"]:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    label_map = {"trades": "T", "patterns": "P",
                                 "predictions": "Pr", "tp_tracking": "TP",
                                 "checkpoints": "Ck"}
                    lbl_text = f"{label_map[tbl]}: {count:,}"
                    self._db_summary_labels[tbl].setText(lbl_text)
                    if tbl == table_name:
                        self._db_summary_labels[tbl].setStyleSheet(
                            f"background-color: {C['border']}; color: {C['cyan']}; "
                            f"border: 1px solid {C['cyan']}; border-radius: 6px; "
                            f"padding: 4px 12px; font-size: 12px; font-weight: bold;")
                    else:
                        self._db_summary_labels[tbl].setStyleSheet(
                            f"background-color: {C['surface2']}; color: {C['subtext']}; "
                            f"border: 1px solid {C['border']}; border-radius: 6px; "
                            f"padding: 4px 12px; font-size: 12px; font-weight: bold;")
                except Exception:
                    self._db_summary_labels[tbl].setText(f"?: err")

            # Get column info
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]

            if not columns:
                conn.close()
                self._db_status.setText(f"Table '{table_name}' not found")
                return

            # Fetch rows (latest 500, newest first)
            rows = conn.execute(
                f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 500"
            ).fetchall()

            total_rows = conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            conn.close()

            # Populate grid
            self._db_grid.setColumnCount(len(columns))
            self._db_grid.setHorizontalHeaderLabels(columns)
            self._db_grid.setRowCount(len(rows))

            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(self._format_db_cell(val, columns[c]))
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                        if isinstance(val, (int, float)) else
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                    # Color profit/loss cells
                    col_name = columns[c].lower()
                    if col_name in ("pnl", "total_pnl", "pnl_pct") and isinstance(val, (int, float)):
                        if val > 0:
                            item.setForeground(pg.mkColor(C['green']))
                        elif val < 0:
                            item.setForeground(pg.mkColor(C['red']))
                    elif col_name == "correct" and val is not None:
                        if val == 1:
                            item.setForeground(pg.mkColor(C['green']))
                        elif val == 0:
                            item.setForeground(pg.mkColor(C['red']))
                    elif col_name == "side":
                        if val == "LONG":
                            item.setForeground(pg.mkColor(C['green']))
                        elif val == "SHORT":
                            item.setForeground(pg.mkColor(C['red']))

                    self._db_grid.setItem(r, c, item)

            # Resize columns to content
            self._db_grid.resizeColumnsToContents()

            # Update labels
            showing = len(rows)
            self._db_row_count.setText(
                f"Rows: {total_rows:,}" + (f" (showing latest {showing})" if showing < total_rows else ""))

            # DB file size
            db_size = db_path.stat().st_size
            if db_size < 1024:
                size_str = f"{db_size} B"
            elif db_size < 1024 * 1024:
                size_str = f"{db_size / 1024:.1f} KB"
            else:
                size_str = f"{db_size / (1024 * 1024):.1f} MB"
            self._db_size_label.setText(f"Size: {size_str}")

            self._db_status.setText(f"Loaded at {time.strftime('%H:%M:%S')}")
            self._db_status.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
            self._db_last_refresh = time.time()

        except Exception as e:
            self._db_status.setText(f"Error: {str(e)[:60]}")
            self._db_status.setStyleSheet(f"color: {C['red']}; font-size: 12px;")

    # ══ Tab 5: TRADE JOURNAL ═════════════════════════════════════════════

    def _build_journal_tab(self) -> QWidget:
        """Trade journal — AI reasoning, lessons, self-reflection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Lesson type summary cards ────────────────────────────────
        summary_row = QHBoxLayout()
        summary_row.setSpacing(6)
        self._jrnl_lesson_labels = {}
        lesson_colors = {
            "GOOD_TRADE": C['green'], "SCALP_WIN": C['green'],
            "WRONG_DIRECTION": C['red'], "EMERGENCY_STOP": C['red'], "CIRCUIT_BREAKER": C['red'],
            "CORRECT_DIR_CLOSED_EARLY": C['yellow'], "CORRECT_DIR_BAD_SL": C['yellow'],
            "BAD_TIMING": C['peach'], "WHIPSAW": C['peach'],
            "BREAKEVEN": C['subtext'], "HELD_TOO_LONG": C['mauve'],
        }
        for ltype in ["GOOD_TRADE", "WRONG_DIRECTION", "EMERGENCY_STOP",
                       "CORRECT_DIR_BAD_SL", "WHIPSAW", "BREAKEVEN"]:
            card = QLabel(f"{ltype}: -")
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            clr = lesson_colors.get(ltype, C['subtext'])
            card.setStyleSheet(
                f"background-color: {C['surface2']}; color: {clr}; "
                f"border: 1px solid {C['border']}; border-radius: 6px; "
                f"padding: 4px 8px; font-size: 11px; font-weight: bold;")
            card.setFixedHeight(28)
            self._jrnl_lesson_labels[ltype] = card
            summary_row.addWidget(card)
        layout.addLayout(summary_row)

        # ── Control bar ──────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['cyan']}; "
            f"font-weight: bold; font-size: 13px;")
        refresh_btn.clicked.connect(self._refresh_journal)
        ctrl.addWidget(refresh_btn)

        self._jrnl_auto_refresh = QPushButton("Auto: OFF")
        self._jrnl_auto_refresh.setFixedWidth(110)
        self._jrnl_auto_refresh.setCheckable(True)
        self._jrnl_auto_refresh.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['dim']}; "
            f"font-weight: bold; font-size: 13px;")
        self._jrnl_auto_refresh.clicked.connect(self._toggle_journal_auto_refresh)
        ctrl.addWidget(self._jrnl_auto_refresh)

        ctrl.addSpacing(10)

        # Lesson type filter
        lbl = QLabel("Filter:")
        lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
        ctrl.addWidget(lbl)

        self._jrnl_filter = QComboBox()
        self._jrnl_filter.addItems([
            "ALL", "GOOD_TRADE", "WRONG_DIRECTION", "EMERGENCY_STOP",
            "CORRECT_DIR_CLOSED_EARLY", "CORRECT_DIR_BAD_SL",
            "BAD_TIMING", "WHIPSAW", "BREAKEVEN", "SCALP_WIN", "HELD_TOO_LONG",
        ])
        self._jrnl_filter.setStyleSheet(f"""
            QComboBox {{
                background-color: {C['surface2']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 4px 12px; font-size: 13px; min-width: 180px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {C['surface']}; color: {C['text']};
                border: 1px solid {C['border']}; selection-background-color: {C['border']};
            }}
        """)
        self._jrnl_filter.currentTextChanged.connect(lambda _: self._refresh_journal())
        ctrl.addWidget(self._jrnl_filter)

        ctrl.addStretch(1)

        self._jrnl_count = QLabel("Entries: -")
        self._jrnl_count.setStyleSheet(
            f"color: {C['cyan']}; font-size: 14px; font-weight: bold;")
        ctrl.addWidget(self._jrnl_count)

        layout.addLayout(ctrl)

        # ── Splitter: journal table (top) + detail view (bottom) ─────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {C['border']}; height: 3px;
            }}
        """)

        # Journal table
        self._jrnl_grid = QTableWidget()
        self._jrnl_grid.setAlternatingRowColors(True)
        self._jrnl_grid.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._jrnl_grid.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._jrnl_grid.horizontalHeader().setStretchLastSection(True)
        self._jrnl_grid.verticalHeader().setDefaultSectionSize(26)
        self._jrnl_grid.setStyleSheet(f"""
            QTableWidget {{
                background-color: {C['bg']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                gridline-color: {C['border']}; font-size: 13px;
                font-family: 'Cascadia Code', Consolas, monospace;
            }}
            QTableWidget::item {{ padding: 2px 8px; }}
            QTableWidget::item:alternate {{ background-color: {C['surface']}; }}
            QTableWidget::item:selected {{ background-color: {C['border']}; color: {C['cyan']}; }}
            QHeaderView::section {{
                background-color: {C['surface2']}; color: {C['cyan']};
                border: 1px solid {C['border']}; padding: 4px 8px;
                font-size: 12px; font-weight: bold;
            }}
        """)
        self._jrnl_grid.currentCellChanged.connect(self._on_journal_row_selected)
        splitter.addWidget(self._jrnl_grid)

        # Detail panel
        detail_box = QGroupBox("Trade Detail")
        detail_lay = QVBoxLayout(detail_box)
        detail_lay.setContentsMargins(10, 8, 10, 8)

        self._jrnl_detail = QTextEdit()
        self._jrnl_detail.setReadOnly(True)
        self._jrnl_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self._jrnl_detail.setMinimumHeight(100)
        self._jrnl_detail.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C['bg']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 14px; padding: 8px; line-height: 1.5;
            }}
        """)
        detail_lay.addWidget(self._jrnl_detail)
        splitter.addWidget(detail_box)

        # Set initial splitter proportions (70% table, 30% detail)
        splitter.setSizes([500, 200])
        layout.addWidget(splitter, stretch=1)

        # State
        self._jrnl_auto_on = False
        self._jrnl_last_refresh = 0
        self._jrnl_entries = []  # Cached full entries for detail view

        return tab

    # ══ Tab 6: MODEL EXPORT ════════════════════════════════════════════════════

    def _build_export_tab(self) -> QWidget:
        """Model export — package trained model for live dashboard deployment."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Row 1: Model Info + Model Selection ─────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        # Model Info panel (left)
        info_box = self._panel("Best Model Info", [
            ("Source Week", "ex_week"),
            ("Val Sharpe", "ex_sharpe"),
            ("Balance", "ex_balance"),
            ("Timestamp", "ex_ts"),
            ("Obs Dim", "ex_obsdim"),
            ("Features", "ex_feats"),
        ], label_width=110)
        row1.addWidget(info_box)

        # Model Selection panel (right)
        sel_box = QGroupBox("Model Selection")
        sel_lay = QVBoxLayout(sel_box)
        sel_lay.setSpacing(8)

        src_row = QHBoxLayout()
        src_lbl = QLabel("Source:")
        src_lbl.setStyleSheet(
            f"color: {C['subtext']}; font-size: 14px; font-weight: bold;")
        src_row.addWidget(src_lbl)

        self._export_source_combo = QComboBox()
        self._export_source_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {C['surface2']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 6px 16px; font-size: 14px; font-weight: bold;
                min-width: 200px;
            }}
            QComboBox::drop-down {{ border: none; width: 30px; }}
            QComboBox QAbstractItemView {{
                background-color: {C['surface']}; color: {C['text']};
                border: 1px solid {C['border']};
                selection-background-color: {C['border']};
                font-size: 14px;
            }}
        """)
        src_row.addWidget(self._export_source_combo, stretch=1)
        sel_lay.addLayout(src_row)

        refresh_btn = QPushButton("Refresh List")
        refresh_btn.setFixedWidth(120)
        refresh_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['cyan']}; "
            f"font-weight: bold; font-size: 13px;")
        refresh_btn.clicked.connect(self._refresh_export_tab)
        sel_lay.addWidget(refresh_btn)
        sel_lay.addStretch(1)

        row1.addWidget(sel_box)
        layout.addLayout(row1)

        # ── Row 2: Validation Checks + Export Action ────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        # Validation checks (left)
        val_box = QGroupBox("Validation Checks")
        val_grid = QGridLayout(val_box)
        val_grid.setSpacing(4)
        self._check_labels: Dict[str, QLabel] = {}
        check_items = [
            ("model_file_exists", "Model file exists"),
            ("meta_file_exists", "Metadata file exists"),
            ("feature_count", "Feature count (67)"),
            ("obs_dim", "Obs dim (670)"),
            ("reward_state_present", "Reward state present"),
            ("stress_results", "Stress results available"),
            ("feature_baseline", "Feature baseline (54 features)"),
            ("correlation_baseline", "Correlation baseline (top-20)"),
            ("model_loads_ok", "Model loads OK"),
        ]
        for i, (key, display) in enumerate(check_items):
            desc = QLabel(display)
            desc.setStyleSheet(
                f"color: {C['subtext']}; font-size: 13px;")
            desc.setFixedWidth(170)
            status = QLabel("--")
            status.setStyleSheet(
                f"color: {C['dim']}; font-size: 13px; font-weight: bold;")
            self._check_labels[key] = status
            val_grid.addWidget(desc, i, 0)
            val_grid.addWidget(status, i, 1)
        row2.addWidget(val_box)

        # Export action (right)
        export_box = QGroupBox("Package for Live Dashboard")
        export_lay = QVBoxLayout(export_box)
        export_lay.setSpacing(8)

        self._export_btn = QPushButton("Package for Live Dashboard")
        self._export_btn.setMinimumHeight(44)
        self._export_btn.setStyleSheet(
            f"background-color: {C['surface2']}; color: {C['blue']}; "
            f"font-weight: bold; font-size: 14px; "
            f"border: 2px solid {C['blue']}; border-radius: 8px;")
        self._export_btn.clicked.connect(self._on_export_clicked)
        export_lay.addWidget(self._export_btn)

        # Output path
        out_row = QHBoxLayout()
        out_lbl = QLabel("Output:")
        out_lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
        out_lbl.setFixedWidth(60)
        self._export_path_label = QLabel("--")
        self._export_path_label.setStyleSheet(
            f"color: {C['text']}; font-size: 12px; "
            f"font-family: 'Cascadia Code', Consolas, monospace;")
        self._export_path_label.setWordWrap(True)
        out_row.addWidget(out_lbl)
        out_row.addWidget(self._export_path_label, stretch=1)
        export_lay.addLayout(out_row)

        # File size
        size_row = QHBoxLayout()
        size_lbl = QLabel("Size:")
        size_lbl.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
        size_lbl.setFixedWidth(60)
        self._export_size_label = QLabel("--")
        self._export_size_label.setStyleSheet(
            f"color: {C['cyan']}; font-size: 13px; font-weight: bold;")
        size_row.addWidget(size_lbl)
        size_row.addWidget(self._export_size_label, stretch=1)
        export_lay.addLayout(size_row)

        # Status message
        self._export_status_label = QLabel("")
        self._export_status_label.setStyleSheet(
            f"color: {C['subtext']}; font-size: 13px;")
        export_lay.addWidget(self._export_status_label)

        export_lay.addStretch(1)
        row2.addWidget(export_box)
        layout.addLayout(row2)

        # ── Row 3: Export Log ───────────────────────────────────────────
        log_box = QGroupBox("Export Log")
        log_lay = QVBoxLayout(log_box)
        self._export_log = QTextEdit()
        self._export_log.setReadOnly(True)
        self._export_log.setMaximumHeight(150)
        self._export_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C['bg']}; color: {C['text']};
                border: 1px solid {C['border']}; border-radius: 6px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 12px; padding: 6px;
            }}
        """)
        log_lay.addWidget(self._export_log)
        layout.addWidget(log_box, stretch=1)

        return tab

    def _refresh_export_tab(self):
        """Load available models and metadata into the export tab."""
        import json as _json

        model_dir = self.cfg.model_dir

        # ── Populate combo box with available models ────────────────────
        options = []
        if (model_dir / "spartus_best.zip").exists():
            options.append(("Best Model", str(model_dir / "spartus_best.zip")))
        if (model_dir / "spartus_latest.zip").exists():
            options.append(("Latest", str(model_dir / "spartus_latest.zip")))
        for p in sorted(model_dir.glob("spartus_week_*.zip")):
            week_num = ModelExporter._parse_week_from_filename(p.stem)
            options.append((f"Week {week_num:04d}", str(p)))

        self._export_source_combo.blockSignals(True)
        self._export_source_combo.clear()
        self._export_combo_paths = {}
        for label, path in options:
            self._export_source_combo.addItem(label)
            self._export_combo_paths[label] = path
        self._export_source_combo.blockSignals(False)

        # ── Load metadata for selected model ───────────────────────────
        selected = self._export_source_combo.currentText()
        model_path = self._export_combo_paths.get(selected, "")
        if model_path:
            meta_path = Path(model_path).with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = _json.load(f)
                    week = meta.get("week", "?")
                    sharpe = meta.get("val_sharpe", 0.0)
                    balance = meta.get("balance", 0.0)
                    ts = meta.get("timestamp", 0)
                    dt_str = datetime.datetime.fromtimestamp(ts).strftime(
                        "%Y-%m-%d %H:%M") if ts else "?"
                    self._v["ex_week"].set(f"W{week}")
                    self._v["ex_sharpe"].set(
                        f"{sharpe:.4f}" if sharpe is not None else "N/A",
                        "green" if sharpe and sharpe > 0.5 else (
                            "yellow" if sharpe and sharpe > 0 else "red"))
                    self._v["ex_balance"].set(
                        f"${balance:.2f}" if balance is not None else "N/A")
                    self._v["ex_ts"].set(dt_str)
                except Exception:
                    for key in ["ex_week", "ex_sharpe", "ex_balance", "ex_ts"]:
                        self._v[key].set("error", "red")
            else:
                # Week checkpoint — show what we know
                week_num = ModelExporter._parse_week_from_filename(
                    Path(model_path).stem)
                self._v["ex_week"].set(f"W{week_num}" if week_num else "?")
                self._v["ex_sharpe"].set("N/A (no meta)", "dim")
                self._v["ex_balance"].set("N/A", "dim")
                self._v["ex_ts"].set("N/A", "dim")
        else:
            for key in ["ex_week", "ex_sharpe", "ex_balance", "ex_ts"]:
                self._v[key].set("No models found", "red")

        # ── Always show config-derived fields ──────────────────────────
        self._v["ex_obsdim"].set(
            str(self.cfg.obs_dim),
            "green" if self.cfg.obs_dim == 670 else "red")
        self._v["ex_feats"].set(
            str(self.cfg.num_features),
            "green" if self.cfg.num_features == 67 else "red")

        # ── Run quick validation ───────────────────────────────────────
        self._run_validation_checks(quick=True)

    def _run_validation_checks(self, quick: bool = True):
        """Run validation checks and update the checklist labels."""
        selected = self._export_source_combo.currentText()
        model_path = self._export_combo_paths.get(selected)
        if not model_path:
            for lbl in self._check_labels.values():
                lbl.setText("--")
                lbl.setStyleSheet(
                    f"color: {C['dim']}; font-size: 13px; font-weight: bold;")
            return

        checks = self._exporter.validate_model(
            model_path=model_path, skip_load=quick)

        for key, lbl in self._check_labels.items():
            if key not in checks:
                lbl.setText("-- skipped")
                lbl.setStyleSheet(
                    f"color: {C['dim']}; font-size: 13px;")
                continue
            passed, detail = checks[key]
            if passed:
                lbl.setText(f"\u2713 {detail}")
                lbl.setStyleSheet(
                    f"color: {C['green']}; font-size: 13px; font-weight: bold;")
            else:
                lbl.setText(f"\u2717 {detail}")
                lbl.setStyleSheet(
                    f"color: {C['red']}; font-size: 13px; font-weight: bold;")

    def _on_export_clicked(self):
        """Package the selected model for live dashboard deployment."""
        self._export_btn.setEnabled(False)
        self._export_btn.setText("Packaging...")
        self._export_status_label.setText("Running validation...")
        self._export_status_label.setStyleSheet(
            f"color: {C['yellow']}; font-size: 13px;")
        QApplication.processEvents()

        selected = self._export_source_combo.currentText()
        model_path = self._export_combo_paths.get(selected)

        if not model_path:
            self._export_status_label.setText("No model selected")
            self._export_status_label.setStyleSheet(
                f"color: {C['red']}; font-size: 13px; font-weight: bold;")
            self._export_btn.setEnabled(True)
            self._export_btn.setText("Package for Live Dashboard")
            return

        try:
            # Full validation (including SAC.load)
            self._run_validation_checks(quick=False)
            QApplication.processEvents()

            self._export_status_label.setText("Creating package...")
            QApplication.processEvents()

            # Package
            out_path = self._exporter.package_model(model_path=model_path)
            out_path = Path(out_path)

            # File size
            size_bytes = out_path.stat().st_size
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

            self._export_path_label.setText(str(out_path))
            self._export_size_label.setText(size_str)
            self._export_status_label.setText(
                f"Packaged successfully at {time.strftime('%H:%M:%S')}")
            self._export_status_label.setStyleSheet(
                f"color: {C['green']}; font-size: 13px; font-weight: bold;")

            # Check what was included
            import zipfile as _zf
            with _zf.ZipFile(str(out_path), "r") as zcheck:
                zip_contents = [i.filename for i in zcheck.infolist()]
            has_bl = "feature_baseline.json" in zip_contents
            has_sr = "stress_results.json" in zip_contents
            extras = []
            if has_bl:
                extras.append("feature_baseline")
            if has_sr:
                extras.append("stress_results")
            extras_str = (" + " + ", ".join(extras)) if extras else ""

            self._export_log.append(
                f'<span style="color:{C["green"]}">'
                f'[{time.strftime("%H:%M:%S")}] SUCCESS: '
                f'{out_path.name} ({size_str}){extras_str}</span>')
            if not has_bl:
                self._export_log.append(
                    f'<span style="color:{C["yellow"]}">'
                    f'  WARNING: feature_baseline.json not included '
                    f'(no val feature caches)</span>')
            if not has_sr:
                self._export_log.append(
                    f'<span style="color:{C["yellow"]}">'
                    f'  WARNING: stress_results.json not included '
                    f'(run stress matrix first)</span>')

        except Exception as e:
            self._export_status_label.setText(f"FAILED: {str(e)[:80]}")
            self._export_status_label.setStyleSheet(
                f"color: {C['red']}; font-size: 13px; font-weight: bold;")
            self._export_log.append(
                f'<span style="color:{C["red"]}">'
                f'[{time.strftime("%H:%M:%S")}] ERROR: {str(e)[:100]}'
                f'</span>')
        finally:
            self._export_btn.setEnabled(True)
            self._export_btn.setText("Package for Live Dashboard")

    def _toggle_journal_auto_refresh(self):
        """Toggle auto-refresh for journal tab."""
        self._jrnl_auto_on = not self._jrnl_auto_on
        if self._jrnl_auto_on:
            self._jrnl_auto_refresh.setText("Auto: ON")
            self._jrnl_auto_refresh.setStyleSheet(
                f"background-color: {C['surface2']}; color: {C['green']}; "
                f"font-weight: bold; font-size: 13px;")
        else:
            self._jrnl_auto_refresh.setText("Auto: OFF")
            self._jrnl_auto_refresh.setStyleSheet(
                f"background-color: {C['surface2']}; color: {C['dim']}; "
                f"font-weight: bold; font-size: 13px;")

    def _refresh_journal(self):
        """Load journal entries from SQLite and display."""
        db_path = Path(self.cfg.memory_db_path)
        if not db_path.exists():
            return

        lesson_filter = self._jrnl_filter.currentText()

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA journal_mode=WAL")

            # Update lesson type counts
            for ltype in self._jrnl_lesson_labels:
                count = conn.execute(
                    "SELECT COUNT(*) FROM trade_journal WHERE lesson_type=?",
                    (ltype,)
                ).fetchone()[0]
                self._jrnl_lesson_labels[ltype].setText(f"{ltype}: {count}")

            # Fetch entries (with optional filter)
            if lesson_filter == "ALL":
                rows = conn.execute("""
                    SELECT j.id, j.trade_id, j.week, j.lesson_type, j.summary,
                           j.direction_correct, j.sl_quality, j.entry_reasoning,
                           j.exit_analysis, j.created_at,
                           t.side, t.pnl, t.hold_bars, t.entry_price, t.exit_price
                    FROM trade_journal j
                    LEFT JOIN trades t ON j.trade_id = t.id
                    ORDER BY j.id DESC LIMIT 500
                """).fetchall()
            else:
                rows = conn.execute("""
                    SELECT j.id, j.trade_id, j.week, j.lesson_type, j.summary,
                           j.direction_correct, j.sl_quality, j.entry_reasoning,
                           j.exit_analysis, j.created_at,
                           t.side, t.pnl, t.hold_bars, t.entry_price, t.exit_price
                    FROM trade_journal j
                    LEFT JOIN trades t ON j.trade_id = t.id
                    WHERE j.lesson_type = ?
                    ORDER BY j.id DESC LIMIT 500
                """, (lesson_filter,)).fetchall()

            total = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
            conn.close()

            # Cache entries for detail view
            self._jrnl_entries = rows

            # Populate table
            columns = ["#", "Trade", "Week", "Lesson", "Summary",
                        "Direction", "SL Quality", "Side", "P/L", "Bars"]
            self._jrnl_grid.setColumnCount(len(columns))
            self._jrnl_grid.setHorizontalHeaderLabels(columns)
            self._jrnl_grid.setRowCount(len(rows))

            lesson_colors = {
                "GOOD_TRADE": C['green'], "SCALP_WIN": C['green'],
                "WRONG_DIRECTION": C['red'], "EMERGENCY_STOP": C['red'], "CIRCUIT_BREAKER": C['red'],
                "CORRECT_DIR_CLOSED_EARLY": C['yellow'],
                "CORRECT_DIR_BAD_SL": C['yellow'],
                "BAD_TIMING": C['peach'], "WHIPSAW": C['peach'],
                "BREAKEVEN": C['subtext'], "HELD_TOO_LONG": C['mauve'],
            }

            for r, row in enumerate(rows):
                (jid, tid, week, lesson, summary,
                 dir_correct, sl_quality, entry_reason, exit_analysis, created,
                 side, pnl, hold_bars, entry_price, exit_price) = row

                vals = [
                    str(jid) if jid is not None else "-",
                    str(tid) if tid is not None else "-",
                    str(week) if week is not None else "-",
                    lesson or "-", summary or "-",
                    "\u2713" if dir_correct == 1 else "\u2717" if dir_correct == 0 else "-",
                    sl_quality or "-",
                    side or "-",
                    f"\u00a3{pnl:+.2f}" if pnl is not None else "-",
                    str(hold_bars) if hold_bars is not None else "-",
                ]

                for c, val in enumerate(vals):
                    item = QTableWidgetItem(val)

                    # Color coding
                    if c == 3:  # Lesson type
                        clr = lesson_colors.get(lesson, C['subtext'])
                        item.setForeground(pg.mkColor(clr))
                    elif c == 5:  # Direction correct
                        if dir_correct == 1:
                            item.setForeground(pg.mkColor(C['green']))
                        elif dir_correct == 0:
                            item.setForeground(pg.mkColor(C['red']))
                    elif c == 7:  # Side
                        if side == "LONG":
                            item.setForeground(pg.mkColor(C['green']))
                        elif side == "SHORT":
                            item.setForeground(pg.mkColor(C['red']))
                    elif c == 8 and pnl is not None:  # P/L
                        if pnl > 0:
                            item.setForeground(pg.mkColor(C['green']))
                        elif pnl < 0:
                            item.setForeground(pg.mkColor(C['red']))

                    self._jrnl_grid.setItem(r, c, item)

            self._jrnl_grid.resizeColumnsToContents()

            showing = len(rows)
            filtered = f" ({lesson_filter})" if lesson_filter != "ALL" else ""
            self._jrnl_count.setText(
                f"Entries: {total:,}{filtered}" +
                (f" (showing {showing})" if showing < total else ""))

            self._jrnl_last_refresh = time.time()

        except Exception:
            pass  # Silently handle DB errors during training

    def _on_journal_row_selected(self, row: int, col: int, prev_row: int, prev_col: int):
        """Show full reasoning detail when a journal row is selected."""
        if row < 0 or row >= len(self._jrnl_entries):
            return

        entry = self._jrnl_entries[row]
        (jid, tid, week, lesson, summary,
         dir_correct, sl_quality, entry_reasoning, exit_analysis, created,
         side, pnl, hold_bars, entry_price, exit_price) = entry

        lesson_colors = {
            "GOOD_TRADE": C['green'], "SCALP_WIN": C['green'],
            "WRONG_DIRECTION": C['red'], "EMERGENCY_STOP": C['red'], "CIRCUIT_BREAKER": C['red'],
            "CORRECT_DIR_CLOSED_EARLY": C['yellow'],
            "CORRECT_DIR_BAD_SL": C['yellow'],
            "BAD_TIMING": C['peach'], "WHIPSAW": C['peach'],
            "BREAKEVEN": C['subtext'], "HELD_TOO_LONG": C['mauve'],
        }
        lclr = lesson_colors.get(lesson, C['subtext'])
        pnl_clr = C['green'] if (pnl or 0) > 0 else C['red'] if (pnl or 0) < 0 else C['subtext']
        dir_icon = f'<span style="color:{C["green"]}">\u2713 Correct</span>' if dir_correct == 1 else \
                   f'<span style="color:{C["red"]}">\u2717 Wrong</span>' if dir_correct == 0 else "N/A"

        html = f"""
        <div style="font-family: 'Cascadia Code', Consolas, monospace;">
            <div style="margin-bottom: 8px;">
                <span style="color:{C['cyan']}; font-size: 16px; font-weight: bold;">
                    Trade #{tid if tid is not None else '?'}  &mdash;  Week {week if week is not None else '?'}
                </span>
                &nbsp;&nbsp;
                <span style="color:{lclr}; font-size: 15px; font-weight: bold;">
                    [{lesson}]
                </span>
            </div>

            <div style="margin-bottom: 6px; color:{C['subtext']}; font-size: 13px;">
                {side or '?'} &nbsp;|&nbsp;
                Entry: {f'{entry_price:.2f}' if entry_price else '?'} &rarr;
                Exit: {f'{exit_price:.2f}' if exit_price else '?'} &nbsp;|&nbsp;
                <span style="color:{pnl_clr}">&pound;{pnl:+.2f}</span> &nbsp;|&nbsp;
                {hold_bars if hold_bars is not None else '?'} bars &nbsp;|&nbsp;
                Direction: {dir_icon} &nbsp;|&nbsp;
                SL Quality: {sl_quality or '?'}
            </div>

            <div style="margin-top: 10px;">
                <div style="color:{C['cyan']}; font-size: 13px; font-weight: bold; margin-bottom: 4px;">
                    ENTRY REASONING
                </div>
                <div style="color:{C['text']}; font-size: 14px; padding-left: 8px; margin-bottom: 10px;">
                    {entry_reasoning or 'No reasoning recorded'}
                </div>
            </div>

            <div style="margin-top: 6px;">
                <div style="color:{C['cyan']}; font-size: 13px; font-weight: bold; margin-bottom: 4px;">
                    EXIT ANALYSIS
                </div>
                <div style="color:{C['text']}; font-size: 14px; padding-left: 8px; margin-bottom: 10px;">
                    {exit_analysis or 'No exit analysis recorded'}
                </div>
            </div>

            <div style="margin-top: 6px;">
                <div style="color:{C['cyan']}; font-size: 13px; font-weight: bold; margin-bottom: 4px;">
                    SUMMARY
                </div>
                <div style="color:{C['text']}; font-size: 14px; padding-left: 8px;">
                    {summary or '-'}
                </div>
            </div>
        </div>
        """
        self._jrnl_detail.setHtml(html)

    def _format_db_cell(self, value, column_name: str) -> str:
        """Format a SQLite cell value for display."""
        if value is None:
            return "-"
        col = column_name.lower()

        # Price columns — show more decimals
        if col in ("entry_price", "exit_price", "tp_price", "sl_price",
                    "price_at_prediction", "max_favorable"):
            return f"{value:.2f}" if isinstance(value, float) else str(value)

        # Percentage / rate columns
        if col in ("pnl_pct", "val_return", "val_max_dd", "val_win_rate",
                    "win_rate"):
            return f"{value:.4f}" if isinstance(value, float) else str(value)

        # Money columns
        if col in ("pnl", "total_pnl", "profit_locked_by_trail"):
            if isinstance(value, float):
                return f"\u00a3{value:+.2f}" if abs(value) > 0.005 else "\u00a30.00"
            return str(value)

        # Sharpe
        if col in ("val_sharpe",):
            return f"{value:.4f}" if isinstance(value, float) else str(value)

        # Lot size, conviction, confidence
        if col in ("lot_size", "conviction", "predicted_confidence",
                    "avg_hold_bars", "rsi_at_entry", "trend_dir_at_entry",
                    "vol_regime_at_entry"):
            return f"{value:.3f}" if isinstance(value, float) else str(value)

        # Generic float
        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)

    # ══ Update Loop ══════════════════════════════════════════════════════════

    def _update(self):
        m = self.metrics
        # Forward trainer reference to live fine-tune tab whenever it becomes available
        if hasattr(self, "_tab_live_ft") and m.get("_trainer_ref"):
            self._tab_live_ft.set_trainer(m["_trainer_ref"])
        week = m.get("current_week", 0)
        total = m.get("total_weeks", 0)
        ts = m.get("timestep", 0)
        stage = 1 if week < self.cfg.stage1_end_week else (
            2 if week < self.cfg.stage2_end_week else 3)
        sname = {1: "Easy", 2: "Normal", 3: "Full Realism"}.get(stage, "?")

        # ── Header ──────────────────────────────────────────────────────
        training_done = m.get("_training_done", False)
        training_error = m.get("_error")
        paused = m.get("_paused", False)

        # Detect training completion → return to IDLE with proper badge
        if training_done and self._dashboard_state != "IDLE":
            self._set_dashboard_state("IDLE")
            if training_error:
                self._hdr_info.setText(f"IDLE — Training stopped: {training_error[:80]}")
                self._health_badge.setText("CRASHED")
                self._health_badge.setStyleSheet(
                    f"color: {C['red']}; font-size: 15px; font-weight: bold; "
                    f"padding: 4px 16px; border: 2px solid {C['red']}; border-radius: 6px;")
            else:
                self._hdr_info.setText("IDLE — Training complete")
                self._health_badge.setText("COMPLETE")
                self._health_badge.setStyleSheet(
                    f"color: {C['green']}; font-size: 15px; font-weight: bold; "
                    f"padding: 4px 16px; border: 2px solid {C['green']}; border-radius: 6px;")

        # Convergence state — always read (used by other sections below)
        state = m.get("convergence_state", "WARMING_UP")

        # Only update header info and badge when actively running
        if self._dashboard_state in ("RUNNING", "PAUSED"):
            self._hdr_info.setText(
                f"Week {week}/{total}  |  Step {ts:,}  |  Stage {stage}/3 ({sname})")
            badge_map = {
                "WARMING_UP": ("EARLY", C['cyan']),
                "IMPROVING": ("LEARNING", C['green']),
                "CONVERGED": ("CONVERGED", C['green']),
                "STABLE": ("STABLE", C['green']),
                "PLATEAU": ("PLATEAU", C['yellow']),
                "OVERFITTING": ("OVERFITTING", C['red']),
                "COLLAPSED": ("COLLAPSED", C['red']),
            }
            if paused:
                btxt, bcol = "PAUSED", C['peach']
            else:
                btxt, bcol = badge_map.get(state, ("?", C['subtext']))
            self._health_badge.setText(btxt)
            self._health_badge.setStyleSheet(
                f"color: {bcol}; font-size: 15px; font-weight: bold; "
                f"padding: 4px 16px; border: 2px solid {bcol}; border-radius: 6px;")

        # Pause button text — only show "Resume" when actively paused, not in IDLE
        if self._dashboard_state == "IDLE":
            self._pause_btn.setText("Pause")
        else:
            self._pause_btn.setText("Resume" if paused else "Pause")

        # ── Progress ────────────────────────────────────────────────────
        self._v["pg_week"].set(f"{week}/{total}")
        self._v["pg_stage"].set(f"{stage}/3 ({sname})")
        # Progress: use total timesteps for smooth progress within weeks
        total_target_steps = total * self.cfg.steps_per_week * self.cfg.n_envs if total > 0 else 1
        pct = min(int(ts / total_target_steps * 100), 100) if total_target_steps > 0 else 0
        self._prog_bar.setValue(pct)
        self._prog_bar.setFormat(f"Week {week}/{total} — {pct}%")
        self._v["pg_steps"].set(f"{ts:,}")
        wstep = ts % self.cfg.steps_per_week if ts > 0 else 0
        self._v["pg_wsteps"].set(f"{wstep:,}/{self.cfg.steps_per_week:,}")
        speed = m.get("steps_per_sec", 0)
        self._v["pg_speed"].set(
            f"{speed:.0f} steps/sec" if speed > 0 else "-")

        elapsed = time.time() - m.get("_start_time", time.time())
        hr, rm = divmod(int(elapsed), 3600)
        mn, s = divmod(rm, 60)
        self._v["pg_time"].set(f"{hr}h {mn:02d}m")

        if week > 0 and total > 0:
            eta_s = (elapsed / max(week, 1)) * (total - week)
            eh, erm = divmod(int(eta_s), 3600)
            em, _ = divmod(erm, 60)
            self._v["pg_eta"].set(f"~{eh}h {em:02d}m")
        else:
            self._v["pg_eta"].set("-")

        resumed = m.get("_resumed", False)
        self._v["pg_session"].set(
            f"RESUMED W{m.get('_resumed_week', '?')}" if resumed else "FRESH start",
            "cyan" if resumed else "green")

        # ── Account (Overview Tab) ──────────────────────────────────────
        bal = m.get("balance", 0.0)
        pk = m.get("peak_balance", 0.0)
        ini = self.cfg.initial_balance
        dd = m.get("drawdown", 0.0)
        ret = ((bal - ini) / ini * 100) if ini > 0 else 0

        self._v["acct_start"].set(f"\u00a3{ini:.2f}")
        self._v["acct_cur"].set(
            f"\u00a3{bal:.2f}", "green" if bal >= ini else "red")
        self._v["acct_peak"].set(f"\u00a3{pk:.2f}", "cyan")
        self._v["acct_dd"].set(
            f"{dd:.1%}", _tc(dd, green_below=0.03, yellow_below=0.07))
        self._v["acct_ret"].set(
            f"{ret:+.1f}%", "green" if ret >= 0 else "red")
        bank = m.get("bankruptcies", 0)
        self._v["acct_bank"].set(
            str(bank), "green" if bank == 0 else ("yellow" if bank <= 2 else "red"))

        # Quick Stats (Overview Tab)
        wr = m.get("win_rate", 0)
        self._v["qs_wr"].set(
            f"{wr:.1%}", "green" if wr >= 0.52 else ("yellow" if wr >= 0.48 else "red"))
        self._v["qs_trades"].set(str(m.get("total_trades", 0)))
        pf = m.get("profit_factor", 0)
        self._v["qs_pf"].set(
            f"{pf:.2f}" if pf > 0 else "-",
            "green" if pf >= 1.2 else ("yellow" if pf >= 1.0 else "red") if pf > 0 else "subtext")
        sh = m.get("sharpe", 0)
        self._v["qs_sh"].set(f"{sh:.3f}", _tc(sh, green_above=0.8, yellow_above=0.3))
        wpnl = m.get("week_pnl", 0)
        self._v["qs_wpnl"].set(
            f"\u00a3{wpnl:+.2f}", "green" if wpnl >= 0 else "red")
        self._v["qs_speed"].set(
            f"{speed:.0f} sps" if speed > 0 else "-")

        # ── Learning Metrics (Metrics Tab) ──────────────────────────────
        self._v["lm_wr"].set(
            f"{wr:.1%}", _tc(wr, green_above=0.52, yellow_above=0.48))
        self._v["lm_pf"].set(
            f"{pf:.2f}" if pf > 0 else "-",
            _tc(pf, green_above=1.2, yellow_above=1.0) if pf > 0 else "subtext")
        avg_pnl = m.get("avg_trade_pnl", 0)
        self._v["lm_avg"].set(
            f"\u00a3{avg_pnl:+.2f}" if avg_pnl != 0 else "-",
            "green" if avg_pnl > 0 else ("red" if avg_pnl < 0 else "subtext"))
        self._v["lm_sh"].set(
            f"{sh:.3f}", _tc(sh, green_above=0.8, yellow_above=0.3))
        self._v["lm_mem"].set(str(m.get("total_trades", 0)))
        self._v["lm_pat"].set(str(m.get("total_patterns", 0)))
        self._v["lm_lot"].set(f"{m.get('avg_lot_size', 0):.2f}")

        # ── This Week (Metrics Tab) ────────────────────────────────────
        wt = m.get("episode_trades", 0)
        self._v["tw_trades"].set(str(wt))
        ww = m.get("week_wins", 0)
        wpct = (ww / wt * 100) if wt > 0 else 0
        self._v["tw_wins"].set(f"{ww} ({wpct:.0f}%)")
        self._v["tw_pnl"].set(
            f"\u00a3{wpnl:+.2f}", "green" if wpnl >= 0 else "red")
        self._v["tw_best"].set(
            f"+\u00a3{m.get('week_best_trade', 0):.2f}", "green")
        self._v["tw_worst"].set(
            f"-\u00a3{abs(m.get('week_worst_trade', 0)):.2f}", "red")
        self._v["tw_hold"].set(f"{m.get('avg_hold_bars', 0):.0f} bars")
        self._v["tw_comm"].set(f"\u00a3{m.get('week_commission', 0):.2f}")

        # ── Predictions & TP (Metrics Tab) ─────────────────────────────
        ta = m.get("trend_accuracy", 0)
        self._v["pr_acc"].set(
            f"{ta:.1%}", _tc(ta, green_above=0.55, yellow_above=0.50))
        self._v["pr_up"].set(f"{m.get('trend_acc_up', 0):.1%}")
        self._v["pr_dn"].set(f"{m.get('trend_acc_down', 0):.1%}")
        self._v["pr_ver"].set(str(m.get("verified_predictions", 0)))
        self._v["pr_pend"].set(str(m.get("pending_predictions", 0)))
        tph = m.get("tp_hit_rate", 0)
        self._v["tp_hit"].set(
            f"{tph:.1%}", _tc(tph, green_above=0.40, yellow_above=0.25))
        tpr = m.get("tp_reachable_rate", 0)
        self._v["tp_reach"].set(
            f"{tpr:.1%}", _tc(tpr, green_above=0.60, yellow_above=0.40))
        slh = m.get("sl_hit_rate", 0)
        self._v["tp_sl"].set(
            f"{slh:.1%}", _tc(slh, green_below=0.30, yellow_below=0.45))

        # ── Reward Breakdown (Metrics Tab) ─────────────────────────────
        for key, mk in [("rw_r1", "r1_position_pnl"), ("rw_r2", "r2_trade_quality"),
                         ("rw_r3", "r3_drawdown"), ("rw_r4", "r4_sharpe"),
                         ("rw_r5", "r5_risk_bonus")]:
            self._v[key].set(f"{m.get(mk, 0):+.4f}")
        self._v["rw_raw"].set(f"{m.get('raw_reward', 0):+.4f}")
        self._v["rw_norm"].set(f"{m.get('reward', 0):+.4f}")
        self._v["rw_mu"].set(f"{m.get('reward_running_mean', 0):.4f}")
        self._v["rw_sig"].set(
            f"{m.get('reward_running_std', 0):.4f}",
            _tc(m.get('reward_running_std'), green_min=0.1, green_max=5.0)
            if m.get('reward_running_std') is not None else "subtext")
        clip = m.get("reward_clip_pct", 0)
        self._v["rw_clip"].set(
            f"{clip:.1%}", _tc(clip, green_below=0.05, yellow_below=0.15))

        # ── SAC Internals (Internals Tab) ──────────────────────────────
        alpha = m.get("entropy_alpha")
        self._v["sac_a"].set(
            f"{alpha:.4f}" if alpha is not None else "-",
            _tc(alpha, green_min=0.01, green_max=1.0) if alpha is not None else "subtext")
        ent_pct = m.get("policy_entropy_pct")
        self._v["sac_ent"].set(
            f"{ent_pct:.0f}% of init" if ent_pct is not None else "-",
            _tc(ent_pct, green_above=40, yellow_above=20) if ent_pct is not None else "subtext")
        qm = m.get("q_value_mean")
        self._v["sac_qm"].set(
            f"{qm:.2f}" if qm is not None else "-",
            _tc(qm, green_below=50, yellow_below=100) if qm is not None else "subtext")
        qx = m.get("q_value_max")
        self._v["sac_qx"].set(f"{qx:.2f}" if qx is not None else "-")
        ag = m.get("actor_grad_norm")
        self._v["sac_ag"].set(
            f"{ag:.3f} / 1.0" if ag is not None else "-",
            _tc(ag, green_below=0.5, yellow_below=0.8) if ag is not None else "subtext")
        cg = m.get("critic_grad_norm")
        self._v["sac_cg"].set(
            f"{cg:.3f} / 1.0" if cg is not None else "-",
            _tc(cg, green_below=0.5, yellow_below=0.8) if cg is not None else "subtext")
        al = m.get("actor_loss")
        self._v["sac_al"].set(f"{al:.4f}" if al is not None else "-")
        cl = m.get("critic_loss")
        self._v["sac_cl"].set(f"{cl:.4f}" if cl is not None else "-")
        lr = m.get("learning_rate")
        self._v["sac_lr"].set(f"{lr:.2e}" if lr is not None else "-")
        buf = m.get("buffer_pct")
        self._v["sac_buf"].set(f"{buf:.0f}%" if buf is not None else "-")

        # ── Convergence (Internals Tab) ────────────────────────────────
        sc = {"WARMING_UP": "cyan", "IMPROVING": "green", "CONVERGED": "green",
              "STABLE": "green", "OVERFITTING": "red", "COLLAPSED": "red",
              "PLATEAU": "yellow"}
        self._v["cv_st"].set(f"\u25cf {state}", sc.get(state, "subtext"))
        vsh = m.get("best_val_sharpe", 0)
        self._v["cv_sh"].set(f"{vsh:.3f}" if vsh > -999 else "-")
        self._v["cv_best"].set(f"W{m.get('best_checkpoint_week', '?')}")
        wsb = m.get("weeks_since_best", 0)
        self._v["cv_since"].set(
            f"{wsb} weeks", _tc(wsb, green_below=20, yellow_below=50))
        astd = m.get("action_std")
        self._v["cv_astd"].set(
            f"{astd:.4f}" if astd is not None else "-",
            _tc(astd, green_above=0.10, yellow_above=0.05)
            if astd is not None else "subtext")
        self._v["cv_etrend"].set(m.get("entropy_trend", "stable"))

        # ── Curriculum & Regime (Internals Tab) ────────────────────────
        self._v["cur_stg"].set(f"{stage}/3 ({sname})")
        diff = m.get("week_difficulty", 0)
        self._v["cur_diff"].set(f"{diff:.2f}" if diff > 0 else "-")
        self._v["cur_reg"].set(m.get("regime", "-"))
        for key, mk in [("cur_bup", "buffer_trending_up"),
                         ("cur_bdn", "buffer_trending_down"),
                         ("cur_brng", "buffer_ranging"),
                         ("cur_bvol", "buffer_volatile")]:
            v = m.get(mk)
            self._v[key].set(
                f"{v:.0%}" if v is not None else "-",
                _tc(v, green_above=0.15, yellow_above=0.10)
                if v is not None else "subtext")
        pat_conf = m.get("pattern_confidence_avg")
        self._v["cur_pat"].set(
            f"{pat_conf:.0%}" if pat_conf is not None else "-",
            _tc(pat_conf, green_above=0.50, yellow_above=0.30)
            if pat_conf is not None else "subtext")

        # ── Safety & Anti-Hack (Internals Tab) ────────────────────────
        dt = m.get("daily_trades", 0)
        self._v["sf_daily"].set(
            f"{dt}/10", _tc(dt, green_below=8, yellow_below=9))
        conv_thresh = 0.6 if dt >= 10 else 0.3
        self._v["sf_conv"].set(
            f"{conv_thresh}", "yellow" if conv_thresh > 0.3 else "green")
        hb = m.get("hold_blocks", 0)
        self._v["sf_hold"].set(str(hb), _tc(hb, green_below=2, yellow_below=5))
        dead = m.get("dead_features", 0)
        exp = m.get("exploding_features", 0)
        nan_f = m.get("nan_features", 0)
        obs_ok = (dead == 0 and exp == 0 and nan_f == 0)
        self._v["sf_obs"].set(
            "\u2713 All OK" if obs_ok else f"\u2717 {dead+exp+nan_f} issues",
            "green" if obs_ok else "red")
        self._v["sf_dead"].set(str(dead), "green" if dead == 0 else "red")
        self._v["sf_nan"].set(str(nan_f), "green" if nan_f == 0 else "red")
        gc = m.get("grad_clip_pct", 0)
        self._v["sf_gclip"].set(
            f"{gc:.1%}", _tc(gc, green_below=0.05, yellow_below=0.30))
        self._v["sf_lr"].set(f"{lr:.2e}" if lr is not None else "-")
        self._v["sf_noise"].set("\u2713 Active", "green")
        lstm_met = m.get("lstm_criteria_met", 0)
        lstm_labels = {0: "Not needed", 1: "Investigate", 2: "Recommended"}
        self._v["sf_lstm"].set(
            lstm_labels.get(lstm_met, f"{lstm_met}/3"),
            "green" if lstm_met == 0 else ("yellow" if lstm_met == 1 else "red"))

        # ── Decision Log (only update if content changed to preserve selection) ─
        decisions = m.get("_decisions", [])
        if decisions:
            html_parts = []
            for d in decisions[-8:]:
                if "WIN" in d:
                    html_parts.append(f'<span style="color:{C["green"]}; font-size:13px">{d}</span>')
                elif "LOSS" in d:
                    html_parts.append(f'<span style="color:{C["red"]}; font-size:13px">{d}</span>')
                elif "B/E" in d:
                    html_parts.append(f'<span style="color:{C["yellow"]}; font-size:13px">{d}</span>')
                elif "OPEN LONG" in d:
                    html_parts.append(f'<span style="color:{C["green"]}; font-size:13px">{d}</span>')
                elif "OPEN SHORT" in d:
                    html_parts.append(f'<span style="color:{C["red"]}; font-size:13px">{d}</span>')
                elif "TRAIL" in d:
                    html_parts.append(f'<span style="color:{C["cyan"]}; font-size:13px">{d}</span>')
                else:
                    html_parts.append(f'<span style="color:{C["subtext"]}; font-size:13px">{d}</span>')
            new_html = "<br>".join(html_parts)
            if new_html != self._decisions_html:
                self._decisions_html = new_html
                self._decisions_edit.setHtml(new_html)

        # ── Alerts (only update if content changed to preserve selection) ──
        alerts = m.get("_alerts", [])
        if alerts:
            html_parts = []
            for a in alerts[-8:]:
                if "CRITICAL" in a:
                    html_parts.append(f'<span style="color:{C["red"]}; font-size:13px">{a}</span>')
                elif "WARNING" in a:
                    html_parts.append(f'<span style="color:{C["yellow"]}; font-size:13px">{a}</span>')
                elif "POSITIVE" in a or "INFO" in a:
                    html_parts.append(f'<span style="color:{C["green"]}; font-size:13px">{a}</span>')
                else:
                    html_parts.append(f'<span style="color:{C["subtext"]}; font-size:13px">{a}</span>')
            new_html = "<br>".join(html_parts)
            if new_html != self._alerts_html:
                self._alerts_html = new_html
                self._alerts_edit.setHtml(new_html)

        # ── Balance Chart ─────────────────────────────────────────────
        if bal > 0:
            self._bal_data.append(bal)
            if len(self._bal_data) > 3000:
                self._bal_data = self._bal_data[-3000:]

            data = self._bal_data
            n = len(data)
            if n >= 2:
                x = list(range(n))
                # Color based on P/L vs initial balance
                ini = self.cfg.initial_balance
                color = C['green'] if data[-1] >= ini else C['red']
                self._bcurve_green.setData(x, data)
                self._bcurve_green.setPen(pg.mkPen(color, width=2.5))
                self._bcurve_green.setVisible(True)
                self._bcurve_red.setVisible(False)
            elif n == 1:
                self._bcurve_green.setData([0], data)

        # ── DB Viewer auto-refresh ────────────────────────────────────
        if (self._db_auto_on
                and self._tabs.currentIndex() == 3
                and time.time() - self._db_last_refresh > 5):
            self._refresh_db_view()

        # ── Trade Journal auto-refresh ────────────────────────────────
        if (self._jrnl_auto_on
                and self._tabs.currentIndex() == 4
                and time.time() - self._jrnl_last_refresh > 5):
            self._refresh_journal()

        # ── Footer ──────────────────────────────────────────────────────
        try:
            import psutil
            ram = psutil.virtual_memory()
            self._foot_ram.setText(
                f"RAM: {ram.used / 1024**3:.1f}/{ram.total / 1024**3:.0f} GB ({ram.percent:.0f}%)")
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                # Use mem_get_info for reliable (free, total) on all platforms
                free, total_mem = torch.cuda.mem_get_info(0)
                used = (total_mem - free) / 1024**3
                total_gb = total_mem / 1024**3
                self._foot_gpu.setText(
                    f"GPU: {name} ({used:.1f}/{total_gb:.0f} GB)")
            else:
                self._foot_gpu.setText("GPU: CPU only")
        except Exception:
            self._foot_gpu.setText("GPU: N/A")

        self._foot_chk.setText(
            f"Checkpoint: W{m.get('last_checkpoint_week', '-')}")

    # ══ Actions ══════════════════════════════════════════════════════════════

    def _on_tab_changed(self, index: int):
        """Auto-load data when switching to DB Viewer, Journal, Export, or Live Fine-Tune tabs."""
        if index == 3:
            self._refresh_db_view()
        elif index == 4:
            self._refresh_journal()
        elif index == 5:
            self._refresh_export_tab()
        elif index == 6:
            pass  # Live Fine-Tune tab uses its own 1Hz timer — nothing extra needed on switch

    # ══ Training Launcher (v3.4) ═════════════════════════════════════════════

    def set_training_launcher(self, train_fn, max_weeks=None, seed=42):
        """Called by train.py to give dashboard the ability to start training."""
        self._train_fn = train_fn
        self._train_max_weeks = max_weeks
        self._train_seed = seed

    def auto_resume_on_show(self):
        """Queue an auto-resume when the dashboard is shown (for --resume CLI)."""
        self._auto_resume = True

    def showEvent(self, event):
        """Override to handle auto-resume after window is visible."""
        super().showEvent(event)
        if self._auto_resume:
            self._auto_resume = False
            # Use a short timer to let the UI render first
            QTimer.singleShot(100, lambda: self._launch_training(resume=True))

    def _on_start_clicked(self):
        """User clicked Start — begin fresh training."""
        reply = QMessageBox.question(
            self, "Start Fresh Training",
            "This will create a new model from scratch.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._launch_training(resume=False)

    def _on_resume_clicked(self):
        """User clicked Resume — show model picker dialog."""
        model_path = self._show_model_picker()
        if model_path is None:
            return  # User cancelled
        self._launch_training(resume=True, model_path=str(model_path))

    def _on_finetune_clicked(self):
        """User clicked Fine-Tune — pick model, load weights, reset counters."""
        model_path = self._show_model_picker()
        if model_path is None:
            return  # User cancelled
        reply = QMessageBox.question(
            self, "Fine-Tune Model",
            f"This will load weights from:\n{model_path.name}\n\n"
            "Week counter, balance, and convergence will reset to zero.\n"
            "The model keeps its learned patterns but trains from Week 0\n"
            "with the current config settings.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._launch_training(resume=True, finetune=True, model_path=str(model_path))

    def _show_model_picker(self) -> Optional[Path]:
        """Show dialog listing available model checkpoints. Returns path or None."""
        model_dir = self.cfg.model_dir
        if not model_dir.exists():
            QMessageBox.warning(self, "No Models", f"Model directory not found:\n{model_dir}")
            return None

        # Collect all .zip model files
        zip_files = sorted(model_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zip_files:
            QMessageBox.warning(self, "No Models", "No model checkpoints found.")
            return None

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model to Resume")
        dialog.setMinimumSize(500, 350)
        dialog.setStyleSheet(f"background-color: {C['bg']}; color: {C['text']};")

        layout = QVBoxLayout(dialog)

        label = QLabel("Select a checkpoint to resume training from:")
        label.setStyleSheet(f"color: {C['subtext']}; font-size: 14px; padding: 8px;")
        layout.addWidget(label)

        list_widget = QListWidget()
        list_widget.setStyleSheet(
            f"background-color: {C['surface']}; color: {C['text']}; "
            f"font-size: 13px; border: 1px solid {C['border']}; padding: 4px;"
        )

        for zf in zip_files:
            size_mb = zf.stat().st_size / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(zf.stat().st_mtime)
            mtime_str = mtime.strftime("%Y-%m-%d %H:%M")
            item = QListWidgetItem(f"{zf.name}    ({size_mb:.1f} MB, {mtime_str})")
            item.setData(Qt.ItemDataRole.UserRole, str(zf))
            list_widget.addItem(item)

        # Pre-select the first item (most recent)
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
            return None

        selected = list_widget.currentItem()
        if selected is None:
            return None

        return Path(selected.data(Qt.ItemDataRole.UserRole))

    def _launch_training(self, resume: bool, finetune: bool = False,
                         model_path: Optional[str] = None):
        """Start the training thread in the background."""
        if self._train_fn is None:
            QMessageBox.warning(self, "Error", "Training launcher not configured.")
            return

        if self._train_thread and self._train_thread.is_alive():
            QMessageBox.warning(self, "Already Running", "Training is already in progress.")
            return

        # Clear ALL stale state from previous runs — start clean
        self.metrics.pop("_training_done", None)
        self.metrics.pop("_error", None)
        self.metrics.pop("_quit_requested", None)
        self.metrics.pop("_paused", None)
        self.metrics.pop("_finetune", None)
        self.metrics.pop("_resume_model_path", None)
        self.metrics.pop("_resumed", None)
        self.metrics.pop("_resumed_week", None)
        self.metrics["_start_time"] = time.time()

        # Set flags AFTER cleanup
        if finetune:
            self.metrics["_finetune"] = True
        if model_path:
            self.metrics["_resume_model_path"] = model_path

        self._train_thread = threading.Thread(
            target=self._train_fn,
            args=(self.cfg, self.metrics, self._train_max_weeks, resume, self._train_seed),
            daemon=False,
        )
        self._train_thread.start()
        self.metrics["_train_thread"] = self._train_thread  # Expose for train.py cleanup
        self._set_dashboard_state("RUNNING")

    def _set_dashboard_state(self, state: str):
        """Update dashboard state and enable/disable buttons accordingly."""
        self._dashboard_state = state
        is_idle = state == "IDLE"
        is_running = state == "RUNNING"

        self._start_btn.setEnabled(is_idle)
        self._resume_btn.setEnabled(is_idle)
        self._pause_btn.setEnabled(is_running or state == "PAUSED")

        if is_idle:
            self._hdr_info.setText("IDLE — Ready")
            self._health_badge.setText("IDLE")
            self._health_badge.setStyleSheet(
                f"color: {C['cyan']}; font-size: 15px; font-weight: bold; "
                f"padding: 4px 16px; border: 2px solid {C['cyan']}; border-radius: 6px;")

    def _toggle_pause(self):
        paused = not self.metrics.get("_paused", False)
        self.metrics["_paused"] = paused
        if paused:
            self._set_dashboard_state("PAUSED")
        else:
            self._set_dashboard_state("RUNNING")

    def _request_quit(self):
        self.metrics["_quit_requested"] = True

    def closeEvent(self, event: QCloseEvent):
        self.metrics["_quit_requested"] = True
        self._timer.stop()
        event.accept()
