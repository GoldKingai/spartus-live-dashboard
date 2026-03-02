"""Tab 6 -- Analytics & Diagnostics for the Spartus Live Trading Dashboard.

The most data-rich tab: action distributions, training vs live comparison,
session/day-of-week breakdowns, feature drift, correlation drift, and
auto-generated weekly reports with CSV/JSON export.

Layout::

    +------------------------------------+------------------------------+
    |  ACTION DISTRIBUTIONS (last 500)   |  TRAINING vs LIVE            |
    |  Direction:  histogram  mu=0.12    |  Metric    Expected  Actual  |
    |  Conviction: histogram  mu=0.68    |  PF        2.24      1.85   |
    |  Exit:       histogram  mu=0.22    |  Win Rate  52%       48%    |
    |  SL Adj:     histogram  mu=0.45    |  MaxDD     20.8%     3.2%   |
    |  Flat rate: 62% of bars            |  T/Day     2.4       1.8    |
    |  Avg trades/day: 1.8               |  TIM%      8.9%      6.1%  |
    +------------------------------------+------------------------------+
    |  SESSION BREAKDOWN                 |  FEATURE DRIFT               |
    |  Session   Trades Win%  PF  AvgPL  |  Feature      Live   Train  |
    |  London      12   58%  1.9  +2.1   |  returns_20   0.02   0.00  |
    |  NY Overlap   8   50%  1.5  +0.8   |  atr_14_norm  1.23   0.95  |
    |  Asia          3   33%  0.8  -1.2   |  * 52/54 within 2sig       |
    |  Off-hours     1    0%  0.0  -3.1   |  ! 2 features drifted      |
    |                                    |                             |
    |  DAY-OF-WEEK BREAKDOWN             |  CORRELATION DRIFT           |
    |  Mon: +$4.20  Tue: +$1.80         |  Score: 0.08  * OK           |
    |  Wed: -$2.10  Thu: +$6.50         |                              |
    |  Fri: +$1.20                       |                              |
    +------------------------------------+------------------------------+
    |  WEEKLY REPORTS (auto-generated)                                  |
    |  Week 1: PF 1.85, 12 trades, +$14.30, MaxDD 3.2%               |
    |  Week 2: PF 2.10, 15 trades, +$22.10, MaxDD 2.8%               |
    |  [Export CSV]  [Export Full JSON]                                  |
    +------------------------------------------------------------------+

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels -- NEVER dark gray on dark backgrounds.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QGroupBox, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg
import numpy as np

from dashboard.theme import C
from dashboard.widgets import HistogramWidget, SectionHeader, CopyableTableWidget
from dashboard import currency

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: styled label (matches other tabs' convention)
# ---------------------------------------------------------------------------

def _make_label(text: str, color: str = C["subtext"], bold: bool = False,
                font_size: int = 13) -> QLabel:
    """Create a QLabel with the given text, color, and optional bold."""
    lbl = QLabel(text)
    style = (
        f"color: {color}; font-size: {font_size}px; "
        f"background: transparent; border: none;"
    )
    if bold:
        style += " font-weight: bold;"
    lbl.setStyleSheet(style)
    return lbl


def _styled_table(columns: list[str], stretch_last: bool = True) -> QTableWidget:
    """Create a QTableWidget pre-styled to match the dark theme."""
    table = CopyableTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

    header = table.horizontalHeader()
    for i in range(len(columns)):
        header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
    if stretch_last and len(columns) > 0:
        header.setSectionResizeMode(
            len(columns) - 1, QHeaderView.ResizeMode.Stretch
        )

    table.setStyleSheet(
        f"alternate-background-color: {C['surface2']}; "
        f"background-color: {C['surface']}; "
        f"color: {C['text']}; "
        f"gridline-color: {C['border']};"
    )
    return table


# ---------------------------------------------------------------------------
# Session definitions (UTC hour ranges)
# ---------------------------------------------------------------------------

SESSION_DEFS = [
    ("London",     7, 12),
    ("NY Overlap", 12, 16),
    ("NY PM",      16, 20),
    ("Asia",       0,  7),
    ("Off-hours",  20, 24),
]


# ---------------------------------------------------------------------------
# AnalyticsTab
# ---------------------------------------------------------------------------

class AnalyticsTab(QWidget):
    """Tab 6: Analytics & Diagnostics.

    Displays action distributions, training vs live comparison, session
    and day-of-week breakdowns, feature and correlation drift, plus
    auto-generated weekly reports with export capabilities.

    Call ``update_data(data)`` each tick with the latest state dict.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Cached weekly reports for export
        self._weekly_reports: list[dict] = []

        # ----- Root layout -----
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ----- Top row: Action Distributions (left) + Training vs Live (right) -----
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {C['border']}; width: 2px; }}"
        )
        top_splitter.addWidget(self._build_action_distributions_box())
        top_splitter.addWidget(self._build_training_vs_live_box())
        top_splitter.setStretchFactor(0, 5)
        top_splitter.setStretchFactor(1, 5)
        root.addWidget(top_splitter, stretch=4)

        # ----- Middle row: Session/DoW (left) + Feature/Corr Drift (right) -----
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)
        mid_splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {C['border']}; width: 2px; }}"
        )
        mid_splitter.addWidget(self._build_session_breakdown_box())
        mid_splitter.addWidget(self._build_drift_box())
        mid_splitter.setStretchFactor(0, 5)
        mid_splitter.setStretchFactor(1, 5)
        root.addWidget(mid_splitter, stretch=4)

        # ----- Bottom row: Weekly Reports (full width) -----
        root.addWidget(self._build_weekly_reports_box(), stretch=3)

    # ==================================================================
    # Section builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. ACTION DISTRIBUTIONS
    # ------------------------------------------------------------------

    def _build_action_distributions_box(self) -> QGroupBox:
        """Panel with 4 mini histograms and summary stats."""
        box = QGroupBox("ACTION DISTRIBUTIONS (last 500)")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(4)

        # 4 mini histogram widgets, each ~80px tall
        self._hist_direction = HistogramWidget("Direction [-1, 1]")
        self._hist_direction.setFixedHeight(80)
        self._hist_direction.setMinimumHeight(60)
        layout.addWidget(self._hist_direction)

        self._hist_conviction = HistogramWidget("Conviction [0, 1]")
        self._hist_conviction.setFixedHeight(80)
        self._hist_conviction.setMinimumHeight(60)
        layout.addWidget(self._hist_conviction)

        self._hist_exit = HistogramWidget("Exit Urgency [0, 1]")
        self._hist_exit.setFixedHeight(80)
        self._hist_exit.setMinimumHeight(60)
        layout.addWidget(self._hist_exit)

        self._hist_sl_adj = HistogramWidget("SL Adjustment [0, 1]")
        self._hist_sl_adj.setFixedHeight(80)
        self._hist_sl_adj.setMinimumHeight(60)
        layout.addWidget(self._hist_sl_adj)

        # Summary stats row
        summary_row = QHBoxLayout()
        summary_row.setSpacing(16)

        self._lbl_flat_rate = _make_label("Flat rate: --", C["text"], font_size=12)
        summary_row.addWidget(self._lbl_flat_rate)

        self._lbl_avg_trades_day = _make_label("Avg trades/day: --", C["text"], font_size=12)
        summary_row.addWidget(self._lbl_avg_trades_day)

        summary_row.addStretch()

        # Red flag indicator
        self._lbl_action_flags = _make_label("", C["red"], bold=True, font_size=12)
        summary_row.addWidget(self._lbl_action_flags)

        layout.addLayout(summary_row)

        return box

    # ------------------------------------------------------------------
    # 2. TRAINING vs LIVE
    # ------------------------------------------------------------------

    def _build_training_vs_live_box(self) -> QGroupBox:
        """Table comparing training validation/test metrics with live."""
        box = QGroupBox("TRAINING vs LIVE")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(4)

        columns = ["Metric", "Train (Val)", "Train (Test)", "Live (30d)", "Status"]
        self._tbl_training_vs_live = _styled_table(columns, stretch_last=True)
        self._tbl_training_vs_live.setMinimumHeight(200)

        # Pre-populate rows for standard metrics
        self._tvl_metrics = [
            "PF", "Win Rate", "MaxDD", "Trades/Day",
            "TIM%", "Avg Hold", "Sharpe",
        ]
        self._tbl_training_vs_live.setRowCount(len(self._tvl_metrics))
        for row, metric_name in enumerate(self._tvl_metrics):
            item = QTableWidgetItem(metric_name)
            item.setForeground(QColor(C["text"]))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_training_vs_live.setItem(row, 0, item)
            # Fill remaining columns with dashes
            for col in range(1, 5):
                dash = QTableWidgetItem("--")
                dash.setForeground(QColor(C["label"]))
                dash.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                self._tbl_training_vs_live.setItem(row, col, dash)

        layout.addWidget(self._tbl_training_vs_live)
        return box

    # ------------------------------------------------------------------
    # 3. SESSION BREAKDOWN + DAY-OF-WEEK
    # ------------------------------------------------------------------

    def _build_session_breakdown_box(self) -> QGroupBox:
        """Session performance table and day-of-week grid."""
        box = QGroupBox("SESSION BREAKDOWN")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(8)

        # Session table
        columns = ["Session", "Trades", "Win%", "PF", "Avg P/L", "Net P/L"]
        self._tbl_sessions = _styled_table(columns)
        self._tbl_sessions.setMinimumHeight(120)
        self._tbl_sessions.setRowCount(len(SESSION_DEFS))
        for row, (name, _, _) in enumerate(SESSION_DEFS):
            item = QTableWidgetItem(name)
            item.setForeground(QColor(C["text"]))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_sessions.setItem(row, 0, item)
            for col in range(1, 6):
                dash = QTableWidgetItem("--")
                dash.setForeground(QColor(C["label"]))
                dash.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                self._tbl_sessions.setItem(row, col, dash)

        layout.addWidget(self._tbl_sessions, stretch=3)

        # Day-of-week breakdown header
        layout.addWidget(SectionHeader("DAY-OF-WEEK BREAKDOWN"))

        # Day-of-week grid: Mon-Fri
        dow_widget = QWidget()
        dow_layout = QGridLayout(dow_widget)
        dow_layout.setContentsMargins(4, 4, 4, 4)
        dow_layout.setSpacing(8)

        self._day_labels: dict[str, QLabel] = {}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for col, day in enumerate(days):
            name_lbl = _make_label(f"{day}:", C["subtext"], bold=True, font_size=12)
            dow_layout.addWidget(name_lbl, 0, col, Qt.AlignmentFlag.AlignCenter)
            val_lbl = _make_label("--", C["text"], bold=True, font_size=13)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._day_labels[day] = val_lbl
            dow_layout.addWidget(val_lbl, 1, col, Qt.AlignmentFlag.AlignCenter)

        dow_widget.setStyleSheet(
            f"background-color: {C['surface']}; "
            f"border: 1px solid {C['border']}; "
            f"border-radius: 4px;"
        )
        layout.addWidget(dow_widget, stretch=1)

        return box

    # ------------------------------------------------------------------
    # 4. FEATURE DRIFT + CORRELATION DRIFT
    # ------------------------------------------------------------------

    def _build_drift_box(self) -> QWidget:
        """Combined panel for feature drift and correlation drift."""
        container = QWidget()
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(6)

        # Feature Drift group
        feat_box = QGroupBox("FEATURE DRIFT")
        feat_layout = QVBoxLayout(feat_box)
        feat_layout.setContentsMargins(8, 20, 8, 8)
        feat_layout.setSpacing(4)

        # Summary line
        self._lbl_drift_summary = _make_label(
            "--/54 within 2sig of training baseline", C["text"], font_size=12
        )
        feat_layout.addWidget(self._lbl_drift_summary)

        # Drifted features table
        drift_cols = ["Feature", "Live Mean", "Train Mean", "Sigma Dist", "Status"]
        self._tbl_feature_drift = _styled_table(drift_cols)
        self._tbl_feature_drift.setMinimumHeight(100)
        feat_layout.addWidget(self._tbl_feature_drift)

        # Note about live features
        self._lbl_drift_note = _make_label(
            "(13 live features: no baseline)", C["label"], font_size=11
        )
        feat_layout.addWidget(self._lbl_drift_note)

        vlayout.addWidget(feat_box, stretch=3)

        # Correlation Drift group
        corr_box = QGroupBox("CORRELATION DRIFT")
        corr_layout = QVBoxLayout(corr_box)
        corr_layout.setContentsMargins(8, 20, 8, 8)
        corr_layout.setSpacing(6)

        # Score + status on one line
        score_row = QHBoxLayout()
        score_row.setSpacing(8)

        score_row.addWidget(_make_label("Score:", C["subtext"], bold=True, font_size=13))
        self._lbl_corr_score = _make_label("--", C["text"], bold=True, font_size=14)
        score_row.addWidget(self._lbl_corr_score)

        # Status dot + text
        self._lbl_corr_dot = QLabel("\u25cf")
        self._lbl_corr_dot.setStyleSheet(
            f"color: {C['label']}; font-size: 14px; "
            f"background: transparent; border: none;"
        )
        self._lbl_corr_dot.setFixedWidth(18)
        self._lbl_corr_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_row.addWidget(self._lbl_corr_dot)

        self._lbl_corr_status = _make_label("--", C["text"], bold=True, font_size=13)
        score_row.addWidget(self._lbl_corr_status)

        score_row.addStretch()
        corr_layout.addLayout(score_row)

        # Consecutive counter line
        self._lbl_corr_consecutive = _make_label(
            "consecutive: --", C["label"], font_size=11
        )
        corr_layout.addWidget(self._lbl_corr_consecutive)

        corr_layout.addStretch()

        vlayout.addWidget(corr_box, stretch=1)

        return container

    # ------------------------------------------------------------------
    # 5. WEEKLY REPORTS
    # ------------------------------------------------------------------

    def _build_weekly_reports_box(self) -> QGroupBox:
        """Weekly summaries table with export buttons."""
        box = QGroupBox("WEEKLY REPORTS (auto-generated)")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(6)

        columns = ["Week", "PF", "Trades", "P/L", "MaxDD", "Win Rate", "Sharpe"]
        self._tbl_weekly = _styled_table(columns)
        self._tbl_weekly.setMinimumHeight(80)
        layout.addWidget(self._tbl_weekly)

        # Export buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setStyleSheet(
            f"background-color: {C['surface2']}; "
            f"color: {C['text']}; "
            f"border: 1px solid {C['cyan']}; "
            f"border-radius: 4px; "
            f"padding: 6px 16px; "
            f"font-weight: bold;"
        )
        self._btn_export_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(self._btn_export_csv)

        self._btn_export_json = QPushButton("Export Full JSON")
        self._btn_export_json.setStyleSheet(
            f"background-color: {C['surface2']}; "
            f"color: {C['text']}; "
            f"border: 1px solid {C['cyan']}; "
            f"border-radius: 4px; "
            f"padding: 6px 16px; "
            f"font-weight: bold;"
        )
        self._btn_export_json.clicked.connect(self._export_json)
        btn_row.addWidget(self._btn_export_json)

        layout.addLayout(btn_row)

        return box

    # ==================================================================
    # update_data -- called every tick with live state
    # ==================================================================

    def update_data(self, data: dict) -> None:
        """Refresh all panels with the latest analytics data.

        Parameters
        ----------
        data : dict
            Expected keys:

            action_history : dict
                direction (list[float]), conviction (list[float]),
                exit (list[float]), sl_adj (list[float])
                Each list contains up to the last 500 raw action values.
                Optional: flat_rate (float), avg_trades_day (float)
            training_comparison : dict
                Mapping metric_name -> dict with:
                    val (str/float), test (str/float), live (str/float),
                    status ("green"/"yellow"/"red")
            session_breakdown : dict
                Mapping session_name -> dict with:
                    trades (int), win_rate (float), pf (float),
                    avg_pnl (float), net_pnl (float)
            day_of_week : dict
                Mapping day_name ("Mon".."Fri") -> net_pnl (float)
            feature_drift : dict
                total_baselined (int), within_threshold (int),
                drifted_features (list[dict]):
                    Each: name (str), live_mean (float),
                          train_mean (float), sigma_distance (float)
            correlation_drift : dict
                score (float), status (str "OK"/"WARNING"/"CRITICAL"),
                yellow_consecutive (int), red_consecutive (int)
            weekly_reports : list[dict]
                Each: week (int/str), pf (float), trades (int),
                      pnl (float), max_dd (float), win_rate (float),
                      sharpe (float, optional)
        """
        if not data:
            return

        self._update_action_distributions(data.get("action_history", {}))
        self._update_training_vs_live(data.get("training_comparison", {}))
        self._update_session_breakdown(data.get("session_breakdown", {}))
        self._update_day_of_week(data.get("day_of_week", {}))
        self._update_feature_drift(data.get("feature_drift", {}))
        self._update_correlation_drift(data.get("correlation_drift", {}))
        self._update_weekly_reports(data.get("weekly_reports", []))

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_action_distributions(self, history: dict) -> None:
        """Update the 4 histograms and summary stats."""
        if not history:
            return

        # Update each histogram
        direction = history.get("direction", [])
        conviction = history.get("conviction", [])
        exit_vals = history.get("exit", [])
        sl_adj = history.get("sl_adj", [])

        if direction:
            self._hist_direction.update_histogram(direction, n_bins=20)
        if conviction:
            self._hist_conviction.update_histogram(conviction, n_bins=20)
        if exit_vals:
            self._hist_exit.update_histogram(exit_vals, n_bins=20)
        if sl_adj:
            self._hist_sl_adj.update_histogram(sl_adj, n_bins=20)

        # Summary stats
        flat_rate = history.get("flat_rate")
        if flat_rate is not None:
            flat_pct = flat_rate * 100 if flat_rate <= 1.0 else flat_rate
            flat_color = C["red"] if flat_pct > 90 else C["text"]
            self._lbl_flat_rate.setText(f"Flat rate: {flat_pct:.0f}% of bars")
            self._lbl_flat_rate.setStyleSheet(
                f"color: {flat_color}; font-size: 12px; "
                f"background: transparent; border: none;"
            )
        elif direction:
            # Compute flat rate from direction values: |direction| < 0.3
            arr = np.asarray(direction, dtype=np.float64)
            flat_count = int(np.sum(np.abs(arr) < 0.3))
            flat_pct = (flat_count / len(arr)) * 100.0
            flat_color = C["red"] if flat_pct > 90 else C["text"]
            self._lbl_flat_rate.setText(f"Flat rate: {flat_pct:.0f}% of bars")
            self._lbl_flat_rate.setStyleSheet(
                f"color: {flat_color}; font-size: 12px; "
                f"background: transparent; border: none;"
            )

        avg_trades = history.get("avg_trades_day")
        if avg_trades is not None:
            self._lbl_avg_trades_day.setText(f"Avg trades/day: {avg_trades:.1f}")
            self._lbl_avg_trades_day.setStyleSheet(
                f"color: {C['text']}; font-size: 12px; "
                f"background: transparent; border: none;"
            )

        # Red flag detection
        flags = []
        if direction:
            flat_arr = np.asarray(direction, dtype=np.float64)
            flat_pct_calc = (np.sum(np.abs(flat_arr) < 0.3) / len(flat_arr)) * 100.0
            if flat_pct_calc > 90:
                flags.append("FLAT >90%")

        # Check action std for all dimensions
        for name, vals in [
            ("dir", direction), ("conv", conviction),
            ("exit", exit_vals), ("sl", sl_adj),
        ]:
            if vals:
                std_val = float(np.std(vals))
                if std_val < 0.1:
                    flags.append(f"{name} std<0.1")

        if flags:
            self._lbl_action_flags.setText("RED FLAGS: " + ", ".join(flags))
            self._lbl_action_flags.setStyleSheet(
                f"color: {C['red']}; font-size: 12px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
        else:
            self._lbl_action_flags.setText("")

    def _update_training_vs_live(self, comparison: dict) -> None:
        """Update the training vs live comparison table."""
        if not comparison:
            return

        for row, metric_name in enumerate(self._tvl_metrics):
            metric_data = comparison.get(metric_name, {})
            if not metric_data:
                continue

            # Columns: Metric(0), Train Val(1), Train Test(2), Live 30d(3), Status(4)
            val = metric_data.get("val", "--")
            test = metric_data.get("test", "--")
            live = metric_data.get("live", "--")
            status = metric_data.get("status", "").lower()

            # Format values
            val_item = QTableWidgetItem(str(val))
            val_item.setForeground(QColor(C["text"]))
            val_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_training_vs_live.setItem(row, 1, val_item)

            test_item = QTableWidgetItem(str(test))
            test_item.setForeground(QColor(C["text"]))
            test_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_training_vs_live.setItem(row, 2, test_item)

            live_item = QTableWidgetItem(str(live))
            live_item.setForeground(QColor(C["text"]))
            live_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_training_vs_live.setItem(row, 3, live_item)

            # Status column: colored indicator
            status_colors = {
                "green":  C["green"],
                "yellow": C["yellow"],
                "red":    C["red"],
            }
            status_color = status_colors.get(status, C["label"])
            status_text = status.upper() if status else "--"

            # Map to descriptive text
            status_labels = {
                "green":  "OK",
                "yellow": "WARN",
                "red":    "DEGRADED",
            }
            status_display = status_labels.get(status, status_text)

            status_item = QTableWidgetItem(f"\u25cf {status_display}")
            status_item.setForeground(QColor(status_color))
            status_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_training_vs_live.setItem(row, 4, status_item)

    def _update_session_breakdown(self, sessions: dict) -> None:
        """Update the session performance table."""
        if not sessions:
            return

        for row, (session_name, _, _) in enumerate(SESSION_DEFS):
            session_data = sessions.get(session_name, {})
            if not session_data:
                continue

            trades = session_data.get("trades", 0)
            win_rate = session_data.get("win_rate", 0.0)
            pf = session_data.get("pf", 0.0)
            avg_pnl = session_data.get("avg_pnl", 0.0)
            net_pnl = session_data.get("net_pnl", 0.0)

            # Trades
            trades_item = QTableWidgetItem(str(trades))
            trades_item.setForeground(QColor(C["text"]))
            trades_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_sessions.setItem(row, 1, trades_item)

            # Win Rate
            wr_item = QTableWidgetItem(f"{win_rate:.0f}%")
            wr_color = C["green"] if win_rate >= 50 else C["red"]
            wr_item.setForeground(QColor(wr_color))
            wr_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_sessions.setItem(row, 2, wr_item)

            # Profit Factor (green if >1.5, red if <1.0)
            pf_item = QTableWidgetItem(f"{pf:.2f}")
            if pf >= 1.5:
                pf_color = C["green"]
            elif pf < 1.0:
                pf_color = C["red"]
            else:
                pf_color = C["text"]
            pf_item.setForeground(QColor(pf_color))
            pf_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_sessions.setItem(row, 3, pf_item)

            # Avg P/L (colored)
            avg_item = QTableWidgetItem(currency.fmt_signed(avg_pnl))
            avg_item.setForeground(
                QColor(C["green"] if avg_pnl >= 0 else C["red"])
            )
            avg_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_sessions.setItem(row, 4, avg_item)

            # Net P/L (colored)
            net_item = QTableWidgetItem(currency.fmt_signed(net_pnl))
            net_item.setForeground(
                QColor(C["green"] if net_pnl >= 0 else C["red"])
            )
            net_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_sessions.setItem(row, 5, net_item)

    def _update_day_of_week(self, dow: dict) -> None:
        """Update the day-of-week P/L breakdown grid."""
        if not dow:
            return

        for day_name, lbl in self._day_labels.items():
            pnl = dow.get(day_name)
            if pnl is not None:
                color = C["green"] if pnl >= 0 else C["red"]
                lbl.setText(currency.fmt_signed(pnl))
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 13px; font-weight: bold; "
                    f"background: transparent; border: none;"
                )
            else:
                lbl.setText("--")
                lbl.setStyleSheet(
                    f"color: {C['text']}; font-size: 13px; font-weight: bold; "
                    f"background: transparent; border: none;"
                )

    def _update_feature_drift(self, drift: dict) -> None:
        """Update the feature drift summary and drifted features table."""
        if not drift:
            return

        total = drift.get("total_baselined", 54)
        within = drift.get("within_threshold", total)
        drifted_count = total - within

        # Summary line
        if drifted_count == 0:
            summary_color = C["green"]
            summary_text = f"{within}/{total} within 2\u03c3 of training baseline"
        else:
            summary_color = C["yellow"] if drifted_count <= 3 else C["red"]
            summary_text = (
                f"{within}/{total} within 2\u03c3  |  "
                f"{drifted_count} feature{'s' if drifted_count != 1 else ''} drifted"
            )

        self._lbl_drift_summary.setText(summary_text)
        self._lbl_drift_summary.setStyleSheet(
            f"color: {summary_color}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Drifted features table
        drifted = drift.get("drifted_features", [])
        self._tbl_feature_drift.setRowCount(len(drifted))

        for row, feat in enumerate(drifted):
            name = feat.get("name", "")
            live_mean = feat.get("live_mean", 0.0)
            train_mean = feat.get("train_mean", 0.0)
            sigma_dist = feat.get("sigma_distance", 0.0)

            # Feature name
            name_item = QTableWidgetItem(name)
            name_item.setForeground(QColor(C["text"]))
            name_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_feature_drift.setItem(row, 0, name_item)

            # Live mean
            live_item = QTableWidgetItem(f"{live_mean:.4f}")
            live_item.setForeground(QColor(C["text"]))
            live_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_feature_drift.setItem(row, 1, live_item)

            # Train mean
            train_item = QTableWidgetItem(f"{train_mean:.4f}")
            train_item.setForeground(QColor(C["subtext"]))
            train_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_feature_drift.setItem(row, 2, train_item)

            # Sigma distance
            sigma_item = QTableWidgetItem(f"{sigma_dist:.2f}\u03c3")
            # Color by severity: green <2, yellow 2-3, red >3
            if sigma_dist < 2.0:
                sigma_color = C["green"]
            elif sigma_dist < 3.0:
                sigma_color = C["yellow"]
            else:
                sigma_color = C["red"]
            sigma_item.setForeground(QColor(sigma_color))
            sigma_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_feature_drift.setItem(row, 3, sigma_item)

            # Status dot
            if sigma_dist < 2.0:
                status_text = "\u25cf OK"
                status_color = C["green"]
            elif sigma_dist < 3.0:
                status_text = "\u25cf DRIFT"
                status_color = C["yellow"]
            else:
                status_text = "\u25cf SEVERE"
                status_color = C["red"]
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor(status_color))
            status_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_feature_drift.setItem(row, 4, status_item)

    def _update_correlation_drift(self, corr: dict) -> None:
        """Update the correlation drift score and status display."""
        if not corr:
            return

        score = corr.get("score", 0.0)
        status = corr.get("status", "OK").upper()
        yellow_consec = corr.get("yellow_consecutive", 0)
        red_consec = corr.get("red_consecutive", 0)

        # Score text
        self._lbl_corr_score.setText(f"{score:.2f}")

        # Determine color based on persistence filter
        if status == "CRITICAL" or (score >= 0.25 and red_consec >= 2):
            dot_color = C["red"]
            status_text = "CRITICAL"
        elif status == "WARNING" or (score >= 0.15 and yellow_consec >= 3):
            dot_color = C["yellow"]
            status_text = "WARNING"
        else:
            dot_color = C["green"]
            status_text = "OK"

        self._lbl_corr_score.setStyleSheet(
            f"color: {dot_color}; font-size: 14px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        self._lbl_corr_dot.setStyleSheet(
            f"color: {dot_color}; font-size: 14px; "
            f"background: transparent; border: none;"
        )
        self._lbl_corr_status.setText(status_text)
        self._lbl_corr_status.setStyleSheet(
            f"color: {dot_color}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Consecutive counter
        self._lbl_corr_consecutive.setText(
            f"consecutive: {yellow_consec}/3 yel, {red_consec}/2 red"
        )
        self._lbl_corr_consecutive.setStyleSheet(
            f"color: {C['label']}; font-size: 11px; "
            f"background: transparent; border: none;"
        )

    def _update_weekly_reports(self, reports: list[dict]) -> None:
        """Update the weekly reports table and cache for export."""
        if not reports:
            return

        self._weekly_reports = reports
        self._tbl_weekly.setRowCount(len(reports))

        for row, report in enumerate(reports):
            week = report.get("week", "")
            pf = report.get("pf", 0.0)
            trades = report.get("trades", 0)
            pnl = report.get("pnl", 0.0)
            max_dd = report.get("max_dd", 0.0)
            win_rate = report.get("win_rate", 0.0)
            sharpe = report.get("sharpe")

            # Week
            week_item = QTableWidgetItem(str(week))
            week_item.setForeground(QColor(C["text"]))
            week_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 0, week_item)

            # PF (colored)
            pf_item = QTableWidgetItem(f"{pf:.2f}")
            pf_color = C["green"] if pf >= 1.5 else C["text"] if pf >= 1.0 else C["red"]
            pf_item.setForeground(QColor(pf_color))
            pf_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 1, pf_item)

            # Trades
            trades_item = QTableWidgetItem(str(trades))
            trades_item.setForeground(QColor(C["text"]))
            trades_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 2, trades_item)

            # P/L (colored)
            pnl_item = QTableWidgetItem(currency.fmt_signed(pnl))
            pnl_item.setForeground(
                QColor(C["green"] if pnl >= 0 else C["red"])
            )
            pnl_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 3, pnl_item)

            # MaxDD
            dd_item = QTableWidgetItem(f"{max_dd:.1f}%")
            dd_color = C["red"] if max_dd > 5.0 else C["yellow"] if max_dd > 3.0 else C["text"]
            dd_item.setForeground(QColor(dd_color))
            dd_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 4, dd_item)

            # Win Rate
            wr_item = QTableWidgetItem(f"{win_rate:.0f}%")
            wr_color = C["green"] if win_rate >= 50 else C["red"]
            wr_item.setForeground(QColor(wr_color))
            wr_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 5, wr_item)

            # Sharpe (optional)
            sharpe_text = f"{sharpe:.2f}" if sharpe is not None else "--"
            sharpe_item = QTableWidgetItem(sharpe_text)
            if sharpe is not None:
                sharpe_color = (
                    C["green"] if sharpe >= 1.0
                    else C["text"] if sharpe >= 0.0
                    else C["red"]
                )
            else:
                sharpe_color = C["label"]
            sharpe_item.setForeground(QColor(sharpe_color))
            sharpe_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._tbl_weekly.setItem(row, 6, sharpe_item)

    # ==================================================================
    # Export methods
    # ==================================================================

    def _export_csv(self) -> None:
        """Export weekly reports to a CSV file via file dialog."""
        if not self._weekly_reports:
            log.warning("No weekly reports to export")
            return

        default_name = f"spartus_weekly_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Weekly Reports as CSV",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )

        if not file_path:
            return

        try:
            fieldnames = ["week", "pf", "trades", "pnl", "max_dd", "win_rate", "sharpe"]
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for report in self._weekly_reports:
                    row_data = {k: report.get(k, "") for k in fieldnames}
                    writer.writerow(row_data)

            log.info(f"Weekly reports exported to CSV: {file_path}")
        except Exception:
            log.exception(f"Failed to export CSV to {file_path}")

    def _export_json(self) -> None:
        """Export weekly reports to a JSON file via file dialog."""
        if not self._weekly_reports:
            log.warning("No weekly reports to export")
            return

        default_name = f"spartus_weekly_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Weekly Reports as JSON",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )

        if not file_path:
            return

        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_weeks": len(self._weekly_reports),
                "reports": self._weekly_reports,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            log.info(f"Weekly reports exported to JSON: {file_path}")
        except Exception:
            log.exception(f"Failed to export JSON to {file_path}")
