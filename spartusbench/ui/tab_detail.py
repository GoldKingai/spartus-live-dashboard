"""Tab 4: Run Detail -- detailed view of a specific benchmark run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QTextEdit, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from .styles import C

if TYPE_CHECKING:
    from .main_window import SpartusBenchWindow


class RunDetailTab(QWidget):
    def __init__(self, parent: SpartusBenchWindow):
        super().__init__()
        self.main = parent
        self._current_run = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.lbl_title = QLabel("RUN DETAIL")
        self.lbl_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet(f"color: {C['cyan']};")
        layout.addWidget(self.lbl_title)

        # Sub-tabs for detail sections
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setFont(QFont("Segoe UI", 10))

        # Summary sub-tab
        self.summary_widget = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_widget)
        self._build_summary()
        self.sub_tabs.addTab(self.summary_widget, "Summary")

        # Stress sub-tab
        self.stress_table = QTableWidget()
        self.stress_table.setColumnCount(6)
        self.stress_table.setHorizontalHeaderLabels(
            ["Scenario", "PF", "Retention", "Trades", "MaxDD", "Win%"]
        )
        self.stress_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stress_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stress_table.verticalHeader().setVisible(False)
        self.sub_tabs.addTab(self.stress_table, "Stress")

        # Regime sub-tab
        self.regime_table = QTableWidget()
        self.regime_table.setColumnCount(7)
        self.regime_table.setHorizontalHeaderLabels(
            ["Type", "Slice", "Trades", "Win%", "Net P/L", "PF", "Avg Hold"]
        )
        self.regime_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.regime_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.regime_table.verticalHeader().setVisible(False)
        self.sub_tabs.addTab(self.regime_table, "Regime")

        # Trades sub-tab
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(10)
        self.trades_table.setHorizontalHeaderLabels([
            "#", "Side", "Entry", "Exit", "Lots", "P/L",
            "Hold", "Conv", "Reason", "Session",
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trades_table.verticalHeader().setVisible(False)
        self.sub_tabs.addTab(self.trades_table, "Trades")

        # Hashes sub-tab
        self.txt_hashes = QTextEdit()
        self.txt_hashes.setReadOnly(True)
        self.sub_tabs.addTab(self.txt_hashes, "Hashes")

        layout.addWidget(self.sub_tabs)

    def _build_summary(self):
        # Score section
        score_group = QGroupBox("SpartusScore")
        score_layout = QVBoxLayout(score_group)
        self.lbl_score = QLabel("--")
        self.lbl_score.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.lbl_score)
        self.lbl_score_breakdown = QLabel("")
        self.lbl_score_breakdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score_breakdown.setStyleSheet(f"color: {C['subtext']};")
        score_layout.addWidget(self.lbl_score_breakdown)
        self.summary_layout.addWidget(score_group)

        # Metrics row
        metrics_row = QHBoxLayout()

        # Quick Stats
        stats_group = QGroupBox("Quick Stats")
        stats_layout = QVBoxLayout(stats_group)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setFont(QFont("Cascadia Code", 12))
        stats_layout.addWidget(self.lbl_stats)
        metrics_row.addWidget(stats_group)

        # Stress Retention
        stress_group = QGroupBox("Stress Retention")
        stress_layout = QVBoxLayout(stress_group)
        self.lbl_stress = QLabel("")
        self.lbl_stress.setFont(QFont("Cascadia Code", 12))
        stress_layout.addWidget(self.lbl_stress)
        metrics_row.addWidget(stress_group)

        # Detectors
        det_group = QGroupBox("Detectors")
        det_layout = QVBoxLayout(det_group)
        self.lbl_detectors = QLabel("")
        self.lbl_detectors.setFont(QFont("Cascadia Code", 12))
        det_layout.addWidget(self.lbl_detectors)
        metrics_row.addWidget(det_group)

        self.summary_layout.addLayout(metrics_row)

        # Gates Funnel
        gates_group = QGroupBox("Gating Funnel")
        gates_layout = QVBoxLayout(gates_group)
        self.lbl_gates = QLabel("")
        self.lbl_gates.setFont(QFont("Cascadia Code", 12))
        gates_layout.addWidget(self.lbl_gates)
        self.summary_layout.addWidget(gates_group)

        # Hard Fails
        self.lbl_hard_fails = QLabel("Hard Fails: --")
        self.lbl_hard_fails.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.summary_layout.addWidget(self.lbl_hard_fails)

        self.summary_layout.addStretch()

    def load_run(self, run_id: str):
        """Load and display a specific benchmark run."""
        db = self.main.db
        if not db:
            return

        run = db.get_run(run_id)
        if not run:
            return

        self._current_run = run
        model_id = run.get("model_id", "?")
        ts = run.get("timestamp", "?")
        self.lbl_title.setText(f"RUN DETAIL: {run_id[:8]}... ({model_id}, {ts})")

        self._populate_summary(run)
        self._populate_stress(run_id)
        self._populate_regime(run_id)
        self._populate_trades(run_id)
        self._populate_hashes(run)

    def _populate_summary(self, run: dict):
        score = run.get("spartus_score", 0) or 0
        color = C["green"] if score >= 60 else C["yellow"] if score >= 40 else C["red"]
        self.lbl_score.setText(f"{score:.1f}")
        self.lbl_score.setStyleSheet(f"color: {color}; font-size: 28px;")

        breakdown = (
            f"sharpe={run.get('score_val_sharpe', 0):.1f} x 0.25  |  "
            f"pf={run.get('score_val_pf', 0):.1f} x 0.20  |  "
            f"stress={run.get('score_stress', 0):.1f} x 0.25  |  "
            f"dd={run.get('score_max_dd', 0):.1f} x 0.15  |  "
            f"quality={run.get('score_quality', 0):.1f} x 0.15"
        )
        self.lbl_score_breakdown.setText(breakdown)

        # Quick stats
        stats = (
            f"Sharpe:   {run.get('val_sharpe', 0):.2f}\n"
            f"PF:       {run.get('val_pf', 0):.2f}\n"
            f"Win%:     {run.get('val_win_pct', 0):.1f}%\n"
            f"MaxDD:    {run.get('val_max_dd_pct', 0):.1f}%\n"
            f"Sortino:  {run.get('val_sortino', 0):.2f}\n"
            f"Trades:   {run.get('val_trades', 0)}\n"
            f"TIM%:     {run.get('val_tim_pct', 0):.1f}%\n"
            f"Expect:   {run.get('val_expectancy', 0):.3f}\n"
            f"ConsecL:  {run.get('val_max_consec_loss', 0)}"
        )
        self.lbl_stats.setText(stats)

        # Stress retention
        stress_lines = (
            f"2x spread:  {run.get('stress_2x_spread_pf', 0):.2f}\n"
            f"Combined:   {run.get('stress_combined_pf', 0):.2f}\n"
            f"3x spread:  {run.get('stress_3x_spread_pf', 0):.2f}\n"
            f"5x spread:  {run.get('stress_5x_spread_pf', 0):.2f}\n"
            f"Score:      {run.get('stress_robustness_score', 0):.1f}"
        )
        self.lbl_stress.setText(stress_lines)

        # Detectors
        det_items = [
            ("Aggression", run.get("detector_aggression")),
            ("Collapse", run.get("detector_collapse")),
            ("Fragility", run.get("detector_fragility")),
            ("Overfitting", run.get("detector_overfitting")),
            ("Rew Hack", run.get("detector_reward_hack")),
        ]
        det_lines = []
        for name, detected in det_items:
            if detected:
                det_lines.append(f'<span style="color:{C["red"]}">{name}: DETECTED</span>')
            else:
                det_lines.append(f'<span style="color:{C["green"]}">{name}: -</span>')
        self.lbl_detectors.setText("<br>".join(det_lines))

        # Gates
        gate_lines = (
            f"Direction:  {run.get('gate_direction_pass', 0):.1f}%\n"
            f"Conviction: {run.get('gate_conviction_pass', 0):.1f}%\n"
            f"Lot Sizing: {run.get('gate_lot_pass', 0):.1f}%\n"
            f"Overall:    {run.get('gate_overall_pass', 0):.1f}%"
        )
        self.lbl_gates.setText(gate_lines)

        # Hard fails
        hard_fails_raw = run.get("hard_fails", "[]")
        try:
            fails = json.loads(hard_fails_raw) if hard_fails_raw else []
        except (json.JSONDecodeError, TypeError):
            fails = []

        if fails:
            self.lbl_hard_fails.setText(f"Hard Fails: {', '.join(fails)}")
            self.lbl_hard_fails.setStyleSheet(f"color: {C['red']};")
        else:
            self.lbl_hard_fails.setText("Hard Fails: NONE")
            self.lbl_hard_fails.setStyleSheet(f"color: {C['green']};")

    def _populate_stress(self, run_id: str):
        db = self.main.db
        details = db.get_stress_details(run_id)
        self.stress_table.setRowCount(len(details))

        for i, d in enumerate(details):
            items = [
                d.get("scenario", ""),
                f"{d.get('pf', 0):.2f}",
                f"{d.get('pf_retention', 0):.2f}",
                str(d.get("trades", 0)),
                f"{d.get('max_dd_pct', 0):.1f}%",
                f"{d.get('win_pct', 0):.1f}%",
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j in (1, 2, 3, 4, 5):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.stress_table.setItem(i, j, item)

    def _populate_regime(self, run_id: str):
        db = self.main.db
        details = db.get_regime_details(run_id)
        self.regime_table.setRowCount(len(details))

        for i, d in enumerate(details):
            items = [
                d.get("slice_type", ""),
                d.get("slice_value", ""),
                str(d.get("trades", 0)),
                f"{d.get('win_pct', 0):.1f}%",
                f"${d.get('net_pnl', 0):.2f}",
                f"{d.get('pf', 0):.2f}",
                f"{d.get('avg_hold', 0):.1f}",
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j in (2, 3, 4, 5, 6):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.regime_table.setItem(i, j, item)

    def _populate_trades(self, run_id: str):
        db = self.main.db
        trades = db.get_trades(run_id, scenario="base")
        self.trades_table.setRowCount(len(trades))

        for i, t in enumerate(trades):
            pnl = t.get("pnl", 0) or 0
            color = C["green"] if pnl > 0 else C["red"] if pnl < 0 else C["text"]

            items = [
                str(t.get("trade_num", "")),
                t.get("side", ""),
                f"{t.get('entry_price', 0):.2f}",
                f"{t.get('exit_price', 0):.2f}",
                f"{t.get('lots', 0):.2f}",
                f"{pnl:.3f}",
                str(t.get("hold_bars", "")),
                f"{t.get('conviction', 0):.2f}",
                t.get("close_reason", ""),
                t.get("session", ""),
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j == 5:
                    item.setForeground(QColor(color))
                if j in (2, 3, 4, 5, 6, 7):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.trades_table.setItem(i, j, item)

    def _populate_hashes(self, run: dict):
        lines = [
            f"Run ID:          {run.get('run_id', '?')}",
            f"Timestamp:       {run.get('timestamp', '?')}",
            f"Model:           {run.get('model_id', '?')}",
            f"Model Path:      {run.get('model_path', '?')}",
            f"Suite:           {run.get('suite', '?')}",
            f"Seed:            {run.get('seed', '?')}",
            f"Operator:        {run.get('operator', '?')}",
            "",
            "Reproducibility Hashes:",
            f"  data_manifest: {run.get('data_manifest_hash', '?')}",
            f"  split:         {run.get('split_hash', '?')}",
            f"  features:      {run.get('feature_hash', '?')}",
            f"  config:        {run.get('config_hash', '?')}",
            f"  model_file:    {run.get('model_file_hash', '?')}",
        ]
        self.txt_hashes.setPlainText("\n".join(lines))
