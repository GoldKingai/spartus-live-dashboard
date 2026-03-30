"""Tab 3: Compare -- side-by-side comparison of two benchmark runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from .styles import C

if TYPE_CHECKING:
    from .main_window import SpartusBenchWindow

COMPARE_METRICS = [
    ("SpartusScore", "spartus_score", False),
    ("Sharpe", "val_sharpe", False),
    ("PF", "val_pf", False),
    ("Win Rate", "val_win_pct", False),
    ("Max DD", "val_max_dd_pct", True),
    ("Stress Score", "stress_robustness_score", False),
    ("TIM%", "val_tim_pct", None),
    ("Trades/Day", "val_trades_day", None),
    ("Expectancy", "val_expectancy", False),
    ("Max Consec Loss", "val_max_consec_loss", True),
    ("Trades", "val_trades", None),
    ("Sortino", "val_sortino", False),
    ("Calmar", "val_calmar", False),
    ("Recovery Factor", "val_recovery_factor", False),
    ("Tail Ratio", "val_tail_ratio", False),
    ("Avg Win", "val_avg_win", False),
    ("Avg Loss", "val_avg_loss", True),
    ("Win/Loss Ratio", "val_win_loss_ratio", False),
]


class CompareTab(QWidget):
    def __init__(self, parent: SpartusBenchWindow):
        super().__init__()
        self.main = parent
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("COMPARE MODELS")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C['cyan']};")
        layout.addWidget(title)

        # Model selection
        sel_group = QGroupBox("Select Models")
        sel_layout = QHBoxLayout(sel_group)

        sel_layout.addWidget(QLabel("Model A:"))
        self.combo_a = QComboBox()
        self.combo_a.setMinimumWidth(200)
        sel_layout.addWidget(self.combo_a)

        sel_layout.addWidget(QLabel("Model B:"))
        self.combo_b = QComboBox()
        self.combo_b.setMinimumWidth(200)
        sel_layout.addWidget(self.combo_b)

        self.btn_compare = QPushButton("Compare")
        self.btn_compare.clicked.connect(self._on_compare)
        sel_layout.addWidget(self.btn_compare)

        sel_layout.addStretch()
        layout.addWidget(sel_group)

        # Comparison table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Metric", "Model A", "Model B", "Delta", "Verdict"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, stretch=2)

        # Detector comparison
        self.det_group = QGroupBox("Detectors")
        self.det_layout = QVBoxLayout(self.det_group)
        self.lbl_detectors = QLabel("")
        self.lbl_detectors.setWordWrap(True)
        self.det_layout.addWidget(self.lbl_detectors)
        layout.addWidget(self.det_group)

    def refresh_models(self):
        """Reload model list for comparison."""
        db = self.main.db
        if not db:
            return

        entries = db.get_leaderboard(top_n=50, include_dq=True)
        self.combo_a.clear()
        self.combo_b.clear()

        for entry in entries:
            model_id = entry.get("model_id", "?")
            run_id = entry.get("run_id", "")
            label = f"{model_id} ({entry.get('spartus_score', 0):.1f})"
            self.combo_a.addItem(label, run_id)
            self.combo_b.addItem(label, run_id)

        # Default: select champion as A, next as B
        if len(entries) >= 2:
            self.combo_b.setCurrentIndex(1)

    def _on_compare(self):
        db = self.main.db
        if not db:
            return

        run_id_a = self.combo_a.currentData()
        run_id_b = self.combo_b.currentData()

        if not run_id_a or not run_id_b:
            return

        run_a = db.get_run(run_id_a)
        run_b = db.get_run(run_id_b)

        if not run_a or not run_b:
            return

        self._populate_table(run_a, run_b)
        self._populate_detectors(run_a, run_b)

    def _populate_table(self, run_a: dict, run_b: dict):
        self.table.setRowCount(len(COMPARE_METRICS))

        id_a = run_a.get("model_id", "A")
        id_b = run_b.get("model_id", "B")
        self.table.setHorizontalHeaderLabels(["Metric", id_a, id_b, "Delta", "Verdict"])

        for i, (name, key, invert) in enumerate(COMPARE_METRICS):
            val_a = float(run_a.get(key, 0) or 0)
            val_b = float(run_b.get(key, 0) or 0)
            delta = val_b - val_a

            if invert is None:
                verdict = "NEUTRAL"
                color = C["subtext"]
            elif invert:
                if delta < -0.01:
                    verdict = "IMPROVE"
                    color = C["green"]
                elif delta > 0.5:
                    verdict = "REGRESS"
                    color = C["red"]
                else:
                    verdict = "NEUTRAL"
                    color = C["subtext"]
            else:
                if delta > 0.01:
                    verdict = "IMPROVE"
                    color = C["green"]
                elif delta < -0.01:
                    verdict = "REGRESS"
                    color = C["red"]
                else:
                    verdict = "NEUTRAL"
                    color = C["subtext"]

            items = [name, f"{val_a:.2f}", f"{val_b:.2f}", f"{delta:+.2f}", verdict]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j == 4:
                    item.setForeground(QColor(color))
                if j in (1, 2, 3):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(i, j, item)

    def _populate_detectors(self, run_a: dict, run_b: dict):
        det_keys = [
            ("Aggression Drift", "detector_aggression"),
            ("Conviction Collapse", "detector_collapse"),
            ("Stress Fragility", "detector_fragility"),
            ("Overfitting", "detector_overfitting"),
            ("Reward Hacking", "detector_reward_hack"),
        ]

        lines = []
        for name, key in det_keys:
            a_val = "DETECTED" if run_a.get(key) else "CLEAR"
            b_val = "DETECTED" if run_b.get(key) else "CLEAR"
            a_color = C["red"] if run_a.get(key) else C["green"]
            b_color = C["red"] if run_b.get(key) else C["green"]
            lines.append(
                f'<b>{name}:</b> '
                f'<span style="color:{a_color}">{run_a.get("model_id","A")}={a_val}</span> | '
                f'<span style="color:{b_color}">{run_b.get("model_id","B")}={b_val}</span>'
            )

        self.lbl_detectors.setText("<br>".join(lines))
