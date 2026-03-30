"""Tab 1: Leaderboard -- champion history and ranked model table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from .styles import C

if TYPE_CHECKING:
    from .main_window import SpartusBenchWindow


class LeaderboardTab(QWidget):
    def __init__(self, parent: SpartusBenchWindow):
        super().__init__()
        self.main = parent
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("LEADERBOARD")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C['cyan']};")
        header.addWidget(title)
        header.addStretch()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        # Leaderboard table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Rank", "Model", "Score", "Sharpe", "PF", "MaxDD", "Stress", "Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, stretch=3)

        # Champion info box
        champ_group = QGroupBox("Current Champion")
        champ_layout = QHBoxLayout(champ_group)
        self.lbl_champion = QLabel("No champion yet")
        self.lbl_champion.setFont(QFont("Segoe UI", 14))
        self.lbl_champion.setStyleSheet(f"color: {C['green']};")
        champ_layout.addWidget(self.lbl_champion)
        layout.addWidget(champ_group, stretch=1)

    def refresh(self):
        """Reload leaderboard from database."""
        db = self.main.db
        if not db:
            return

        entries = db.get_leaderboard(top_n=50, include_dq=True)
        self.table.setRowCount(len(entries))

        for i, entry in enumerate(entries):
            is_dq = entry.get("is_disqualified", 0)
            is_champ = entry.get("is_champion", 0)
            score = entry.get("spartus_score", 0) or 0
            sharpe = entry.get("val_sharpe", 0) or 0
            pf = entry.get("val_pf", 0) or 0
            dd = entry.get("val_max_dd_pct", 0) or 0
            stress = entry.get("stress_robustness_score", 0) or 0

            if is_dq:
                rank_str = "-"
                status = "DISQUALIFIED"
                color = C["red"]
            elif is_champ:
                rank_str = f"{i+1}*"
                status = "CHAMPION"
                color = C["green"]
            elif entry.get("dethroned_at"):
                rank_str = str(i + 1)
                status = "(dethroned)"
                color = C["subtext"]
            else:
                rank_str = str(i + 1)
                status = ""
                color = C["text"]

            items = [
                rank_str,
                entry.get("model_id", "?"),
                f"{score:.1f}" if not is_dq else "DQ",
                f"{sharpe:.2f}" if not is_dq else "--",
                f"{pf:.2f}",
                f"{dd:.1f}%",
                f"{stress:.1f}" if not is_dq else "--",
                status,
            ]

            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(color))
                item.setData(Qt.ItemDataRole.UserRole, entry.get("run_id"))
                if j in (2, 3, 4, 5, 6):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(i, j, item)

        # Update champion info
        champion = db.get_current_champion()
        if champion:
            champ_id = champion.get("model_id", "?")
            champ_score = champion.get("spartus_score", 0) or 0
            champ_sharpe = champion.get("val_sharpe", 0) or 0
            champ_pf = champion.get("val_pf", 0) or 0
            self.lbl_champion.setText(
                f"{champ_id}  |  Score: {champ_score:.1f}  |  "
                f"Sharpe: {champ_sharpe:.2f}  |  PF: {champ_pf:.2f}"
            )
        else:
            self.lbl_champion.setText("No champion yet. Run a benchmark to establish one.")

    def _on_double_click(self, index):
        """Open run detail on double-click."""
        item = self.table.item(index.row(), 0)
        if item:
            run_id = item.data(Qt.ItemDataRole.UserRole)
            if run_id:
                self.main.open_run_detail(run_id)
