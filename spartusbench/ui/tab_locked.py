"""Tab 5: Locked Test -- gated test set evaluation with permanent audit trail."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QCheckBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from .styles import C

if TYPE_CHECKING:
    from .main_window import SpartusBenchWindow


class LockedTestTab(QWidget):
    def __init__(self, parent: SpartusBenchWindow):
        super().__init__()
        self.main = parent
        self._running = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("LOCKED TEST")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C['cyan']};")
        layout.addWidget(title)

        # Warning box
        warn_group = QGroupBox("")
        warn_group.setStyleSheet(
            f"QGroupBox {{ background-color: #2a1a0a; border: 2px solid {C['yellow']}; "
            f"border-radius: 8px; }}"
        )
        warn_layout = QVBoxLayout(warn_group)
        warn_text = QLabel(
            "<b>WARNING:</b> Test set evaluation is permanently recorded.<br><br>"
            "The test set is reserved for final go-live assessment. Every run is "
            "logged in the audit trail with operator, timestamp, and result hashes.<br><br>"
            "<b>This action cannot be undone or hidden.</b>"
        )
        warn_text.setWordWrap(True)
        warn_text.setStyleSheet(f"color: {C['yellow']}; font-size: 13px;")
        warn_layout.addWidget(warn_text)
        layout.addWidget(warn_group)

        # Model & options
        opts_group = QGroupBox("Configuration")
        opts_layout = QHBoxLayout(opts_group)

        opts_layout.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(250)
        opts_layout.addWidget(self.combo_model)

        opts_layout.addWidget(QLabel("Seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(1, 99999)
        self.spin_seed.setValue(42)
        opts_layout.addWidget(self.spin_seed)

        opts_layout.addStretch()
        layout.addWidget(opts_group)

        # Confirmation
        self.chk_confirm = QCheckBox("I understand this is permanently recorded")
        self.chk_confirm.setStyleSheet(f"color: {C['yellow']}; font-size: 13px;")
        self.chk_confirm.stateChanged.connect(self._on_confirm_changed)
        layout.addWidget(self.chk_confirm)

        # Run button
        self.btn_run = QPushButton("RUN LOCKED TEST")
        self.btn_run.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            f"QPushButton {{ background-color: {C['peach']}; color: #0d1117; "
            f"border-radius: 8px; }} "
            f"QPushButton:hover {{ background-color: #e0944d; }} "
            f"QPushButton:disabled {{ background-color: {C['dim']}; color: {C['surface']}; }}"
        )
        self.btn_run.clicked.connect(self._on_run)
        layout.addWidget(self.btn_run)

        # Audit trail
        audit_group = QGroupBox("Audit Trail")
        audit_layout = QVBoxLayout(audit_group)

        self.audit_table = QTableWidget()
        self.audit_table.setColumnCount(6)
        self.audit_table.setHorizontalHeaderLabels([
            "Date", "Operator", "Model", "Seed", "Result Hash", "Run ID"
        ])
        self.audit_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.audit_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.audit_table.verticalHeader().setVisible(False)
        audit_layout.addWidget(self.audit_table)

        self.btn_refresh_audit = QPushButton("Refresh Audit Trail")
        self.btn_refresh_audit.clicked.connect(self._refresh_audit)
        audit_layout.addWidget(self.btn_refresh_audit)

        layout.addWidget(audit_group, stretch=1)

        # Initial load
        self._refresh_audit()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_models()
        self._refresh_audit()

    def _refresh_models(self):
        from spartusbench.discovery import discover_models
        models = discover_models()
        self.combo_model.clear()
        champion = self.main.db.get_current_champion() if self.main.db else None
        champ_id = champion.get("model_id") if champion else None

        for m in models:
            label = m["model_id"]
            if label == champ_id:
                label += " (champion)"
            self.combo_model.addItem(label, m["model_id"])

    def _on_confirm_changed(self, state):
        self.btn_run.setEnabled(state == Qt.CheckState.Checked.value)

    def _on_run(self):
        if self._running:
            return

        model_id = self.combo_model.currentData()
        if not model_id:
            return

        # Double confirmation
        reply = QMessageBox.warning(
            self, "Confirm Locked Test",
            f"You are about to run a LOCKED TEST on model {model_id}.\n\n"
            "This will be permanently recorded in the audit trail.\n\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        seed = self.spin_seed.value()
        self._running = True
        self.btn_run.setEnabled(False)
        self.btn_run.setText("RUNNING...")

        thread = threading.Thread(
            target=self._run_locked_test,
            args=(model_id, seed),
            daemon=True,
        )
        thread.start()

    def _run_locked_test(self, model_id, seed):
        try:
            from spartusbench.runner import BenchmarkRunner
            runner = BenchmarkRunner()
            result = runner.run(
                model_ref=model_id,
                suite="locked_test",
                seed=seed,
                confirm_test=True,
                generate_plots=False,
                compare_vs_champion=False,
            )
            self.main.signals.benchmark_complete.emit(result.run_id)
        except Exception as e:
            self.main.signals.benchmark_error.emit(str(e))
        finally:
            self._running = False
            self.btn_run.setEnabled(self.chk_confirm.isChecked())
            self.btn_run.setText("RUN LOCKED TEST")
            self._refresh_audit()

    def _refresh_audit(self):
        db = self.main.db
        if not db:
            return

        entries = db.get_locked_test_audit()
        self.audit_table.setRowCount(len(entries))

        for i, entry in enumerate(entries):
            items = [
                entry.get("timestamp", "?")[:19],
                entry.get("operator", "?"),
                entry.get("model_id", "?"),
                str(entry.get("seed", 42)),
                (entry.get("result_hash", "?") or "?")[:16] + "...",
                (entry.get("run_id", "?") or "?")[:8] + "...",
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                self.audit_table.setItem(i, j, item)
