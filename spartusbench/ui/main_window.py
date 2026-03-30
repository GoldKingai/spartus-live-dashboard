"""SpartusBench main UI window with 5 tabs.

Tab 1: Leaderboard
Tab 2: Run Benchmark
Tab 3: Compare
Tab 4: Run Detail
Tab 5: Locked Test
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QLabel, QMessageBox,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont

from .styles import C, DARK_STYLE
from .tab_leaderboard import LeaderboardTab
from .tab_run import RunBenchmarkTab
from .tab_compare import CompareTab
from .tab_detail import RunDetailTab
from .tab_locked import LockedTestTab

log = logging.getLogger("spartusbench.ui")


class BenchmarkSignals(QObject):
    """Signals for cross-thread communication."""
    benchmark_complete = pyqtSignal(str)  # run_id
    benchmark_progress = pyqtSignal(str, int, int)  # tier, current, total
    benchmark_error = pyqtSignal(str)  # error message
    status_update = pyqtSignal(str)  # status bar message


class SpartusBenchWindow(QMainWindow):
    """Main SpartusBench UI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpartusBench - Benchmark & Model Progression")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        self.signals = BenchmarkSignals()
        self._db = None
        self._init_db()

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        # Create tabs
        self.tab_leaderboard = LeaderboardTab(self)
        self.tab_run = RunBenchmarkTab(self)
        self.tab_compare = CompareTab(self)
        self.tab_detail = RunDetailTab(self)
        self.tab_locked = LockedTestTab(self)

        self.tabs.addTab(self.tab_leaderboard, "Leaderboard")
        self.tabs.addTab(self.tab_run, "Run Benchmark")
        self.tabs.addTab(self.tab_compare, "Compare")
        self.tabs.addTab(self.tab_detail, "Run Detail")
        self.tabs.addTab(self.tab_locked, "Locked Test")

        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status_bar()

        # Apply style
        self.setStyleSheet(DARK_STYLE)

        # Signals
        self.signals.benchmark_complete.connect(self._on_benchmark_complete)
        self.signals.benchmark_error.connect(self._on_benchmark_error)
        self.signals.status_update.connect(self._on_status_update)

        # Initial data load
        QTimer.singleShot(100, self._refresh_all)

    def _init_db(self):
        from spartusbench.database import BenchmarkDB
        self._db = BenchmarkDB()

    @property
    def db(self):
        return self._db

    def _refresh_all(self):
        """Refresh all tabs with latest data."""
        self.tab_leaderboard.refresh()
        self.tab_compare.refresh_models()
        self.tab_run.refresh_models()
        self._update_status_bar()

    def _update_status_bar(self):
        champion = self._db.get_current_champion() if self._db else None
        run_count = self._db.get_run_count() if self._db else 0

        if champion:
            champ_id = champion.get("model_id", "?")
            champ_score = champion.get("spartus_score", 0) or 0
            self.status_bar.showMessage(
                f"Ready | Champion: {champ_id} ({champ_score:.1f}) | DB: {run_count} runs"
            )
        else:
            self.status_bar.showMessage(f"Ready | No champion yet | DB: {run_count} runs")

    def _on_benchmark_complete(self, run_id: str):
        self._refresh_all()
        self.tab_detail.load_run(run_id)
        self.tabs.setCurrentWidget(self.tab_detail)
        self.signals.status_update.emit(f"Benchmark complete: {run_id}")

    def _on_benchmark_error(self, error_msg: str):
        QMessageBox.critical(self, "Benchmark Error", error_msg)
        self.signals.status_update.emit("Benchmark failed")

    def _on_status_update(self, msg: str):
        self.status_bar.showMessage(msg, 10000)

    def open_run_detail(self, run_id: str):
        """Switch to Run Detail tab and load a specific run."""
        self.tab_detail.load_run(run_id)
        self.tabs.setCurrentWidget(self.tab_detail)

    def closeEvent(self, event):
        if self._db:
            self._db.close()
        event.accept()
