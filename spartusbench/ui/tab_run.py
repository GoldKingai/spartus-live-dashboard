"""Tab 2: Run Benchmark -- model selection, suite choice, and execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QRadioButton, QPushButton, QProgressBar,
    QCheckBox, QSpinBox, QTextEdit, QButtonGroup,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .styles import C

if TYPE_CHECKING:
    from .main_window import SpartusBenchWindow


class RunBenchmarkTab(QWidget):
    def __init__(self, parent: SpartusBenchWindow):
        super().__init__()
        self.main = parent
        self._running = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("RUN BENCHMARK")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C['cyan']};")
        layout.addWidget(title)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout(model_group)

        model_layout.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(300)
        model_layout.addWidget(self.combo_model, stretch=1)

        self.btn_discover = QPushButton("Discover Models")
        self.btn_discover.clicked.connect(self.refresh_models)
        model_layout.addWidget(self.btn_discover)
        layout.addWidget(model_group)

        # Suite selection
        suite_group = QGroupBox("Suite")
        suite_layout = QVBoxLayout(suite_group)

        self.suite_group = QButtonGroup()
        suites = [
            ("Full (T1-T6)", "full", "Recommended for champion evaluation"),
            ("Validation Only (T1)", "validation_only", "Quick check (~2 min)"),
            ("Stress Only (T2)", "stress_only", "Cost robustness (~10 min)"),
        ]
        for label, value, desc in suites:
            row = QHBoxLayout()
            radio = QRadioButton(label)
            radio.setProperty("suite_value", value)
            if value == "full":
                radio.setChecked(True)
            self.suite_group.addButton(radio)
            row.addWidget(radio)
            lbl = QLabel(desc)
            lbl.setStyleSheet(f"color: {C['label']};")
            row.addWidget(lbl)
            row.addStretch()
            suite_layout.addLayout(row)
        layout.addWidget(suite_group)

        # Options
        opts_group = QGroupBox("Options")
        opts_layout = QHBoxLayout(opts_group)

        opts_layout.addWidget(QLabel("Seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(1, 99999)
        self.spin_seed.setValue(42)
        opts_layout.addWidget(self.spin_seed)

        self.chk_plots = QCheckBox("Generate plots")
        self.chk_plots.setChecked(True)
        opts_layout.addWidget(self.chk_plots)

        self.chk_compare = QCheckBox("Compare vs champion")
        self.chk_compare.setChecked(True)
        opts_layout.addWidget(self.chk_compare)

        opts_layout.addStretch()
        layout.addWidget(opts_group)

        # Run button
        self.btn_run = QPushButton("RUN BENCHMARK")
        self.btn_run.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet(
            f"QPushButton {{ background-color: {C['blue']}; color: white; "
            f"border-radius: 8px; }} "
            f"QPushButton:hover {{ background-color: #4090d0; }} "
            f"QPushButton:disabled {{ background-color: {C['dim']}; }}"
        )
        self.btn_run.clicked.connect(self._on_run)
        layout.addWidget(self.btn_run)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("Ready")
        self.lbl_progress.setStyleSheet(f"color: {C['subtext']};")
        progress_layout.addWidget(self.lbl_progress)

        layout.addWidget(progress_group)

        # Live output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setMaximumHeight(200)
        output_layout.addWidget(self.txt_output)
        layout.addWidget(output_group, stretch=1)

    def refresh_models(self):
        """Reload available models."""
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

    def _get_selected_suite(self) -> str:
        btn = self.suite_group.checkedButton()
        if btn:
            return btn.property("suite_value")
        return "full"

    def _on_run(self):
        if self._running:
            return

        model_id = self.combo_model.currentData()
        if not model_id:
            self.txt_output.append("No model selected.")
            return

        suite = self._get_selected_suite()
        seed = self.spin_seed.value()
        plots = self.chk_plots.isChecked()
        compare = self.chk_compare.isChecked()

        self._running = True
        self.btn_run.setEnabled(False)
        self.btn_run.setText("RUNNING...")
        self.progress_bar.setValue(0)
        self.txt_output.clear()
        self.txt_output.append(f"Starting benchmark: {model_id} (suite={suite}, seed={seed})")

        # Run in background thread
        thread = threading.Thread(
            target=self._run_benchmark,
            args=(model_id, suite, seed, plots, compare),
            daemon=True,
        )
        thread.start()

    def _run_benchmark(self, model_id, suite, seed, plots, compare):
        """Execute benchmark in background thread."""
        import logging

        # Capture log output
        class QtLogHandler(logging.Handler):
            def __init__(self, signals):
                super().__init__()
                self.signals = signals
            def emit(self, record):
                msg = self.format(record)
                self.signals.status_update.emit(msg)

        handler = QtLogHandler(self.main.signals)
        handler.setFormatter(logging.Formatter("%(message)s"))
        bench_logger = logging.getLogger("spartusbench")
        bench_logger.addHandler(handler)

        try:
            from spartusbench.runner import BenchmarkRunner
            runner = BenchmarkRunner()
            result = runner.run(
                model_ref=model_id,
                suite=suite,
                seed=seed,
                generate_plots=plots,
                compare_vs_champion=compare,
            )
            self.main.signals.benchmark_complete.emit(result.run_id)
        except Exception as e:
            self.main.signals.benchmark_error.emit(str(e))
        finally:
            bench_logger.removeHandler(handler)
            self._running = False
            # Re-enable button from main thread
            self.btn_run.setEnabled(True)
            self.btn_run.setText("RUN BENCHMARK")
