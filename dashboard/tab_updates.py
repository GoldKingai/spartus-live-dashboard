"""Tab 7 -- Updates for the Spartus Live Trading Dashboard.

Shows current version, checks for updates, displays release notes,
and provides one-click update with progress bar and restart.

Layout::

    +-------------------------------------------------------------+
    |  CURRENT VERSION                                            |
    |  v1.1.0  |  Last checked: 2026-03-08 14:30                 |
    |                                                             |
    +-------------------------------------------------------------+
    |  UPDATE STATUS                                              |
    |  [Check for Updates]                                        |
    |                                                             |
    |  Available: v1.2.0  (released 2026-03-07)                   |
    |  +-------------------------------------------------+       |
    |  |  Release Notes:                                  |       |
    |  |  - Added new analytics tab                       |       |
    |  |  - Fixed SL trailing bug                         |       |
    |  +-------------------------------------------------+       |
    |                                                             |
    |  [Update Now]                                               |
    |  [========================================] 60%             |
    |  Pulling latest version from GitHub...                      |
    |                                                             |
    |  [Restart Dashboard]                                        |
    +-------------------------------------------------------------+
    |  UPDATE HISTORY                                             |
    |  v1.1.0  Current  (installed)                               |
    |  v1.0.0  Previous                                           |
    +-------------------------------------------------------------+

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels.
"""

import logging
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QTextEdit, QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from dashboard.theme import C

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_label(text: str, color: str = C["subtext"], bold: bool = False,
                font_size: int = 13) -> QLabel:
    lbl = QLabel(text)
    weight = "bold" if bold else "normal"
    lbl.setStyleSheet(
        f"color: {color}; font-size: {font_size}px; "
        f"font-weight: {weight}; background: transparent; border: none;"
    )
    return lbl


def _make_value(text: str, color: str = C["text"], bold: bool = True,
                font_size: int = 13) -> QLabel:
    lbl = QLabel(text)
    weight = "bold" if bold else "normal"
    lbl.setStyleSheet(
        f"color: {color}; font-size: {font_size}px; "
        f"font-weight: {weight}; background: transparent; border: none;"
    )
    return lbl


# ---------------------------------------------------------------------------
# UpdatesTab
# ---------------------------------------------------------------------------

class UpdatesTab(QWidget):
    """Tab 7: Software update management.

    Signals:
        check_requested  -- user clicked Check for Updates
        update_requested -- user clicked Update Now
        restart_requested -- user clicked Restart Dashboard
    """

    check_requested = pyqtSignal()
    update_requested = pyqtSignal()
    restart_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_check_time: Optional[str] = None
        self._update_available: bool = False
        self._update_in_progress: bool = False
        self._update_complete: bool = False
        self._update_success: bool = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- Current Version group ----
        version_group = QGroupBox("CURRENT VERSION")
        vg_layout = QHBoxLayout(version_group)
        vg_layout.setContentsMargins(16, 20, 16, 12)
        vg_layout.setSpacing(16)

        self._version_label = _make_value("v0.0.0", C["cyan"], bold=True, font_size=20)
        vg_layout.addWidget(self._version_label)

        self._version_status = _make_label("Up to date", C["green"], bold=True, font_size=13)
        vg_layout.addWidget(self._version_status)

        vg_layout.addStretch()

        self._last_check_label = _make_label("Last checked: Never", C["label"], font_size=11)
        vg_layout.addWidget(self._last_check_label)

        root.addWidget(version_group)

        # ---- Update Status group ----
        update_group = QGroupBox("UPDATE STATUS")
        ug_layout = QVBoxLayout(update_group)
        ug_layout.setContentsMargins(16, 20, 16, 12)
        ug_layout.setSpacing(10)

        # Check button row
        check_row = QHBoxLayout()
        self._btn_check = QPushButton("Check for Updates")
        self._btn_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_check.setStyleSheet(
            f"QPushButton {{ background-color: {C['surface2']}; color: {C['text']}; "
            f"font-weight: bold; border: 1px solid {C['border']}; border-radius: 4px; "
            f"padding: 8px 20px; font-size: 13px; }} "
            f"QPushButton:hover {{ background-color: {C['border']}; }} "
            f"QPushButton:disabled {{ color: {C['dim']}; }}"
        )
        self._btn_check.clicked.connect(self._on_check_clicked)
        check_row.addWidget(self._btn_check)

        self._check_status = _make_label("", C["label"], font_size=11)
        check_row.addWidget(self._check_status)
        check_row.addStretch()
        ug_layout.addLayout(check_row)

        # Available update info (hidden by default)
        self._update_info_widget = QWidget()
        self._update_info_widget.setVisible(False)
        ui_layout = QVBoxLayout(self._update_info_widget)
        ui_layout.setContentsMargins(0, 8, 0, 0)
        ui_layout.setSpacing(8)

        self._new_version_label = _make_value(
            "New version available: v0.0.0", C["green"], bold=True, font_size=15
        )
        ui_layout.addWidget(self._new_version_label)

        self._release_date_label = _make_label("Released: ---", C["label"], font_size=11)
        ui_layout.addWidget(self._release_date_label)

        notes_header = _make_label("Release Notes:", C["subtext"], bold=True, font_size=12)
        ui_layout.addWidget(notes_header)

        self._release_notes = QTextEdit()
        self._release_notes.setReadOnly(True)
        self._release_notes.setPlainText("")
        self._release_notes.setMaximumHeight(180)
        self._release_notes.setMinimumHeight(80)
        ui_layout.addWidget(self._release_notes)

        # Update button
        btn_row = QHBoxLayout()
        self._btn_update = QPushButton("Update Now")
        self._btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_update.setStyleSheet(
            f"QPushButton {{ background-color: {C['green']}; color: {C['bg']}; "
            f"font-weight: bold; border: none; border-radius: 4px; "
            f"padding: 10px 28px; font-size: 14px; }} "
            f"QPushButton:hover {{ background-color: #40e640; }} "
            f"QPushButton:pressed {{ background-color: #1fa01f; }} "
            f"QPushButton:disabled {{ background-color: {C['dim']}; color: {C['label']}; }}"
        )
        self._btn_update.clicked.connect(self._on_update_clicked)
        btn_row.addWidget(self._btn_update)
        btn_row.addStretch()
        ui_layout.addLayout(btn_row)

        ug_layout.addWidget(self._update_info_widget)

        # Progress section (hidden by default)
        self._progress_widget = QWidget()
        self._progress_widget.setVisible(False)
        pw_layout = QVBoxLayout(self._progress_widget)
        pw_layout.setContentsMargins(0, 8, 0, 0)
        pw_layout.setSpacing(6)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.setFixedHeight(22)
        pw_layout.addWidget(self._progress_bar)

        self._progress_label = _make_label("", C["yellow"], font_size=12)
        pw_layout.addWidget(self._progress_label)

        ug_layout.addWidget(self._progress_widget)

        # Result section (hidden by default)
        self._result_widget = QWidget()
        self._result_widget.setVisible(False)
        rw_layout = QVBoxLayout(self._result_widget)
        rw_layout.setContentsMargins(0, 8, 0, 0)
        rw_layout.setSpacing(8)

        self._result_label = _make_value("", C["green"], bold=True, font_size=13)
        rw_layout.addWidget(self._result_label)

        btn_restart_row = QHBoxLayout()
        self._btn_restart = QPushButton("Restart Dashboard")
        self._btn_restart.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_restart.setStyleSheet(
            f"QPushButton {{ background-color: {C['cyan']}; color: {C['bg']}; "
            f"font-weight: bold; border: none; border-radius: 4px; "
            f"padding: 10px 28px; font-size: 14px; }} "
            f"QPushButton:hover {{ background-color: #5be0ff; }} "
            f"QPushButton:pressed {{ background-color: #20b0d0; }} "
            f"QPushButton:disabled {{ background-color: {C['dim']}; color: {C['label']}; }}"
        )
        self._btn_restart.clicked.connect(self._on_restart_clicked)
        self._btn_restart.setVisible(False)
        btn_restart_row.addWidget(self._btn_restart)
        btn_restart_row.addStretch()
        rw_layout.addLayout(btn_restart_row)

        ug_layout.addWidget(self._result_widget)

        # "No updates" label (shown when check finds nothing)
        self._no_update_label = _make_label("", C["green"], bold=True, font_size=13)
        self._no_update_label.setVisible(False)
        ug_layout.addWidget(self._no_update_label)

        root.addWidget(update_group)

        # ---- Stretch ----
        root.addStretch()

    # ------------------------------------------------------------------
    # Button handlers (emit signals to orchestrator)
    # ------------------------------------------------------------------

    def _on_check_clicked(self) -> None:
        self._btn_check.setEnabled(False)
        self._check_status.setText("Checking...")
        self._no_update_label.setVisible(False)
        self.check_requested.emit()

    def _on_update_clicked(self) -> None:
        self._btn_update.setEnabled(False)
        self._btn_check.setEnabled(False)
        self._update_in_progress = True
        self._progress_widget.setVisible(True)
        self._progress_label.setText("Starting update...")
        self.update_requested.emit()

    def _on_restart_clicked(self) -> None:
        self._btn_restart.setEnabled(False)
        self.restart_requested.emit()

    # ------------------------------------------------------------------
    # Public API (called by orchestrator to update UI)
    # ------------------------------------------------------------------

    def set_current_version(self, version: str) -> None:
        """Set the displayed current version."""
        self._version_label.setText(f"v{version}")

    def set_update_available(self, update_info) -> None:
        """Show available update details."""
        self._update_available = True
        self._last_check_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._last_check_label.setText(f"Last checked: {self._last_check_time}")
        self._btn_check.setEnabled(True)
        self._check_status.setText("")

        # Update status
        self._version_status.setText("Update available")
        self._version_status.setStyleSheet(
            f"color: {C['yellow']}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Show update info
        self._new_version_label.setText(
            f"New version available: v{update_info.latest_version}"
        )
        if update_info.published_at:
            date_str = update_info.published_at[:10]
            self._release_date_label.setText(f"Released: {date_str}")
        self._release_notes.setPlainText(
            update_info.release_notes or "No release notes."
        )
        self._update_info_widget.setVisible(True)
        self._btn_update.setEnabled(True)
        self._no_update_label.setVisible(False)

    def set_no_update(self) -> None:
        """Show that no update is available (already on latest)."""
        self._update_available = False
        self._last_check_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._last_check_label.setText(f"Last checked: {self._last_check_time}")
        self._btn_check.setEnabled(True)
        self._check_status.setText("")

        self._version_status.setText("Up to date")
        self._version_status.setStyleSheet(
            f"color: {C['green']}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        self._update_info_widget.setVisible(False)
        self._no_update_label.setText("You are running the latest version.")
        self._no_update_label.setVisible(True)

    def set_check_failed(self) -> None:
        """Show that the update check failed (network error etc.)."""
        self._last_check_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._last_check_label.setText(f"Last checked: {self._last_check_time}")
        self._btn_check.setEnabled(True)
        self._check_status.setText("Check failed (no internet?)")
        self._check_status.setStyleSheet(
            f"color: {C['red']}; font-size: 11px; background: transparent; border: none;"
        )

    def set_progress(self, message: str) -> None:
        """Update the progress bar status text during an update."""
        self._progress_widget.setVisible(True)
        self._progress_label.setText(message)

    def set_complete(self, success: bool, message: str) -> None:
        """Show update result."""
        self._update_in_progress = False
        self._update_complete = True
        self._update_success = success
        self._progress_widget.setVisible(False)
        self._result_widget.setVisible(True)

        if success:
            self._result_label.setText(f"Update successful: {message}")
            self._result_label.setStyleSheet(
                f"color: {C['green']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
            self._btn_restart.setVisible(True)
            self._btn_update.setVisible(False)

            self._version_status.setText("Restart required")
            self._version_status.setStyleSheet(
                f"color: {C['cyan']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
        else:
            self._result_label.setText(f"Update failed: {message}")
            self._result_label.setStyleSheet(
                f"color: {C['red']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
            self._btn_update.setEnabled(True)
            self._btn_check.setEnabled(True)

    def update_data(self, data: dict) -> None:
        """Called every second when this tab is active.

        Currently only refreshes the version label (in case it changed).
        The heavy lifting is done by the signal-driven methods above.
        """
        version = data.get("current_version", "")
        if version:
            self._version_label.setText(f"v{version}")
