"""Update notification dialog for Spartus Live Dashboard.

Shows available update info and lets the user apply or skip.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QProgressBar, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class UpdateDialog(QDialog):
    """Modal dialog showing update availability and progress."""

    update_requested = pyqtSignal()
    restart_requested = pyqtSignal()

    def __init__(self, update_info, parent=None):
        super().__init__(parent)
        self.update_info = update_info
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Update Available")
        self.setMinimumWidth(500)
        self.setMinimumHeight(350)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a2e; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                background-color: #16213e; color: #e0e0e0;
                border: 1px solid #0f3460; padding: 8px 16px;
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #0f3460; }
            QPushButton#updateBtn {
                background-color: #00b894; color: #1a1a2e;
                border: none; font-size: 14px;
            }
            QPushButton#updateBtn:hover { background-color: #00cec9; }
            QPushButton#updateBtn:disabled {
                background-color: #636e72; color: #b2bec3;
            }
            QTextEdit {
                background-color: #16213e; color: #b2bec3;
                border: 1px solid #0f3460; border-radius: 4px;
                padding: 8px;
            }
            QProgressBar {
                background-color: #16213e; border: 1px solid #0f3460;
                border-radius: 4px; text-align: center; color: #e0e0e0;
            }
            QProgressBar::chunk { background-color: #00b894; border-radius: 3px; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("A new version is available!")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #00b894;")
        layout.addWidget(header)

        # Version info
        info = self.update_info
        version_label = QLabel(
            f"Current version: <b>v{info.current_version}</b> &nbsp;&rarr;&nbsp; "
            f"Latest version: <b style='color:#00b894;'>v{info.latest_version}</b>"
        )
        version_label.setFont(QFont("Segoe UI", 11))
        layout.addWidget(version_label)

        if info.published_at:
            date_str = info.published_at[:10]
            date_label = QLabel(f"Released: {date_str}")
            date_label.setStyleSheet("color: #636e72; font-size: 10px;")
            layout.addWidget(date_label)

        # Release notes
        notes_label = QLabel("Release Notes:")
        notes_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(notes_label)

        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setPlainText(info.release_notes or "No release notes.")
        self.notes_text.setMaximumHeight(150)
        layout.addWidget(self.notes_text)

        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label (hidden initially)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #fdcb6e; font-size: 11px;")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.skip_btn = QPushButton("Skip This Version")
        self.skip_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.skip_btn)

        self.later_btn = QPushButton("Remind Me Later")
        self.later_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.later_btn)

        btn_layout.addStretch()

        self.update_btn = QPushButton("Update Now")
        self.update_btn.setObjectName("updateBtn")
        self.update_btn.clicked.connect(self._on_update_clicked)
        btn_layout.addWidget(self.update_btn)

        # Restart button (hidden until update complete)
        self.restart_btn = QPushButton("Restart Dashboard")
        self.restart_btn.setObjectName("updateBtn")
        self.restart_btn.setVisible(False)
        self.restart_btn.clicked.connect(self._on_restart_clicked)
        btn_layout.addWidget(self.restart_btn)

        layout.addLayout(btn_layout)

    def _on_update_clicked(self):
        """User clicked Update Now."""
        self.update_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.later_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.status_label.setText("Starting update...")
        self.update_requested.emit()

    def _on_restart_clicked(self):
        """User clicked Restart."""
        self.restart_requested.emit()
        self.accept()

    def set_progress(self, message: str):
        """Update the progress status text."""
        self.status_label.setText(message)

    def set_complete(self, success: bool, message: str):
        """Update completed — show result."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)

        if success:
            self.status_label.setStyleSheet("color: #00b894; font-size: 11px;")
            self.restart_btn.setVisible(True)
            self.update_btn.setVisible(False)
        else:
            self.status_label.setStyleSheet("color: #e17055; font-size: 11px;")
            self.update_btn.setEnabled(True)
            self.skip_btn.setEnabled(True)
            self.later_btn.setEnabled(True)


class UpdateNotificationBar(QLabel):
    """Small notification bar that sits at the top of the main window.

    Shows "Update available: v1.2.0 — Click to update" with a dismiss button.
    """

    clicked = pyqtSignal()

    def __init__(self, version: str, parent=None):
        super().__init__(parent)
        self.setText(
            f"  Update available: <b>v{version}</b> &nbsp;&mdash;&nbsp; "
            f"<a style='color:#00b894;' href='#'>Click to update</a>"
        )
        self.setStyleSheet(
            "background-color: #0f3460; color: #e0e0e0; padding: 6px 12px; "
            "border-bottom: 2px solid #00b894; font-size: 11px;"
        )
        self.setOpenExternalLinks(False)
        self.linkActivated.connect(lambda: self.clicked.emit())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
