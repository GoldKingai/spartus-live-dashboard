"""Shared UI components for the Spartus Live Dashboard.

Reusable widgets used across all 6 tabs.  Every widget follows the dark
theme rules: bright white (#e6edf3) or light gray (#b1bac4) text on
dark backgrounds -- NEVER dark gray (#656d76) on dark.
"""

from PyQt6.QtWidgets import (
    QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
    QWidget, QFrame, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QTextEdit,
    QMenu, QApplication,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QKeySequence, QAction
import pyqtgraph as pg
import numpy as np

from dashboard.theme import C
from dashboard import currency


# ---------------------------------------------------------------------------
# StatusIndicator
# ---------------------------------------------------------------------------

class StatusIndicator(QWidget):
    """Colored dot + text label showing connection / system status.

    Usage::

        indicator = StatusIndicator("MT5 Connection")
        indicator.set_status("Connected", C["green"])
    """

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Colored dot (Unicode filled circle)
        self._dot = QLabel("\u25cf")
        self._dot.setStyleSheet(f"color: {C['label']}; font-size: 14px;")
        self._dot.setFixedWidth(18)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._dot)

        # Label
        self._label = QLabel(label_text)
        self._label.setStyleSheet(f"color: {C['subtext']}; font-size: 13px;")
        layout.addWidget(self._label)

        # Status text
        self._status = QLabel("--")
        self._status.setStyleSheet(f"color: {C['text']}; font-size: 13px;")
        layout.addWidget(self._status)

        layout.addStretch()

    def set_status(self, status_text: str, color: str) -> None:
        """Update the dot color and status text."""
        self._dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._status.setText(status_text)
        self._status.setStyleSheet(f"color: {color}; font-size: 13px;")


# ---------------------------------------------------------------------------
# MetricCard
# ---------------------------------------------------------------------------

class MetricCard(QWidget):
    """Card displaying a label and a large value.

    Usage::

        card = MetricCard("Balance")
        card.set_value("$12,450.00", C["green"])
    """

    def __init__(self, label: str, initial_value: str = "--", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        # Label (subtext color)
        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {C['subtext']}; font-size: 12px;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._label)

        # Value (bright white, large)
        self._value = QLabel(initial_value)
        value_font = QFont()
        value_font.setPointSize(18)
        value_font.setBold(True)
        self._value.setFont(value_font)
        self._value.setStyleSheet(f"color: {C['text']};")
        self._value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._value)

        # Panel styling
        self.setStyleSheet(
            f"background-color: {C['surface']}; "
            f"border: 1px solid {C['border']}; "
            f"border-radius: 6px;"
        )

    def set_value(self, value: str, color: str | None = None) -> None:
        """Update displayed value.  Optional *color* override."""
        self._value.setText(str(value))
        c = color if color else C["text"]
        self._value.setStyleSheet(f"color: {c};")


# ---------------------------------------------------------------------------
# ActionLogWidget
# ---------------------------------------------------------------------------

class ActionLogWidget(QWidget):
    """Scrolling text area showing recent AI decisions.

    New entries appear at the top.  Oldest entries beyond *max_lines*
    are automatically discarded.
    """

    def __init__(self, max_lines: int = 20, parent=None):
        super().__init__(parent)
        self._max_lines = max_lines
        self._entries: list[str] = []
        self._last_html: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Cascadia Code", 11))
        self._text.setStyleSheet(
            f"background-color: {C['surface']}; "
            f"color: {C['text']}; "
            f"border: 1px solid {C['border']}; "
            f"border-radius: 4px; "
            f"padding: 4px;"
        )
        layout.addWidget(self._text)

    def add_entry(
        self, timestamp: str, text: str, color: str | None = None
    ) -> None:
        """Add a colored entry at the top of the log."""
        c = color if color else C["text"]
        html = (
            f'<span style="color:{C["subtext"]}">{timestamp}</span> '
            f'<span style="color:{c}">{text}</span>'
        )
        self._entries.insert(0, html)
        if len(self._entries) > self._max_lines:
            self._entries = self._entries[: self._max_lines]

        new_html = "<br>".join(self._entries)
        if new_html != self._last_html:
            self._text.setHtml(new_html)
            self._last_html = new_html

    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
        self._last_html = ""
        self._text.clear()


# ---------------------------------------------------------------------------
# CopyableTableWidget
# ---------------------------------------------------------------------------

class CopyableTableWidget(QTableWidget):
    """QTableWidget with Ctrl+A / Ctrl+C and right-click Copy support.

    Drop-in replacement for QTableWidget in read-only dashboard panels.
    """

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_selection()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            self.selectAll()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self._copy_selection)
        menu.addAction(copy_action)

        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.selectAll)
        menu.addAction(select_all_action)

        menu.exec(event.globalPos())

    def _copy_selection(self) -> None:
        """Copy selected cells to clipboard as tab-separated text."""
        selection = self.selectedRanges()
        if not selection:
            return

        rows: list[str] = []
        for sel_range in selection:
            for row in range(sel_range.topRow(), sel_range.bottomRow() + 1):
                cells: list[str] = []
                for col in range(sel_range.leftColumn(), sel_range.rightColumn() + 1):
                    item = self.item(row, col)
                    cells.append(item.text() if item else "")
                rows.append("\t".join(cells))

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(rows))


# ---------------------------------------------------------------------------
# TradeTable
# ---------------------------------------------------------------------------

class TradeTable(CopyableTableWidget):
    """Table widget for displaying trade history.

    Rows are inserted at the top so the most recent trade is always
    visible without scrolling.  Inherits copy/select from CopyableTableWidget.
    """

    def __init__(self, columns: list[str], parent=None):
        super().__init__(0, len(columns), parent)
        self.setHorizontalHeaderLabels(columns)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        # Auto-resize columns to contents, then stretch the last section
        header = self.horizontalHeader()
        for i in range(len(columns)):
            header.setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
            )
        if len(columns) > 0:
            header.setSectionResizeMode(
                len(columns) - 1, QHeaderView.ResizeMode.Stretch
            )

        self.setStyleSheet(
            f"alternate-background-color: {C['surface2']}; "
            f"background-color: {C['surface']}; "
            f"color: {C['text']}; "
            f"gridline-color: {C['border']};"
        )

    def add_trade(self, values: list) -> None:
        """Insert a row at position 0 with the given cell values."""
        self.insertRow(0)
        for col, val in enumerate(values):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            # Color profit/loss cells green/red if the value looks numeric
            try:
                numeric = float(str(val).replace(",", "").replace(currency.sym(), ""))
                if numeric > 0:
                    item.setForeground(QColor(C["green"]))
                elif numeric < 0:
                    item.setForeground(QColor(C["red"]))
            except (ValueError, TypeError):
                pass
            self.setItem(0, col, item)

    def clear(self) -> None:
        """Remove all rows."""
        self.setRowCount(0)


# ---------------------------------------------------------------------------
# BalanceChart
# ---------------------------------------------------------------------------

class BalanceChart(pg.PlotWidget):
    """Pyqtgraph PlotWidget for equity / balance curve.

    Dark background, green line, subtle grid.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(C["bg"])
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("left", f"Equity ({currency.sym()})", color=C["subtext"])
        self.setLabel("bottom", "Time", color=C["subtext"])
        self.getAxis("left").setTextPen(C["subtext"])
        self.getAxis("bottom").setTextPen(C["subtext"])

        self._curve = self.plot(
            [], [],
            pen=pg.mkPen(color=C["green"], width=2),
        )

    def update_data(self, timestamps: list, values: list) -> None:
        """Replace the line data with new *timestamps* and *values*."""
        self._curve.setData(timestamps, values)


# ---------------------------------------------------------------------------
# HistogramWidget
# ---------------------------------------------------------------------------

class HistogramWidget(pg.PlotWidget):
    """Pyqtgraph plot for action distributions.

    Shows a bar histogram plus mean/std text overlay.
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setBackground(C["bg"])
        self.showGrid(x=True, y=True, alpha=0.15)
        if title:
            self.setTitle(title, color=C["cyan"], size="11pt")
        self.getAxis("left").setTextPen(C["subtext"])
        self.getAxis("bottom").setTextPen(C["subtext"])

        self._bars: pg.BarGraphItem | None = None

        # Text overlay for mean / std
        self._stats_label = pg.TextItem("", color=C["text"], anchor=(0, 0))
        self._stats_label.setPos(0, 0)
        self.addItem(self._stats_label)

    def update_histogram(self, values: list | np.ndarray, n_bins: int = 20) -> None:
        """Recompute and display histogram bars with mean/std overlay."""
        if len(values) == 0:
            return

        arr = np.asarray(values, dtype=np.float64)
        counts, edges = np.histogram(arr, bins=n_bins)

        # Remove old bars
        if self._bars is not None:
            self.removeItem(self._bars)

        width = (edges[1] - edges[0]) * 0.85
        self._bars = pg.BarGraphItem(
            x=(edges[:-1] + edges[1:]) / 2,
            height=counts,
            width=width,
            brush=pg.mkBrush(C["cyan"] + "99"),  # semi-transparent
            pen=pg.mkPen(C["cyan"], width=0.5),
        )
        self.addItem(self._bars)

        # Update stats overlay
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        self._stats_label.setText(f"mean={mean:.3f}  std={std:.3f}")
        self._stats_label.setPos(edges[0], max(counts) * 0.95)


# ---------------------------------------------------------------------------
# SectionHeader
# ---------------------------------------------------------------------------

class SectionHeader(QLabel):
    """Styled section header label.

    Cyan color, bold, slightly larger font for visual grouping.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        self.setFont(font)
        self.setStyleSheet(
            f"color: {C['cyan']}; "
            f"padding: 4px 0px; "
            f"border: none; "
            f"background: transparent;"
        )
