"""Tab 2 -- Performance for the Spartus Live Trading Dashboard.

Shows the balance/equity curve, rolling performance metrics, and
full trade history in a scrollable table.

Layout::

    +------------------------------------------------------------------+
    |  BALANCE CHART (pyqtgraph line plot, ~50% height)                 |
    |  +------------------------------------------------------------+  |
    |  |     /-\    /---\                                            |  |
    |  |  __/   \__/     \___/-\_____/------                        |  |
    |  +------------------------------------------------------------+  |
    +----------------------+-------------------------------------------+
    |  ROLLING METRICS     |  TRADE HISTORY TABLE                      |
    |  Sharpe (30d): 0.85  |  #  Time   Side  Lots  P/L  Reason       |
    |  Win Rate: 54%       |  47 14:25  LONG  0.02  +5.0 TP_HIT       |
    |  Profit Factor: 1.62 |  46 13:10  SHORT 0.01  -2.3 SL_HIT       |
    |  Avg Trade: +$1.23   |  45 12:40  LONG  0.02  +3.1 AGENT        |
    |  Max DD: 4.2%        |  ...                                      |
    |  Total Trades: 47    |                                           |
    |  Best: +$8.50        |                                           |
    |  Worst: -$4.20       |                                           |
    +----------------------+-------------------------------------------+

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels -- NEVER dark gray on dark backgrounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QGroupBox, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg
import numpy as np

from dashboard.theme import C
from dashboard import currency


# ---------------------------------------------------------------------------
# Helper: styled label (matches Tab 1 convention)
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


# ---------------------------------------------------------------------------
# PerformanceTab
# ---------------------------------------------------------------------------

class PerformanceTab(QWidget):
    """Tab 2: Performance overview.

    Shows a balance/equity curve chart (top half), rolling performance
    metrics (bottom-left), and a scrollable trade history table
    (bottom-right).

    Call ``update_data(data)`` each tick with the latest state dict.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ----- Root layout -----
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ----- Top: Balance Chart (~50% height) -----
        root.addWidget(self._build_balance_chart(), stretch=3)

        # ----- Bottom: Metrics (left ~30%) + Trade History (right ~70%) -----
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {C['border']}; width: 2px; }}"
        )
        bottom_splitter.addWidget(self._build_metrics_box())
        bottom_splitter.addWidget(self._build_trade_history_box())
        bottom_splitter.setStretchFactor(0, 3)   # ~30%
        bottom_splitter.setStretchFactor(1, 7)   # ~70%

        root.addWidget(bottom_splitter, stretch=2)

    # ==================================================================
    # Section builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. BALANCE CHART
    # ------------------------------------------------------------------

    def _build_balance_chart(self) -> QGroupBox:
        """Pyqtgraph line chart for the balance/equity curve."""
        box = QGroupBox("BALANCE CHART")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(0)

        self._chart = pg.PlotWidget()
        self._chart.setBackground(C["bg"])
        self._chart.showGrid(x=True, y=True, alpha=0.15)
        self._chart.setLabel("left", f"Balance ({currency.sym()})", color=C["subtext"])
        self._chart.setLabel("bottom", "Bar #", color=C["subtext"])
        self._chart.getAxis("left").setTextPen(C["subtext"])
        self._chart.getAxis("bottom").setTextPen(C["subtext"])
        self._chart.enableAutoRange()

        # The equity curve line
        self._balance_curve = self._chart.plot(
            [], [],
            pen=pg.mkPen(color=C["green"], width=2),
            name="Balance",
        )

        layout.addWidget(self._chart)
        return box

    # ------------------------------------------------------------------
    # 2. ROLLING METRICS
    # ------------------------------------------------------------------

    def _build_metrics_box(self) -> QGroupBox:
        """Panel of key rolling performance statistics."""
        box = QGroupBox("ROLLING METRICS")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        # Define metric rows: (label_text, attribute_name, initial_value)
        self._metric_fields = [
            ("Sharpe (30d):", "_lbl_sharpe",       "--"),
            ("Win Rate:",     "_lbl_win_rate",     "--"),
            ("Profit Factor:", "_lbl_pf",          "--"),
            ("Avg Trade:",    "_lbl_avg_trade",    "--"),
            ("Max DD:",       "_lbl_max_dd",       "--"),
            ("Total Trades:", "_lbl_total_trades", "--"),
            ("Best:",         "_lbl_best",         "--"),
            ("Worst:",        "_lbl_worst",        "--"),
        ]

        for row, (label_text, attr_name, initial) in enumerate(self._metric_fields):
            lbl = _make_label(label_text, C["subtext"])
            layout.addWidget(lbl, row, 0)

            val = _make_label(initial, C["text"], bold=True)
            setattr(self, attr_name, val)
            layout.addWidget(val, row, 1)

        # Push content upwards
        layout.setRowStretch(len(self._metric_fields), 1)

        return box

    # ------------------------------------------------------------------
    # 3. TRADE HISTORY TABLE
    # ------------------------------------------------------------------

    def _build_trade_history_box(self) -> QGroupBox:
        """Scrollable trade history table with colored Side and P/L columns."""
        box = QGroupBox("TRADE HISTORY")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(0)

        columns = ["#", "Time", "Side", "Lots", "P/L", "Reason"]
        self._trade_table = QTableWidget(0, len(columns))
        self._trade_table.setHorizontalHeaderLabels(columns)
        self._trade_table.setAlternatingRowColors(True)
        self._trade_table.verticalHeader().setVisible(False)
        self._trade_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._trade_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        # Disable auto-scroll so the user can scroll manually
        self._trade_table.setAutoScroll(False)

        # Column resize modes
        header = self._trade_table.horizontalHeader()
        # Fixed widths for compact columns, stretch for Reason
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # #
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Time
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Side
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Lots
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # P/L
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)           # Reason

        # Dark theme styling (supplements the global stylesheet)
        self._trade_table.setStyleSheet(
            f"alternate-background-color: {C['surface2']}; "
            f"background-color: {C['surface']}; "
            f"color: {C['text']}; "
            f"gridline-color: {C['border']};"
        )

        layout.addWidget(self._trade_table)
        return box

    # ==================================================================
    # update_data -- called every tick with live state
    # ==================================================================

    def update_data(self, data: dict) -> None:
        """Refresh all panels with the latest data dict.

        Parameters
        ----------
        data : dict
            Expected keys:

            balance_history : list[float]
                Sequential balance values for the equity curve chart.
            metrics : dict
                sharpe (float), win_rate (float), pf (float),
                avg_trade (float), max_dd (float), total_trades (int),
                best (float), worst (float)
            trades : list[dict]
                Each dict: id (int), time (str), side (str "LONG"/"SHORT"),
                lots (float), pnl (float), reason (str)
        """
        if not data:
            return

        self._update_balance_chart(data.get("balance_history", []))
        self._update_metrics(data.get("metrics", {}))
        self._update_trade_history(data.get("trades", []))

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_balance_chart(self, balance_history: list) -> None:
        """Redraw the balance/equity curve."""
        if not balance_history:
            return

        x = list(range(len(balance_history)))
        self._balance_curve.setData(x, balance_history)

    def _update_metrics(self, metrics: dict) -> None:
        """Update the rolling metrics panel with color coding."""
        if not metrics:
            return

        # --- Sharpe (30d) ---
        sharpe = metrics.get("sharpe")
        if sharpe is not None:
            sharpe_color = (
                C["green"] if sharpe >= 1.0
                else C["text"] if sharpe >= 0.0
                else C["red"]
            )
            self._lbl_sharpe.setText(f"{sharpe:.2f}")
            self._lbl_sharpe.setStyleSheet(
                f"color: {sharpe_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Win Rate ---
        win_rate = metrics.get("win_rate")
        if win_rate is not None:
            wr_color = (
                C["green"] if win_rate >= 55.0
                else C["text"] if win_rate >= 45.0
                else C["red"]
            )
            self._lbl_win_rate.setText(f"{win_rate:.0f}%")
            self._lbl_win_rate.setStyleSheet(
                f"color: {wr_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Profit Factor ---
        pf = metrics.get("pf")
        if pf is not None:
            pf_color = (
                C["green"] if pf >= 1.5
                else C["text"] if pf >= 1.0
                else C["red"]
            )
            self._lbl_pf.setText(f"{pf:.2f}")
            self._lbl_pf.setStyleSheet(
                f"color: {pf_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Average Trade P/L ---
        avg_trade = metrics.get("avg_trade")
        if avg_trade is not None:
            avg_color = C["green"] if avg_trade >= 0 else C["red"]
            self._lbl_avg_trade.setText(currency.fmt_signed(avg_trade))
            self._lbl_avg_trade.setStyleSheet(
                f"color: {avg_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Max Drawdown ---
        max_dd = metrics.get("max_dd")
        if max_dd is not None:
            dd_color = (
                C["red"] if max_dd > 5.0
                else C["yellow"] if max_dd > 2.5
                else C["text"]
            )
            self._lbl_max_dd.setText(f"{max_dd:.1f}%")
            self._lbl_max_dd.setStyleSheet(
                f"color: {dd_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Total Trades ---
        total_trades = metrics.get("total_trades")
        if total_trades is not None:
            self._lbl_total_trades.setText(str(total_trades))
            self._lbl_total_trades.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Best Trade ---
        best = metrics.get("best")
        if best is not None:
            self._lbl_best.setText(currency.fmt_signed(best))
            self._lbl_best.setStyleSheet(
                f"color: {C['green']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # --- Worst Trade ---
        worst = metrics.get("worst")
        if worst is not None:
            self._lbl_worst.setText(currency.fmt_signed(worst))
            self._lbl_worst.setStyleSheet(
                f"color: {C['red']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

    def _update_trade_history(self, trades: list) -> None:
        """Rebuild the trade history table from scratch.

        Trades are expected as a list of dicts, newest first.
        Each dict: id, time, side, lots, pnl, reason.
        """
        if trades is None:
            return

        # Remember current scroll position so we don't auto-scroll
        scrollbar = self._trade_table.verticalScrollBar()
        scroll_pos = scrollbar.value()

        self._trade_table.setRowCount(0)
        self._trade_table.setRowCount(len(trades))

        for row_idx, trade in enumerate(trades):
            trade_id = trade.get("id", "")
            time_str = trade.get("time", "")
            side = trade.get("side", "")
            lots = trade.get("lots", 0.0)
            pnl = trade.get("pnl", 0.0)
            reason = trade.get("reason", "")

            # Column 0: Trade # (ID)
            item_id = QTableWidgetItem(str(trade_id))
            item_id.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            item_id.setForeground(QColor(C["text"]))
            self._trade_table.setItem(row_idx, 0, item_id)

            # Column 1: Time
            item_time = QTableWidgetItem(str(time_str))
            item_time.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            item_time.setForeground(QColor(C["text"]))
            self._trade_table.setItem(row_idx, 1, item_time)

            # Column 2: Side (colored: LONG = green, SHORT = red)
            item_side = QTableWidgetItem(str(side))
            item_side.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            if side.upper() == "LONG":
                item_side.setForeground(QColor(C["green"]))
            elif side.upper() == "SHORT":
                item_side.setForeground(QColor(C["red"]))
            else:
                item_side.setForeground(QColor(C["text"]))
            font_side = item_side.font()
            font_side.setBold(True)
            item_side.setFont(font_side)
            self._trade_table.setItem(row_idx, 2, item_side)

            # Column 3: Lots
            item_lots = QTableWidgetItem(f"{lots:.2f}")
            item_lots.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            item_lots.setForeground(QColor(C["text"]))
            self._trade_table.setItem(row_idx, 3, item_lots)

            # Column 4: P/L (colored: positive = green, negative = red)
            item_pnl = QTableWidgetItem(currency.fmt_signed(pnl))
            item_pnl.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            pnl_color = C["green"] if pnl >= 0 else C["red"]
            item_pnl.setForeground(QColor(pnl_color))
            font_pnl = item_pnl.font()
            font_pnl.setBold(True)
            item_pnl.setFont(font_pnl)
            self._trade_table.setItem(row_idx, 4, item_pnl)

            # Column 5: Reason
            item_reason = QTableWidgetItem(str(reason))
            item_reason.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            item_reason.setForeground(QColor(C["subtext"]))
            self._trade_table.setItem(row_idx, 5, item_reason)

        # Restore scroll position (prevents auto-scroll to top on rebuild)
        scrollbar.setValue(scroll_pos)
