"""Tab 3 -- Trade Journal for the Spartus Live Trading Dashboard.

Shows lesson summary (left), a clickable trade table (left-bottom),
and detailed trade info for the selected trade (right).

Layout::

    +---------------------------+----------------------------------------+
    |  LESSON SUMMARY           |  TRADE DETAIL                          |
    |  GOOD_TRADE: 18           |  Trade #47 -- GOOD_TRADE               |
    |  WRONG_DIRECTION: 8       |  Entry: LONG @ $2,651.30               |
    |  CORRECT_DIR_EARLY: 6     |  Exit: TP_HIT @ $2,658.00              |
    |  BAD_TIMING: 4            |  P/L: +$5.00 (0.40%)                   |
    |  WHIPSAW: 3               |  Hold: 45 min (9 bars)                 |
    |  HELD_TOO_LONG: 2         |  SL Quality: TRAILED (+$2.10)          |
    |  ...                      |  Pattern: rsi=4/trend=5/london/vol=5   |
    +---------------------------+  Pattern W/R: 62% (13W / 8L)           |
    |  TRADE LIST (clickable)   |                                        |
    |  #47  LONG  +$5.00  GOOD  |                                        |
    |  #46  SHORT -$2.30  WRONG |                                        |
    +---------------------------+----------------------------------------+

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

from dashboard.theme import C
from dashboard.widgets import CopyableTableWidget
from dashboard import currency


# ---------------------------------------------------------------------------
# Lesson type -> color mapping
# ---------------------------------------------------------------------------

LESSON_COLORS = {
    "GOOD_TRADE":        C["green"],
    "WRONG_DIRECTION":   C["red"],
    "BAD_TIMING":        C["peach"],       # orange/yellow
    "CORRECT_DIR_EARLY": C["cyan"],
    "WHIPSAW":           C["red"],
    "HELD_TOO_LONG":     C["yellow"],
    "SMALL_WIN":         "#88cc88",        # light green
    "NEUTRAL":           C["text"],
}


# ---------------------------------------------------------------------------
# Helper: styled label (matches Tab 1 / Tab 2 convention)
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
# TradeJournalTab
# ---------------------------------------------------------------------------

class TradeJournalTab(QWidget):
    """Tab 3: Trade Journal.

    Left side shows a lesson summary (counts by lesson type) and a
    clickable trade table.  Right side shows full detail for the
    selected trade.

    Call ``update_data(data)`` each tick with the latest state dict.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Currently selected trade (for detail pane)
        self._selected_trade: dict | None = None
        # Full trade list (kept for click lookup)
        self._trades: list[dict] = []

        # ----- Root layout with horizontal splitter -----
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {C['border']}; width: 2px; }}"
        )

        # Left pane: Lesson Summary (top) + Trade List (bottom)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._build_lesson_summary_box(), stretch=1)
        left_layout.addWidget(self._build_trade_list_box(), stretch=2)

        # Right pane: Trade Detail
        right_pane = self._build_trade_detail_box()

        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 4)   # ~40%
        splitter.setStretchFactor(1, 6)   # ~60%

        root.addWidget(splitter)

    # ==================================================================
    # Section builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. LESSON SUMMARY
    # ------------------------------------------------------------------

    def _build_lesson_summary_box(self) -> QGroupBox:
        """Grid of lesson types with colored counts."""
        box = QGroupBox("LESSON SUMMARY")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(6)

        # Build a label pair for each lesson type
        self._lesson_labels: dict[str, QLabel] = {}

        lesson_types = [
            "GOOD_TRADE",
            "WRONG_DIRECTION",
            "CORRECT_DIR_EARLY",
            "BAD_TIMING",
            "WHIPSAW",
            "HELD_TOO_LONG",
            "SMALL_WIN",
            "NEUTRAL",
        ]

        for row, lesson in enumerate(lesson_types):
            color = LESSON_COLORS.get(lesson, C["text"])

            # Lesson name label
            name_lbl = _make_label(f"{lesson}:", color, bold=True)
            layout.addWidget(name_lbl, row, 0)

            # Count label
            count_lbl = _make_label("0", color, bold=True)
            count_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._lesson_labels[lesson] = count_lbl
            layout.addWidget(count_lbl, row, 1)

        layout.setRowStretch(len(lesson_types), 1)
        return box

    # ------------------------------------------------------------------
    # 2. TRADE LIST (clickable table)
    # ------------------------------------------------------------------

    def _build_trade_list_box(self) -> QGroupBox:
        """Scrollable table of all trades, clickable to show detail."""
        box = QGroupBox("TRADE LIST")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(0)

        columns = ["#", "Side", "P/L", "Lesson"]
        self._trade_table = CopyableTableWidget(0, len(columns))
        self._trade_table.setHorizontalHeaderLabels(columns)
        self._trade_table.setAlternatingRowColors(True)
        self._trade_table.verticalHeader().setVisible(False)
        self._trade_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._trade_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._trade_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

        # Column resize
        header = self._trade_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self._trade_table.setStyleSheet(
            f"alternate-background-color: {C['surface2']}; "
            f"background-color: {C['surface']}; "
            f"color: {C['text']}; "
            f"gridline-color: {C['border']};"
        )

        # Connect row click
        self._trade_table.cellClicked.connect(self._on_trade_clicked)

        layout.addWidget(self._trade_table)
        return box

    # ------------------------------------------------------------------
    # 3. TRADE DETAIL
    # ------------------------------------------------------------------

    def _build_trade_detail_box(self) -> QGroupBox:
        """Right-side panel showing detailed info for the selected trade."""
        box = QGroupBox("TRADE DETAIL")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        # Row 0: Trade ID + Lesson
        layout.addWidget(_make_label("Trade:"), 0, 0)
        self._lbl_trade_id = _make_label("--", C["text"], bold=True, font_size=15)
        layout.addWidget(self._lbl_trade_id, 0, 1)

        # Row 1: Entry side + price
        layout.addWidget(_make_label("Entry:"), 1, 0)
        self._lbl_entry = _make_label("--", C["text"])
        layout.addWidget(self._lbl_entry, 1, 1)

        # Row 2: Exit reason + price
        layout.addWidget(_make_label("Exit:"), 2, 0)
        self._lbl_exit = _make_label("--", C["text"])
        layout.addWidget(self._lbl_exit, 2, 1)

        # Row 3: P/L
        layout.addWidget(_make_label("P/L:"), 3, 0)
        self._lbl_pnl = _make_label("--", C["text"], bold=True)
        layout.addWidget(self._lbl_pnl, 3, 1)

        # Row 4: Hold duration
        layout.addWidget(_make_label("Hold:"), 4, 0)
        self._lbl_hold = _make_label("--", C["text"])
        layout.addWidget(self._lbl_hold, 4, 1)

        # Row 5: SL Quality
        layout.addWidget(_make_label("SL Quality:"), 5, 0)
        self._lbl_sl_quality = _make_label("--", C["text"])
        layout.addWidget(self._lbl_sl_quality, 5, 1)

        # Row 6: Entry conditions / pattern
        layout.addWidget(_make_label("Pattern:"), 6, 0)
        self._lbl_pattern = _make_label("--", C["text"])
        self._lbl_pattern.setWordWrap(True)
        layout.addWidget(self._lbl_pattern, 6, 1)

        # Row 7: Pattern win rate
        layout.addWidget(_make_label("Pattern W/R:"), 7, 0)
        self._lbl_pattern_wr = _make_label("--", C["text"])
        layout.addWidget(self._lbl_pattern_wr, 7, 1)

        # Push content up
        layout.setRowStretch(8, 1)
        layout.setColumnStretch(1, 1)

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

            lesson_summary : dict
                Mapping of lesson type (str) -> count (int).
                e.g. {"GOOD_TRADE": 18, "WRONG_DIRECTION": 8, ...}
            trades : list[dict]
                Each dict: id (int), side (str), entry_price (float),
                exit_price (float), exit_reason (str), pnl (float),
                pnl_pct (float), hold_min (int), hold_bars (int),
                lesson (str), sl_quality (str), sl_saved (float),
                pattern (str), pattern_wins (int), pattern_losses (int),
                rsi (int), trend (int), session (str), volatility (int)
        """
        if not data:
            return

        self._update_lesson_summary(data.get("lesson_summary", {}))
        self._update_trade_list(data.get("trades", []))

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_lesson_summary(self, summary: dict) -> None:
        """Update lesson type counts."""
        if not summary:
            return

        for lesson_type, count_label in self._lesson_labels.items():
            count = summary.get(lesson_type, 0)
            color = LESSON_COLORS.get(lesson_type, C["text"])
            count_label.setText(str(count))
            count_label.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

    def _update_trade_list(self, trades: list) -> None:
        """Rebuild the trade list table and preserve selection if possible."""
        if trades is None:
            return

        self._trades = trades

        # Remember selected trade ID so we can re-select after rebuild
        selected_id = None
        if self._selected_trade is not None:
            selected_id = self._selected_trade.get("id")

        # Remember scroll position
        scrollbar = self._trade_table.verticalScrollBar()
        scroll_pos = scrollbar.value()

        self._trade_table.setRowCount(0)
        self._trade_table.setRowCount(len(trades))

        reselect_row = -1

        for row_idx, trade in enumerate(trades):
            trade_id = trade.get("id", "")
            side = trade.get("side", "")
            pnl = trade.get("pnl", 0.0)
            lesson = trade.get("lesson", "")

            # Check if this was the previously selected row
            if trade_id == selected_id:
                reselect_row = row_idx

            # Column 0: Trade #
            item_id = QTableWidgetItem(str(trade_id))
            item_id.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            item_id.setForeground(QColor(C["text"]))
            self._trade_table.setItem(row_idx, 0, item_id)

            # Column 1: Side (colored)
            item_side = QTableWidgetItem(side)
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
            self._trade_table.setItem(row_idx, 1, item_side)

            # Column 2: P/L (colored)
            item_pnl = QTableWidgetItem(currency.fmt_signed(pnl))
            item_pnl.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            pnl_color = C["green"] if pnl >= 0 else C["red"]
            item_pnl.setForeground(QColor(pnl_color))
            font_pnl = item_pnl.font()
            font_pnl.setBold(True)
            item_pnl.setFont(font_pnl)
            self._trade_table.setItem(row_idx, 2, item_pnl)

            # Column 3: Lesson (colored by type)
            lesson_color = LESSON_COLORS.get(lesson, C["text"])
            item_lesson = QTableWidgetItem(lesson)
            item_lesson.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            item_lesson.setForeground(QColor(lesson_color))
            self._trade_table.setItem(row_idx, 3, item_lesson)

        # Restore scroll position
        scrollbar.setValue(scroll_pos)

        # Re-select previously selected row (or auto-select first)
        if reselect_row >= 0:
            self._trade_table.selectRow(reselect_row)
            self._show_trade_detail(trades[reselect_row])
        elif len(trades) > 0 and self._selected_trade is None:
            # Auto-select first trade if nothing was selected
            self._trade_table.selectRow(0)
            self._show_trade_detail(trades[0])

    def _on_trade_clicked(self, row: int, _column: int) -> None:
        """Handle click on a trade row to show its detail."""
        if 0 <= row < len(self._trades):
            self._show_trade_detail(self._trades[row])

    # ------------------------------------------------------------------
    # Trade detail display
    # ------------------------------------------------------------------

    def _show_trade_detail(self, trade: dict) -> None:
        """Display full information for the selected trade.

        Parameters
        ----------
        trade : dict
            id, side, entry_price, exit_price, exit_reason, pnl,
            pnl_pct, hold_min, hold_bars, lesson, sl_quality,
            sl_saved, pattern, pattern_wins, pattern_losses,
            rsi, trend, session, volatility
        """
        self._selected_trade = trade

        trade_id = trade.get("id", "--")
        lesson = trade.get("lesson", "--")
        lesson_color = LESSON_COLORS.get(lesson, C["text"])

        # Trade ID + lesson (colored)
        self._lbl_trade_id.setText(f"#{trade_id}")
        self._lbl_trade_id.setStyleSheet(
            f"color: {lesson_color}; font-size: 15px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Entry
        side = trade.get("side", "--")
        entry_price = trade.get("entry_price")
        side_color = C["green"] if side == "LONG" else C["red"] if side == "SHORT" else C["text"]
        entry_text = f"{side}"
        if entry_price is not None:
            entry_text += f" @ ${entry_price:,.2f}"
        self._lbl_entry.setText(entry_text)
        self._lbl_entry.setStyleSheet(
            f"color: {side_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Exit
        exit_reason = trade.get("exit_reason", "--")
        exit_price = trade.get("exit_price")
        exit_text = f"{exit_reason}"
        if exit_price is not None:
            exit_text += f" @ ${exit_price:,.2f}"
        self._lbl_exit.setText(exit_text)
        self._lbl_exit.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # P/L
        pnl = trade.get("pnl", 0.0)
        pnl_pct = trade.get("pnl_pct", 0.0)
        pnl_color = C["green"] if pnl >= 0 else C["red"]
        self._lbl_pnl.setText(f"{currency.fmt_signed(pnl)} ({pnl_pct:+.2f}%)")
        self._lbl_pnl.setStyleSheet(
            f"color: {pnl_color}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Hold duration
        hold_min = trade.get("hold_min", 0)
        hold_bars = trade.get("hold_bars", 0)
        self._lbl_hold.setText(f"{hold_min} min ({hold_bars} bars)")
        self._lbl_hold.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # SL Quality
        sl_quality = trade.get("sl_quality", "--")
        sl_saved = trade.get("sl_saved", 0.0)
        sl_text = sl_quality
        if sl_saved != 0:
            sl_text += f" ({currency.fmt_signed(sl_saved)})"
        sl_color = C["green"] if "TRAIL" in sl_quality.upper() else C["text"]
        self._lbl_sl_quality.setText(sl_text)
        self._lbl_sl_quality.setStyleSheet(
            f"color: {sl_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Pattern / entry conditions
        rsi = trade.get("rsi", "--")
        trend = trade.get("trend", "--")
        session = trade.get("session", "--")
        volatility = trade.get("volatility", "--")
        pattern_str = trade.get("pattern", "")
        if not pattern_str:
            pattern_str = f"rsi={rsi}/trend={trend}/{session}/vol={volatility}"
        self._lbl_pattern.setText(pattern_str)
        self._lbl_pattern.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Pattern win rate
        pattern_wins = trade.get("pattern_wins", 0)
        pattern_losses = trade.get("pattern_losses", 0)
        pattern_total = pattern_wins + pattern_losses
        if pattern_total > 0:
            wr = (pattern_wins / pattern_total) * 100
            wr_color = C["green"] if wr >= 55 else C["text"] if wr >= 45 else C["red"]
            self._lbl_pattern_wr.setText(
                f"{wr:.0f}% ({pattern_wins}W / {pattern_losses}L)"
            )
            self._lbl_pattern_wr.setStyleSheet(
                f"color: {wr_color}; font-size: 13px; "
                f"background: transparent; border: none;"
            )
        else:
            self._lbl_pattern_wr.setText("--")
            self._lbl_pattern_wr.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; "
                f"background: transparent; border: none;"
            )
