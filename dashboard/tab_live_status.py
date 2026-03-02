"""Tab 1 — Live Status for the Spartus Live Trading Dashboard.

Shows the main live trading view: MT5 connection status, account info,
open position details, today's summary, and the AI decision log.

Layout::

    ┌──────────────────────┬────────────────────────────────────┐
    │  MT5 CONNECTION       │  ACCOUNT                          │
    ├──────────────────────┼────────────────────────────────────┤
    │  OPEN POSITION        │  TODAY'S SUMMARY                  │
    ├──────────────────────┴────────────────────────────────────┤
    │  AI DECISION LOG (last 20 actions)                        │
    └───────────────────────────────────────────────────────────┘

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels — NEVER dark gray on dark backgrounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from dashboard.theme import C
from dashboard.widgets import StatusIndicator, ActionLogWidget
from dashboard import currency


# ---------------------------------------------------------------------------
# Helper: create a styled label pair (label + value) on one row
# ---------------------------------------------------------------------------

def _make_label(text: str, color: str = C["subtext"], bold: bool = False,
                font_size: int = 13) -> QLabel:
    """Create a QLabel with the given text, color, and optional bold."""
    lbl = QLabel(text)
    style = f"color: {color}; font-size: {font_size}px; background: transparent; border: none;"
    if bold:
        style += " font-weight: bold;"
    lbl.setStyleSheet(style)
    return lbl


# ---------------------------------------------------------------------------
# LiveStatusTab
# ---------------------------------------------------------------------------

class LiveStatusTab(QWidget):
    """Tab 1: Live trading status overview.

    Shows MT5 connection health, account balances, open position details,
    today's performance summary, and the AI decision log.

    Call ``update_data(data)`` each tick with the latest state dict.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._prev_decisions: list = []

        # ----- Root layout -----
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ----- Top row: MT5 Connection (left) + Account (right) -----
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        top_row.addWidget(self._build_connection_box(), 1)
        top_row.addWidget(self._build_account_box(), 1)
        root.addLayout(top_row)

        # ----- Middle row: Open Position (left) + Today Summary (right) -----
        mid_row = QHBoxLayout()
        mid_row.setSpacing(6)
        mid_row.addWidget(self._build_position_box(), 1)
        mid_row.addWidget(self._build_summary_box(), 1)
        root.addLayout(mid_row)

        # ----- Bottom row: AI Decision Log (full width) -----
        root.addWidget(self._build_decision_log_box(), 1)

    # ==================================================================
    # Section builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. MT5 CONNECTION
    # ------------------------------------------------------------------

    def _build_connection_box(self) -> QGroupBox:
        box = QGroupBox("MT5 CONNECTION")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        # Status indicator (dot + text)
        self._conn_indicator = StatusIndicator("Status")
        self._conn_indicator.set_status("Disconnected", C["red"])
        self._conn_indicator.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self._conn_indicator, 0, 0, 1, 2)

        # Server
        layout.addWidget(_make_label("Server:"), 1, 0)
        self._lbl_server = _make_label("--", C["text"])
        layout.addWidget(self._lbl_server, 1, 1)

        # Latency
        layout.addWidget(_make_label("Latency:"), 2, 0)
        self._lbl_latency = _make_label("-- ms", C["text"])
        layout.addWidget(self._lbl_latency, 2, 1)

        # Spread
        layout.addWidget(_make_label("Spread:"), 3, 0)
        self._lbl_spread = _make_label("-- pips", C["text"])
        layout.addWidget(self._lbl_spread, 3, 1)

        # Market status
        layout.addWidget(_make_label("Market:"), 4, 0)
        self._lbl_market_status = _make_label("--", C["label"])
        layout.addWidget(self._lbl_market_status, 4, 1)

        # Last bar / bars processed
        layout.addWidget(_make_label("Last Bar:"), 5, 0)
        self._lbl_last_bar = _make_label("--", C["text"])
        layout.addWidget(self._lbl_last_bar, 5, 1)

        # Bars processed
        layout.addWidget(_make_label("Bars:"), 6, 0)
        self._lbl_bars_processed = _make_label("0", C["text"])
        layout.addWidget(self._lbl_bars_processed, 6, 1)

        # --- Broker Constraints section ---
        _broker_hdr = _make_label("BROKER CONSTRAINTS", C["cyan"], bold=True, font_size=11)
        layout.addWidget(_broker_hdr, 7, 0, 1, 2)

        # Min Lot / Lot Step
        layout.addWidget(_make_label("Min Lot / Step:"), 8, 0)
        self._lbl_min_lot = _make_label("-- / --", C["text"])
        layout.addWidget(self._lbl_min_lot, 8, 1)

        # Stop Level / Freeze Level
        layout.addWidget(_make_label("Stop/Freeze Lvl:"), 9, 0)
        self._lbl_stops = _make_label("-- / -- pts", C["text"])
        layout.addWidget(self._lbl_stops, 9, 1)

        # Tick Value / VPP
        layout.addWidget(_make_label("Tick Val / VPP:"), 10, 0)
        self._lbl_tick_vpp = _make_label("-- / --", C["text"])
        layout.addWidget(self._lbl_tick_vpp, 10, 1)

        # Spread detail (current / EMA / max)
        layout.addWidget(_make_label("Spread Detail:"), 11, 0)
        self._lbl_spread_detail = _make_label("-- / -- / --", C["text"])
        layout.addWidget(self._lbl_spread_detail, 11, 1)

        # Min SL Distance
        layout.addWidget(_make_label("Min SL Dist:"), 12, 0)
        self._lbl_min_sl = _make_label("-- pts", C["text"])
        layout.addWidget(self._lbl_min_sl, 12, 1)

        # Push content to the top
        layout.setRowStretch(13, 1)

        return box

    # ------------------------------------------------------------------
    # 2. ACCOUNT
    # ------------------------------------------------------------------

    def _build_account_box(self) -> QGroupBox:
        box = QGroupBox("ACCOUNT")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        fields = [
            ("Currency:", "_lbl_currency"),
            ("Balance:", "_lbl_balance"),
            ("Equity:", "_lbl_equity"),
            ("Margin:", "_lbl_margin"),
            ("Free Margin:", "_lbl_free_margin"),
        ]

        for row, (label_text, attr_name) in enumerate(fields):
            layout.addWidget(_make_label(label_text), row, 0)
            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row, 1)

        layout.setRowStretch(len(fields), 1)

        return box

    # ------------------------------------------------------------------
    # 3. OPEN POSITION
    # ------------------------------------------------------------------

    def _build_position_box(self) -> QGroupBox:
        box = QGroupBox("OPEN POSITION")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        # Side + lots (first row, spans both columns for emphasis)
        self._lbl_side = _make_label("FLAT", C["label"], bold=True, font_size=16)
        layout.addWidget(self._lbl_side, 0, 0, 1, 2)

        # Detail fields
        detail_fields = [
            ("Entry:", "_lbl_entry"),
            ("Current:", "_lbl_current"),
            ("P/L:", "_lbl_pnl"),
            ("SL:", "_lbl_sl"),
            ("TP:", "_lbl_tp"),
            ("Duration:", "_lbl_duration"),
        ]

        for row_offset, (label_text, attr_name) in enumerate(detail_fields, start=1):
            layout.addWidget(_make_label(label_text), row_offset, 0)
            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row_offset, 1)

        layout.setRowStretch(len(detail_fields) + 1, 1)

        return box

    # ------------------------------------------------------------------
    # 4. TODAY'S SUMMARY
    # ------------------------------------------------------------------

    def _build_summary_box(self) -> QGroupBox:
        box = QGroupBox("TODAY'S SUMMARY")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        fields = [
            ("Trades:", "_lbl_trades"),
            ("P/L:", "_lbl_day_pnl"),
            ("Win Rate:", "_lbl_winrate"),
            ("Max DD:", "_lbl_max_dd"),
            ("Profit Factor:", "_lbl_pf"),
        ]

        for row, (label_text, attr_name) in enumerate(fields):
            layout.addWidget(_make_label(label_text), row, 0)
            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row, 1)

        layout.setRowStretch(len(fields), 1)

        return box

    # ------------------------------------------------------------------
    # 5. AI DECISION LOG
    # ------------------------------------------------------------------

    def _build_decision_log_box(self) -> QGroupBox:
        box = QGroupBox("AI DECISION LOG")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 20, 12, 8)
        layout.setSpacing(4)

        self._decision_log = ActionLogWidget(max_lines=20)
        layout.addWidget(self._decision_log)

        return box

    # ==================================================================
    # update_data — called every tick with live state
    # ==================================================================

    def update_data(self, data: dict) -> None:
        """Refresh all panels with the latest data dict.

        Parameters
        ----------
        data : dict
            Expected keys:

            connection : dict
                connected (bool), server (str), latency_ms (float), spread (float)
            account : dict
                currency (str), balance (float), equity (float),
                margin (float), free_margin (float)
            position : dict or None
                side (str "LONG"/"SHORT"/"FLAT"), lots (float),
                entry_price (float), current_price (float), pnl (float),
                sl (float), tp (float), duration_min (int),
                trailing (bool)
            today : dict
                trades (int), wins (int), losses (int), pnl (float),
                win_rate (float), max_dd (float), profit_factor (float)
            decisions : list[dict]
                Each dict: timestamp (str), action (str), details (str)
        """
        if not data:
            return

        self._update_connection(data.get("connection", {}))
        self._update_market_status(data.get("market", {}))
        self._update_account(data.get("account", {}))
        self._update_position(data.get("position"))
        self._update_summary(data.get("today", {}))
        self._update_decisions(data.get("decisions", []))
        self._update_broker(data.get("broker"))

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_connection(self, conn: dict) -> None:
        """Update MT5 connection panel."""
        if not conn:
            return

        connected = conn.get("connected", False)
        if connected:
            self._conn_indicator.set_status("Connected", C["green"])
        else:
            self._conn_indicator.set_status("Disconnected", C["red"])

        server = conn.get("server", "--")
        self._lbl_server.setText(str(server))

        latency = conn.get("latency_ms")
        if latency is not None:
            lat_color = C["text"]
            if latency > 200:
                lat_color = C["red"]
            elif latency > 100:
                lat_color = C["yellow"]
            self._lbl_latency.setText(f"{latency:.0f} ms")
            self._lbl_latency.setStyleSheet(
                f"color: {lat_color}; font-size: 13px; background: transparent; border: none;"
            )
        else:
            self._lbl_latency.setText("-- ms")

        spread = conn.get("spread")
        if spread is not None:
            self._lbl_spread.setText(f"{spread:.1f} pips")
        else:
            self._lbl_spread.setText("-- pips")

    def _update_market_status(self, market: dict) -> None:
        """Update market status (OPEN / CLOSED / INITIALIZING) and last bar info."""
        if not market:
            return

        status = market.get("status", "INITIALIZING")
        status_colors = {
            "OPEN": C["green"],
            "CLOSED": C["yellow"],
            "DISCONNECTED": C["red"],
            "INITIALIZING": C["label"],
        }
        color = status_colors.get(status, C["label"])
        self._lbl_market_status.setText(status)
        self._lbl_market_status.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        last_bar = market.get("last_bar_time", "--")
        self._lbl_last_bar.setText(str(last_bar))

        bars = market.get("bars_processed", 0)
        self._lbl_bars_processed.setText(str(bars))

    def _update_account(self, acct: dict) -> None:
        """Update account info panel."""
        if not acct:
            return

        self._lbl_currency.setText(str(acct.get("currency", "--")))

        for key, lbl_attr in [
            ("balance", "_lbl_balance"),
            ("equity", "_lbl_equity"),
            ("margin", "_lbl_margin"),
            ("free_margin", "_lbl_free_margin"),
        ]:
            value = acct.get(key)
            lbl: QLabel = getattr(self, lbl_attr)
            if value is not None:
                lbl.setText(currency.fmt(value))
                lbl.setStyleSheet(
                    f"color: {C['text']}; font-size: 13px; background: transparent; border: none;"
                )
            else:
                lbl.setText("--")

    def _update_position(self, pos: dict | None) -> None:
        """Update open position panel."""
        if pos is None or pos.get("side", "FLAT") == "FLAT":
            # No position
            self._lbl_side.setText("FLAT")
            self._lbl_side.setStyleSheet(
                f"color: {C['label']}; font-size: 16px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
            for attr in ("_lbl_entry", "_lbl_current", "_lbl_pnl",
                         "_lbl_sl", "_lbl_tp", "_lbl_duration"):
                getattr(self, attr).setText("--")
                getattr(self, attr).setStyleSheet(
                    f"color: {C['text']}; font-size: 13px; background: transparent; border: none;"
                )
            return

        # Side + lots
        side = pos.get("side", "FLAT")
        lots = pos.get("lots", 0.0)
        if side == "LONG":
            side_color = C["green"]
        elif side == "SHORT":
            side_color = C["red"]
        else:
            side_color = C["label"]
        self._lbl_side.setText(f"{side} {lots:.2f} lots")
        self._lbl_side.setStyleSheet(
            f"color: {side_color}; font-size: 16px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Entry price
        entry = pos.get("entry_price")
        if entry is not None:
            self._lbl_entry.setText(f"${entry:,.2f}")

        # Current price
        current = pos.get("current_price")
        if current is not None:
            self._lbl_current.setText(f"${current:,.2f}")

        # Unrealized P/L (colored)
        pnl = pos.get("pnl")
        if pnl is not None:
            pnl_color = C["green"] if pnl >= 0 else C["red"]
            self._lbl_pnl.setText(currency.fmt_signed(pnl))
            self._lbl_pnl.setStyleSheet(
                f"color: {pnl_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # Stop Loss (with trailing indicator)
        sl = pos.get("sl")
        trailing = pos.get("trailing", False)
        if sl is not None:
            trailing_marker = " (\u25b2)" if trailing else ""
            self._lbl_sl.setText(f"${sl:,.2f}{trailing_marker}")

        # Take Profit
        tp = pos.get("tp")
        if tp is not None:
            self._lbl_tp.setText(f"${tp:,.2f}")

        # Duration
        duration = pos.get("duration_min")
        if duration is not None:
            self._lbl_duration.setText(f"{duration} min")

    def _update_summary(self, today: dict) -> None:
        """Update today's trading summary panel."""
        if not today:
            return

        # Trades (with W/L breakdown)
        trades = today.get("trades", 0)
        wins = today.get("wins", 0)
        losses = today.get("losses", 0)
        self._lbl_trades.setText(f"{trades}  ({wins}W / {losses}L)")

        # P/L (colored)
        pnl = today.get("pnl")
        if pnl is not None:
            pnl_color = C["green"] if pnl >= 0 else C["red"]
            self._lbl_day_pnl.setText(currency.fmt_signed(pnl))
            self._lbl_day_pnl.setStyleSheet(
                f"color: {pnl_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # Win Rate
        win_rate = today.get("win_rate")
        if win_rate is not None:
            self._lbl_winrate.setText(f"{win_rate:.0f}%")

        # Max Drawdown
        max_dd = today.get("max_dd")
        if max_dd is not None:
            dd_color = C["red"] if max_dd > 3.0 else C["yellow"] if max_dd > 1.5 else C["text"]
            self._lbl_max_dd.setText(f"{max_dd:.1f}%")
            self._lbl_max_dd.setStyleSheet(
                f"color: {dd_color}; font-size: 13px; background: transparent; border: none;"
            )

        # Profit Factor
        pf = today.get("profit_factor")
        if pf is not None:
            pf_color = C["green"] if pf >= 1.5 else C["text"] if pf >= 1.0 else C["red"]
            self._lbl_pf.setText(f"{pf:.2f}")
            self._lbl_pf.setStyleSheet(
                f"color: {pf_color}; font-size: 13px; background: transparent; border: none;"
            )

    def _update_decisions(self, decisions: list[dict]) -> None:
        """Rebuild the AI decision log from the latest decisions list.

        Each decision dict has: timestamp, action, details.

        Color coding:
          - Opens (BUY, SELL, OPEN) -> green
          - Closes (CLOSE, EXIT, STOP_OUT) -> red
          - Holds / adjustments (HOLD, TRAIL_SL, MODIFY) -> white
        """
        if not decisions:
            return

        # Skip rebuild if decisions haven't changed (preserves text selection)
        if decisions == self._prev_decisions:
            return
        self._prev_decisions = list(decisions)

        # Clear and re-populate so it always reflects the latest snapshot
        self._decision_log.clear()

        # Action -> color mapping
        _ACTION_COLORS = {
            "BUY": C["green"],
            "SELL": C["green"],
            "OPEN_LONG": C["green"],
            "OPEN_SHORT": C["green"],
            "CLOSE": C["red"],
            "EXIT": C["red"],
            "STOP_OUT": C["red"],
            "MARGIN_CALL": C["red"],
            "HOLD": C["text"],
            "TRAIL_SL": C["cyan"],
            "MODIFY": C["cyan"],
            "SKIP": C["label"],
            "FLAT": C["label"],
            "HOLD_FLAT": C["label"],
            "BELOW_THRESHOLD": C["label"],
            "LOTS_ZERO": C["yellow"],
        }

        # Decisions are expected newest-first; ActionLogWidget.add_entry
        # also inserts at the top, so we iterate in reverse (oldest first)
        # to end up with newest at the top.
        for entry in reversed(decisions):
            ts = entry.get("timestamp", "")
            action = entry.get("action", "")
            details = entry.get("details", "")

            action_upper = action.upper()
            if action_upper in _ACTION_COLORS:
                color = _ACTION_COLORS[action_upper]
            elif action_upper.startswith("WK_BLOCKED"):
                color = C["yellow"]
            elif action_upper.startswith("CB_BLOCKED"):
                color = "#ff8800"  # orange
            elif action_upper.startswith("BLOCKED_"):
                color = "#ff8800"  # orange
            else:
                color = C["text"]
            text = f"{action} {details}".strip() if details else action

            self._decision_log.add_entry(ts, text, color)

    def _update_broker(self, broker) -> None:
        """Update broker constraints panel from a BrokerSnapshot."""
        if broker is None:
            return

        # Min Lot / Lot Step
        self._lbl_min_lot.setText(
            f"{broker.volume_min:.3f} / {broker.volume_step:.3f}"
        )

        # Stop Level / Freeze Level
        self._lbl_stops.setText(
            f"{broker.stops_level} / {broker.freeze_level} pts"
        )

        # Tick Value / VPP
        self._lbl_tick_vpp.setText(
            f"{broker.tick_value:.4f} / {broker.value_per_point:.2f}"
        )

        # Spread detail: current / EMA / max(1h)
        spread_color = C["text"]
        if broker.spread_current_points > 40:
            spread_color = C["red"]
        elif broker.spread_current_points > 30:
            spread_color = C["yellow"]
        self._lbl_spread_detail.setText(
            f"{broker.spread_current:.2f} / "
            f"{broker.spread_ema:.2f} / "
            f"{broker.spread_max_1h:.2f}"
        )
        self._lbl_spread_detail.setStyleSheet(
            f"color: {spread_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Min SL Distance
        min_sl_pts = (
            int(broker.min_sl_distance / broker.point)
            if broker.point > 0 else 0
        )
        self._lbl_min_sl.setText(
            f"{broker.min_sl_distance:.2f} ({min_sl_pts} pts)"
        )
