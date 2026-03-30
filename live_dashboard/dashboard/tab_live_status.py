"""Tab 1 — Live Status for the Spartus Live Trading Dashboard.

Shows the main live trading view: MT5 connection status, account info,
open position details, AI trade protection progress, today's summary,
and the AI decision log.

Layout::

    ┌──────────────────────┬────────────────────────────────────┐
    │  MT5 CONNECTION       │  ACCOUNT                          │
    ├──────────────┬─────────────────────────────────────────── ┤
    │  OPEN         │  AI TRADE PROTECTION  (big, prominent)    │
    │  POSITION     │  ● Breakeven   +£2.00  ✓ SL at entry      │
    │               │  ● Profit Lock +£3.00  ✓ £1.50 locked     │
    │               │  ○ Trailing    +£4.00  1.0× ATR           │
    ├──────────────┴────────────────────────────────────────────┤
    │  TODAY'S SUMMARY  │  AI DECISION LOG (last 20 actions)    │
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

        # ----- Middle row: Open Position (left, compact) + AI Protection (right, prominent) -----
        mid_row = QHBoxLayout()
        mid_row.setSpacing(6)
        mid_row.addWidget(self._build_position_box(), 2)
        mid_row.addWidget(self._build_protection_box(), 3)
        root.addLayout(mid_row)

        # ----- Bottom row: Today Summary (left) + AI Decision Log (right) -----
        bot_row = QHBoxLayout()
        bot_row.setSpacing(6)
        bot_row.addWidget(self._build_summary_box(), 1)
        bot_row.addWidget(self._build_decision_log_box(), 2)
        root.addLayout(bot_row, 1)

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
        layout.setSpacing(10)

        fields = [
            ("Currency:", "_lbl_currency"),
            ("Balance:", "_lbl_balance"),
            ("Equity:", "_lbl_equity"),
            ("Margin:", "_lbl_margin"),
            ("Free Margin:", "_lbl_free_margin"),
        ]

        for row, (label_text, attr_name) in enumerate(fields):
            layout.addWidget(_make_label(label_text, font_size=15), row, 0)
            value_label = _make_label("--", C["text"], font_size=15)
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
        layout.setSpacing(6)

        # Row 0: Side + lots (prominent)
        self._lbl_side = _make_label("FLAT", C["label"], bold=True, font_size=16)
        layout.addWidget(self._lbl_side, 0, 0, 1, 2)

        # Rows 1-6: Trade details
        detail_fields = [
            ("Entry:",    "_lbl_entry"),
            ("Current:",  "_lbl_current"),
            ("P/L:",      "_lbl_pnl"),
            ("SL:",       "_lbl_sl"),
            ("TP:",       "_lbl_tp"),
            ("Duration:", "_lbl_duration"),
        ]
        for row_offset, (label_text, attr_name) in enumerate(detail_fields, start=1):
            layout.addWidget(_make_label(label_text), row_offset, 0)
            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row_offset, 1)

        layout.setRowStretch(len(detail_fields) + 1, 1)
        layout.setColumnStretch(1, 1)

        return box


    # ------------------------------------------------------------------
    # 4. AI TRADE PROTECTION  (standalone, prominent panel)
    # ------------------------------------------------------------------

    def _build_protection_box(self) -> QGroupBox:
        """Build the AI Trade Protection panel — large, centre-right position."""
        box = QGroupBox("AI TRADE PROTECTION")
        outer = QVBoxLayout(box)
        outer.setContentsMargins(16, 20, 16, 14)
        outer.setSpacing(10)

        # --- Active stage badge (top-right area) ---
        badge_row = QHBoxLayout()
        badge_row.setSpacing(0)
        self._lbl_stage_badge = _make_label("Stage 0 / 3", C["label"], bold=True, font_size=13)
        badge_row.addStretch()
        badge_row.addWidget(self._lbl_stage_badge)
        outer.addLayout(badge_row)

        # --- Three protection stage rows ---
        # Each row: big dot | stage name | threshold | status
        stage_grid = QGridLayout()
        stage_grid.setContentsMargins(0, 0, 0, 0)
        stage_grid.setHorizontalSpacing(12)
        stage_grid.setVerticalSpacing(14)
        stage_grid.setColumnStretch(3, 1)

        def _add_stage_row(row: int, dot_attr: str, name_attr: str,
                           thresh_attr: str, stat_attr: str) -> None:
            dot = QLabel("○")
            dot.setStyleSheet(
                f"color: {C['label']}; font-size: 22px; background: transparent; border: none;"
            )
            dot.setFixedWidth(28)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            setattr(self, dot_attr, dot)
            stage_grid.addWidget(dot, row, 0)

            name_lbl = _make_label("--", C["label"], bold=True, font_size=17)
            setattr(self, name_attr, name_lbl)
            stage_grid.addWidget(name_lbl, row, 1)

            thresh_lbl = _make_label("--", C["subtext"], font_size=14)
            setattr(self, thresh_attr, thresh_lbl)
            stage_grid.addWidget(thresh_lbl, row, 2)

            stat_lbl = _make_label("", C["subtext"], font_size=13)
            setattr(self, stat_attr, stat_lbl)
            stage_grid.addWidget(stat_lbl, row, 3)

        _add_stage_row(0, "_dot_be",    "_lbl_be_name",    "_lbl_be_thresh",    "_lbl_be_stat")
        _add_stage_row(1, "_dot_lock",  "_lbl_lock_name",  "_lbl_lock_thresh",  "_lbl_lock_stat")
        _add_stage_row(2, "_dot_trail", "_lbl_trail_name", "_lbl_trail_thresh", "_lbl_trail_stat")

        outer.addLayout(stage_grid)

        # --- Horizontal divider line ---
        div = QLabel()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C['border']}; border: none;")
        outer.addWidget(div)

        # --- Summary / next-trigger line ---
        self._lbl_prot_summary = _make_label("", C["label"], bold=True, font_size=14)
        self._lbl_prot_summary.setWordWrap(True)
        outer.addWidget(self._lbl_prot_summary)

        outer.addStretch()
        return box

    # ------------------------------------------------------------------
    # 5. TODAY'S SUMMARY
    # ------------------------------------------------------------------

    def _build_summary_box(self) -> QGroupBox:
        box = QGroupBox("TODAY'S SUMMARY")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)

        fields = [
            ("Trades:", "_lbl_trades"),
            ("P/L:", "_lbl_day_pnl"),
            ("Win Rate:", "_lbl_winrate"),
            ("Max DD:", "_lbl_max_dd"),
            ("Profit Factor:", "_lbl_pf"),
        ]

        for row, (label_text, attr_name) in enumerate(fields):
            layout.addWidget(_make_label(label_text, font_size=15), row, 0)
            value_label = _make_label("--", C["text"], font_size=15)
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
                    f"color: {C['text']}; font-size: 15px; background: transparent; border: none;"
                )
            else:
                lbl.setText("--")

    def _update_position(self, pos: dict | None) -> None:
        """Update open position panel."""
        _FLAT_ATTRS = (
            "_lbl_entry", "_lbl_current", "_lbl_pnl",
            "_lbl_sl", "_lbl_tp", "_lbl_duration",
            "_lbl_be_name", "_lbl_be_thresh", "_lbl_be_stat",
            "_lbl_lock_name", "_lbl_lock_thresh", "_lbl_lock_stat",
            "_lbl_trail_name", "_lbl_trail_thresh", "_lbl_trail_stat",
            "_lbl_prot_summary",
        )

        if pos is None or pos.get("side", "FLAT") == "FLAT":
            self._lbl_side.setText("FLAT")
            self._lbl_side.setStyleSheet(
                f"color: {C['label']}; font-size: 16px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
            for attr in _FLAT_ATTRS:
                lbl = getattr(self, attr, None)
                if lbl:
                    lbl.setText("--")
                    lbl.setStyleSheet(
                        f"color: {C['text']}; font-size: 12px; background: transparent; border: none;"
                    )
            for dot_attr in ("_dot_be", "_dot_lock", "_dot_trail"):
                dot = getattr(self, dot_attr, None)
                if dot:
                    dot.setText("○")
                    dot.setStyleSheet(
                        f"color: {C['label']}; font-size: 22px; background: transparent; border: none;"
                    )
            self._lbl_stage_badge.setText("Stage 0 / 3  ○○○")
            self._lbl_stage_badge.setStyleSheet(
                f"color: {C['label']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
            self._lbl_prot_summary.setText("No open position")
            self._lbl_prot_summary.setStyleSheet(
                f"color: {C['subtext']}; font-size: 14px; font-weight: bold; "
                f"background: transparent; border: none;"
            )
            return

        # --- Side + lots ---
        side = pos.get("side", "FLAT")
        lots = pos.get("lots", 0.0)
        side_color = C["green"] if side == "LONG" else C["red"] if side == "SHORT" else C["label"]
        self._lbl_side.setText(f"{side}  {lots:.2f} lots")
        self._lbl_side.setStyleSheet(
            f"color: {side_color}; font-size: 16px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # --- Detail fields ---
        entry = pos.get("entry_price")
        if entry is not None:
            self._lbl_entry.setText(f"${entry:,.2f}")

        current = pos.get("current_price")
        if current is not None:
            self._lbl_current.setText(f"${current:,.2f}")

        pnl = pos.get("pnl", 0.0)
        if pnl is not None:
            pnl_color = C["green"] if pnl >= 0 else C["red"]
            self._lbl_pnl.setText(currency.fmt_signed(pnl))
            self._lbl_pnl.setStyleSheet(
                f"color: {pnl_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        sl = pos.get("sl")
        trailing = pos.get("trailing", False)
        if sl is not None:
            trailing_marker = " (\u25b2)" if trailing else ""
            self._lbl_sl.setText(f"${sl:,.2f}{trailing_marker}")

        tp = pos.get("tp")
        if tp is not None:
            self._lbl_tp.setText(f"${tp:,.2f}")

        duration = pos.get("duration_min")
        if duration is not None:
            self._lbl_duration.setText(f"{duration} min")

        # --- Protection Progress (displayed in the standalone AI TRADE PROTECTION box) ---
        stage = pos.get("protection_stage", 0)
        be_gbp = pos.get("be_trigger_gbp", 2.0)
        lock_gbp = pos.get("lock_trigger_gbp", 3.0)
        lock_amount_gbp = pos.get("lock_amount_gbp", 1.5)
        trail_gbp = pos.get("trail_trigger_gbp", 4.0)
        trail_atr = pos.get("trail_atr_mult", 1.0)
        locked_pnl = pos.get("locked_pnl", 0.0)

        def _fmt_gbp(v: float) -> str:
            return currency.fmt(v)

        _GREEN = C["green"]   # All triggered stages glow green
        _GRAY  = C["label"]   # Untriggered / inactive

        # Stage badge
        badge_color = _GREEN if stage > 0 else _GRAY
        self._lbl_stage_badge.setText(f"Stage {stage} / 3  {'●' * stage}{'○' * (3 - stage)}")
        self._lbl_stage_badge.setStyleSheet(
            f"color: {badge_color}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Stage 1: Breakeven
        if stage >= 1:
            self._dot_be.setText("●")
            self._dot_be.setStyleSheet(
                f"color: {_GREEN}; font-size: 22px; background: transparent; border: none;"
            )
            self._lbl_be_name.setText("Breakeven")
            self._lbl_be_name.setStyleSheet(
                f"color: {_GREEN}; font-size: 17px; font-weight: bold; background: transparent; border: none;"
            )
            self._lbl_be_thresh.setText(f"at +{_fmt_gbp(be_gbp)}")
            self._lbl_be_thresh.setStyleSheet(
                f"color: {C['text']}; font-size: 14px; background: transparent; border: none;"
            )
            self._lbl_be_stat.setText("✓ SL at entry")
            self._lbl_be_stat.setStyleSheet(
                f"color: {_GREEN}; font-size: 13px; background: transparent; border: none;"
            )
        else:
            self._dot_be.setText("○")
            self._dot_be.setStyleSheet(
                f"color: {_GRAY}; font-size: 22px; background: transparent; border: none;"
            )
            self._lbl_be_name.setText("Breakeven")
            self._lbl_be_name.setStyleSheet(
                f"color: {_GRAY}; font-size: 17px; font-weight: bold; background: transparent; border: none;"
            )
            self._lbl_be_thresh.setText(f"at +{_fmt_gbp(be_gbp)}")
            self._lbl_be_thresh.setStyleSheet(
                f"color: {C['subtext']}; font-size: 14px; background: transparent; border: none;"
            )
            dist_to_be = be_gbp - pnl
            be_stat_txt = f"→ {_fmt_gbp(dist_to_be)} away" if dist_to_be > 0 else "(nearly there...)"
            self._lbl_be_stat.setText(be_stat_txt)
            self._lbl_be_stat.setStyleSheet(
                f"color: {C['subtext']}; font-size: 13px; background: transparent; border: none;"
            )

        # Stage 2: Profit Lock
        if stage >= 2:
            self._dot_lock.setText("●")
            self._dot_lock.setStyleSheet(
                f"color: {_GREEN}; font-size: 22px; background: transparent; border: none;"
            )
            self._lbl_lock_name.setText("Profit Lock")
            self._lbl_lock_name.setStyleSheet(
                f"color: {_GREEN}; font-size: 17px; font-weight: bold; background: transparent; border: none;"
            )
            self._lbl_lock_thresh.setText(f"at +{_fmt_gbp(lock_gbp)}")
            self._lbl_lock_thresh.setStyleSheet(
                f"color: {C['text']}; font-size: 14px; background: transparent; border: none;"
            )
            locked_display = currency.fmt(locked_pnl) if locked_pnl > 0 else _fmt_gbp(lock_amount_gbp)
            self._lbl_lock_stat.setText(f"✓ locked {locked_display}")
            self._lbl_lock_stat.setStyleSheet(
                f"color: {_GREEN}; font-size: 13px; background: transparent; border: none;"
            )
        else:
            self._dot_lock.setText("○")
            self._dot_lock.setStyleSheet(
                f"color: {_GRAY}; font-size: 22px; background: transparent; border: none;"
            )
            self._lbl_lock_name.setText("Profit Lock")
            self._lbl_lock_name.setStyleSheet(
                f"color: {_GRAY}; font-size: 17px; font-weight: bold; background: transparent; border: none;"
            )
            self._lbl_lock_thresh.setText(f"at +{_fmt_gbp(lock_gbp)}  (locks {_fmt_gbp(lock_amount_gbp)})")
            self._lbl_lock_thresh.setStyleSheet(
                f"color: {C['subtext']}; font-size: 14px; background: transparent; border: none;"
            )
            dist_to_lock = lock_gbp - pnl
            stat_txt = f"→ {_fmt_gbp(dist_to_lock)} away" if dist_to_lock > 0 else "(nearly there...)"
            self._lbl_lock_stat.setText(stat_txt)
            self._lbl_lock_stat.setStyleSheet(
                f"color: {C['subtext']}; font-size: 13px; background: transparent; border: none;"
            )

        # Stage 3: Trailing Stop
        if stage >= 3:
            self._dot_trail.setText("●")
            self._dot_trail.setStyleSheet(
                f"color: {_GREEN}; font-size: 22px; background: transparent; border: none;"
            )
            self._lbl_trail_name.setText("Trailing Stop")
            self._lbl_trail_name.setStyleSheet(
                f"color: {_GREEN}; font-size: 17px; font-weight: bold; background: transparent; border: none;"
            )
            self._lbl_trail_thresh.setText(f"at +{_fmt_gbp(trail_gbp)}")
            self._lbl_trail_thresh.setStyleSheet(
                f"color: {C['text']}; font-size: 14px; background: transparent; border: none;"
            )
            self._lbl_trail_stat.setText(f"✓ Active  ({trail_atr:.1f}× ATR)")
            self._lbl_trail_stat.setStyleSheet(
                f"color: {_GREEN}; font-size: 13px; background: transparent; border: none;"
            )
        else:
            self._dot_trail.setText("○")
            self._dot_trail.setStyleSheet(
                f"color: {_GRAY}; font-size: 22px; background: transparent; border: none;"
            )
            self._lbl_trail_name.setText("Trailing Stop")
            self._lbl_trail_name.setStyleSheet(
                f"color: {_GRAY}; font-size: 17px; font-weight: bold; background: transparent; border: none;"
            )
            self._lbl_trail_thresh.setText(f"at +{_fmt_gbp(trail_gbp)}  ({trail_atr:.1f}× ATR)")
            self._lbl_trail_thresh.setStyleSheet(
                f"color: {C['subtext']}; font-size: 14px; background: transparent; border: none;"
            )
            dist_to_trail = trail_gbp - pnl
            stat_txt = f"→ {_fmt_gbp(dist_to_trail)} away" if dist_to_trail > 0 else "(nearly there...)"
            self._lbl_trail_stat.setText(stat_txt)
            self._lbl_trail_stat.setStyleSheet(
                f"color: {C['subtext']}; font-size: 13px; background: transparent; border: none;"
            )

        # --- Summary line ---
        if stage >= 3:
            summary_txt = f"Trailing active  |  Min guaranteed: {currency.fmt(locked_pnl) if locked_pnl > 0 else '?'}"
            summary_color = _GREEN
        elif stage == 2:
            dist_to_t = trail_gbp - pnl
            summary_txt = f"Profit locked: {currency.fmt(locked_pnl) if locked_pnl > 0 else _fmt_gbp(lock_amount_gbp)}  |  Trail in {_fmt_gbp(max(dist_to_t, 0))}"
            summary_color = _GREEN
        elif stage == 1:
            dist_to_l = lock_gbp - pnl
            summary_txt = f"BE active  |  Lock profit in {_fmt_gbp(max(dist_to_l, 0))}"
            summary_color = _GREEN
        elif pnl >= 0:
            dist_to_b = be_gbp - pnl
            summary_txt = f"Next: Breakeven in {_fmt_gbp(max(dist_to_b, 0))}"
            summary_color = C["label"]
        else:
            summary_txt = f"P/L: {currency.fmt_signed(pnl)}  |  BE triggers at +{_fmt_gbp(be_gbp)}"
            summary_color = C["label"]

        self._lbl_prot_summary.setText(summary_txt)
        self._lbl_prot_summary.setStyleSheet(
            f"color: {summary_color}; font-size: 14px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

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
                f"color: {pnl_color}; font-size: 15px; font-weight: bold; "
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
                f"color: {dd_color}; font-size: 15px; background: transparent; border: none;"
            )

        # Profit Factor
        pf = today.get("profit_factor")
        if pf is not None:
            pf_color = C["green"] if pf >= 1.5 else C["text"] if pf >= 1.0 else C["red"]
            self._lbl_pf.setText(f"{pf:.2f}")
            self._lbl_pf.setStyleSheet(
                f"color: {pf_color}; font-size: 15px; background: transparent; border: none;"
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
