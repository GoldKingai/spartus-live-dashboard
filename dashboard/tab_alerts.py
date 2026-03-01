"""Tab 5 -- Alerts & Safety for the Spartus Live Trading Dashboard.

Shows safety status and controls (left), plus alert log and daily
risk summary (right).

Layout::

    +---------------------------+----------------------------------------+
    |  SAFETY STATUS            |  ALERT LOG                             |
    |  * RUNNING                |  14:35 [INFO] Trailing SL              |
    |  * Circuit Breaker: OFF   |  14:25 [INFO] Opened LONG 0.02        |
    |  * Weekend Close: 5h away |  13:15 [WARN] Spread spike             |
    |  * Daily DD: 1.2% / 3.0% |  ...                                   |
    |  * Consec. Losses: 1      |                                        |
    |  * Connection: Stable     |                                        |
    |                           |                                        |
    |  CONTROLS                 |  DAILY RISK                            |
    |  [> Start Trading]        |  Trades: 4 / 10 soft / 20 hard        |
    |  [- Wind Down]            |  DD: 1.2% / 3.0% daily halt           |
    |  [# Stop Now]             |  DD: 2.1% / 10% total limit           |
    |  [! EMERGENCY STOP]       |  Equity: $1,245.89                     |
    |  [Reset Circuit Breaker]  |                                        |
    +---------------------------+----------------------------------------+

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels -- NEVER dark gray on dark backgrounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QTextEdit, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from dashboard.theme import C, STATE_COLORS
from dashboard import currency


# ---------------------------------------------------------------------------
# Alert level -> color mapping
# ---------------------------------------------------------------------------

ALERT_COLORS = {
    "INFO":  C["text"],
    "WARN":  C["yellow"],
    "ERROR": C["red"],
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
# AlertsTab
# ---------------------------------------------------------------------------

class AlertsTab(QWidget):
    """Tab 5: Alerts & Safety.

    Left side shows safety status (top) and control buttons (bottom).
    Right side shows a scrolling alert log (top) and daily risk
    summary (bottom).

    Signals
    -------
    start_requested : emitted when Start Trading button is clicked
    wind_down_requested : emitted when Wind Down button is clicked
    stop_requested : emitted when Stop Now button is clicked
    emergency_stop_requested : emitted when Emergency Stop button is clicked
    reset_cb_requested : emitted when Reset Circuit Breaker button is clicked

    Call ``update_data(data)`` each tick with the latest state dict.
    """

    # Signals for button actions -- the orchestrator connects to these
    start_requested = pyqtSignal()
    wind_down_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    emergency_stop_requested = pyqtSignal()
    reset_cb_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current trading state (used for button enable/disable logic)
        self._trading_state: str = "STOPPED"

        # ----- Root layout with horizontal splitter -----
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {C['border']}; width: 2px; }}"
        )

        # Left pane: Safety Status (top) + Controls (bottom)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._build_safety_status_box(), stretch=3)
        left_layout.addWidget(self._build_controls_box(), stretch=2)

        # Right pane: Alert Log (top) + Daily Risk (bottom)
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self._build_alert_log_box(), stretch=3)
        right_layout.addWidget(self._build_daily_risk_box(), stretch=2)

        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        root.addWidget(splitter)

    # ==================================================================
    # Section builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. SAFETY STATUS
    # ------------------------------------------------------------------

    def _build_safety_status_box(self) -> QGroupBox:
        """Panel showing trading state, circuit breaker, weekend, DD, etc."""
        box = QGroupBox("SAFETY STATUS")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        # Row 0: Trading state (big colored dot + text)
        self._state_dot = QLabel("\u2b24")  # large filled circle
        self._state_dot.setStyleSheet(
            f"color: {C['label']}; font-size: 18px; "
            f"background: transparent; border: none;"
        )
        self._state_dot.setFixedWidth(24)
        self._state_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._state_dot, 0, 0)

        self._lbl_state = _make_label("STOPPED", C["label"], bold=True, font_size=16)
        layout.addWidget(self._lbl_state, 0, 1)

        # Status detail rows
        detail_fields = [
            ("Circuit Breaker:", "_lbl_circuit_breaker"),
            ("Weekend Close:",   "_lbl_weekend"),
            ("Daily DD:",        "_lbl_daily_dd"),
            ("Consec. Losses:",  "_lbl_consec_losses"),
            ("Connection:",      "_lbl_connection"),
        ]

        for row_offset, (label_text, attr_name) in enumerate(detail_fields, start=1):
            # Dot indicator
            dot = QLabel("\u25cf")
            dot.setStyleSheet(
                f"color: {C['label']}; font-size: 12px; "
                f"background: transparent; border: none;"
            )
            dot.setFixedWidth(16)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            setattr(self, f"{attr_name}_dot", dot)
            layout.addWidget(dot, row_offset, 0)

            layout.addWidget(_make_label(label_text), row_offset, 1)

            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row_offset, 2)

        layout.setRowStretch(len(detail_fields) + 1, 1)
        layout.setColumnStretch(2, 1)
        return box

    # ------------------------------------------------------------------
    # 2. CONTROLS
    # ------------------------------------------------------------------

    def _build_controls_box(self) -> QGroupBox:
        """Panel with Start / Wind Down / Stop / Emergency Stop buttons."""
        box = QGroupBox("CONTROLS")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        # Start Trading
        self._btn_start = QPushButton("Start Trading")
        self._btn_start.setStyleSheet(
            f"QPushButton {{ "
            f"  background-color: {C['surface2']}; "
            f"  color: {C['green']}; "
            f"  border: 2px solid {C['green']}; "
            f"  border-radius: 4px; "
            f"  padding: 10px 16px; "
            f"  font-size: 13px; font-weight: bold; "
            f"}} "
            f"QPushButton:hover {{ background-color: {C['border']}; }} "
            f"QPushButton:disabled {{ "
            f"  color: {C['dim']}; border-color: {C['border']}; "
            f"  background-color: {C['surface']}; "
            f"}}"
        )
        self._btn_start.clicked.connect(self.start_requested.emit)
        layout.addWidget(self._btn_start)

        # Wind Down
        self._btn_wind_down = QPushButton("Wind Down")
        self._btn_wind_down.setStyleSheet(
            f"QPushButton {{ "
            f"  background-color: {C['surface2']}; "
            f"  color: {C['yellow']}; "
            f"  border: 2px solid {C['yellow']}; "
            f"  border-radius: 4px; "
            f"  padding: 10px 16px; "
            f"  font-size: 13px; font-weight: bold; "
            f"}} "
            f"QPushButton:hover {{ background-color: {C['border']}; }} "
            f"QPushButton:disabled {{ "
            f"  color: {C['dim']}; border-color: {C['border']}; "
            f"  background-color: {C['surface']}; "
            f"}}"
        )
        self._btn_wind_down.setEnabled(False)
        self._btn_wind_down.clicked.connect(self.wind_down_requested.emit)
        layout.addWidget(self._btn_wind_down)

        # Stop Now
        self._btn_stop = QPushButton("Stop Now")
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ "
            f"  background-color: {C['surface2']}; "
            f"  color: {C['red']}; "
            f"  border: 2px solid {C['red']}; "
            f"  border-radius: 4px; "
            f"  padding: 10px 16px; "
            f"  font-size: 13px; font-weight: bold; "
            f"}} "
            f"QPushButton:hover {{ background-color: {C['border']}; }} "
            f"QPushButton:disabled {{ "
            f"  color: {C['dim']}; border-color: {C['border']}; "
            f"  background-color: {C['surface']}; "
            f"}}"
        )
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self._btn_stop)

        # Emergency Stop -- bright red background, ALWAYS enabled
        self._btn_emergency = QPushButton("EMERGENCY STOP")
        self._btn_emergency.setStyleSheet(
            f"QPushButton {{ "
            f"  background-color: #cc0000; "
            f"  color: {C['white']}; "
            f"  border: 2px solid {C['red']}; "
            f"  border-radius: 4px; "
            f"  padding: 10px 16px; "
            f"  font-size: 14px; font-weight: bold; "
            f"}} "
            f"QPushButton:hover {{ background-color: {C['red']}; }}"
        )
        self._btn_emergency.clicked.connect(self.emergency_stop_requested.emit)
        layout.addWidget(self._btn_emergency)

        # Reset Circuit Breaker
        self._btn_reset_cb = QPushButton("Reset Circuit Breaker")
        self._btn_reset_cb.setStyleSheet(
            f"QPushButton {{ "
            f"  background-color: {C['surface2']}; "
            f"  color: {C['peach']}; "
            f"  border: 2px solid {C['peach']}; "
            f"  border-radius: 4px; "
            f"  padding: 8px 16px; "
            f"  font-size: 12px; font-weight: bold; "
            f"}} "
            f"QPushButton:hover {{ background-color: {C['border']}; }} "
            f"QPushButton:disabled {{ "
            f"  color: {C['dim']}; border-color: {C['border']}; "
            f"  background-color: {C['surface']}; "
            f"}}"
        )
        self._btn_reset_cb.setEnabled(False)
        self._btn_reset_cb.clicked.connect(self.reset_cb_requested.emit)
        layout.addWidget(self._btn_reset_cb)

        layout.addStretch()
        return box

    # ------------------------------------------------------------------
    # 3. ALERT LOG
    # ------------------------------------------------------------------

    def _build_alert_log_box(self) -> QGroupBox:
        """Scrolling log of timestamped alerts, colored by level."""
        box = QGroupBox("ALERT LOG")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(0)

        self._alert_log = QTextEdit()
        self._alert_log.setReadOnly(True)
        self._alert_log.setFont(QFont("Cascadia Code", 11))
        self._alert_log.setStyleSheet(
            f"background-color: {C['surface']}; "
            f"color: {C['text']}; "
            f"border: 1px solid {C['border']}; "
            f"border-radius: 4px; "
            f"padding: 4px;"
        )

        layout.addWidget(self._alert_log)
        return box

    # ------------------------------------------------------------------
    # 4. DAILY RISK
    # ------------------------------------------------------------------

    def _build_daily_risk_box(self) -> QGroupBox:
        """Panel showing daily risk usage: trades, DD, equity."""
        box = QGroupBox("DAILY RISK")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        risk_fields = [
            ("Trades:",    "_lbl_risk_trades"),
            ("Daily DD:",  "_lbl_risk_daily_dd"),
            ("Total DD:",  "_lbl_risk_total_dd"),
            ("Equity:",    "_lbl_risk_equity"),
        ]

        for row, (label_text, attr_name) in enumerate(risk_fields):
            layout.addWidget(_make_label(label_text), row, 0)
            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row, 1)

        layout.setRowStretch(len(risk_fields), 1)
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

            safety_status : dict
                state (str "STOPPED"/"RUNNING"/"WINDING DOWN"/"CB PAUSED"),
                circuit_breaker (str "OFF" or "PAUSED (2:30 remaining)"),
                weekend_close (str e.g. "5h away" or "N/A"),
                daily_dd_current (float), daily_dd_limit (float),
                consec_losses (int),
                connection (str "Stable"/"Unstable"/"Disconnected")
            alerts : list[dict]
                Each dict: timestamp (str), level (str "INFO"/"WARN"/"ERROR"),
                message (str)
            daily_risk : dict
                trades_current (int), trades_soft_cap (int),
                trades_hard_cap (int),
                daily_dd_current (float), daily_dd_halt (float),
                total_dd_current (float), total_dd_limit (float),
                equity (float)
            trading_state : str
                Current state for button enable/disable logic
        """
        if not data:
            return

        self._update_safety_status(data.get("safety_status", {}))
        self._update_alerts(data.get("alerts", []))
        self._update_daily_risk(data.get("daily_risk", {}))
        self._update_button_states(data.get("trading_state", self._trading_state))

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_safety_status(self, status: dict) -> None:
        """Update the safety status panel."""
        if not status:
            return

        # Trading state
        state = status.get("state", "STOPPED")
        state_key_map = {
            "STOPPED": "stopped",
            "RUNNING": "running",
            "WINDING DOWN": "winding_down",
            "CB PAUSED": "cb_paused",
        }
        color_key = state_key_map.get(state, "stopped")
        state_color = STATE_COLORS.get(color_key, C["label"])

        self._state_dot.setStyleSheet(
            f"color: {state_color}; font-size: 18px; "
            f"background: transparent; border: none;"
        )
        self._lbl_state.setText(state)
        self._lbl_state.setStyleSheet(
            f"color: {state_color}; font-size: 16px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        # Circuit Breaker
        cb = status.get("circuit_breaker", "--")
        cb_is_off = "OFF" in str(cb).upper()
        cb_color = C["green"] if cb_is_off else C["peach"]
        self._lbl_circuit_breaker.setText(str(cb))
        self._lbl_circuit_breaker.setStyleSheet(
            f"color: {cb_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_circuit_breaker_dot.setStyleSheet(
            f"color: {cb_color}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Weekend close
        weekend = status.get("weekend_close", "--")
        self._lbl_weekend.setText(str(weekend))
        self._lbl_weekend.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_weekend_dot.setStyleSheet(
            f"color: {C['text']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Daily DD
        dd_current = status.get("daily_dd_current", 0.0)
        dd_limit = status.get("daily_dd_limit", 3.0)
        dd_pct = (dd_current / dd_limit * 100) if dd_limit > 0 else 0
        dd_color = C["red"] if dd_pct > 80 else C["yellow"] if dd_pct > 50 else C["green"]
        self._lbl_daily_dd.setText(f"{dd_current:.1f}% / {dd_limit:.1f}%")
        self._lbl_daily_dd.setStyleSheet(
            f"color: {dd_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_daily_dd_dot.setStyleSheet(
            f"color: {dd_color}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Consecutive losses
        consec = status.get("consec_losses", 0)
        consec_color = C["red"] if consec >= 3 else C["yellow"] if consec >= 2 else C["green"]
        self._lbl_consec_losses.setText(str(consec))
        self._lbl_consec_losses.setStyleSheet(
            f"color: {consec_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_consec_losses_dot.setStyleSheet(
            f"color: {consec_color}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Connection
        conn = status.get("connection", "--")
        conn_upper = str(conn).upper()
        if conn_upper == "STABLE":
            conn_color = C["green"]
        elif conn_upper == "UNSTABLE":
            conn_color = C["yellow"]
        else:
            conn_color = C["red"]
        self._lbl_connection.setText(str(conn))
        self._lbl_connection.setStyleSheet(
            f"color: {conn_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_connection_dot.setStyleSheet(
            f"color: {conn_color}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

    def _update_alerts(self, alerts: list) -> None:
        """Rebuild the alert log from the latest alerts list.

        Each alert dict has: timestamp, level, message.
        Most recent alerts should be at the top.
        """
        if alerts is None:
            return

        # Build HTML content (newest first)
        html_lines = []
        for alert in alerts:
            ts = alert.get("timestamp", "")
            level = alert.get("level", "INFO").upper()
            message = alert.get("message", "")
            color = ALERT_COLORS.get(level, C["text"])

            html_lines.append(
                f'<span style="color:{C["subtext"]}">{ts}</span> '
                f'<span style="color:{color}">[{level}]</span> '
                f'<span style="color:{C["text"]}">{message}</span>'
            )

        self._alert_log.setHtml("<br>".join(html_lines))

    def _update_daily_risk(self, risk: dict) -> None:
        """Update the daily risk panel."""
        if not risk:
            return

        # Trades: current / soft cap / hard cap
        trades_cur = risk.get("trades_current", 0)
        trades_soft = risk.get("trades_soft_cap", 10)
        trades_hard = risk.get("trades_hard_cap", 20)
        trades_color = (
            C["red"] if trades_cur >= trades_hard
            else C["yellow"] if trades_cur >= trades_soft
            else C["text"]
        )
        self._lbl_risk_trades.setText(
            f"{trades_cur} / {trades_soft} soft / {trades_hard} hard"
        )
        self._lbl_risk_trades.setStyleSheet(
            f"color: {trades_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Daily DD: current / halt threshold
        dd_cur = risk.get("daily_dd_current", 0.0)
        dd_halt = risk.get("daily_dd_halt", 3.0)
        dd_pct = (dd_cur / dd_halt * 100) if dd_halt > 0 else 0
        dd_color = C["red"] if dd_pct > 80 else C["yellow"] if dd_pct > 50 else C["text"]
        self._lbl_risk_daily_dd.setText(f"{dd_cur:.1f}% / {dd_halt:.1f}% daily halt")
        self._lbl_risk_daily_dd.setStyleSheet(
            f"color: {dd_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Total DD: current / max limit
        total_dd_cur = risk.get("total_dd_current", 0.0)
        total_dd_limit = risk.get("total_dd_limit", 10.0)
        total_pct = (total_dd_cur / total_dd_limit * 100) if total_dd_limit > 0 else 0
        total_color = (
            C["red"] if total_pct > 80
            else C["yellow"] if total_pct > 50
            else C["text"]
        )
        self._lbl_risk_total_dd.setText(
            f"{total_dd_cur:.1f}% / {total_dd_limit:.1f}% total limit"
        )
        self._lbl_risk_total_dd.setStyleSheet(
            f"color: {total_color}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        # Equity
        equity = risk.get("equity")
        if equity is not None:
            self._lbl_risk_equity.setText(currency.fmt(equity))
            self._lbl_risk_equity.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

    def _update_button_states(self, state: str) -> None:
        """Enable/disable buttons based on the current trading state.

        State machine rules:
          STOPPED       -> Start enabled, others disabled (except Emergency)
          RUNNING       -> Wind Down + Stop enabled, Start disabled
          WINDING DOWN  -> Stop only
          CB PAUSED     -> Wind Down + Stop enabled, Reset CB enabled

        Emergency Stop is ALWAYS enabled.
        """
        self._trading_state = state

        if state == "STOPPED":
            self._btn_start.setEnabled(True)
            self._btn_wind_down.setEnabled(False)
            self._btn_stop.setEnabled(False)
            self._btn_reset_cb.setEnabled(False)
        elif state == "RUNNING":
            self._btn_start.setEnabled(False)
            self._btn_wind_down.setEnabled(True)
            self._btn_stop.setEnabled(True)
            self._btn_reset_cb.setEnabled(False)
        elif state == "WINDING DOWN":
            self._btn_start.setEnabled(False)
            self._btn_wind_down.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._btn_reset_cb.setEnabled(False)
        elif state == "CB PAUSED":
            self._btn_start.setEnabled(False)
            self._btn_wind_down.setEnabled(True)
            self._btn_stop.setEnabled(True)
            self._btn_reset_cb.setEnabled(True)
        else:
            # Unknown state -- disable all except emergency
            self._btn_start.setEnabled(False)
            self._btn_wind_down.setEnabled(False)
            self._btn_stop.setEnabled(False)
            self._btn_reset_cb.setEnabled(False)

        # Emergency Stop is ALWAYS enabled, regardless of state
        self._btn_emergency.setEnabled(True)
