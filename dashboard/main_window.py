"""Main window for the Spartus Live Trading Dashboard.

Provides a 6-tab layout with header bar, trading state controls,
and a 1 Hz update timer that pulls live data into every tab.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pyqtgraph as pg

from dashboard.theme import C, STATE_COLORS, get_stylesheet
from dashboard.widgets import StatusIndicator

# Pyqtgraph global dark config -- must be set before any PlotWidget is created
pg.setConfigOption("background", C["bg"])
pg.setConfigOption("foreground", C["text"])

log = logging.getLogger(__name__)

# File-based command channel: screenshot tool writes a tab index here,
# the dashboard reads it and switches tabs, then deletes the file.
_CMD_DIR = Path(__file__).resolve().parent.parent / "storage" / "state"
_TAB_CMD_FILE = _CMD_DIR / "_tab_switch_cmd.json"


# ---------------------------------------------------------------------------
# Trading states (matches core/orchestrator state machine)
# ---------------------------------------------------------------------------

class TradingState:
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    WINDING_DOWN = "WINDING DOWN"
    CB_PAUSED = "CB PAUSED"


# ---------------------------------------------------------------------------
# LiveDashboard
# ---------------------------------------------------------------------------

class LiveDashboard(QMainWindow):
    """Main application window for the Spartus Live Trading Dashboard.

    Contains 6 tabs:
    1. LIVE STATUS   -- Connection, account, position, AI decisions
    2. PERFORMANCE   -- Balance chart, rolling metrics, trade history
    3. TRADE JOURNAL -- Lesson summary, trade detail
    4. MODEL & FEATURES -- Model info, feature health, correlated feeds
    5. ALERTS & SAFETY  -- Safety status, controls, alert log
    6. ANALYTICS     -- Action distributions, training vs live, drift
    """

    def __init__(self) -> None:
        super().__init__()

        # ---- Backend component references (set later via set_components) ----
        self._mt5_bridge = None
        self._feature_pipeline = None
        self._inference_engine = None
        self._trade_executor = None
        self._risk_manager = None
        self._position_manager = None
        self._memory = None
        self._model_info: dict = {}

        # ---- Trading state ----
        self._trading_state: str = TradingState.STOPPED
        self._paper_trading: bool = True

        # ---- Tab widgets (populated by tab builders later) ----
        self._tab_widgets: dict = {}
        self._tabs: Optional[QTabWidget] = None

        # ---- Window setup ----
        self.setWindowTitle("SPARTUS LIVE TRADING \u2014 XAUUSD")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_stylesheet())

        # ---- Central widget ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Header bar (always visible)
        main_layout.addWidget(self._create_header_bar())

        # Tab widget
        self._create_tabs()
        main_layout.addWidget(self._tabs)

        # ---- Update timer ----
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update)

    # ------------------------------------------------------------------
    # Component injection
    # ------------------------------------------------------------------

    def set_components(
        self,
        mt5_bridge,
        feature_pipeline,
        inference_engine,
        trade_executor,
        risk_manager,
        position_manager,
        memory,
        model_info: dict,
    ) -> None:
        """Inject all backend components after initialisation.

        Must be called before ``start_update_timer()`` so the UI
        update loop has data to pull from.
        """
        self._mt5_bridge = mt5_bridge
        self._feature_pipeline = feature_pipeline
        self._inference_engine = inference_engine
        self._trade_executor = trade_executor
        self._risk_manager = risk_manager
        self._position_manager = position_manager
        self._memory = memory
        self._model_info = model_info or {}
        log.info("Dashboard components injected")

        # Re-evaluate button states now that components are available
        self._update_button_states()

    # ------------------------------------------------------------------
    # Header bar
    # ------------------------------------------------------------------

    def _create_header_bar(self) -> QWidget:
        """Top bar with title, state indicator, paper banner, and buttons."""
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {C['surface']}; "
            f"border: 1px solid {C['border']}; "
            f"border-radius: 6px;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        # ---- Title ----
        title = QLabel("SPARTUS LIVE TRADING")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(
            f"color: {C['cyan']}; border: none; background: transparent;"
        )
        layout.addWidget(title)

        # ---- Symbol badge ----
        symbol = QLabel("XAUUSD")
        symbol.setStyleSheet(
            f"color: {C['peach']}; font-weight: bold; font-size: 14px; "
            f"border: none; background: transparent;"
        )
        layout.addWidget(symbol)

        # ---- Trading state indicator ----
        self._state_indicator = StatusIndicator("State")
        self._state_indicator.set_status(TradingState.STOPPED, STATE_COLORS["stopped"])
        self._state_indicator.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self._state_indicator)

        # ---- Paper trading banner ----
        self._paper_banner = QLabel("  PAPER TRADING  ")
        self._paper_banner.setStyleSheet(
            f"color: {C['bg']}; "
            f"background-color: {C['yellow']}; "
            f"font-weight: bold; "
            f"font-size: 12px; "
            f"border-radius: 4px; "
            f"padding: 2px 8px; "
            f"border: none;"
        )
        self._paper_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._paper_banner)
        # Visibility toggled when paper_trading flag is set
        self._paper_banner.setVisible(self._paper_trading)

        layout.addItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        # ---- Control buttons ----

        # Start Trading
        self._btn_start = QPushButton("Start Trading")
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.clicked.connect(self.start_trading)
        layout.addWidget(self._btn_start)

        # Wind Down
        self._btn_wind_down = QPushButton("Wind Down")
        self._btn_wind_down.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_wind_down.clicked.connect(self.wind_down)
        self._btn_wind_down.setEnabled(False)
        layout.addWidget(self._btn_wind_down)

        # Stop Now
        self._btn_stop = QPushButton("Stop Now")
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.clicked.connect(self.stop_trading)
        self._btn_stop.setEnabled(False)
        layout.addWidget(self._btn_stop)

        # Emergency Stop -- always visible & enabled
        self._btn_emergency = QPushButton("Emergency Stop")
        self._btn_emergency.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_emergency.setStyleSheet(
            f"QPushButton {{ "
            f"  background-color: #cc0000; color: {C['white']}; "
            f"  font-weight: bold; border: 2px solid {C['red']}; "
            f"  border-radius: 4px; padding: 6px 16px; "
            f"}} "
            f"QPushButton:hover {{ background-color: #e60000; border-color: #ff5555; }} "
            f"QPushButton:pressed {{ background-color: #990000; border-color: #cc0000; }}"
        )
        self._btn_emergency.clicked.connect(self.emergency_stop)
        layout.addWidget(self._btn_emergency)

        # Apply initial styles based on STOPPED state
        self._update_button_states()

        return bar

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _create_tabs(self) -> None:
        """Create the QTabWidget with 6 placeholder tabs.

        Each tab is an empty QWidget with a QVBoxLayout.  The actual
        per-tab content is built by dedicated tab modules that populate
        these containers (tab builders receive the container widget).
        """
        self._tabs = QTabWidget()

        tab_names = [
            "LIVE STATUS",
            "PERFORMANCE",
            "TRADE JOURNAL",
            "MODEL & FEATURES",
            "ALERTS & SAFETY",
            "ANALYTICS",
        ]

        for name in tab_names:
            container = QWidget()
            container.setLayout(QVBoxLayout())
            container.layout().setContentsMargins(6, 6, 6, 6)
            self._tabs.addTab(container, name)
            self._tab_widgets[name] = container

    def get_tab(self, name: str) -> QWidget:
        """Return the container QWidget for a named tab.

        Used by tab builder modules to populate tab content.
        """
        return self._tab_widgets.get(name)

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def _start_update_timer(self) -> None:
        """Start the 1 Hz UI update timer."""
        self._update_timer.start(1000)
        log.info("Dashboard update timer started (1 Hz)")

    def _stop_update_timer(self) -> None:
        """Stop the update timer."""
        self._update_timer.stop()

    def _update(self) -> None:
        """Pull latest state from components and refresh all visible tabs.

        Called every second by the QTimer.  Silently catches exceptions
        so a single tab error does not crash the whole dashboard.
        """
        try:
            self._check_tab_command()
            self._update_state_indicator()
            self._update_paper_banner()
            self._update_button_states()
        except Exception:
            log.exception("Error in dashboard _update")

    def _check_tab_command(self) -> None:
        """Check for external tab-switch commands from the screenshot tool.

        The screenshot script writes a JSON file with ``{"tab": N}``.
        We switch to that tab and delete the file to acknowledge.
        """
        try:
            if _TAB_CMD_FILE.exists():
                data = json.loads(_TAB_CMD_FILE.read_text(encoding="utf-8"))
                tab_idx = data.get("tab")
                if tab_idx is not None and self._tabs is not None:
                    if 0 <= tab_idx < self._tabs.count():
                        self._tabs.setCurrentIndex(tab_idx)
                        log.debug("Tab switched to %d via command file", tab_idx)
                _TAB_CMD_FILE.unlink(missing_ok=True)
        except Exception:
            # Never let command file errors affect the main loop
            try:
                _TAB_CMD_FILE.unlink(missing_ok=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # State display helpers
    # ------------------------------------------------------------------

    def _update_state_indicator(self) -> None:
        """Refresh the header state dot + text."""
        state_to_color_key = {
            TradingState.STOPPED: "stopped",
            TradingState.RUNNING: "running",
            TradingState.WINDING_DOWN: "winding_down",
            TradingState.CB_PAUSED: "cb_paused",
        }
        color_key = state_to_color_key.get(self._trading_state, "stopped")
        self._state_indicator.set_status(
            self._trading_state, STATE_COLORS[color_key]
        )

    def _update_paper_banner(self) -> None:
        """Show / hide the paper trading banner."""
        self._paper_banner.setVisible(self._paper_trading)

    # -- Header button style helpers -----------------------------------------

    @staticmethod
    def _hdr_btn_interactive(bg: str, fg: str, hover_bg: str, pressed_bg: str) -> str:
        """Stylesheet for an enabled, clickable header button."""
        return (
            f"QPushButton {{ background-color: {bg}; color: {fg}; "
            f"font-weight: bold; border: none; border-radius: 4px; "
            f"padding: 6px 16px; }} "
            f"QPushButton:hover {{ background-color: {hover_bg}; }} "
            f"QPushButton:pressed {{ background-color: {pressed_bg}; "
            f"padding-top: 7px; padding-bottom: 5px; }}"
        )

    @staticmethod
    def _hdr_btn_active(color: str, tint_bg: str) -> str:
        """Stylesheet for a disabled button showing the active state."""
        return (
            f"QPushButton {{ background-color: {tint_bg}; color: {color}; "
            f"font-weight: bold; border: 2px solid {color}; "
            f"border-radius: 4px; padding: 6px 16px; }}"
        )

    @staticmethod
    def _hdr_btn_disabled() -> str:
        """Stylesheet for a greyed-out disabled button."""
        return (
            f"QPushButton {{ background-color: {C['surface']}; color: {C['dim']}; "
            f"font-weight: bold; border: 1px solid {C['border']}; "
            f"border-radius: 4px; padding: 6px 16px; }}"
        )

    # -------------------------------------------------------------------------

    def _update_button_states(self) -> None:
        """Enable / disable buttons and apply visual styles for state feedback.

        State machine rules:
          STOPPED       -> Start enabled (interactive), others disabled (grey)
          RUNNING       -> Start shows 'Running' (active glow), Wind Down + Stop interactive
          WINDING_DOWN  -> Wind Down shows 'Winding Down...' (active glow), Stop interactive
          CB_PAUSED     -> Start shows 'CB Paused' (active glow, orange), Wind Down + Stop interactive

        The active-state button uses a coloured border + tinted background so
        the user can tell the current state just from the buttons, without
        needing to read the status indicator.

        Emergency Stop is always visible, enabled, and styled.
        """
        state = self._trading_state
        components_ready = self._mt5_bridge is not None

        # Reusable interactive styles
        green = self._hdr_btn_interactive(C['green'], C['bg'], '#40e640', '#1fa01f')
        yellow = self._hdr_btn_interactive(C['yellow'], C['bg'], '#ffe04d', '#d4aa00')
        red = self._hdr_btn_interactive(C['red'], C['white'], '#ff5555', '#cc2222')
        disabled = self._hdr_btn_disabled()

        if state == TradingState.STOPPED:
            self._btn_start.setText("\u25b6  Start Trading")
            self._btn_start.setEnabled(components_ready)
            self._btn_start.setStyleSheet(green if components_ready else disabled)

            self._btn_wind_down.setText("Wind Down")
            self._btn_wind_down.setEnabled(False)
            self._btn_wind_down.setStyleSheet(disabled)

            self._btn_stop.setText("Stop Now")
            self._btn_stop.setEnabled(False)
            self._btn_stop.setStyleSheet(disabled)

        elif state == TradingState.RUNNING or state == TradingState.CB_PAUSED:
            # Start button -> active indicator
            if state == TradingState.RUNNING:
                self._btn_start.setText("\u25cf  Running")
                self._btn_start.setStyleSheet(
                    self._hdr_btn_active(C['green'], '#0a2a0a')
                )
            else:
                self._btn_start.setText("\u25cf  CB Paused")
                self._btn_start.setStyleSheet(
                    self._hdr_btn_active(C['peach'], '#2a1a0a')
                )
            self._btn_start.setEnabled(False)

            self._btn_wind_down.setText("Wind Down")
            self._btn_wind_down.setEnabled(True)
            self._btn_wind_down.setStyleSheet(yellow)

            self._btn_stop.setText("Stop Now")
            self._btn_stop.setEnabled(True)
            self._btn_stop.setStyleSheet(red)

        elif state == TradingState.WINDING_DOWN:
            self._btn_start.setText("\u25b6  Start Trading")
            self._btn_start.setEnabled(False)
            self._btn_start.setStyleSheet(disabled)

            # Wind Down button -> active indicator
            self._btn_wind_down.setText("\u25cf  Winding Down\u2026")
            self._btn_wind_down.setEnabled(False)
            self._btn_wind_down.setStyleSheet(
                self._hdr_btn_active(C['yellow'], '#2a2500')
            )

            self._btn_stop.setText("Stop Now")
            self._btn_stop.setEnabled(True)
            self._btn_stop.setStyleSheet(red)

        # Emergency Stop is always enabled
        self._btn_emergency.setEnabled(True)

    # ------------------------------------------------------------------
    # Public state setter
    # ------------------------------------------------------------------

    def set_trading_state(self, state: str) -> None:
        """Update the trading state from the orchestrator."""
        self._trading_state = state
        self._update_state_indicator()
        self._update_button_states()
        log.info(f"Dashboard trading state -> {state}")

    def set_paper_trading(self, enabled: bool) -> None:
        """Toggle paper trading mode display."""
        self._paper_trading = enabled
        self._update_paper_banner()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def start_trading(self) -> None:
        """Callback for the Start Trading button.

        Sets state to RUNNING and starts the update timer.
        The orchestrator should connect to this signal or call this
        method after its own startup sequence completes.
        """
        if self._mt5_bridge is None:
            log.warning("Cannot start trading: components not injected")
            return

        self._trading_state = TradingState.RUNNING
        self._update_state_indicator()
        self._update_button_states()
        self._start_update_timer()
        log.info("Trading started from dashboard")

    def stop_trading(self) -> None:
        """Callback for the Stop Now button.

        Immediately sets state to STOPPED and stops the update timer.
        The orchestrator should close any open positions.
        """
        self._trading_state = TradingState.STOPPED
        self._update_state_indicator()
        self._update_button_states()
        self._stop_update_timer()
        log.info("Trading stopped from dashboard")

    def wind_down(self) -> None:
        """Callback for the Wind Down button.

        Sets state to WINDING_DOWN.  The orchestrator should stop
        opening new positions and close existing ones as they hit TP/SL.
        """
        self._trading_state = TradingState.WINDING_DOWN
        self._update_state_indicator()
        self._update_button_states()
        log.info("Wind down initiated from dashboard")

    def emergency_stop(self) -> None:
        """Callback for the Emergency Stop button.

        Forces STOPPED state, stops the timer, and signals the
        orchestrator to close all positions immediately.
        """
        self._trading_state = TradingState.STOPPED
        self._update_state_indicator()
        self._update_button_states()
        self._stop_update_timer()

        # Attempt immediate close-all via trade executor
        if self._trade_executor is not None:
            try:
                self._trade_executor.close_all_positions()
            except Exception:
                log.exception("Error during emergency close-all")

        log.critical("EMERGENCY STOP activated from dashboard")
