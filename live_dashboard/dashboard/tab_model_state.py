"""Tab 4 -- Model & Features for the Spartus Live Trading Dashboard.

Shows model metadata and last action (left), plus feature health
and correlated feed status (right).

Layout::

    +---------------------------+----------------------------------------+
    |  MODEL INFO               |  FEATURE HEALTH                        |
    |  Model: (auto-discovered) |  * N/N features active                 |
    |  Trained: Week NNN        |  * Normalizer: 200/200 bars            |
    |  Val Sharpe: X.XXX        |  * Frame buffer: 10/10 frames          |
    |  Architecture: [256,256]  |  * NaN rate: 0.0%                      |
    |  Features: N (N*F obs)    |  * Inf values: 0                       |
    |  Actions: N continuous    |  * Constant feats: 0                   |
    |                           |                                        |
    |  LAST ACTION              |  CORRELATED FEEDS                      |
    |  direction: 0.65          |  * EURUSD: OK (2.1s ago)               |
    |  conviction: 0.72         |  * XAGUSD: OK (2.1s ago)               |
    |  exit: 0.23               |  * USDJPY: OK (2.1s ago)               |
    |  sl_adj: 0.41             |  * US500: OK (4.3s ago)                |
    |                           |  * USOIL: STALE (neutral fill)         |
    +---------------------------+----------------------------------------+

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels -- NEVER dark gray on dark backgrounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QGroupBox, QLabel, QDoubleSpinBox, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from dashboard.theme import C
from dashboard import currency


# ---------------------------------------------------------------------------
# Correlated instrument list (order matches project spec)
# ---------------------------------------------------------------------------

CORRELATED_SYMBOLS = ["EURUSD", "XAGUSD", "USDJPY", "US500", "USOIL"]


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
# ModelStateTab
# ---------------------------------------------------------------------------

class ModelStateTab(QWidget):
    """Tab 4: Model & Features.

    Left side shows model metadata and the last raw action values.
    Right side shows feature pipeline health and correlated data
    feed status.

    Call ``update_data(data)`` each tick with the latest state dict.
    """

    # Emitted when user clicks Save in AI Trade Protection section.
    # Dict keys: be_trigger_gbp, be_buffer_pips, lock_trigger_gbp,
    #            lock_amount_gbp, trail_trigger_gbp, trail_atr_mult
    ai_protection_save_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ----- Root layout with horizontal splitter -----
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {C['border']}; width: 2px; }}"
        )

        # Left pane: Model Info (top) + Last Action (mid) + AI Protection (bottom)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._build_model_info_box(), stretch=2)
        left_layout.addWidget(self._build_last_action_box(), stretch=2)
        left_layout.addWidget(self._build_ai_protection_box(), stretch=4)

        # Right pane: Feature Health + Correlated Feeds + Calendar + Reward
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self._build_feature_health_box(), stretch=3)
        right_layout.addWidget(self._build_correlated_feeds_box(), stretch=2)
        right_layout.addWidget(self._build_calendar_box(), stretch=1)
        right_layout.addWidget(self._build_reward_state_box(), stretch=1)

        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        root.addWidget(splitter)

    # ==================================================================
    # Section builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. MODEL INFO
    # ------------------------------------------------------------------

    def _build_model_info_box(self) -> QGroupBox:
        """Panel showing model metadata."""
        box = QGroupBox("MODEL INFO")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        fields = [
            ("Model:",        "_lbl_model_name"),
            ("Trained:",      "_lbl_trained_week"),
            ("Val Sharpe:",   "_lbl_val_sharpe"),
            ("Architecture:", "_lbl_architecture"),
            ("Features:",     "_lbl_features"),
            ("Actions:",      "_lbl_actions"),
        ]

        for row, (label_text, attr_name) in enumerate(fields):
            layout.addWidget(_make_label(label_text), row, 0)
            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row, 1)

        layout.setRowStretch(len(fields), 1)
        layout.setColumnStretch(1, 1)
        return box

    # ------------------------------------------------------------------
    # 2. LAST ACTION
    # ------------------------------------------------------------------

    def _build_last_action_box(self) -> QGroupBox:
        """Panel showing the last raw action values from the model."""
        box = QGroupBox("LAST ACTION")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        action_fields = [
            ("direction:",  "_lbl_action_direction"),
            ("conviction:", "_lbl_action_conviction"),
            ("exit:",       "_lbl_action_exit"),
            ("sl_adj:",     "_lbl_action_sl_adj"),
        ]

        for row, (label_text, attr_name) in enumerate(action_fields):
            layout.addWidget(_make_label(label_text), row, 0)
            value_label = _make_label("--", C["text"], bold=True)
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row, 1)

        layout.setRowStretch(len(action_fields), 1)
        layout.setColumnStretch(1, 1)
        return box

    # ------------------------------------------------------------------
    # 2b. AI TRADE PROTECTION (user-adjustable)
    # ------------------------------------------------------------------

    def _build_ai_protection_box(self) -> QGroupBox:
        """Panel with spinboxes for AI trade protection settings."""
        box = QGroupBox("AI TRADE PROTECTION")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(6)

        row = 0

        # Section label: Breakeven
        lbl_be = _make_label("BREAKEVEN", C["cyan"], bold=True, font_size=11)
        layout.addWidget(lbl_be, row, 0, 1, 3)
        row += 1

        layout.addWidget(_make_label("Trigger:"), row, 0)
        self._spin_ai_be_trigger = self._make_protection_spin(0.10, 100.0, 2.00, currency.sym())
        layout.addWidget(self._spin_ai_be_trigger, row, 1)
        layout.addWidget(_make_label("profit to move SL to breakeven",
                                     C["subtext"], font_size=11), row, 2)
        row += 1

        layout.addWidget(_make_label("Buffer:"), row, 0)
        self._spin_ai_be_buffer = self._make_protection_spin(0.0, 2.0, 0.5, "pips")
        layout.addWidget(self._spin_ai_be_buffer, row, 1)
        layout.addWidget(_make_label("pips above entry for breakeven SL",
                                     C["subtext"], font_size=11), row, 2)
        row += 1

        # Section label: Profit Lock
        lbl_lock = _make_label("PROFIT LOCK", C["cyan"], bold=True, font_size=11)
        layout.addWidget(lbl_lock, row, 0, 1, 3)
        row += 1

        layout.addWidget(_make_label("Trigger:"), row, 0)
        self._spin_ai_lock_trigger = self._make_protection_spin(0.10, 200.0, 3.00, currency.sym())
        layout.addWidget(self._spin_ai_lock_trigger, row, 1)
        layout.addWidget(_make_label("profit to activate lock",
                                     C["subtext"], font_size=11), row, 2)
        row += 1

        layout.addWidget(_make_label("Lock:"), row, 0)
        self._spin_ai_lock_amount = self._make_protection_spin(0.10, 100.0, 1.50, currency.sym())
        layout.addWidget(self._spin_ai_lock_amount, row, 1)
        layout.addWidget(_make_label("guaranteed profit locked in",
                                     C["subtext"], font_size=11), row, 2)
        row += 1

        # Section label: Trailing Stop
        lbl_trail = _make_label("TRAILING STOP", C["cyan"], bold=True, font_size=11)
        layout.addWidget(lbl_trail, row, 0, 1, 3)
        row += 1

        layout.addWidget(_make_label("Trigger:"), row, 0)
        self._spin_ai_trail_trigger = self._make_protection_spin(0.10, 200.0, 4.00, currency.sym())
        layout.addWidget(self._spin_ai_trail_trigger, row, 1)
        layout.addWidget(_make_label("profit to start trailing stop",
                                     C["subtext"], font_size=11), row, 2)
        row += 1

        layout.addWidget(_make_label("Distance:"), row, 0)
        self._spin_ai_trail_atr = self._make_protection_spin(0.3, 3.0, 1.0, "ATR")
        layout.addWidget(self._spin_ai_trail_atr, row, 1)
        layout.addWidget(_make_label("ATR multiplier for trail distance",
                                     C["subtext"], font_size=11), row, 2)
        row += 1

        # Button row: Save + Reset
        btn_row = QHBoxLayout()

        self._btn_ai_save = QPushButton("Save AI Protection")
        self._btn_ai_save.setFixedHeight(30)
        self._btn_ai_save.setStyleSheet(
            f"QPushButton {{ background: {C['bg']}; color: {C['cyan']}; "
            f"border: 1px solid {C['cyan']}; border-radius: 4px; "
            f"font-size: 12px; font-weight: bold; padding: 4px 16px; }}"
            f"QPushButton:hover {{ background: {C['cyan']}; color: {C['bg']}; }}"
        )
        self._btn_ai_save.clicked.connect(self._on_ai_save_clicked)
        btn_row.addWidget(self._btn_ai_save)

        self._btn_ai_reset = QPushButton("Reset to Default")
        self._btn_ai_reset.setFixedHeight(30)
        self._btn_ai_reset.setStyleSheet(
            f"QPushButton {{ background: {C['bg']}; color: {C['label']}; "
            f"border: 1px solid {C['border']}; border-radius: 4px; "
            f"font-size: 12px; padding: 4px 12px; }}"
            f"QPushButton:hover {{ background: {C['surface2']}; color: {C['text']}; }}"
        )
        self._btn_ai_reset.clicked.connect(self._on_ai_reset_clicked)
        btn_row.addWidget(self._btn_ai_reset)

        self._lbl_ai_save_status = _make_label("", C["subtext"], font_size=11)
        btn_row.addWidget(self._lbl_ai_save_status)
        btn_row.addStretch()

        layout.addLayout(btn_row, row, 0, 1, 3)
        row += 1

        layout.setRowStretch(row, 1)
        layout.setColumnStretch(2, 1)
        return box

    def _make_protection_spin(self, lo: float, hi: float, default: float,
                              suffix: str) -> QDoubleSpinBox:
        """Create a styled spinbox for AI protection settings."""
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setValue(default)
        spin.setSuffix(f" {suffix}")
        spin.setFixedWidth(100)
        spin.setStyleSheet(
            f"QDoubleSpinBox {{ background-color: {C['surface2']}; color: {C['text']}; "
            f"border: 1px solid {C['border']}; border-radius: 3px; "
            f"padding: 2px 4px; font-size: 12px; min-height: 26px; }}"
            f"QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ "
            f"background-color: {C['surface2']}; width: 20px; "
            f"border-left: 1px solid {C['border']}; }}"
            f"QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ "
            f"background-color: {C['border']}; }}"
            f"QDoubleSpinBox::up-arrow {{ image: none; border-left: 5px solid transparent; "
            f"border-right: 5px solid transparent; border-bottom: 6px solid {C['text']}; "
            f"width: 0; height: 0; }}"
            f"QDoubleSpinBox::down-arrow {{ image: none; border-left: 5px solid transparent; "
            f"border-right: 5px solid transparent; border-top: 6px solid {C['text']}; "
            f"width: 0; height: 0; }}"
        )
        spin.valueChanged.connect(self._on_ai_setting_changed)
        return spin

    def _on_ai_setting_changed(self) -> None:
        """Mark AI protection settings as unsaved."""
        self._lbl_ai_save_status.setText("(unsaved changes)")
        self._lbl_ai_save_status.setStyleSheet(
            f"color: {C['yellow']}; font-size: 11px; "
            f"background: transparent; border: none;"
        )

    def _on_ai_save_clicked(self) -> None:
        """Emit AI protection settings for persistence."""
        settings = self.get_ai_protection_settings()
        self.ai_protection_save_requested.emit(settings)
        self._lbl_ai_save_status.setText("Settings saved")
        self._lbl_ai_save_status.setStyleSheet(
            f"color: {C['green']}; font-size: 11px; "
            f"background: transparent; border: none;"
        )

    def _on_ai_reset_clicked(self) -> None:
        """Reset AI protection spinboxes to default £ values."""
        defaults = {
            "be_trigger_gbp": 2.00,
            "be_buffer_pips": 0.5,
            "lock_trigger_gbp": 3.00,
            "lock_amount_gbp": 1.50,
            "trail_trigger_gbp": 4.00,
            "trail_atr_mult": 1.0,
        }
        self.load_ai_protection_settings(defaults)
        self._lbl_ai_save_status.setText("(unsaved — defaults loaded)")
        self._lbl_ai_save_status.setStyleSheet(
            f"color: {C['yellow']}; font-size: 11px; "
            f"background: transparent; border: none;"
        )

    def get_ai_protection_settings(self) -> dict:
        """Return current AI protection spinbox values as a dict."""
        return {
            "be_trigger_gbp": self._spin_ai_be_trigger.value(),
            "be_buffer_pips": self._spin_ai_be_buffer.value(),
            "lock_trigger_gbp": self._spin_ai_lock_trigger.value(),
            "lock_amount_gbp": self._spin_ai_lock_amount.value(),
            "trail_trigger_gbp": self._spin_ai_trail_trigger.value(),
            "trail_atr_mult": self._spin_ai_trail_atr.value(),
        }

    def load_ai_protection_settings(self, settings: dict) -> None:
        """Load AI protection values into spinboxes (blocks signals)."""
        mapping = {
            "be_trigger_gbp": self._spin_ai_be_trigger,
            "be_buffer_pips": self._spin_ai_be_buffer,
            "lock_trigger_gbp": self._spin_ai_lock_trigger,
            "lock_amount_gbp": self._spin_ai_lock_amount,
            "trail_trigger_gbp": self._spin_ai_trail_trigger,
            "trail_atr_mult": self._spin_ai_trail_atr,
        }
        for key, spin in mapping.items():
            if key in settings:
                spin.blockSignals(True)
                spin.setValue(float(settings[key]))
                spin.blockSignals(False)

    # ------------------------------------------------------------------
    # 3. FEATURE HEALTH
    # ------------------------------------------------------------------

    def _build_feature_health_box(self) -> QGroupBox:
        """Panel showing feature pipeline health metrics."""
        box = QGroupBox("FEATURE HEALTH")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        health_fields = [
            ("Features active:",  "_lbl_features_active"),
            ("Normalizer:",       "_lbl_normalizer_fill"),
            ("Frame buffer:",     "_lbl_frame_buffer"),
            ("NaN rate:",         "_lbl_nan_rate"),
            ("Inf values:",       "_lbl_inf_count"),
            ("Constant feats:",   "_lbl_constant_feats"),
        ]

        for row, (label_text, attr_name) in enumerate(health_fields):
            # Dot indicator + label text
            dot = QLabel("\u25cf")
            dot.setStyleSheet(
                f"color: {C['label']}; font-size: 12px; "
                f"background: transparent; border: none;"
            )
            dot.setFixedWidth(16)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            setattr(self, f"{attr_name}_dot", dot)
            layout.addWidget(dot, row, 0)

            layout.addWidget(_make_label(label_text), row, 1)

            value_label = _make_label("--", C["text"])
            setattr(self, attr_name, value_label)
            layout.addWidget(value_label, row, 2)

        layout.setRowStretch(len(health_fields), 1)
        layout.setColumnStretch(2, 1)
        return box

    # ------------------------------------------------------------------
    # 4. CORRELATED FEEDS
    # ------------------------------------------------------------------

    def _build_correlated_feeds_box(self) -> QGroupBox:
        """Panel showing status of each correlated instrument data feed."""
        box = QGroupBox("CORRELATED FEEDS")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        self._feed_dots: dict[str, QLabel] = {}
        self._feed_labels: dict[str, QLabel] = {}

        for row, symbol in enumerate(CORRELATED_SYMBOLS):
            # Dot
            dot = QLabel("\u25cf")
            dot.setStyleSheet(
                f"color: {C['label']}; font-size: 12px; "
                f"background: transparent; border: none;"
            )
            dot.setFixedWidth(16)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._feed_dots[symbol] = dot
            layout.addWidget(dot, row, 0)

            # Symbol name
            layout.addWidget(_make_label(f"{symbol}:", C["subtext"], bold=True), row, 1)

            # Status text
            status_lbl = _make_label("--", C["text"])
            self._feed_labels[symbol] = status_lbl
            layout.addWidget(status_lbl, row, 2)

        layout.setRowStretch(len(CORRELATED_SYMBOLS), 1)
        layout.setColumnStretch(2, 1)
        return box

    # ------------------------------------------------------------------
    # 5. CALENDAR INFO
    # ------------------------------------------------------------------

    def _build_calendar_box(self) -> QGroupBox:
        """Panel showing calendar event summary."""
        box = QGroupBox("CALENDAR")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(_make_label("Events today:"), 0, 0)
        self._lbl_events_today = _make_label("--", C["text"])
        layout.addWidget(self._lbl_events_today, 0, 1)

        layout.addWidget(_make_label("Next event:"), 1, 0)
        self._lbl_next_event = _make_label("--", C["text"])
        layout.addWidget(self._lbl_next_event, 1, 1)

        layout.setColumnStretch(1, 1)
        return box

    # ------------------------------------------------------------------
    # 6. REWARD NORMALIZER STATE
    # ------------------------------------------------------------------

    def _build_reward_state_box(self) -> QGroupBox:
        """Panel showing reward normalizer state from model package."""
        box = QGroupBox("REWARD NORMALIZER")
        layout = QGridLayout(box)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(_make_label("Running mean:"), 0, 0)
        self._lbl_reward_mean = _make_label("--", C["text"])
        layout.addWidget(self._lbl_reward_mean, 0, 1)

        layout.addWidget(_make_label("Running var:"), 1, 0)
        self._lbl_reward_var = _make_label("--", C["text"])
        layout.addWidget(self._lbl_reward_var, 1, 1)

        layout.addWidget(_make_label("Count:"), 2, 0)
        self._lbl_reward_count = _make_label("--", C["text"])
        layout.addWidget(self._lbl_reward_count, 2, 1)

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

            model_info : dict
                name (str), trained_week (int), val_sharpe (float),
                architecture (str), feature_count (int),
                obs_dim (int), action_count (int)
            last_action : dict
                direction (float), conviction (float),
                exit_urgency (float), sl_adjustment (float)
            feature_health : dict
                active (int), total (int),
                normalizer_fill (int), normalizer_capacity (int),
                frame_fill (int), frame_capacity (int),
                nan_rate (float), inf_count (int),
                constant_count (int)
            correlated_feeds : dict
                Mapping symbol -> dict with:
                    status (str "OK" or "STALE"),
                    age_sec (float), note (str optional)
        """
        if not data:
            return

        self._update_model_info(data.get("model_info", {}))
        self._update_last_action(data.get("last_action", {}))
        self._update_feature_health(data.get("feature_health", {}))
        self._update_correlated_feeds(data.get("correlated_feeds", {}))
        self._update_calendar(data.get("calendar_info", {}))
        self._update_reward_state(data.get("reward_state", {}))

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_model_info(self, info: dict) -> None:
        """Update model metadata panel."""
        if not info:
            return

        name = info.get("name", "--")
        self._lbl_model_name.setText(str(name))
        self._lbl_model_name.setStyleSheet(
            f"color: {C['cyan']}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        week = info.get("trained_week")
        if week is not None:
            self._lbl_trained_week.setText(f"Week {week}")
        self._lbl_trained_week.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        sharpe = info.get("val_sharpe")
        if sharpe is not None:
            sharpe_color = (
                C["green"] if sharpe >= 1.0
                else C["text"] if sharpe >= 0.0
                else C["red"]
            )
            self._lbl_val_sharpe.setText(f"{sharpe:.3f}")
            self._lbl_val_sharpe.setStyleSheet(
                f"color: {sharpe_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        arch = info.get("architecture", "--")
        self._lbl_architecture.setText(str(arch))
        self._lbl_architecture.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        feat_count = info.get("feature_count", "?")
        obs_dim = info.get("obs_dim", "?")
        self._lbl_features.setText(f"{feat_count} ({obs_dim} obs)")
        self._lbl_features.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

        action_count = info.get("action_count", "?")
        self._lbl_actions.setText(f"{action_count} continuous")
        self._lbl_actions.setStyleSheet(
            f"color: {C['text']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )

    def _update_last_action(self, action: dict) -> None:
        """Update last action panel with color-coded values."""
        if not action:
            return

        # Direction: green if positive (bullish), red if negative (bearish)
        direction = action.get("direction")
        if direction is not None:
            dir_color = C["green"] if direction >= 0 else C["red"]
            self._lbl_action_direction.setText(f"{direction:+.3f}")
            self._lbl_action_direction.setStyleSheet(
                f"color: {dir_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # Conviction: cyan when high (>0.6), text otherwise
        conviction = action.get("conviction")
        if conviction is not None:
            conv_color = C["cyan"] if conviction > 0.6 else C["text"]
            self._lbl_action_conviction.setText(f"{conviction:.3f}")
            self._lbl_action_conviction.setStyleSheet(
                f"color: {conv_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # Exit urgency: yellow/red when high, text when low
        exit_val = action.get("exit_urgency")
        if exit_val is not None:
            exit_color = (
                C["red"] if exit_val > 0.7
                else C["yellow"] if exit_val > 0.4
                else C["text"]
            )
            self._lbl_action_exit.setText(f"{exit_val:.3f}")
            self._lbl_action_exit.setStyleSheet(
                f"color: {exit_color}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

        # SL adjustment: text color (neutral info)
        sl_adj = action.get("sl_adjustment")
        if sl_adj is not None:
            self._lbl_action_sl_adj.setText(f"{sl_adj:.3f}")
            self._lbl_action_sl_adj.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; font-weight: bold; "
                f"background: transparent; border: none;"
            )

    def _update_feature_health(self, health: dict) -> None:
        """Update feature health panel with colored status dots."""
        if not health:
            return

        # Features active
        active = health.get("active", 0)
        total = health.get("total", 67)
        is_full = active == total
        self._lbl_features_active.setText(f"{active}/{total} features active")
        self._lbl_features_active.setStyleSheet(
            f"color: {C['green'] if is_full else C['red']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_features_active_dot.setStyleSheet(
            f"color: {C['green'] if is_full else C['red']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Normalizer fill
        norm_fill = health.get("normalizer_fill", 0)
        norm_cap = health.get("normalizer_capacity", 200)
        norm_ok = norm_fill >= norm_cap
        self._lbl_normalizer_fill.setText(f"{norm_fill}/{norm_cap} bars")
        self._lbl_normalizer_fill.setStyleSheet(
            f"color: {C['green'] if norm_ok else C['yellow']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_normalizer_fill_dot.setStyleSheet(
            f"color: {C['green'] if norm_ok else C['yellow']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Frame buffer fill
        frame_fill = health.get("frame_fill", 0)
        frame_cap = health.get("frame_capacity", 10)
        frame_ok = frame_fill >= frame_cap
        self._lbl_frame_buffer.setText(f"{frame_fill}/{frame_cap} frames")
        self._lbl_frame_buffer.setStyleSheet(
            f"color: {C['green'] if frame_ok else C['yellow']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_frame_buffer_dot.setStyleSheet(
            f"color: {C['green'] if frame_ok else C['yellow']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # NaN rate
        nan_rate = health.get("nan_rate", 0.0)
        nan_ok = nan_rate == 0.0
        self._lbl_nan_rate.setText(f"{nan_rate:.1f}%")
        self._lbl_nan_rate.setStyleSheet(
            f"color: {C['green'] if nan_ok else C['red']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_nan_rate_dot.setStyleSheet(
            f"color: {C['green'] if nan_ok else C['red']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Inf count
        inf_count = health.get("inf_count", 0)
        inf_ok = inf_count == 0
        self._lbl_inf_count.setText(str(inf_count))
        self._lbl_inf_count.setStyleSheet(
            f"color: {C['green'] if inf_ok else C['red']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_inf_count_dot.setStyleSheet(
            f"color: {C['green'] if inf_ok else C['red']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

        # Constant features
        const_count = health.get("constant_count", 0)
        const_ok = const_count == 0
        self._lbl_constant_feats.setText(str(const_count))
        self._lbl_constant_feats.setStyleSheet(
            f"color: {C['green'] if const_ok else C['yellow']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        self._lbl_constant_feats_dot.setStyleSheet(
            f"color: {C['green'] if const_ok else C['yellow']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )

    def _update_correlated_feeds(self, feeds: dict) -> None:
        """Update correlated instrument feed status indicators."""
        if not feeds:
            return

        for symbol in CORRELATED_SYMBOLS:
            feed_info = feeds.get(symbol, {})
            status = feed_info.get("status", "UNKNOWN")
            age_sec = feed_info.get("age_sec")
            note = feed_info.get("note", "")

            dot = self._feed_dots.get(symbol)
            lbl = self._feed_labels.get(symbol)

            if dot is None or lbl is None:
                continue

            if status.upper() == "OK":
                color = C["green"]
                if age_sec is not None:
                    text = f"OK ({age_sec:.1f}s ago)"
                else:
                    text = "OK"
            elif status.upper() == "STALE":
                color = C["red"]
                text = f"STALE"
                if note:
                    text += f" ({note})"
                elif age_sec is not None:
                    text += f" ({age_sec:.0f}s ago)"
            else:
                color = C["label"]
                text = status
                if note:
                    text += f" ({note})"

            dot.setStyleSheet(
                f"color: {color}; font-size: 12px; "
                f"background: transparent; border: none;"
            )
            lbl.setText(text)
            lbl.setStyleSheet(
                f"color: {color}; font-size: 13px; "
                f"background: transparent; border: none;"
            )

    def _update_calendar(self, info: dict) -> None:
        """Update calendar events panel."""
        if not info:
            return

        events_today = info.get("events_today", 0)
        color = C["yellow"] if events_today > 0 else C["text"]
        self._lbl_events_today.setText(str(events_today))
        self._lbl_events_today.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )

        next_time = info.get("next_event_time")
        next_name = info.get("next_event_name")
        if next_time and next_name:
            self._lbl_next_event.setText(f"{next_time} - {next_name}")
            self._lbl_next_event.setStyleSheet(
                f"color: {C['peach']}; font-size: 13px; "
                f"background: transparent; border: none;"
            )
        else:
            self._lbl_next_event.setText("None upcoming")
            self._lbl_next_event.setStyleSheet(
                f"color: {C['label']}; font-size: 13px; "
                f"background: transparent; border: none;"
            )

    def _update_reward_state(self, state: dict) -> None:
        """Update reward normalizer state panel."""
        if not state:
            return

        mean = state.get("running_mean")
        var = state.get("running_var") or state.get("running_variance")
        count = state.get("count") or state.get("n")

        if mean is not None:
            self._lbl_reward_mean.setText(f"{mean:.4f}")
            self._lbl_reward_mean.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; "
                f"background: transparent; border: none;"
            )

        if var is not None:
            self._lbl_reward_var.setText(f"{var:.4f}")
            self._lbl_reward_var.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; "
                f"background: transparent; border: none;"
            )

        if count is not None:
            self._lbl_reward_count.setText(f"{count:,}")
            self._lbl_reward_count.setStyleSheet(
                f"color: {C['text']}; font-size: 13px; "
                f"background: transparent; border: none;"
            )
