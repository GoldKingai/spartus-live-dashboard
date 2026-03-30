"""Tab 7: Manual Trade Management -- Live position tracking & SL protection.

Detects manually-opened positions and applies staged profit protection.
Shows real-time P/L, R-multiple, protection stage, and SL movement.
Protection settings are INDEPENDENT from AI trade protection.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QScrollArea,
)

from dashboard.theme import C
from dashboard import currency

log = logging.getLogger(__name__)

# Font sizes (centralised for easy tuning)
_FS_HEADING = 15     # Section headings
_FS_BODY = 13        # Body text / descriptions
_FS_LABEL = 13       # Labels next to controls
_FS_NOTE = 12        # Tips and notes
_FS_SPIN = 14        # Spinbox value text
_FS_TABLE = 11       # Table cell text
_FS_TRACKING = 13    # Tracking summary line


class SettingsTab(QWidget):
    """Tab 7: Manual Trade Management.

    Signals
    -------
    manual_trade_toggled(bool)
        Emitted when the user toggles manual trade management on/off.
    manual_protection_changed(dict)
        Emitted when any protection setting slider is changed.
    """

    manual_trade_toggled = pyqtSignal(bool)
    manual_protection_changed = pyqtSignal(dict)
    save_settings_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manual_trade_enabled = False

        # Wrap everything in a scroll area so it works on smaller screens
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(14)

        # --- Manual Trade Management Section ---
        root.addWidget(self._build_manual_trade_box())

        # --- Protection Settings (user-adjustable) ---
        root.addWidget(self._build_protection_settings_box())

        # --- Live Positions Table ---
        root.addWidget(self._build_manual_positions_box())

        root.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_manual_trade_box(self) -> QGroupBox:
        """Manual trade management toggle and description."""
        box = QGroupBox("MANUAL TRADE MANAGEMENT")
        box.setStyleSheet(
            f"QGroupBox {{ font-size: {_FS_HEADING}px; font-weight: bold; }}"
        )
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        # Description -- plain English
        desc = QLabel(
            "When you open a trade yourself (not placed by the AI), this system "
            "automatically watches it and moves your stop loss to protect your "
            "profits as the trade goes in your favour. It will never open or "
            "close trades for you -- it only adjusts the stop loss."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {C['subtext']}; font-size: {_FS_BODY}px; line-height: 1.4;"
        )
        layout.addWidget(desc)

        # Separation notice
        sep_note = QLabel(
            "These settings only apply to YOUR manual trades. "
            "They do NOT affect any trades the AI places or manages."
        )
        sep_note.setWordWrap(True)
        sep_note.setStyleSheet(
            f"color: {C['cyan']}; font-size: {_FS_BODY}px; font-weight: bold;"
        )
        layout.addWidget(sep_note)

        # Toggle row
        toggle_row = QHBoxLayout()

        self._lbl_status = QLabel("OFF")
        self._lbl_status.setStyleSheet(
            f"color: {C['red']}; font-size: 18px; font-weight: bold;"
        )
        self._lbl_status.setFixedWidth(55)
        toggle_row.addWidget(self._lbl_status)

        self._btn_toggle = QPushButton("Enable Manual Trade Management")
        self._btn_toggle.setFixedHeight(44)
        self._btn_toggle.setStyleSheet(
            f"QPushButton {{ background-color: {C['surface2']}; "
            f"color: {C['text']}; font-size: 15px; font-weight: bold; "
            f"border: 1px solid {C['border']}; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {C['border']}; }}"
        )
        self._btn_toggle.clicked.connect(self._on_toggle)
        toggle_row.addWidget(self._btn_toggle)

        layout.addLayout(toggle_row)

        # Tracking summary
        self._lbl_tracking = QLabel("No manual positions tracked")
        self._lbl_tracking.setStyleSheet(
            f"color: {C['label']}; font-size: {_FS_TRACKING}px; margin-top: 4px;"
        )
        layout.addWidget(self._lbl_tracking)

        return box

    def _build_protection_settings_box(self) -> QGroupBox:
        """User-adjustable protection stage settings with spinboxes."""
        box = QGroupBox("PROTECTION SETTINGS")
        box.setStyleSheet(
            f"QGroupBox {{ border: 2px solid {C['cyan']}; "
            f"font-size: {_FS_HEADING}px; font-weight: bold; }}"
            f"QGroupBox::title {{ color: {C['cyan']}; }}"
        )
        layout = QVBoxLayout(box)
        layout.setSpacing(14)

        # Spinbox style
        spin_style = (
            f"QDoubleSpinBox {{ background-color: {C['surface2']}; "
            f"color: {C['text']}; border: 1px solid {C['border']}; "
            f"border-radius: 4px; padding: 5px; font-size: {_FS_SPIN}px; "
            f"font-weight: bold; min-height: 28px; }}"
            f"QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ "
            f"width: 20px; }}"
        )

        # === Stage 1: Breakeven ===
        layout.addWidget(self._make_stage_header(
            "Stage 1 — Move Stop Loss to Breakeven",
            C["yellow"],
        ))
        layout.addWidget(self._make_description(
            "When your trade reaches this much profit, the stop loss moves "
            "to your entry price so you can't lose money on this trade anymore."
        ))

        s1_row = QHBoxLayout()
        s1_row.addWidget(self._make_label("Activate when profit reaches:"))
        self._spin_be_trigger = self._make_spin(0.10, 100.0, 2.00, currency.sym(), spin_style, 110)
        s1_row.addWidget(self._spin_be_trigger)
        _csy = currency.sym()
        s1_row.addWidget(self._make_label(
            f"  (e.g. {_csy}2.00 = move SL to entry when up {_csy}2)", C["label"],
        ))
        s1_row.addStretch()
        layout.addLayout(s1_row)

        # Divider
        layout.addWidget(self._make_divider())

        # === Stage 2: Lock Profit ===
        layout.addWidget(self._make_stage_header(
            "Stage 2 — Lock in Guaranteed Profit",
            C["peach"],
        ))
        layout.addWidget(self._make_description(
            "When your trade goes further into profit, the stop loss moves "
            "past your entry price to lock in a guaranteed profit. Even if "
            "price reverses, you walk away with money."
        ))

        s2_row1 = QHBoxLayout()
        s2_row1.addWidget(self._make_label("Activate when profit reaches:"))
        self._spin_lock_trigger = self._make_spin(0.10, 200.0, 3.00, currency.sym(), spin_style, 110)
        s2_row1.addWidget(self._spin_lock_trigger)
        s2_row1.addStretch()
        layout.addLayout(s2_row1)

        s2_row2 = QHBoxLayout()
        s2_row2.addWidget(self._make_label("Guaranteed profit to lock in:"))
        self._spin_lock_amount = self._make_spin(0.10, 100.0, 1.50, currency.sym(), spin_style, 110)
        s2_row2.addWidget(self._spin_lock_amount)
        s2_row2.addWidget(self._make_label(
            f"  (e.g. {currency.sym()}1.50 = SL moved to lock {currency.sym()}1.50 profit)", C["label"],
        ))
        s2_row2.addStretch()
        layout.addLayout(s2_row2)

        # Divider
        layout.addWidget(self._make_divider())

        # === Stage 3: Trailing Stop ===
        layout.addWidget(self._make_stage_header(
            "Stage 3 — Trailing Stop (Follow the Price)",
            C["green"],
        ))
        layout.addWidget(self._make_description(
            "When your trade is deep in profit, the stop loss starts "
            "following the price automatically. If price keeps going your "
            "way, the stop loss follows. If price turns around, the stop "
            "loss stays where it is and you keep your profit."
        ))

        s3_row1 = QHBoxLayout()
        s3_row1.addWidget(self._make_label("Activate when profit reaches:"))
        self._spin_trail_trigger = self._make_spin(0.10, 200.0, 4.00, currency.sym(), spin_style, 110)
        s3_row1.addWidget(self._spin_trail_trigger)
        s3_row1.addStretch()
        layout.addLayout(s3_row1)

        s3_row2 = QHBoxLayout()
        s3_row2.addWidget(self._make_label("Trail distance:"))
        self._spin_trail_atr = self._make_spin(0.3, 5.0, 1.0, "x ATR", spin_style, 120)
        s3_row2.addWidget(self._spin_trail_atr)
        s3_row2.addWidget(self._make_label(
            "  (lower = tighter trail, locks more profit but may exit sooner)",
            C["label"],
        ))
        s3_row2.addStretch()
        layout.addLayout(s3_row2)

        # Divider
        layout.addWidget(self._make_divider())

        # Tips
        tip = QLabel(
            "How to make it more aggressive: Lower the trigger values to "
            "lock profit sooner. Lower the trail distance to keep the stop "
            "loss closer to the price. Changes apply immediately to any "
            "active trades."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            f"color: {C['yellow']}; font-size: {_FS_NOTE}px; "
            f"font-weight: bold; margin-top: 2px;"
        )
        layout.addWidget(tip)

        note = QLabel(
            "Important: Your trade must have a stop loss set for this to work. "
            "If you open a trade without a stop loss, the system will wait and "
            "start protecting it as soon as you set one."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {C['peach']}; font-size: {_FS_NOTE}px; margin-top: 2px;"
        )
        layout.addWidget(note)

        layout.addWidget(self._make_divider())

        # Save + Reset buttons row
        save_row = QHBoxLayout()

        self._btn_save = QPushButton("Save Settings")
        self._btn_save.setFixedHeight(40)
        self._btn_save.setFixedWidth(160)
        self._btn_save.setStyleSheet(
            f"QPushButton {{ background-color: {C['surface2']}; "
            f"color: {C['cyan']}; font-size: 14px; font-weight: bold; "
            f"border: 2px solid {C['cyan']}; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: #1a2a3a; }}"
        )
        self._btn_save.clicked.connect(self._on_save_clicked)
        save_row.addWidget(self._btn_save)

        self._btn_reset = QPushButton("Reset to Default")
        self._btn_reset.setFixedHeight(40)
        self._btn_reset.setFixedWidth(160)
        self._btn_reset.setStyleSheet(
            f"QPushButton {{ background-color: {C['surface2']}; "
            f"color: {C['label']}; font-size: 14px; "
            f"border: 1px solid {C['border']}; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {C['border']}; color: {C['text']}; }}"
        )
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        save_row.addWidget(self._btn_reset)

        _export_import_style = (
            f"QPushButton {{ background-color: {C['surface2']}; "
            f"color: {C['label']}; font-size: 13px; "
            f"border: 1px solid {C['border']}; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {C['border']}; color: {C['text']}; }}"
        )

        self._btn_export = QPushButton("Export JSON")
        self._btn_export.setFixedHeight(40)
        self._btn_export.setFixedWidth(130)
        self._btn_export.setStyleSheet(_export_import_style)
        self._btn_export.clicked.connect(self._on_export_clicked)
        save_row.addWidget(self._btn_export)

        self._btn_import = QPushButton("Import JSON")
        self._btn_import.setFixedHeight(40)
        self._btn_import.setFixedWidth(130)
        self._btn_import.setStyleSheet(_export_import_style)
        self._btn_import.clicked.connect(self._on_import_clicked)
        save_row.addWidget(self._btn_import)

        self._lbl_save_status = QLabel("")
        self._lbl_save_status.setStyleSheet(
            f"color: {C['green']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
        )
        save_row.addWidget(self._lbl_save_status)
        save_row.addStretch()

        layout.addLayout(save_row)

        return box

    def _build_manual_positions_box(self) -> QGroupBox:
        """Table showing currently tracked manual positions with live data."""
        box = QGroupBox("LIVE POSITION MONITOR")
        box.setStyleSheet(
            f"QGroupBox {{ font-size: {_FS_HEADING}px; font-weight: bold; }}"
        )
        layout = QVBoxLayout(box)

        self._tbl_manual = QTableWidget(0, 12)
        self._tbl_manual.setHorizontalHeaderLabels([
            "Ticket", "Side", "Entry", "Lots", "Current\nPrice",
            "P/L", "Initial\nSL", "Current\nSL", "Stage",
            "R-Multiple", "Max R", "Status",
        ])
        header = self._tbl_manual.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(11):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(11, QHeaderView.ResizeMode.Stretch)

        self._tbl_manual.setAlternatingRowColors(True)
        self._tbl_manual.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._tbl_manual.verticalHeader().setVisible(False)
        font = QFont()
        font.setPointSize(_FS_TABLE)
        self._tbl_manual.setFont(font)
        layout.addWidget(self._tbl_manual)

        self._manual_positions_box = box
        return box

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stage_header(text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: {_FS_HEADING}px;"
        )
        return lbl

    @staticmethod
    def _make_description(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {C['subtext']}; font-size: {_FS_BODY}px; "
            f"margin-left: 8px; line-height: 1.3;"
        )
        return lbl

    @staticmethod
    def _make_label(text: str, color: str | None = None) -> QLabel:
        lbl = QLabel(text)
        c = color or C["text"]
        lbl.setStyleSheet(f"color: {c}; font-size: {_FS_LABEL}px;")
        return lbl

    def _make_spin(
        self, lo: float, hi: float, default: float,
        suffix: str, style: str, width: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setValue(default)
        spin.setSuffix(suffix)
        spin.setFixedWidth(width)
        spin.setStyleSheet(style)
        spin.valueChanged.connect(self._on_setting_changed)
        return spin

    @staticmethod
    def _make_divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {C['border']};")
        line.setFixedHeight(1)
        return line

    # ------------------------------------------------------------------
    # Settings change handler
    # ------------------------------------------------------------------

    def _on_setting_changed(self, _value=None) -> None:
        """Emit current protection settings whenever any spinbox changes."""
        overrides = self.get_protection_settings()
        self.manual_protection_changed.emit(overrides)
        log.info("Manual protection settings changed: %s", overrides)
        # Mark save button as needing save
        self._lbl_save_status.setText("(unsaved changes)")
        self._lbl_save_status.setStyleSheet(
            f"color: {C['yellow']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
        )

    def _on_save_clicked(self) -> None:
        """Save current settings so they persist across restarts."""
        settings = self.get_protection_settings()
        settings["manage_manual_trades"] = self._manual_trade_enabled
        self.save_settings_requested.emit(settings)
        self._lbl_save_status.setText("Settings saved")
        self._lbl_save_status.setStyleSheet(
            f"color: {C['green']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
        )

    def _on_reset_clicked(self) -> None:
        """Reset protection settings to default £ values."""
        defaults = {
            "be_trigger_gbp": 2.00,
            "lock_trigger_gbp": 3.00,
            "lock_amount_gbp": 1.50,
            "trail_trigger_gbp": 4.00,
            "trail_atr_mult": 1.0,
        }
        self.load_protection_settings(defaults)
        self._lbl_save_status.setText("(unsaved — defaults loaded)")
        self._lbl_save_status.setStyleSheet(
            f"color: {C['yellow']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
        )

    def get_protection_settings(self) -> dict:
        """Return current UI protection settings as a dict."""
        return {
            "be_trigger_gbp": self._spin_be_trigger.value(),
            "lock_trigger_gbp": self._spin_lock_trigger.value(),
            "lock_amount_gbp": self._spin_lock_amount.value(),
            "trail_trigger_gbp": self._spin_trail_trigger.value(),
            "trail_atr_mult": self._spin_trail_atr.value(),
        }

    def load_protection_settings(self, settings: dict) -> None:
        """Load protection settings from config (called at startup).

        Blocks signals to avoid triggering change events during init.
        """
        for spin, key in [
            (self._spin_be_trigger, "be_trigger_gbp"),
            (self._spin_lock_trigger, "lock_trigger_gbp"),
            (self._spin_lock_amount, "lock_amount_gbp"),
            (self._spin_trail_trigger, "trail_trigger_gbp"),
            (self._spin_trail_atr, "trail_atr_mult"),
        ]:
            spin.blockSignals(True)
            spin.setValue(settings.get(key, spin.value()))
            spin.blockSignals(False)

    def _on_export_clicked(self) -> None:
        """Export manual trade management settings to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Manual Trade Settings",
            str(pathlib.Path.home() / "manual_trade_settings.json"),
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            settings = self.get_protection_settings()
            settings["manage_manual_trades"] = self._manual_trade_enabled
            settings["_source"] = "spartus_manual_trade_management"
            pathlib.Path(path).write_text(
                json.dumps(settings, indent=2), encoding="utf-8"
            )
            self._lbl_save_status.setText(f"Exported → {pathlib.Path(path).name}")
            self._lbl_save_status.setStyleSheet(
                f"color: {C['green']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
            )
        except Exception as exc:
            self._lbl_save_status.setText(f"Export failed: {exc}")
            self._lbl_save_status.setStyleSheet(
                f"color: {C['red']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
            )

    def _on_import_clicked(self) -> None:
        """Import manual trade management settings from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Manual Trade Settings",
            str(pathlib.Path.home()),
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            self.load_protection_settings(data)
            self._lbl_save_status.setText("Imported (unsaved)")
            self._lbl_save_status.setStyleSheet(
                f"color: {C['yellow']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
            )
        except Exception as exc:
            self._lbl_save_status.setText(f"Import failed: {exc}")
            self._lbl_save_status.setStyleSheet(
                f"color: {C['red']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
            )

    # ------------------------------------------------------------------
    # Toggle handler
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        """Handle the enable/disable button click."""
        self._manual_trade_enabled = not self._manual_trade_enabled
        self._update_toggle_ui()
        self.manual_trade_toggled.emit(self._manual_trade_enabled)
        log.info(
            "Manual trade management toggled: %s",
            "ON" if self._manual_trade_enabled else "OFF",
        )
        # Mark save button as needing save
        self._lbl_save_status.setText("(unsaved changes)")
        self._lbl_save_status.setStyleSheet(
            f"color: {C['yellow']}; font-size: {_FS_NOTE}px; margin-left: 10px;"
        )

    def _update_toggle_ui(self) -> None:
        """Update button text and status label."""
        if self._manual_trade_enabled:
            self._lbl_status.setText("ON")
            self._lbl_status.setStyleSheet(
                f"color: {C['green']}; font-size: 18px; font-weight: bold;"
            )
            self._btn_toggle.setText("Disable Manual Trade Management")
            self._btn_toggle.setStyleSheet(
                f"QPushButton {{ background-color: #1a3320; "
                f"color: {C['green']}; font-size: 15px; font-weight: bold; "
                f"border: 1px solid {C['green']}; border-radius: 6px; }}"
                f"QPushButton:hover {{ background-color: #254530; }}"
            )
        else:
            self._lbl_status.setText("OFF")
            self._lbl_status.setStyleSheet(
                f"color: {C['red']}; font-size: 18px; font-weight: bold;"
            )
            self._btn_toggle.setText("Enable Manual Trade Management")
            self._btn_toggle.setStyleSheet(
                f"QPushButton {{ background-color: {C['surface2']}; "
                f"color: {C['text']}; font-size: 15px; font-weight: bold; "
                f"border: 1px solid {C['border']}; border-radius: 6px; }}"
                f"QPushButton:hover {{ background-color: {C['border']}; }}"
            )

    # ------------------------------------------------------------------
    # External state setters
    # ------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        """Set the toggle state from config (called at startup)."""
        self._manual_trade_enabled = enabled
        self._update_toggle_ui()

    def is_enabled(self) -> bool:
        """Return current toggle state."""
        return self._manual_trade_enabled

    # ------------------------------------------------------------------
    # Update from main loop
    # ------------------------------------------------------------------

    def update_data(self, data: dict) -> None:
        """Refresh the manual positions table with live data.

        Args:
            data: Dict with keys:
                - ``manual_positions``: tracking dict from TradeExecutor
                - ``current_price``: current market price (float)
                - ``tick_value``: MT5 tick value in account currency
                - ``tick_size``: MT5 tick size (e.g. 0.01)
        """
        positions = data.get("manual_positions", {})
        current_price = data.get("current_price", 0.0)
        tick_value = data.get("tick_value", 0.745)
        tick_size = data.get("tick_size", 0.01) or 0.01

        if not positions:
            self._lbl_tracking.setText(
                "No manual positions tracked"
                if self._manual_trade_enabled
                else "Manual trade management is OFF"
            )
            self._lbl_tracking.setStyleSheet(
                f"color: {C['label']}; font-size: {_FS_TRACKING}px; margin-top: 4px;"
            )
            self._tbl_manual.setRowCount(0)
            return

        self._lbl_tracking.setText(
            f"Tracking {len(positions)} manual position(s)  |  "
            f"Price: {current_price:.2f}"
        )
        self._lbl_tracking.setStyleSheet(
            f"color: {C['green']}; font-size: {_FS_TRACKING}px; "
            f"font-weight: bold; margin-top: 4px;"
        )

        self._tbl_manual.setRowCount(len(positions))
        stage_names = {0: "Monitoring", 1: "Breakeven", 2: "Locked", 3: "Trailing"}

        for row, (ticket, mp) in enumerate(positions.items()):
            entry = mp.get("entry_price", 0)
            side = mp.get("side", "?")
            lots = mp.get("lots", 0)
            initial_sl = mp.get("initial_sl", 0)
            current_sl = mp.get("current_sl", 0)
            r_dist = abs(entry - initial_sl) if initial_sl > 0 else 0
            max_favorable = mp.get("max_favorable", 0)
            max_r = max_favorable / r_dist if r_dist > 0 else 0
            stage = mp.get("protection_stage", 0)

            # Live P/L calculation
            if current_price > 0 and entry > 0:
                if side == "LONG":
                    price_move = current_price - entry
                else:
                    price_move = entry - current_price
                ticks = price_move / tick_size
                pnl = ticks * tick_value * lots
            else:
                price_move = 0
                pnl = 0

            # Current R-multiple (live, based on current price not MFE)
            if r_dist > 0:
                current_r = price_move / r_dist
            else:
                current_r = 0

            # Status text
            if initial_sl <= 0:
                status = "WAITING FOR SL"
            elif stage >= 3:
                status = "TRAILING"
            elif stage >= 1:
                status = "PROTECTED"
            else:
                status = "MONITORING"

            items = [
                (str(ticket), None),
                (side, C["green"] if side == "LONG" else C["red"]),
                (f"{entry:.2f}", None),
                (f"{lots:.3f}", None),
                (f"{current_price:.2f}", None),
                (f"{pnl:+.2f}", C["green"] if pnl >= 0 else C["red"]),
                (f"{initial_sl:.2f}" if initial_sl > 0 else "NONE", C["red"] if initial_sl <= 0 else None),
                (f"{current_sl:.2f}" if current_sl > 0 else "NONE", None),
                (stage_names.get(stage, str(stage)), self._stage_color(stage)),
                (f"{current_r:+.2f}R", C["green"] if current_r >= 0 else C["red"]),
                (f"{max_r:.2f}R", C["cyan"] if max_r >= 1.0 else None),
                (status, self._status_color(status)),
            ]

            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(QColor(color))
                # Bold P/L column
                if col == 5:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self._tbl_manual.setItem(row, col, item)

    @staticmethod
    def _stage_color(stage: int) -> str:
        if stage >= 3:
            return C["green"]
        elif stage == 2:
            return C["peach"]
        elif stage == 1:
            return C["yellow"]
        return C["dim"]

    @staticmethod
    def _status_color(status: str) -> str:
        if status == "TRAILING":
            return C["green"]
        elif status == "PROTECTED":
            return C["yellow"]
        elif status == "WAITING FOR SL":
            return C["red"]
        return C["dim"]
