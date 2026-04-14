"""Tab 8 — Settings for the Spartus Live Trading Dashboard.

Two sections:
  1. REMOTE ACCESS (MCP SERVER) — start/stop the MCP server, manage the
     bearer token, auto-detect the Tailscale IP, and copy connection
     credentials for the FHAE dashboard.
  2. SOFTWARE UPDATES — version check, one-click update, restart
     (formerly the standalone Updates tab).

Layout::

    +-------------------------------------------------------------+
    |  REMOTE ACCESS (MCP SERVER)                                 |
    |  Tailscale IP:  100.x.x.x        [Refresh]                  |
    |  Server URL:    http://100.x.x.x:7474   [Copy]              |
    |  Port:          [7474]                                       |
    |  Token:         ●●●●●●●●●●●●   [Show] [Copy] [Generate]     |
    |  Status:  ● RUNNING  on port 7474                           |
    |  [  Start MCP Server  ]  [  Stop  ]                         |
    +-------------------------------------------------------------+
    |  SOFTWARE UPDATES                                           |
    |   ... (existing UpdatesTab content) ...                     |
    +-------------------------------------------------------------+

All text follows dark-theme rules: bright white (#e6edf3) values,
light gray (#b1bac4) labels.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QScrollArea,
    QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QClipboard
from PyQt6.QtWidgets import QApplication

from dashboard.theme import C
from dashboard.tab_updates import UpdatesTab

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent.parent   # live_dashboard root
_SETTINGS_FILE = _HERE / "config" / "user_settings.json"
_MCP_SERVER = _HERE / "mcp_server.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lbl(text: str, color: str = C["subtext"], bold: bool = False,
         size: int = 13) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet(
        f"color: {color}; font-size: {size}px; "
        f"font-weight: {'bold' if bold else 'normal'}; "
        f"background: transparent; border: none;"
    )
    return w


def _val(text: str, color: str = C["text"], bold: bool = True,
         size: int = 13) -> QLabel:
    return _lbl(text, color=color, bold=bold, size=size)


def _btn(text: str, color: str = C["surface2"], fg: str = C["text"],
         accent: str = C["border"], bold: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(
        f"QPushButton {{ background-color: {color}; color: {fg}; "
        f"font-weight: {'bold' if bold else 'normal'}; "
        f"border: 1px solid {accent}; border-radius: 4px; "
        f"padding: 7px 18px; font-size: 13px; }} "
        f"QPushButton:hover {{ background-color: {accent}; }} "
        f"QPushButton:disabled {{ color: {C['dim']}; }}"
    )
    return b


def _detect_tailscale_ip() -> Optional[str]:
    """Return the first 100.x.x.x address found on any interface (Tailscale range)."""
    pattern = re.compile(r"100\.\d{1,3}\.\d{1,3}\.\d{1,3}")
    # Try ip addr (Linux / macOS)
    for cmd in (["ip", "addr"], ["ifconfig"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
            m = pattern.search(out)
            if m:
                return m.group()
        except Exception:
            pass
    # Try ipconfig (Windows)
    try:
        out = subprocess.run(
            "ipconfig", capture_output=True, text=True, timeout=3, shell=True
        ).stdout
        m = pattern.search(out)
        if m:
            return m.group()
    except Exception:
        pass
    return None


def _detect_lan_ip() -> Optional[str]:
    """Return the best local LAN IP (fallback when Tailscale is not running)."""
    import socket
    try:
        # Connect to an external address to discover which local interface is used
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return None


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    try:
        _SETTINGS_FILE.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Failed to save settings: %s", exc)


# ---------------------------------------------------------------------------
# McpSettingsTab
# ---------------------------------------------------------------------------

class McpSettingsTab(QWidget):
    """Combined Settings tab: MCP remote-access + software updates.

    Exposes the same public API as the old ``UpdatesTab`` so ``main.py``
    requires minimal changes.

    Signals (forwarded from embedded UpdatesTab):
        check_requested
        update_requested
        restart_requested
    """

    # Forward UpdatesTab signals
    check_requested = pyqtSignal()
    update_requested = pyqtSignal()
    restart_requested = pyqtSignal()

    # Internal: marshal IP detection result back to main thread safely
    _ip_detected = pyqtSignal(str, str)   # (tailscale_ip_or_empty, lan_ip_or_empty)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mcp_proc: Optional[subprocess.Popen] = None
        self._tailscale_ip: Optional[str] = None
        self._lan_ip: Optional[str] = None
        self._token_visible: bool = False
        self._port: int = 7474

        # Load persisted token
        settings = _load_settings()
        self._token: str = settings.get("mcp_token", "")

        # Detect LAN IP immediately (fast: just a socket) so URL is never blank
        self._lan_ip = _detect_lan_ip()

        self._build_ui()

        # Wire internal signal so background thread can safely update the UI
        self._ip_detected.connect(self._on_ip_detected)

        # Poll MCP process status every 2 s
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(2000)

        # Check for Tailscale in background (may take a moment)
        self._do_detect_ip()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Scrollable root so nothing is ever clipped ───────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(16)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        scroll.setWidget(inner)

        field_style = (
            f"QLineEdit {{ background: {C['surface2']}; color: {C['text']}; "
            f"border: 1px solid {C['border']}; border-radius: 4px; "
            f"padding: 6px 10px; font-size: 13px; }} "
            f"QLineEdit:read-only {{ color: {C['cyan']}; }}"
        )

        # ── Section 1: Remote Access ─────────────────────────────────
        mcp_group = QGroupBox("REMOTE ACCESS  —  MCP SERVER")
        mcp_group.setStyleSheet(
            f"QGroupBox {{ color: {C['cyan']}; font-size: 12px; font-weight: bold; "
            f"border: 1px solid {C['border']}; border-radius: 6px; margin-top: 8px; "
            f"padding-top: 6px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}"
        )
        mcp_layout = QVBoxLayout(mcp_group)
        mcp_layout.setContentsMargins(16, 22, 16, 16)
        mcp_layout.setSpacing(14)

        # ── Tailscale / LAN IP ───────────────────────────────────────
        mcp_layout.addWidget(_lbl("IP Address  (Tailscale preferred, LAN fallback)", C["label"], size=11))
        row_ip = QHBoxLayout()
        # Show LAN IP immediately; background thread will upgrade to Tailscale if found
        _ip_initial = self._lan_ip or "Checking…"
        self._ip_val = QLineEdit(_ip_initial)
        self._ip_val.setReadOnly(True)
        self._ip_val.setStyleSheet(field_style)
        row_ip.addWidget(self._ip_val)
        btn_refresh_ip = _btn("Refresh", C["surface2"], C["text"])
        btn_refresh_ip.setFixedWidth(90)
        btn_refresh_ip.clicked.connect(self._do_detect_ip)
        row_ip.addWidget(btn_refresh_ip)
        mcp_layout.addLayout(row_ip)

        # ── Port ──────────────────────────────────────────────────────
        mcp_layout.addWidget(_lbl("Port", C["label"], size=11))
        self._port_edit = QLineEdit(str(self._port))
        self._port_edit.setFixedWidth(100)
        self._port_edit.setStyleSheet(
            f"QLineEdit {{ background: {C['surface2']}; color: {C['text']}; "
            f"border: 1px solid {C['border']}; border-radius: 4px; "
            f"padding: 6px 10px; font-size: 13px; }}"
        )
        self._port_edit.textChanged.connect(self._on_port_changed)
        mcp_layout.addWidget(self._port_edit)

        # ── Server URL (full-width, read-only, selectable) ────────────
        mcp_layout.addWidget(_lbl("Server URL  (paste this into FHAE dashboard)", C["label"], size=11))
        row_url = QHBoxLayout()
        _url_initial = f"http://{self._lan_ip}:{self._port}" if self._lan_ip else "—"
        self._url_val = QLineEdit(_url_initial)
        self._url_val.setReadOnly(True)
        self._url_val.setStyleSheet(field_style)
        row_url.addWidget(self._url_val)
        btn_copy_url = _btn("Copy URL", C["surface2"], C["blue"])
        btn_copy_url.setFixedWidth(100)
        btn_copy_url.clicked.connect(self._copy_url)
        row_url.addWidget(btn_copy_url)
        mcp_layout.addLayout(row_url)

        # ── Bearer Token ──────────────────────────────────────────────
        mcp_layout.addWidget(_lbl("Bearer Token  (SPARTUS_MCP_TOKEN)", C["label"], size=11))

        # Token input row
        row_token_input = QHBoxLayout()
        self._token_edit = QLineEdit(self._token)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("Generate or enter a strong secret token…")
        self._token_edit.setStyleSheet(
            f"QLineEdit {{ background: {C['surface2']}; color: {C['text']}; "
            f"border: 1px solid {C['border']}; border-radius: 4px; "
            f"padding: 6px 10px; font-size: 13px; }}"
        )
        self._token_edit.textChanged.connect(self._on_token_changed)
        row_token_input.addWidget(self._token_edit)

        self._btn_show_token = _btn("Show", C["surface2"], C["label"])
        self._btn_show_token.setFixedWidth(70)
        self._btn_show_token.clicked.connect(self._toggle_token_visibility)
        row_token_input.addWidget(self._btn_show_token)
        mcp_layout.addLayout(row_token_input)

        # Token action row — separate line so buttons have space
        row_token_btns = QHBoxLayout()
        btn_copy_token = _btn("Copy Token", C["surface2"], C["blue"])
        btn_copy_token.clicked.connect(self._copy_token)
        row_token_btns.addWidget(btn_copy_token)

        btn_gen_token = _btn("Generate New Token", C["surface2"], C["mauve"])
        btn_gen_token.clicked.connect(self._generate_token)
        row_token_btns.addWidget(btn_gen_token)

        row_token_btns.addStretch()
        mcp_layout.addLayout(row_token_btns)

        self._token_saved_lbl = _lbl("", C["green"], size=11)
        mcp_layout.addWidget(self._token_saved_lbl)

        # ── Separator ─────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background: {C['border']}; max-height: 1px;")
        mcp_layout.addWidget(sep)

        # ── Status indicator ──────────────────────────────────────────
        status_row = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(
            f"color: {C['dim']}; font-size: 20px; background: transparent; border: none;"
        )
        status_row.addWidget(self._status_dot)
        self._status_lbl = QLabel("STOPPED")
        self._status_lbl.setStyleSheet(
            f"color: {C['dim']}; font-size: 14px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        mcp_layout.addLayout(status_row)

        # ── Start / Stop buttons ──────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start MCP Server")
        self._btn_start.setFixedHeight(42)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.setStyleSheet(
            f"QPushButton {{ background-color: {C['green']}; color: {C['bg']}; "
            f"font-weight: bold; border: none; border-radius: 5px; "
            f"padding: 10px 28px; font-size: 14px; }} "
            f"QPushButton:hover {{ background-color: #40e640; }} "
            f"QPushButton:disabled {{ background-color: {C['dim']}; color: {C['surface']}; }}"
        )
        self._btn_start.clicked.connect(self._start_server)
        btn_row.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedHeight(42)
        self._btn_stop.setFixedWidth(110)
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ background-color: {C['red']}; color: {C['bg']}; "
            f"font-weight: bold; border: none; border-radius: 5px; "
            f"padding: 10px 20px; font-size: 14px; }} "
            f"QPushButton:hover {{ background-color: #ff5555; }} "
            f"QPushButton:disabled {{ background-color: {C['dim']}; color: {C['surface']}; }}"
        )
        self._btn_stop.clicked.connect(self._stop_server)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        mcp_layout.addLayout(btn_row)

        # ── Hint ──────────────────────────────────────────────────────
        hint = _lbl(
            "Start the MCP server here, then copy the URL and token into: "
            "FHAE Dashboard → Trading Marketplace → SpartusTradeAI → Add Instance. "
            "Use the Tailscale IP for cross-network access, or the LAN IP if FHAE "
            "and Spartus are on the same home network.",
            C["dim"], size=11
        )
        hint.setWordWrap(True)
        mcp_layout.addWidget(hint)

        root.addWidget(mcp_group)

        # ── Section 2: Software Updates ──────────────────────────────
        updates_group = QGroupBox("SOFTWARE UPDATES")
        updates_group.setStyleSheet(
            f"QGroupBox {{ color: {C['subtext']}; font-size: 12px; font-weight: bold; "
            f"border: 1px solid {C['border']}; border-radius: 6px; margin-top: 8px; "
            f"padding-top: 6px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}"
        )
        ug_outer = QVBoxLayout(updates_group)
        ug_outer.setContentsMargins(4, 8, 4, 4)

        self._updates_tab = UpdatesTab()
        self._updates_tab.check_requested.connect(self.check_requested)
        self._updates_tab.update_requested.connect(self.update_requested)
        self._updates_tab.restart_requested.connect(self.restart_requested)
        ug_outer.addWidget(self._updates_tab)

        root.addWidget(updates_group)
        root.addStretch()

    # ------------------------------------------------------------------
    # Tailscale IP detection
    # ------------------------------------------------------------------

    def _do_detect_ip(self) -> None:
        self._ip_val.setText("Detecting…")
        self._ip_val.setStyleSheet(
            f"color: {C['label']}; font-size: 13px; "
            f"background: transparent; border: none;"
        )
        # Run in a short-lived thread to avoid freezing the UI
        import threading
        threading.Thread(target=self._detect_ip_thread, daemon=True).start()

    def _detect_ip_thread(self) -> None:
        tailscale = _detect_tailscale_ip() or ""
        lan = _detect_lan_ip() or ""
        self._ip_detected.emit(tailscale, lan)   # safe cross-thread signal

    def _on_ip_detected(self, tailscale_ip: str, lan_ip: str) -> None:
        self._set_ip(tailscale_ip or None, lan_ip or None)

    def _set_ip(self, tailscale_ip: Optional[str], lan_ip: Optional[str] = None) -> None:
        self._tailscale_ip = tailscale_ip
        self._lan_ip = lan_ip

        if tailscale_ip:
            self._ip_val.setText(tailscale_ip)
            self._ip_val.setStyleSheet(
                f"QLineEdit {{ background: {C['surface2']}; color: {C['cyan']}; "
                f"border: 1px solid {C['border']}; border-radius: 4px; "
                f"padding: 6px 10px; font-size: 13px; font-weight: bold; }}"
            )
        elif lan_ip:
            self._ip_val.setText(f"{lan_ip}  (LAN only — Tailscale not detected)")
            self._ip_val.setStyleSheet(
                f"QLineEdit {{ background: {C['surface2']}; color: {C['yellow']}; "
                f"border: 1px solid {C['border']}; border-radius: 4px; "
                f"padding: 6px 10px; font-size: 13px; }}"
            )
        else:
            self._ip_val.setText("Could not detect IP address")
            self._ip_val.setStyleSheet(
                f"QLineEdit {{ background: {C['surface2']}; color: {C['red']}; "
                f"border: 1px solid {C['border']}; border-radius: 4px; "
                f"padding: 6px 10px; font-size: 13px; }}"
            )
        self._refresh_url()

    def _refresh_url(self) -> None:
        # Use Tailscale IP if available, else LAN IP
        ip = self._tailscale_ip or self._lan_ip
        if ip:
            url = f"http://{ip}:{self._port}"
        else:
            url = f"http://<ip-not-detected>:{self._port}"
        self._url_val.setText(url)

    # ------------------------------------------------------------------
    # Port
    # ------------------------------------------------------------------

    def _on_port_changed(self, text: str) -> None:
        try:
            self._port = int(text)
        except ValueError:
            pass
        self._refresh_url()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _on_token_changed(self, text: str) -> None:
        self._token = text
        settings = _load_settings()
        settings["mcp_token"] = text
        _save_settings(settings)
        self._token_saved_lbl.setText("Saved")
        QTimer.singleShot(1500, lambda: self._token_saved_lbl.setText(""))

    def _toggle_token_visibility(self) -> None:
        self._token_visible = not self._token_visible
        if self._token_visible:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._btn_show_token.setText("Hide")
        else:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._btn_show_token.setText("Show")

    def _generate_token(self) -> None:
        token = secrets.token_urlsafe(32)
        self._token_edit.setText(token)   # triggers _on_token_changed

    def _copy_token(self) -> None:
        if self._token:
            QApplication.clipboard().setText(self._token)
            self._token_saved_lbl.setText("Token copied to clipboard")
            QTimer.singleShot(2000, lambda: self._token_saved_lbl.setText(""))

    def _copy_url(self) -> None:
        url = self._url_val.text()
        if url and "<" not in url:
            QApplication.clipboard().setText(url)
            self._token_saved_lbl.setText("URL copied to clipboard")
            QTimer.singleShot(2000, lambda: self._token_saved_lbl.setText(""))

    # ------------------------------------------------------------------
    # MCP server process management
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        if self._mcp_proc and self._mcp_proc.poll() is None:
            return  # already running

        token = self._token.strip()
        if not token:
            self._set_status(False, "Set a token before starting")
            return

        # Ensure fastapi + uvicorn are installed into this Python
        try:
            import importlib
            missing = [p for p in ("fastapi", "uvicorn") if importlib.util.find_spec(p) is None]
            if missing:
                self._set_status(False, f"Installing {', '.join(missing)}…")
                QApplication.processEvents()
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing,
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    self._set_status(False, f"pip install failed: {result.stderr[:120]}")
                    return
        except Exception as exc:
            self._set_status(False, f"Dependency check failed: {exc}")
            return

        import os
        env = {**os.environ, "SPARTUS_MCP_TOKEN": token}
        try:
            self._mcp_proc = subprocess.Popen(
                [sys.executable, str(_MCP_SERVER), "--port", str(self._port)],
                cwd=str(_HERE),
                env=env,
            )
            log.info("MCP server started (PID %d, port %d)", self._mcp_proc.pid, self._port)
        except Exception as exc:
            log.error("Failed to start MCP server: %s", exc)
            self._set_status(False, f"Failed to start: {exc}")
            return

        self._refresh_status()

    def _stop_server(self) -> None:
        if self._mcp_proc and self._mcp_proc.poll() is None:
            self._mcp_proc.terminate()
            log.info("MCP server terminated")
        self._refresh_status()

    def _refresh_status(self) -> None:
        running = self._mcp_proc is not None and self._mcp_proc.poll() is None
        if running:
            self._set_status(True, f"RUNNING  —  port {self._port}")
        else:
            self._set_status(False, "STOPPED")

    def _set_status(self, running: bool, text: str) -> None:
        color = C["green"] if running else C["dim"]
        self._status_dot.setStyleSheet(
            f"color: {color}; font-size: 16px; "
            f"background: transparent; border: none;"
        )
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    # ------------------------------------------------------------------
    # Forwarded UpdatesTab public API
    # ------------------------------------------------------------------

    def set_current_version(self, version: str) -> None:
        self._updates_tab.set_current_version(version)

    def set_update_available(self, update_info) -> None:
        self._updates_tab.set_update_available(update_info)

    def set_no_update(self) -> None:
        self._updates_tab.set_no_update()

    def set_check_failed(self) -> None:
        self._updates_tab.set_check_failed()

    def set_progress(self, message: str) -> None:
        self._updates_tab.set_progress(message)

    def set_complete(self, success: bool, message: str) -> None:
        self._updates_tab.set_complete(success, message)

    def update_data(self, data: dict) -> None:
        self._updates_tab.update_data(data)
