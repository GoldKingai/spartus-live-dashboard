"""Screenshot capture utility for the Spartus Live Dashboard.

Captures the dashboard window (or individual tabs) as PNG images
that can be inspected by Claude Code for visual analysis.

Usage from CLI:
    python -m utils.screenshot                  # Capture running dashboard
    python -m utils.screenshot --tab 0          # Capture specific tab
    python -m utils.screenshot --output my.png  # Custom output path
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default screenshot directory
_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "storage" / "screenshots"


def capture_dashboard_screenshot(
    output_path: Optional[str] = None,
    tab_index: Optional[int] = None,
    window_title: str = "SPARTUS LIVE TRADING",
) -> Optional[str]:
    """Capture a screenshot of the Spartus Live Dashboard window.

    Uses platform-native methods to find and capture the window.
    Falls back to full-screen crop if window-specific capture fails.

    Args:
        output_path: Where to save the PNG. If None, auto-generates
                     a timestamped path in storage/screenshots/.
        tab_index:   If set, switch to this tab before capturing.
        window_title: Window title substring to match.

    Returns:
        Absolute path to the saved PNG, or None on failure.
    """
    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tab_suffix = f"_tab{tab_index}" if tab_index is not None else ""
        output_path = str(_SCREENSHOT_DIR / f"live_dashboard_{ts}{tab_suffix}.png")

    # Try PyQt6-internal capture first (most reliable when dashboard is in-process)
    result = _capture_via_pyqt6(output_path, tab_index, window_title)
    if result:
        return result

    # Fallback: platform-native window capture
    result = _capture_via_platform(output_path, window_title)
    if result:
        return result

    logger.error("All screenshot methods failed")
    return None


def _capture_via_pyqt6(
    output_path: str,
    tab_index: Optional[int],
    window_title: str,
) -> Optional[str]:
    """Capture via PyQt6 QScreen.grabWindow -- works when QApplication exists."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer

        app = QApplication.instance()
        if app is None:
            return None

        # Find the dashboard window
        target_window = None
        for widget in app.topLevelWidgets():
            if hasattr(widget, 'windowTitle') and window_title in widget.windowTitle():
                target_window = widget
                break

        if target_window is None:
            logger.debug("No window matching '%s' found in QApplication", window_title)
            return None

        # Switch tab if requested
        if tab_index is not None:
            tabs = target_window.findChild(type(target_window).findChild(target_window, None).__class__) if False else None
            # Find QTabWidget
            from PyQt6.QtWidgets import QTabWidget
            for child in target_window.findChildren(QTabWidget):
                if 0 <= tab_index < child.count():
                    child.setCurrentIndex(tab_index)
                    # Process events so the tab actually renders
                    app.processEvents()
                    break

        # Grab the window
        screen = target_window.screen()
        if screen is None:
            screen = app.primaryScreen()

        pixmap = screen.grabWindow(int(target_window.winId()))
        if pixmap.isNull():
            logger.warning("grabWindow returned null pixmap")
            return None

        pixmap.save(output_path, "PNG")
        logger.info("Screenshot saved (PyQt6): %s", output_path)
        return output_path

    except Exception as exc:
        logger.debug("PyQt6 capture failed: %s", exc)
        return None


def _capture_via_platform(
    output_path: str,
    window_title: str,
) -> Optional[str]:
    """Capture via platform-native methods (Windows win32 API or PIL)."""
    if sys.platform == "win32":
        return _capture_win32(output_path, window_title)
    return None


def _capture_win32(output_path: str, window_title: str) -> Optional[str]:
    """Windows-specific: find window by title and capture it."""
    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32

        # Find window by partial title match
        hwnd = [None]

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_callback(h, _):
            length = user32.GetWindowTextLengthW(h)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(h, buf, length + 1)
                if window_title.upper() in buf.value.upper():
                    hwnd[0] = h
                    return False  # Stop enumerating
            return True

        user32.EnumWindows(enum_callback, 0)

        if hwnd[0] is None:
            logger.debug("No window found with title containing '%s'", window_title)
            return None

        h = hwnd[0]

        # Bring window to front briefly
        user32.SetForegroundWindow(h)
        time.sleep(0.3)

        # Get window rect
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(h, ctypes.byref(rect))
        x, y, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
        w, h_px = x2 - x, y2 - y

        if w <= 0 or h_px <= 0:
            logger.warning("Window has zero dimensions")
            return None

        # Use PIL to grab the region
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(bbox=(x, y, x2, y2))
            img.save(output_path, "PNG")
            logger.info("Screenshot saved (win32+PIL): %s (%dx%d)", output_path, w, h_px)
            return output_path
        except ImportError:
            pass

        # Fallback: use win32 GDI (no PIL needed)
        return _capture_win32_gdi(output_path, hwnd[0], x, y, w, h_px)

    except Exception as exc:
        logger.debug("Win32 capture failed: %s", exc)
        return None


def _capture_win32_gdi(
    output_path: str,
    hwnd,
    x: int, y: int, w: int, h: int,
) -> Optional[str]:
    """Pure win32 GDI capture without PIL."""
    try:
        import ctypes
        from ctypes import wintypes

        gdi32 = ctypes.windll.gdi32
        user32 = ctypes.windll.user32

        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        old_bmp = gdi32.SelectObject(hdc_mem, hbmp)

        gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020)  # SRCCOPY

        # Save as BMP then convert header for PNG
        # Actually, let's write a proper BMP file
        import struct

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # Top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        img_size = w * h * 4
        buf = ctypes.create_string_buffer(img_size)
        gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bmi), 0)

        # Clean up GDI
        gdi32.SelectObject(hdc_mem, old_bmp)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        # Write BMP file (we'll write .bmp, then note it)
        bmp_path = output_path.replace(".png", ".bmp")
        row_size = w * 4
        file_size = 54 + img_size

        with open(bmp_path, "wb") as f:
            # BMP header
            f.write(b"BM")
            f.write(struct.pack("<I", file_size))
            f.write(struct.pack("<HH", 0, 0))
            f.write(struct.pack("<I", 54))
            # DIB header
            f.write(struct.pack("<I", 40))
            f.write(struct.pack("<i", w))
            f.write(struct.pack("<i", -h))  # top-down
            f.write(struct.pack("<HH", 1, 32))
            f.write(struct.pack("<I", 0))  # BI_RGB
            f.write(struct.pack("<I", img_size))
            f.write(struct.pack("<ii", 0, 0))
            f.write(struct.pack("<II", 0, 0))
            f.write(buf.raw)

        logger.info("Screenshot saved (GDI BMP): %s (%dx%d)", bmp_path, w, h)
        return bmp_path

    except Exception as exc:
        logger.debug("GDI capture failed: %s", exc)
        return None


# ── Standalone CLI entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Capture Spartus Dashboard screenshot")
    parser.add_argument("--output", "-o", help="Output PNG path")
    parser.add_argument("--tab", "-t", type=int, help="Tab index to capture (0-7)")
    parser.add_argument(
        "--title", default="SPARTUS",
        help="Window title substring to match",
    )
    args = parser.parse_args()

    path = capture_dashboard_screenshot(
        output_path=args.output,
        tab_index=args.tab,
        window_title=args.title,
    )
    if path:
        print(f"Screenshot saved: {path}")
    else:
        print("Failed to capture screenshot", file=sys.stderr)
        sys.exit(1)
