"""Standalone screenshot script for the Spartus Live Dashboard.

Can be called from Claude Code or any terminal to capture the dashboard
window without needing to be inside the dashboard process.

Uses PrintWindow API to capture the actual window content directly,
even if other windows are on top of it.

Supports tab switching via --tab to capture specific tabs.

Usage:
    python scripts/take_screenshot.py                    # Current tab
    python scripts/take_screenshot.py --tab 0            # LIVE STATUS
    python scripts/take_screenshot.py --tab 1            # PERFORMANCE
    python scripts/take_screenshot.py --tab 2            # TRADE JOURNAL
    python scripts/take_screenshot.py --tab 3            # MODEL & FEATURES
    python scripts/take_screenshot.py --tab 4            # ALERTS & SAFETY
    python scripts/take_screenshot.py --tab 5            # ANALYTICS
    python scripts/take_screenshot.py --all              # All 6 tabs
    python scripts/take_screenshot.py -o my_capture.png  # Custom output path
"""

import json
import sys
import time
import ctypes
import ctypes.wintypes
import struct
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = BASE_DIR / "storage" / "screenshots"
_TAB_CMD_FILE = BASE_DIR / "storage" / "state" / "_tab_switch_cmd.json"

TAB_NAMES = [
    "LIVE_STATUS", "PERFORMANCE", "TRADE_JOURNAL",
    "MODEL_FEATURES", "ALERTS_SAFETY", "ANALYTICS",
    "MANUAL_TRADE_MGMT", "UPDATES",
]


def find_window(title_substring: str):
    """Find a window handle by partial title match (Windows).

    Prefers exact start-of-title matches and larger windows to avoid
    capturing VSCode or terminal windows that contain the project name.
    """
    user32 = ctypes.windll.user32
    candidates = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if title_substring.upper() in title.upper():
                    starts = title.upper().startswith(title_substring.upper())
                    is_ide = any(x in title for x in [
                        "Visual Studio", "Code", "cmd.exe",
                        "PowerShell", "Terminal", "Windows Terminal",
                    ])
                    rect = ctypes.wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    is_large = w >= 600 and h >= 400
                    score = (2 if starts else 0) + (3 if is_large else 0) - (2 if is_ide else 0)
                    candidates.append((score, hwnd, title, w, h))
        return True

    user32.EnumWindows(callback, 0)

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    chosen = candidates[0]
    log.info("Matched window: '%s' (%dx%d, score=%d)", chosen[2], chosen[3], chosen[4], chosen[0])
    return chosen[1]


def switch_tab(hwnd, tab_index: int) -> None:
    """Switch to a specific tab via file-based command to the dashboard.

    Writes a JSON command file that the dashboard's QTimer reads every
    second, then switches the QTabWidget index directly.  This is fully
    reliable regardless of DPI scaling or window overlap.
    """
    _TAB_CMD_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TAB_CMD_FILE.write_text(
        json.dumps({"tab": tab_index}), encoding="utf-8",
    )

    # Wait for the dashboard to read the command (1 Hz timer = ~1s worst case)
    deadline = time.time() + 3.0
    while _TAB_CMD_FILE.exists() and time.time() < deadline:
        time.sleep(0.2)

    if _TAB_CMD_FILE.exists():
        # Dashboard didn't pick it up -- clean up and warn
        log.warning("Dashboard did not process tab switch command (is it running?)")
        try:
            _TAB_CMD_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        log.info("Switched to tab %d (%s)", tab_index, TAB_NAMES[tab_index])

    # Extra delay for the tab content to render after switching
    time.sleep(0.5)


def capture_window(hwnd, output_path: str) -> bool:
    """Capture a window using PrintWindow API.

    PrintWindow captures the window's own rendering directly,
    so it works even when other windows overlap it.
    """
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top

    if w <= 0 or h <= 0:
        log.error("Window has zero dimensions (%dx%d)", w, h)
        return False

    hdc_window = user32.GetWindowDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_window, w, h)
    old = gdi32.SelectObject(hdc_mem, hbmp)

    PW_RENDERFULLCONTENT = 0x00000002
    ok = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)
    if not ok:
        user32.PrintWindow(hwnd, hdc_mem, 0)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.wintypes.DWORD),
            ("biWidth", ctypes.wintypes.LONG),
            ("biHeight", ctypes.wintypes.LONG),
            ("biPlanes", ctypes.wintypes.WORD),
            ("biBitCount", ctypes.wintypes.WORD),
            ("biCompression", ctypes.wintypes.DWORD),
            ("biSizeImage", ctypes.wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.wintypes.LONG),
            ("biYPelsPerMeter", ctypes.wintypes.LONG),
            ("biClrUsed", ctypes.wintypes.DWORD),
            ("biClrImportant", ctypes.wintypes.DWORD),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0

    img_size = w * h * 4
    buf = ctypes.create_string_buffer(img_size)
    gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bmi), 0)

    gdi32.SelectObject(hdc_mem, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc_window)

    try:
        from PIL import Image
        img = Image.frombytes("RGBA", (w, h), buf.raw, "raw", "BGRA")
        img = img.convert("RGB")
        img.save(output_path, "PNG")
        log.info("Captured %dx%d -> %s (PNG)", w, h, output_path)
        return True
    except ImportError:
        pass

    bmp_path = output_path if output_path.endswith(".bmp") else output_path.replace(".png", ".bmp")
    file_size = 54 + img_size
    with open(bmp_path, "wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HH", 0, 0))
        f.write(struct.pack("<I", 54))
        f.write(struct.pack("<I", 40))
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", -h))
        f.write(struct.pack("<HH", 1, 32))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", img_size))
        f.write(struct.pack("<ii", 0, 0))
        f.write(struct.pack("<II", 0, 0))
        f.write(buf.raw)

    log.info("Captured %dx%d -> %s (BMP)", w, h, bmp_path)
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture a screenshot of the Spartus Live Dashboard"
    )
    parser.add_argument("--output", "-o", help="Output file path (PNG preferred)")
    parser.add_argument("--tab", "-t", type=int, choices=range(8),
                        help="Tab index: 0=Live Status, 1=Performance, 2=Journal, "
                             "3=Model, 4=Alerts, 5=Analytics, 6=Manual Trade, 7=Updates")
    parser.add_argument("--all", action="store_true",
                        help="Capture all 8 tabs")
    parser.add_argument(
        "--title", default="SPARTUS LIVE TRADING",
        help="Window title to search for (default: SPARTUS LIVE TRADING)",
    )
    args = parser.parse_args()

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Find window
    hwnd = find_window(args.title)
    if hwnd is None:
        log.error("No visible window found with title containing '%s'", args.title)
        log.error("Is the dashboard running?")
        sys.exit(1)

    if args.all:
        # Capture all 8 tabs
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        paths = []
        for i in range(8):
            switch_tab(hwnd, i)
            out = str(SCREENSHOT_DIR / f"live_{ts}_tab{i}_{TAB_NAMES[i]}.png")
            capture_window(hwnd, out)
            paths.append(out)
        for p in paths:
            print(p)
    else:
        # Single capture
        if args.tab is not None:
            switch_tab(hwnd, args.tab)

        if args.output:
            out = args.output
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            tab_suffix = f"_tab{args.tab}_{TAB_NAMES[args.tab]}" if args.tab is not None else ""
            out = str(SCREENSHOT_DIR / f"live_{ts}{tab_suffix}.png")

        ok = capture_window(hwnd, out)
        if ok:
            print(out)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
