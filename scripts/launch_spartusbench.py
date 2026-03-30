#!/usr/bin/env python
"""Launch the SpartusBench benchmark dashboard UI.

Usage:
    python scripts/launch_spartusbench.py
"""

import sys
import os

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from spartusbench.ui.main_window import SpartusBenchWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpartusBench")
    app.setFont(QFont("Segoe UI", 10))

    window = SpartusBenchWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
