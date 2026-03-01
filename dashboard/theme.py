"""Dark theme for Spartus Live Dashboard.

Matches the training dashboard's color scheme for consistency.
IMPORTANT: Bright white fonts on dark backgrounds — never dark grey on dark.
"""

# Color palette — GitHub-dark inspired
C = {
    "bg":       "#0d1117",      # Deep dark background
    "surface":  "#161b22",      # Panel background
    "surface2": "#1c2333",      # Slightly lighter panels
    "border":   "#30363d",      # Panel borders
    "text":     "#e6edf3",      # Bright white text (primary)
    "subtext":  "#b1bac4",      # Light gray (secondary — still readable)
    "label":    "#8b949e",      # Label gray
    "dim":      "#656d76",      # Unimportant items (ONLY on light backgrounds)
    "green":    "#2dcc2d",      # Bright profit green
    "red":      "#ff3333",      # Bright loss red
    "yellow":   "#ffcc00",      # Bright warning yellow
    "blue":     "#58a6ff",      # Link/accent blue
    "cyan":     "#39d5ff",      # Bright accent cyan
    "mauve":    "#bc8cff",      # Purple accent
    "peach":    "#ffa657",      # Orange accent
    "white":    "#ffffff",      # Pure white for emphasis
}

# Font families
FONT_PRIMARY = "'Segoe UI', 'Cascadia Code', Consolas, monospace"
FONT_CODE = "'Cascadia Code', Consolas, monospace"

# Trading state colors
STATE_COLORS = {
    "stopped":      C["label"],     # Gray
    "running":      C["green"],     # Green
    "winding_down": C["yellow"],    # Yellow
    "cb_paused":    C["peach"],     # Orange
    "emergency":    C["red"],       # Red
}


def get_stylesheet() -> str:
    """Return the full application stylesheet."""
    return f"""
    /* === Global === */
    QMainWindow, QWidget {{
        background-color: {C['bg']};
        color: {C['text']};
        font-family: {FONT_PRIMARY};
        font-size: 13px;
    }}

    /* === Tab Widget === */
    QTabWidget::pane {{
        border: 1px solid {C['border']};
        background-color: {C['bg']};
    }}
    QTabBar::tab {{
        background-color: {C['surface']};
        color: {C['subtext']};
        border: 1px solid {C['border']};
        border-bottom: none;
        padding: 8px 20px;
        margin-right: 2px;
        font-size: 13px;
        font-weight: bold;
    }}
    QTabBar::tab:selected {{
        background-color: {C['surface2']};
        color: {C['cyan']};
        border-bottom: 2px solid {C['cyan']};
    }}
    QTabBar::tab:hover {{
        background-color: {C['surface2']};
        color: {C['text']};
    }}

    /* === Group Box === */
    QGroupBox {{
        background-color: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px 8px 8px 8px;
        font-size: 13px;
        font-weight: bold;
        color: {C['cyan']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {C['cyan']};
    }}

    /* === Labels === */
    QLabel {{
        color: {C['text']};
        font-size: 13px;
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {C['surface2']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 4px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {C['border']};
    }}
    QPushButton:pressed {{
        background-color: {C['surface']};
    }}
    QPushButton:disabled {{
        color: {C['dim']};
        background-color: {C['surface']};
    }}

    /* === Tables === */
    QTableWidget, QTableView {{
        background-color: {C['surface']};
        color: {C['text']};
        border: 1px solid {C['border']};
        gridline-color: {C['border']};
        alternate-background-color: {C['surface2']};
        selection-background-color: {C['blue']};
        selection-color: {C['white']};
        font-size: 12px;
    }}
    QHeaderView::section {{
        background-color: {C['surface2']};
        color: {C['cyan']};
        border: 1px solid {C['border']};
        padding: 4px 8px;
        font-size: 12px;
        font-weight: bold;
    }}

    /* === Scroll Bars === */
    QScrollBar:vertical {{
        background-color: {C['surface']};
        width: 10px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {C['border']};
        border-radius: 5px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {C['label']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background-color: {C['surface']};
        height: 10px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {C['border']};
        border-radius: 5px;
        min-width: 20px;
    }}

    /* === Text Edit / Plain Text === */
    QTextEdit, QPlainTextEdit {{
        background-color: {C['surface']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 4px;
        font-family: {FONT_CODE};
        font-size: 12px;
    }}

    /* === Line Edit === */
    QLineEdit {{
        background-color: {C['surface']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    /* === Combo Box === */
    QComboBox {{
        background-color: {C['surface2']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QComboBox::drop-down {{
        border: none;
    }}

    /* === Progress Bar === */
    QProgressBar {{
        background-color: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 4px;
        text-align: center;
        color: {C['text']};
        font-size: 11px;
    }}
    QProgressBar::chunk {{
        background-color: {C['green']};
        border-radius: 3px;
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {C['border']};
    }}

    /* === Tooltip === */
    QToolTip {{
        background-color: {C['surface2']};
        color: {C['text']};
        border: 1px solid {C['border']};
        padding: 4px;
        font-size: 12px;
    }}
    """
