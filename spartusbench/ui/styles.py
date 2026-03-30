"""Color palette and stylesheet for SpartusBench UI.

Matches the dark theme from the training dashboard for visual consistency.
"""

C = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "surface2": "#1c2333",
    "border":   "#30363d",
    "text":     "#e6edf3",
    "subtext":  "#b1bac4",
    "label":    "#8b949e",
    "dim":      "#656d76",
    "green":    "#2dcc2d",
    "red":      "#ff3333",
    "yellow":   "#ffcc00",
    "blue":     "#58a6ff",
    "cyan":     "#39d5ff",
    "mauve":    "#bc8cff",
    "peach":    "#ffa657",
}

DARK_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']}; color: {C['text']};
    font-family: 'Segoe UI', 'Cascadia Code', Consolas, monospace;
}}
QGroupBox {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    border-radius: 8px; margin-top: 18px; padding: 14px 10px 10px 10px;
    font-size: 13px; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 8px;
    color: {C['cyan']}; font-size: 13px;
}}
QLabel {{ font-size: 14px; }}
QPushButton {{
    background-color: {C['surface2']}; border: 1px solid {C['border']};
    border-radius: 6px; padding: 8px 24px; font-size: 13px;
    color: {C['text']}; font-weight: bold;
}}
QPushButton:hover {{ background-color: {C['border']}; }}
QPushButton:disabled {{ color: {C['dim']}; }}
QTextEdit, QPlainTextEdit {{
    background-color: {C['bg']}; border: 1px solid {C['border']};
    border-radius: 6px; font-family: 'Cascadia Code', Consolas, monospace;
    font-size: 13px; color: {C['text']}; padding: 6px;
}}
QTableWidget {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    border-radius: 6px; gridline-color: {C['border']};
    font-size: 13px; color: {C['text']};
}}
QTableWidget::item {{ padding: 4px 8px; }}
QTableWidget::item:selected {{
    background-color: {C['surface2']}; color: {C['cyan']};
}}
QHeaderView::section {{
    background-color: {C['surface2']}; color: {C['cyan']};
    border: 1px solid {C['border']}; padding: 4px 8px;
    font-weight: bold; font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {C['border']}; background-color: {C['bg']};
}}
QTabBar::tab {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    padding: 8px 20px; margin-right: 2px; font-weight: bold;
    font-size: 12px; color: {C['subtext']};
}}
QTabBar::tab:selected {{
    background-color: {C['surface2']}; color: {C['cyan']};
    border-bottom: 2px solid {C['cyan']};
}}
QTabBar::tab:hover {{ color: {C['text']}; }}
QComboBox {{
    background-color: {C['surface2']}; border: 1px solid {C['border']};
    border-radius: 6px; padding: 6px 12px; font-size: 13px;
    color: {C['text']};
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    color: {C['text']}; selection-background-color: {C['surface2']};
}}
QProgressBar {{
    background-color: {C['surface']}; border: 1px solid {C['border']};
    border-radius: 4px; text-align: center; font-size: 12px;
    color: {C['text']}; min-height: 20px;
}}
QProgressBar::chunk {{
    background-color: {C['blue']}; border-radius: 3px;
}}
QCheckBox {{ font-size: 13px; color: {C['text']}; spacing: 8px; }}
QSpinBox, QLineEdit {{
    background-color: {C['surface2']}; border: 1px solid {C['border']};
    border-radius: 4px; padding: 4px 8px; font-size: 13px;
    color: {C['text']};
}}
QStatusBar {{
    background-color: {C['surface']}; color: {C['subtext']};
    border-top: 1px solid {C['border']}; font-size: 12px;
}}
QRadioButton {{ font-size: 13px; color: {C['text']}; spacing: 6px; }}
"""
