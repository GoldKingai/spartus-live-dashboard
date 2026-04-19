#!/usr/bin/env bash
# ============================================================
# Spartus Live Trading Dashboard -- Linux / macOS Launcher
# ============================================================
# Equivalent to launch.bat for Windows.
#
# Usage:
#   ./launch.sh              Normal launch
#   ./launch.sh --paper      Force paper trading mode
#   ./launch.sh --config X   Use custom config file
# ============================================================

set -e

# Navigate to dashboard directory (works regardless of caller's cwd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Find Python (prefer venv > system) ----
PYTHON=""
if [ -f "../venv/bin/python" ]; then
    PYTHON="../venv/bin/python"
    echo "[OK] Using project venv: ../venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    echo "[OK] Using local venv: venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    echo "[OK] Using system python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
    echo "[OK] Using system python"
else
    echo "[ERROR] Python not found. Install Python 3.10-3.12 or run install.sh."
    exit 1
fi

# ---- Validate Python version (3.10-3.12; torch DLL issues above 3.12) ----
$PYTHON -c "import sys; v=sys.version_info; exit(0 if (3,10)<=v[:2]<=(3,12) else 1)" 2>/dev/null || {
    echo "[ERROR] Python version not supported. Need 3.10-3.12. Run install.sh."
    exit 1
}

# ---- Check critical dependencies ----
echo ""
echo "Checking dependencies..."
$PYTHON -c "import torch; torch.zeros(1)" 2>/dev/null || {
    echo "[ERROR] PyTorch not working. Run install.sh to fix."
    exit 1
}
$PYTHON -c "import PyQt6" 2>/dev/null || {
    echo "[ERROR] PyQt6 not installed. Run install.sh first."
    exit 1
}
$PYTHON -c "import stable_baselines3" 2>/dev/null || {
    echo "[ERROR] stable-baselines3 not installed. Run install.sh first."
    exit 1
}

# ---- MT5 informational check (not fatal on Linux/macOS) ----
if ! $PYTHON -c "import MetaTrader5" 2>/dev/null; then
    case "$(uname -s)" in
        Linux*|Darwin*)
            echo "[INFO] MetaTrader5 not installed (expected on $(uname -s))."
            echo "       Dashboard will run in offline mode -- no live trading."
            ;;
        *)
            echo "[WARN] MetaTrader5 not installed. Run install.sh first."
            ;;
    esac
fi

# ---- Create storage directories ----
mkdir -p storage/logs storage/memory storage/models storage/state \
         storage/screenshots storage/reports/weekly

# ---- Qt platform plugin path (Linux distros sometimes need this hint) ----
# Lets PyQt6 find its xcb/wayland plugins when bundled in a venv. Harmless
# on macOS / Windows-bash if the path doesn't exist.
QT_PLUGIN_DIR=""
for cand in "venv/lib/python3.12/site-packages/PyQt6/Qt6/plugins" \
            "venv/lib/python3.11/site-packages/PyQt6/Qt6/plugins" \
            "venv/lib/python3.10/site-packages/PyQt6/Qt6/plugins" \
            "../venv/lib/python3.12/site-packages/PyQt6/Qt6/plugins" \
            "../venv/lib/python3.11/site-packages/PyQt6/Qt6/plugins" \
            "../venv/lib/python3.10/site-packages/PyQt6/Qt6/plugins"; do
    if [ -d "$cand" ]; then
        QT_PLUGIN_DIR="$(cd "$cand" && pwd)"
        break
    fi
done
if [ -n "$QT_PLUGIN_DIR" ]; then
    export QT_PLUGIN_PATH="$QT_PLUGIN_DIR"
fi

# ---- Launch ----
echo ""
echo "============================================================"
echo "  SPARTUS LIVE TRADING DASHBOARD"
echo "============================================================"
echo ""
$PYTHON main.py "$@"
