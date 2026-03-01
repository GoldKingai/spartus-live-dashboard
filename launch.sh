#!/usr/bin/env bash
# ============================================================
# Spartus Live Trading Dashboard -- Unix/Mac Launcher
# ============================================================
# Usage:
#   ./launch.sh              Normal launch
#   ./launch.sh --paper      Force paper trading mode
#   ./launch.sh --config X   Use custom config file
# ============================================================

set -e

# Navigate to dashboard directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Find Python ----
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
    echo "[ERROR] Python not found. Install Python 3.11+ or activate a virtualenv."
    exit 1
fi

# ---- Check critical dependencies ----
echo ""
echo "Checking dependencies..."
$PYTHON -c "import PyQt6" 2>/dev/null || {
    echo "[ERROR] PyQt6 not installed. Run: pip install -r requirements.txt"
    exit 1
}
$PYTHON -c "import stable_baselines3" 2>/dev/null || {
    echo "[ERROR] stable-baselines3 not installed. Run: pip install -r requirements.txt"
    exit 1
}

# ---- Create storage directories ----
mkdir -p storage/logs storage/memory storage/models storage/state

# ---- Launch ----
echo ""
echo "============================================================"
echo "  SPARTUS LIVE TRADING DASHBOARD"
echo "============================================================"
echo ""
$PYTHON main.py "$@"
