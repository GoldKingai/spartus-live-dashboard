#!/usr/bin/env bash
# ============================================================
# Spartus Live Trading Dashboard -- One-Click Installer
# ============================================================
# This script:
#   1. Checks for Python 3.10+
#   2. Creates a virtual environment
#   3. Installs all dependencies
#   4. Creates required directories
#   5. Verifies the installation
# ============================================================

set -e

echo ""
echo "============================================================"
echo "  SPARTUS LIVE TRADING DASHBOARD -- INSTALLER"
echo "============================================================"
echo ""

# Navigate to this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Step 1: Find Python ----
echo "[1/5] Checking for Python..."

PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
fi

if [ -z "$PYTHON" ]; then
    echo ""
    echo "[ERROR] Python is not installed."
    echo ""
    echo "Please install Python 3.10 or higher:"
    echo "  macOS:  brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
    echo ""
    exit 1
fi

# Check Python version
$PYTHON -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null || {
    echo ""
    echo "[ERROR] Python 3.10 or higher is required."
    $PYTHON --version
    exit 1
}

echo "       Found: $($PYTHON --version)"
echo "       [OK]"
echo ""

# ---- Step 2: Create virtual environment ----
echo "[2/5] Creating virtual environment..."

if [ -f "venv/bin/python" ]; then
    echo "       Already exists -- skipping."
else
    $PYTHON -m venv venv
    echo "       [OK] Created venv/"
fi
echo ""

PYTHON="venv/bin/python"
PIP="venv/bin/pip"

# ---- Step 3: Install dependencies ----
echo "[3/5] Installing dependencies (this may take a few minutes)..."
echo ""

$PIP install --upgrade pip > /dev/null 2>&1
$PIP install -r requirements.txt

echo ""

# ---- Step 4: Create directories ----
echo "[4/5] Creating directory structure..."

mkdir -p storage/logs storage/memory storage/models storage/state
mkdir -p storage/screenshots storage/reports/weekly

echo "       [OK] storage/ directories ready."
echo ""

# ---- Step 5: Verify installation ----
echo "[5/5] Verifying installation..."

VERIFY_OK=1

for pkg in PyQt6 stable_baselines3 numpy pandas ta yaml; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo "       [OK] $pkg"
    else
        echo "       [FAIL] $pkg"
        VERIFY_OK=0
    fi
done

echo ""

if [ "$VERIFY_OK" -eq 0 ]; then
    echo "============================================================"
    echo "  INSTALLATION INCOMPLETE -- Some packages failed."
    echo "  Try: venv/bin/pip install -r requirements.txt"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "  INSTALLATION COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Place your trained model .zip in storage/models/"
echo "  2. Make sure MetaTrader 5 is running and logged in"
echo "  3. Run ./launch.sh to start the dashboard"
echo ""
echo "Configuration: config/default_config.yaml"
echo "Logs:          storage/logs/dashboard.log"
echo ""
