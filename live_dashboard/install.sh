#!/usr/bin/env bash
# ============================================================
# Spartus Live Trading Dashboard -- One-Click Installer
# ============================================================
# This script:
#   1. Checks for Python 3.10-3.12 (3.13+ has torch DLL issues)
#   2. Creates a virtual environment
#   3. Installs PyTorch CPU (must come BEFORE other deps)
#   4. Installs all remaining dependencies
#   5. Creates required directories
#   6. Verifies the installation
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

# ---- Step 1: Find Python 3.10-3.12 ----
echo "[1/6] Checking for Python 3.10-3.12..."

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        if $candidate -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "[ERROR] Python 3.10-3.12 is required (3.13+ is NOT supported)."
    echo ""
    echo "Please install a supported Python version:"
    echo "  macOS:  brew install python@3.12"
    echo "  Ubuntu: sudo apt install python3.12 python3.12-venv"
    echo ""
    echo "NOTE: Python 3.13+ causes PyTorch compatibility issues."
    echo ""
    exit 1
fi

echo "       Found: $($PYTHON --version)"
echo "       [OK]"
echo ""

# ---- Step 2: Create virtual environment ----
echo "[2/6] Creating virtual environment..."

if [ -f "venv/bin/python" ]; then
    # Check existing venv Python version
    if venv/bin/python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" 2>/dev/null; then
        echo "       Already exists with compatible Python -- skipping."
    else
        echo "       Existing venv has incompatible Python -- recreating..."
        rm -rf venv
        $PYTHON -m venv venv
        echo "       [OK] Recreated venv/"
    fi
else
    $PYTHON -m venv venv
    echo "       [OK] Created venv/"
fi
echo ""

PYTHON="venv/bin/python"
PIP="venv/bin/pip"

# ---- Step 3: Install PyTorch CPU ----
echo "[3/6] Installing PyTorch (CPU version)..."
echo "       This is ~200 MB and may take a few minutes."
echo ""

# Check if torch is already installed and working
if $PYTHON -c "import torch; torch.zeros(1)" 2>/dev/null; then
    echo "       torch $($PYTHON -c 'import torch; print(torch.__version__)') -- already working"
    echo "       [OK] Skipping reinstall."
else
    $PIP install --upgrade pip > /dev/null 2>&1
    $PIP install torch --index-url https://download.pytorch.org/whl/cpu
    echo "       [OK]"
fi
echo ""

# ---- Step 4: Install remaining dependencies ----
echo "[4/6] Installing remaining dependencies..."
echo ""

$PIP install -r requirements.txt

echo ""

# ---- Step 5: Create directories ----
echo "[5/6] Creating directory structure..."

mkdir -p model
mkdir -p storage/logs storage/memory storage/models storage/state
mkdir -p storage/screenshots storage/reports/weekly

echo "       [OK] All directories ready."
echo ""

# ---- Step 6: Verify installation ----
echo "[6/6] Verifying installation..."

VERIFY_OK=1

# Critical: torch must actually load
if $PYTHON -c "import torch; torch.zeros(1)" 2>/dev/null; then
    echo "       [OK] PyTorch $($PYTHON -c 'import torch; print(torch.__version__)')"
else
    echo "       [FAIL] PyTorch -- cannot load"
    VERIFY_OK=0
fi

for pkg in PyQt6 stable_baselines3 numpy pandas ta yaml fastapi uvicorn; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo "       [OK] $pkg"
    else
        echo "       [FAIL] $pkg"
        VERIFY_OK=0
    fi
done

# Optional
if $PYTHON -c "import MetaTrader5" 2>/dev/null; then
    echo "       [OK] MetaTrader5"
else
    echo "       [WARN] MetaTrader5 -- requires MT5 terminal to be installed."
fi

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
echo "  1. Place your trained model .zip in the model/ folder"
echo "  2. Make sure MetaTrader 5 is running and logged in"
echo "  3. In MT5, right-click Market Watch and click 'Show All'"
echo "     (XAUUSD must be visible for the dashboard to work)"
echo "  4. Run ./launch.sh to start the dashboard"
echo ""
echo "Configuration: config/default_config.yaml"
echo "Logs:          storage/logs/dashboard.log"
echo ""
