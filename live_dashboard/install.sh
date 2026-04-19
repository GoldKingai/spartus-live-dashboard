#!/usr/bin/env bash
# ============================================================
# Spartus Live Trading Dashboard -- One-Click Installer
# ============================================================
# Cross-platform installer for Linux / macOS.
# Windows users: run install.bat instead.
#
# Steps:
#   1. Checks for Python 3.10-3.12 (3.13+ has torch DLL issues)
#   2. Creates a virtual environment
#   3. Installs PyTorch CPU (must come BEFORE other deps)
#   4. Installs all remaining dependencies (MetaTrader5 auto-skipped on
#      non-Windows via PEP 508 marker in requirements.txt)
#   5. Creates required directories
#   6. Verifies the installation
#
# ──────────────────────────────────────────────────────────────
# IMPORTANT: MT5 + Linux/macOS
# ──────────────────────────────────────────────────────────────
# MetaTrader5 only ships a Windows binary. Pip will skip it on
# Linux/macOS automatically. Without MT5, the dashboard runs in
# offline mode — UI loads, post-trade analysis works, but live
# broker connection is disabled.
#
# To enable live trading on Linux you can:
#   (a) Run MT5 + Python under Wine
#   (b) Use a remote MT5 bridge on a Windows host
# ============================================================

set -e

# Detect platform for clearer error messages later
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
case "$OS_NAME" in
    Linux*)   PLATFORM="Linux" ;;
    Darwin*)  PLATFORM="macOS" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="Windows-Bash" ;;
    *)        PLATFORM="Unknown" ;;
esac
echo "Detected platform: $PLATFORM"

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

# Optional — MT5 is Windows-only; on Linux/macOS this is expected to be absent
if $PYTHON -c "import MetaTrader5" 2>/dev/null; then
    echo "       [OK] MetaTrader5"
else
    if [ "$PLATFORM" = "Linux" ] || [ "$PLATFORM" = "macOS" ]; then
        echo "       [INFO] MetaTrader5 -- not available on $PLATFORM (expected)."
        echo "              Dashboard will run in offline mode (no live trading)."
    else
        echo "       [WARN] MetaTrader5 -- install MT5 terminal first."
    fi
fi

# Make launch.sh executable so user can `./launch.sh` directly
chmod +x launch.sh 2>/dev/null || true

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
if [ "$PLATFORM" = "Linux" ] || [ "$PLATFORM" = "macOS" ]; then
    echo "  2. (LINUX/MACOS) Live trading needs MT5. Options:"
    echo "       - Run via Wine + Windows MT5 + Windows Python"
    echo "       - Use a remote MT5 bridge running on a Windows host"
    echo "       - Or stay in offline/replay mode (dashboard UI only)"
    echo "  3. Run ./launch.sh to start the dashboard"
else
    echo "  2. Make sure MetaTrader 5 is running and logged in"
    echo "  3. In MT5, right-click Market Watch and click 'Show All'"
    echo "     (XAUUSD must be visible for the dashboard to work)"
    echo "  4. Run ./launch.sh to start the dashboard"
fi
echo ""
echo "Configuration: config/default_config.yaml"
echo "Logs:          storage/logs/dashboard.log"
echo ""

# ─────────────────────────────────────────────────────────────────
# Linux/macOS: optional MT5-in-Wine bridge instructions
# ─────────────────────────────────────────────────────────────────
if [ "$PLATFORM" = "Linux" ] || [ "$PLATFORM" = "macOS" ]; then
    echo "============================================================"
    echo "  LINUX LIVE TRADING (optional) — MT5 bridge setup"
    echo "============================================================"
    echo ""
    echo "Want live trading on $PLATFORM? MT5 needs to run in Wine and"
    echo "expose its API over the mt5linux RPC bridge. Steps:"
    echo ""
    echo "  1. Install Wine and create an MT5 Wine prefix"
    echo "  2. Install Windows Python 3.11 inside that Wine prefix"
    echo "  3. Inside Wine Python:  pip install MetaTrader5 mt5linux"
    echo "  4. Start the bridge daemon (inside Wine):"
    echo "        python -m mt5linux --host localhost --port 18812"
    echo "  5. Launch the dashboard — it auto-detects the bridge."
    echo ""
    echo "  Override host/port via env vars before launch:"
    echo "        export MT5_BRIDGE_HOST=192.168.1.50"
    echo "        export MT5_BRIDGE_PORT=18812"
    echo ""
    echo "  Full walkthrough: docs/LINUX_BRIDGE.md"
    echo ""
fi
