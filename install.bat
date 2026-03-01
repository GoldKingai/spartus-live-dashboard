@echo off
:: ============================================================
:: Spartus Live Trading Dashboard -- One-Click Installer
:: ============================================================
:: This script:
::   1. Checks for Python 3.10+
::   2. Creates a virtual environment
::   3. Installs all dependencies
::   4. Creates required directories
::   5. Verifies the installation
:: ============================================================

title Spartus Installer

echo.
echo ============================================================
echo   SPARTUS LIVE TRADING DASHBOARD -- INSTALLER
echo ============================================================
echo.

:: Navigate to this script's directory (works from any location)
cd /d "%~dp0"

:: ---- Step 1: Find Python ----
echo [1/5] Checking for Python...

set "PYTHON="

:: Check common Python locations
where python >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=python"
) else (
    where python3 >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=python3"
    )
)

if "%PYTHON%"=="" (
    echo.
    echo [ERROR] Python is not installed or not in your PATH.
    echo.
    echo Please install Python 3.10 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check the box that says
    echo   "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: Check Python version (must be 3.10+)
%PYTHON% -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python 3.10 or higher is required.
    echo.
    %PYTHON% --version
    echo.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('%PYTHON% --version') do echo        Found: %%i
echo        [OK]
echo.

:: ---- Step 2: Create virtual environment ----
echo [2/5] Creating virtual environment...

if exist "venv\Scripts\python.exe" (
    echo        Already exists -- skipping.
) else (
    %PYTHON% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        [OK] Created venv/
)
echo.

:: Activate venv for the rest of this script
set "PYTHON=venv\Scripts\python.exe"
set "PIP=venv\Scripts\pip.exe"

:: ---- Step 3: Install dependencies ----
echo [3/5] Installing dependencies (this may take a few minutes)...
echo.

%PIP% install --upgrade pip >nul 2>&1
%PIP% install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo        Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

:: ---- Step 4: Create directories ----
echo [4/5] Creating directory structure...

if not exist "storage\logs" mkdir "storage\logs"
if not exist "storage\memory" mkdir "storage\memory"
if not exist "storage\models" mkdir "storage\models"
if not exist "storage\state" mkdir "storage\state"
if not exist "storage\screenshots" mkdir "storage\screenshots"
if not exist "storage\reports\weekly" mkdir "storage\reports\weekly"

echo        [OK] storage/ directories ready.
echo.

:: ---- Step 5: Verify installation ----
echo [5/5] Verifying installation...

set "VERIFY_OK=1"

%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [WARN] MetaTrader5 package not available -- install requires MT5 terminal.
)

%PYTHON% -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] PyQt6
    set "VERIFY_OK=0"
) else (
    echo        [OK] PyQt6
)

%PYTHON% -c "import stable_baselines3" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] stable-baselines3
    set "VERIFY_OK=0"
) else (
    echo        [OK] stable-baselines3
)

%PYTHON% -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] numpy
    set "VERIFY_OK=0"
) else (
    echo        [OK] numpy
)

%PYTHON% -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] pandas
    set "VERIFY_OK=0"
) else (
    echo        [OK] pandas
)

%PYTHON% -c "import ta" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] ta
    set "VERIFY_OK=0"
) else (
    echo        [OK] ta
)

%PYTHON% -c "import yaml" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] PyYAML
    set "VERIFY_OK=0"
) else (
    echo        [OK] PyYAML
)

echo.

if "%VERIFY_OK%"=="0" (
    echo ============================================================
    echo   INSTALLATION INCOMPLETE -- Some packages failed to install.
    echo   Try running: venv\Scripts\pip install -r requirements.txt
    echo ============================================================
    pause
    exit /b 1
)

echo ============================================================
echo   INSTALLATION COMPLETE
echo ============================================================
echo.
echo Next steps:
echo   1. Place your trained model .zip in storage\models\
echo   2. Make sure MetaTrader 5 is running and logged in
echo   3. Double-click launch.bat to start the dashboard
echo.
echo Configuration: config\default_config.yaml
echo Logs:          storage\logs\dashboard.log
echo.
pause
