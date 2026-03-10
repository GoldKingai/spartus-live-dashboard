@echo off
:: ============================================================
:: Spartus Live Trading Dashboard -- One-Click Installer
:: ============================================================
:: This script:
::   1. Checks for Python 3.10-3.12 (3.13+ has torch DLL issues)
::   2. Creates a virtual environment
::   3. Installs PyTorch CPU (must come BEFORE other deps)
::   4. Installs all remaining dependencies
::   5. Creates required directories
::   6. Verifies the installation
:: ============================================================

title Spartus Installer

echo.
echo ============================================================
echo   SPARTUS LIVE TRADING DASHBOARD -- INSTALLER
echo ============================================================
echo.

:: Navigate to this script's directory (works from any location)
cd /d "%~dp0"

:: ---- Step 1: Find Python 3.10-3.12 ----
echo [1/6] Checking for Python 3.10-3.12...
echo.

set "PYTHON="

:: Try specific versions first (most reliable)
where py >nul 2>&1
if %errorlevel%==0 (
    :: py launcher -- try 3.12, 3.11, 3.10 in order
    py -3.12 -c "pass" >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=py -3.12"
        echo        Found Python 3.12 via py launcher
        goto :python_found
    )
    py -3.11 -c "pass" >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=py -3.11"
        echo        Found Python 3.11 via py launcher
        goto :python_found
    )
    py -3.10 -c "pass" >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=py -3.10"
        echo        Found Python 3.10 via py launcher
        goto :python_found
    )
)

:: Fallback: check system python
where python >nul 2>&1
if %errorlevel%==0 (
    :: Check it's 3.10-3.12
    python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=python"
        goto :python_found
    )
    :: Wrong version -- show what they have
    echo.
    echo [ERROR] System Python found but wrong version:
    python --version
    echo.
    echo Spartus requires Python 3.10, 3.11, or 3.12.
    echo Python 3.13+ is NOT supported (PyTorch DLL compatibility issues).
    echo.
    echo Please install Python 3.12 from:
    echo   https://www.python.org/downloads/release/python-3129/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo.
echo [ERROR] Python is not installed or not in your PATH.
echo.
echo Please install Python 3.12 from:
echo   https://www.python.org/downloads/release/python-3129/
echo.
echo IMPORTANT: During installation, check the box that says
echo   "Add Python to PATH"
echo.
echo NOTE: Python 3.13+ is NOT supported due to PyTorch issues.
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%i in ('%PYTHON% --version') do echo        Version: %%i
echo        [OK]
echo.

:: ---- Step 2: Create virtual environment ----
echo [2/6] Creating virtual environment...

if exist "venv\Scripts\python.exe" (
    :: Check existing venv Python version is 3.10-3.12
    venv\Scripts\python.exe -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
    if %errorlevel% neq 0 (
        echo        Existing venv has incompatible Python -- recreating...
        rmdir /s /q venv >nul 2>&1
        %PYTHON% -m venv venv
        if %errorlevel% neq 0 (
            echo [ERROR] Failed to create virtual environment.
            pause
            exit /b 1
        )
        echo        [OK] Recreated venv/
    ) else (
        echo        Already exists with compatible Python -- skipping.
    )
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

:: ---- Step 3: Install PyTorch CPU ----
echo [3/6] Installing PyTorch (CPU version)...
echo        This is ~200 MB and may take a few minutes.
echo.

:: Check if torch is already installed and working
%PYTHON% -c "import torch; print(f'torch {torch.__version__} already installed')" >nul 2>&1
if %errorlevel%==0 (
    :: Verify it actually loads (catches the c10.dll crash)
    %PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
    if %errorlevel%==0 (
        for /f "tokens=*" %%i in ('%PYTHON% -c "import torch; print(f'torch {torch.__version__}')"') do echo        %%i -- working
        echo        [OK] Skipping reinstall.
        goto :torch_done
    ) else (
        echo        Existing torch installation is broken -- reinstalling...
        %PIP% uninstall torch -y >nul 2>&1
    )
)

%PIP% install --upgrade pip >nul 2>&1
%PIP% install torch --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch installation failed.
    echo        Check your internet connection and try again.
    pause
    exit /b 1
)
echo        [OK]

:torch_done
echo.

:: ---- Step 4: Install remaining dependencies ----
echo [4/6] Installing remaining dependencies...
echo.

%PIP% install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo        Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

:: ---- Step 5: Create directories ----
echo [5/6] Creating directory structure...

if not exist "model" mkdir "model"
if not exist "storage\logs" mkdir "storage\logs"
if not exist "storage\memory" mkdir "storage\memory"
if not exist "storage\models" mkdir "storage\models"
if not exist "storage\state" mkdir "storage\state"
if not exist "storage\screenshots" mkdir "storage\screenshots"
if not exist "storage\reports\weekly" mkdir "storage\reports\weekly"

echo        [OK] All directories ready.
echo.

:: ---- Step 6: Verify installation ----
echo [6/6] Verifying installation...

set "VERIFY_OK=1"
set "WARN_COUNT=0"

:: Critical packages
%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] PyTorch -- cannot load (DLL error?)
    set "VERIFY_OK=0"
) else (
    for /f "tokens=*" %%i in ('%PYTHON% -c "import torch; print(torch.__version__)"') do echo        [OK] PyTorch %%i
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

:: Optional packages
%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [WARN] MetaTrader5 -- requires MT5 terminal to be installed.
    set /a WARN_COUNT+=1
) else (
    echo        [OK] MetaTrader5
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
echo   1. Place your trained model .zip in the model\ folder
echo   2. Make sure MetaTrader 5 is running and logged in
echo   3. In MT5, right-click Market Watch and click "Show All"
echo      (XAUUSD must be visible for the dashboard to work)
echo   4. Double-click launch.bat to start the dashboard
echo.
echo Configuration: config\default_config.yaml
echo Logs:          storage\logs\dashboard.log
echo.
pause
