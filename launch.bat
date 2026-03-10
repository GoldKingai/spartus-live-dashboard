@echo off
:: ============================================================
:: Spartus Live Trading Dashboard -- Windows Launcher
:: ============================================================
:: Usage:
::   launch.bat                Normal launch
::   launch.bat --paper        Force paper trading mode
::   launch.bat --config X     Use custom config file
::   launch.bat --skip-update  Skip automatic update check
:: ============================================================

title Spartus Live Trading Dashboard

:: Navigate to dashboard directory
cd /d "%~dp0"

:: ---- Find Python ----
:: Priority: venv in parent project > venv in dashboard > system python
set "PYTHON="

if exist "..\venv\Scripts\python.exe" (
    set "PYTHON=..\venv\Scripts\python.exe"
    echo [OK] Using project venv: ..\venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    echo [OK] Using local venv: venv\Scripts\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=python"
        echo [OK] Using system Python
    ) else (
        echo [ERROR] Python not found. Run install.bat first, or install Python 3.11+
        pause
        exit /b 1
    )
)

:: ---- Validate Python version (3.10-3.12) ----
%PYTHON% -c "import sys; v=sys.version_info; exit(0 if (3,10)<=v[:2]<=(3,12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python version not supported.
    %PYTHON% --version
    echo.
    echo Spartus requires Python 3.10, 3.11, or 3.12.
    echo Python 3.13+ causes PyTorch DLL crashes on Windows.
    echo.
    echo Fix: Run install.bat to create a venv with the right Python,
    echo      or install Python 3.12 from python.org
    echo.
    pause
    exit /b 1
)

:: ---- Check dependencies ----
echo.
echo Checking dependencies...

:: Check torch loads without DLL crash
%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch failed to load. This usually means:
    echo   - CUDA version installed on CPU-only machine
    echo   - Python version incompatible with installed torch
    echo.
    echo Fix: Run install.bat to reinstall, or manually:
    echo   venv\Scripts\pip uninstall torch
    echo   venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo.
    pause
    exit /b 1
)

%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] MetaTrader5 not installed. Run: pip install MetaTrader5
)
%PYTHON% -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyQt6 not installed. Run install.bat first.
    pause
    exit /b 1
)
%PYTHON% -c "import stable_baselines3" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] stable-baselines3 not installed. Run install.bat first.
    pause
    exit /b 1
)

:: ---- Check for model ----
if not exist "model\*.zip" (
    echo.
    echo [WARN] No model file found in model\ folder.
    echo        Place your spartus_live_*.zip model file in the model\ folder.
    echo        The dashboard will launch in limited mode without a model.
    echo.
)

:: ---- Create storage directories ----
if not exist "storage\logs" mkdir "storage\logs"
if not exist "storage\memory" mkdir "storage\memory"
if not exist "storage\models" mkdir "storage\models"
if not exist "storage\state" mkdir "storage\state"
if not exist "storage\screenshots" mkdir "storage\screenshots"
if not exist "storage\reports\weekly" mkdir "storage\reports\weekly"

:: ---- Launch ----
echo.
echo ============================================================
echo   SPARTUS LIVE TRADING DASHBOARD
echo ============================================================
echo.
%PYTHON% main.py %*

:: If dashboard closes with an error, pause so user can read it
if %errorlevel% neq 0 (
    echo.
    echo [Dashboard exited with error code %errorlevel%]
    pause
)
