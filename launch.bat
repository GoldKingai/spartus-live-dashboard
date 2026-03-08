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
        echo [ERROR] Python not found. Install Python 3.11+ or activate a virtualenv.
        pause
        exit /b 1
    )
)

:: ---- Check dependencies ----
echo.
echo Checking dependencies...
%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] MetaTrader5 not installed. Run: pip install MetaTrader5
)
%PYTHON% -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyQt6 not installed. Run: pip install -r requirements.txt
    pause
    exit /b 1
)
%PYTHON% -c "import stable_baselines3" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] stable-baselines3 not installed. Run: pip install -r requirements.txt
    pause
    exit /b 1
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
