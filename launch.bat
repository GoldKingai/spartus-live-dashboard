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
:: Priority: embedded python > parent venv > local venv > system python
set "PYTHON="

if exist "python\python.exe" (
    set "PYTHON=python\python.exe"
    echo [OK] Using embedded Python: python\python.exe
) else if exist "..\venv\Scripts\python.exe" (
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
        echo.
        echo [ERROR] Python not found. Run install.bat first.
        echo         install.bat will download Python automatically.
        echo.
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
    echo Spartus requires Python 3.10-3.12.
    echo Run install.bat to set up the correct version automatically.
    echo.
    pause
    exit /b 1
)

:: ---- Quick dependency check ----
echo.
echo Checking dependencies...

%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch not working. Run install.bat to fix.
    pause
    exit /b 1
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

%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] MetaTrader5 not installed. MT5 features will be unavailable.
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

:: ---- Set Qt plugin path for embedded Python ----
if exist "python\Lib\site-packages\PyQt6\Qt6\plugins" (
    set "QT_PLUGIN_PATH=%~dp0python\Lib\site-packages\PyQt6\Qt6\plugins"
)

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
