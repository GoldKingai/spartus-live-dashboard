@echo off
title Spartus Live Trading Dashboard

cd /d "%~dp0"

:: ---- Find Python ----
set "PYTHON="

if exist "python\python.exe" (
    set "PYTHON=python\python.exe"
    echo [OK] Using embedded Python
    goto :python_found
)
if exist "..\venv\Scripts\python.exe" (
    set "PYTHON=..\venv\Scripts\python.exe"
    echo [OK] Using project venv
    goto :python_found
)
if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    echo [OK] Using local venv
    goto :python_found
)

where python >nul 2>&1
if %errorlevel% neq 0 goto :no_python
set "PYTHON=python"
echo [OK] Using system Python
goto :python_found

:no_python
echo.
echo [ERROR] Python not found. Run install.bat first.
pause
exit /b 1

:python_found

:: ---- Validate Python version ----
%PYTHON% -c "import sys; v=sys.version_info; exit(0 if (3,10)<=v[:2]<=(3,12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python version not supported. Run install.bat.
    pause
    exit /b 1
)

:: ---- Quick dependency check ----
echo Checking dependencies...

%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% neq 0 (
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

if %errorlevel% neq 0 (
    echo.
    echo [Dashboard exited with error code %errorlevel%]
    pause
)
