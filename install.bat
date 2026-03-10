@echo off
:: ============================================================
:: Spartus Live Trading Dashboard -- One-Click Installer
:: ============================================================
:: This script sets up EVERYTHING from scratch:
::   1. Downloads Python 3.11 embeddable package (if no Python found)
::   2. Bootstraps pip into the embedded Python
::   3. Installs PyTorch CPU
::   4. Installs all remaining dependencies
::   5. Creates required directories
::   6. Verifies the installation
::
:: The user does NOT need Python installed on their system.
:: ============================================================

title Spartus Installer

echo.
echo ============================================================
echo   SPARTUS LIVE TRADING DASHBOARD -- INSTALLER
echo ============================================================
echo.

:: Navigate to this script's directory (works from any location)
cd /d "%~dp0"

:: ============================================================
:: Step 1: Find or download Python
:: ============================================================
echo [1/6] Setting up Python...
echo.

:: --- Check if we already have an embedded Python ---
if exist "python\python.exe" (
    echo        Found embedded Python in python\ folder.
    python\python.exe --version 2>nul
    if %errorlevel%==0 (
        echo        [OK] Using embedded Python.
        set "PYTHON=python\python.exe"
        goto :python_ready
    ) else (
        echo        [WARN] Embedded Python is broken -- will re-download.
        rmdir /s /q python >nul 2>&1
    )
)

:: --- Check if we already have a working venv ---
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
    if %errorlevel%==0 (
        echo        Found existing venv with compatible Python.
        set "PYTHON=venv\Scripts\python.exe"
        set "USE_VENV=1"
        goto :python_ready
    ) else (
        echo        Existing venv has incompatible Python -- will set up fresh.
        rmdir /s /q venv >nul 2>&1
    )
)

:: --- Try to find a system Python 3.10-3.12 ---
set "SYS_PYTHON="

:: Try py launcher first (most reliable on Windows)
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.12 -c "pass" >nul 2>&1
    if %errorlevel%==0 (
        set "SYS_PYTHON=py -3.12"
        goto :found_system_python
    )
    py -3.11 -c "pass" >nul 2>&1
    if %errorlevel%==0 (
        set "SYS_PYTHON=py -3.11"
        goto :found_system_python
    )
    py -3.10 -c "pass" >nul 2>&1
    if %errorlevel%==0 (
        set "SYS_PYTHON=py -3.10"
        goto :found_system_python
    )
)

:: Try plain python command
where python >nul 2>&1
if %errorlevel%==0 (
    python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
    if %errorlevel%==0 (
        set "SYS_PYTHON=python"
        goto :found_system_python
    )
)

:: --- No system Python found -- download embedded Python ---
echo        No compatible Python found on system.
echo        Downloading Python 3.11 embeddable package...
echo.
goto :download_python

:found_system_python
echo        Found system Python: %SYS_PYTHON%
for /f "tokens=*" %%i in ('%SYS_PYTHON% --version') do echo        Version: %%i
echo.
echo        Creating virtual environment...
%SYS_PYTHON% -m venv venv
if %errorlevel% neq 0 (
    echo        [WARN] venv creation failed -- falling back to embedded Python.
    goto :download_python
)
set "PYTHON=venv\Scripts\python.exe"
set "USE_VENV=1"
echo        [OK] Created venv/
goto :python_ready

:: ============================================================
:: Download Python embeddable package
:: ============================================================
:download_python

set "PY_VERSION=3.11.9"
set "PY_ZIP=python-%PY_VERSION%-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/%PY_ZIP%"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

echo        Downloading Python %PY_VERSION% (~8 MB)...
echo        URL: %PY_URL%
echo.

:: Use PowerShell to download (available on all modern Windows)
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ZIP%' -UseBasicParsing }" 2>nul
if not exist "%PY_ZIP%" (
    :: Try curl as fallback
    curl -L -o "%PY_ZIP%" "%PY_URL%" 2>nul
)
if not exist "%PY_ZIP%" (
    echo.
    echo [ERROR] Failed to download Python.
    echo        Please check your internet connection.
    echo        Or install Python 3.11 manually from https://www.python.org
    pause
    exit /b 1
)

echo        Extracting Python...
mkdir python 2>nul
powershell -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath 'python' -Force" 2>nul
if not exist "python\python.exe" (
    echo [ERROR] Failed to extract Python.
    pause
    exit /b 1
)

:: Clean up zip
del "%PY_ZIP%" 2>nul

:: --- Configure the embeddable Python for pip ---
echo        Configuring embedded Python for package management...

:: Find the ._pth file (e.g., python311._pth)
for %%f in (python\python*._pth) do set "PTH_FILE=%%f"

if "%PTH_FILE%"=="" (
    echo [ERROR] Could not find ._pth file in python\ directory.
    pause
    exit /b 1
)

:: Rewrite the ._pth file to enable site-packages and import site
echo python311.zip> "%PTH_FILE%"
echo .>> "%PTH_FILE%"
echo Lib\site-packages>> "%PTH_FILE%"
echo import site>> "%PTH_FILE%"

:: Create Lib\site-packages directory
mkdir "python\Lib\site-packages" 2>nul

:: Download and run get-pip.py
echo        Installing pip...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile 'get-pip.py' -UseBasicParsing }" 2>nul
if not exist "get-pip.py" (
    curl -L -o "get-pip.py" "%GETPIP_URL%" 2>nul
)
if not exist "get-pip.py" (
    echo [ERROR] Failed to download get-pip.py
    pause
    exit /b 1
)

python\python.exe get-pip.py --no-warn-script-location >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install pip.
    pause
    exit /b 1
)

del "get-pip.py" 2>nul

set "PYTHON=python\python.exe"
set "USE_VENV=0"
echo        [OK] Embedded Python %PY_VERSION% ready.

:python_ready
echo.
for /f "tokens=*" %%i in ('%PYTHON% --version') do echo        Python: %%i
echo.

:: ============================================================
:: Step 2: Upgrade pip
:: ============================================================
echo [2/6] Upgrading pip...
%PYTHON% -m pip install --upgrade pip --no-warn-script-location >nul 2>&1
echo        [OK]
echo.

:: ============================================================
:: Step 3: Install PyTorch CPU
:: ============================================================
echo [3/6] Installing PyTorch (CPU version)...
echo        This is ~200 MB and may take a few minutes.
echo.

:: Check if torch is already installed and working
%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('%PYTHON% -c "import torch; print(torch.__version__)"') do echo        torch %%i already installed and working.
    echo        [OK] Skipping reinstall.
    goto :torch_done
)

:: Uninstall any broken torch first
%PYTHON% -m pip uninstall torch -y >nul 2>&1

%PYTHON% -m pip install torch --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
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

:: ============================================================
:: Step 4: Install remaining dependencies
:: ============================================================
echo [4/6] Installing remaining dependencies...
echo.

%PYTHON% -m pip install -r requirements.txt --no-warn-script-location
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo        Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

:: ============================================================
:: Step 5: Create directories
:: ============================================================
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

:: ============================================================
:: Step 6: Verify installation
:: ============================================================
echo [6/6] Verifying installation...

set "VERIFY_OK=1"

:: Critical: torch must actually load
%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] PyTorch -- cannot load
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

:: Optional
%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [WARN] MetaTrader5 -- requires MT5 terminal to be installed.
) else (
    echo        [OK] MetaTrader5
)

echo.

if "%VERIFY_OK%"=="0" (
    echo ============================================================
    echo   INSTALLATION INCOMPLETE -- Some packages failed.
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
