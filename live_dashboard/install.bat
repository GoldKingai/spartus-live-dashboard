@echo off
title Spartus Installer

echo.
echo ============================================================
echo   SPARTUS LIVE TRADING DASHBOARD -- INSTALLER
echo ============================================================
echo.

cd /d "%~dp0"

:: ============================================================
:: Step 1: Find or download Python
:: ============================================================
echo [1/7] Setting up Python...
echo.

:: --- Check for embedded Python ---
if not exist "python\python.exe" goto :check_venv
echo        Found embedded Python in python\ folder.
python\python.exe --version 2>nul
if %errorlevel%==0 (
    echo        [OK] Using embedded Python.
    set "PYTHON=python\python.exe"
    goto :python_ready
)
echo        [WARN] Embedded Python is broken -- will re-download.
rmdir /s /q python >nul 2>&1

:check_venv
if not exist "venv\Scripts\python.exe" goto :check_system_python
venv\Scripts\python.exe -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
if %errorlevel%==0 (
    echo        Found existing venv with compatible Python.
    set "PYTHON=venv\Scripts\python.exe"
    goto :python_ready
)
echo        Existing venv has incompatible Python -- will set up fresh.
rmdir /s /q venv >nul 2>&1

:check_system_python
set "SYS_PYTHON="

where py >nul 2>&1
if %errorlevel% neq 0 goto :try_plain_python

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

:try_plain_python
where python >nul 2>&1
if %errorlevel% neq 0 goto :download_python
python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
if %errorlevel%==0 (
    set "SYS_PYTHON=python"
    goto :found_system_python
)
goto :download_python

:found_system_python
echo        Found system Python: %SYS_PYTHON%
echo.
echo        Creating virtual environment...
%SYS_PYTHON% -m venv venv
if %errorlevel% neq 0 (
    echo        [WARN] venv creation failed -- falling back to embedded Python.
    goto :download_python
)
set "PYTHON=venv\Scripts\python.exe"
echo        [OK] Created venv/
goto :python_ready

:: ============================================================
:: Download Python embeddable package
:: ============================================================
:download_python
echo        No compatible Python found on system.
echo        Downloading Python 3.11.9 embeddable package (~8 MB)...
echo.

set "PY_VERSION=3.11.9"
set "PY_ZIP=python-%PY_VERSION%-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/%PY_ZIP%"

powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ZIP%' -UseBasicParsing" 2>nul
if not exist "%PY_ZIP%" curl -L -o "%PY_ZIP%" "%PY_URL%" 2>nul
if not exist "%PY_ZIP%" (
    echo [ERROR] Failed to download Python. Check your internet connection.
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
del "%PY_ZIP%" 2>nul

:: Configure embedded Python for pip
echo        Configuring for package management...
set "PTH_FILE="
for %%f in (python\python*._pth) do set "PTH_FILE=%%f"
if "%PTH_FILE%"=="" (
    echo [ERROR] Could not find ._pth file.
    pause
    exit /b 1
)

echo python311.zip> "%PTH_FILE%"
echo.>> "%PTH_FILE%"
echo Lib\site-packages>> "%PTH_FILE%"
echo import site>> "%PTH_FILE%"
mkdir "python\Lib\site-packages" 2>nul

:: Install pip
echo        Installing pip...
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile 'get-pip.py' -UseBasicParsing" 2>nul
if not exist "get-pip.py" curl -L -o "get-pip.py" "%GETPIP_URL%" 2>nul
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
echo        [OK] Embedded Python %PY_VERSION% ready.

:python_ready
echo.
echo        Python ready.
echo.

:: ============================================================
:: Step 2: Upgrade pip
:: ============================================================
echo [2/7] Upgrading pip...
%PYTHON% -m pip install --upgrade pip --no-warn-script-location >nul 2>&1
echo        [OK]
echo.

:: ============================================================
:: Step 3: Install Visual C++ Runtime (required by PyTorch)
:: ============================================================
echo [3/7] Checking Visual C++ Runtime...

if exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    echo        [OK] Visual C++ Runtime already installed.
    goto :vcpp_done
)

echo        Visual C++ Runtime not found -- downloading...
set "VCREDIST_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%VCREDIST_URL%' -OutFile 'vc_redist.x64.exe' -UseBasicParsing" 2>nul
if not exist "vc_redist.x64.exe" curl -L -o "vc_redist.x64.exe" "%VCREDIST_URL%" 2>nul
if not exist "vc_redist.x64.exe" (
    echo        [WARN] Could not download VC++ Runtime.
    echo        If PyTorch fails, install from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    goto :vcpp_done
)

echo        Installing VC++ Redistributable (may need admin)...
vc_redist.x64.exe /install /quiet /norestart
if %errorlevel% neq 0 (
    echo        Trying with progress bar...
    vc_redist.x64.exe /install /passive /norestart
)
del "vc_redist.x64.exe" 2>nul
echo        [OK] Visual C++ Runtime installed.

:vcpp_done
echo.

:: ============================================================
:: Step 4: Install PyTorch CPU
:: ============================================================
echo [4/7] Installing PyTorch (CPU version)...
echo        This may take a few minutes (~200 MB).
echo.

%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel%==0 (
    echo        [OK] PyTorch already working. Skipping.
    goto :torch_done
)

echo        Removing broken/incompatible torch...
%PYTHON% -m pip uninstall torch -y >nul 2>&1

echo        Installing torch (CPU only)...
%PYTHON% -m pip install torch --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo        [OK]

:torch_done
echo.

:: ============================================================
:: Step 5: Install remaining dependencies
:: ============================================================
echo [5/7] Installing remaining dependencies...
echo.

%PYTHON% -m pip install -r requirements.txt --no-warn-script-location
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo.

:: ============================================================
:: Step 6: Create directories
:: ============================================================
echo [6/7] Creating directory structure...

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
:: Step 7: Verify installation
:: ============================================================
echo [7/7] Verifying installation...
echo.

set "VERIFY_OK=1"

%PYTHON% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAIL] PyTorch -- cannot load
    set "VERIFY_OK=0"
) else (
    echo        [OK] PyTorch
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

%PYTHON% -c "import MetaTrader5" >nul 2>&1
if %errorlevel% neq 0 (
    echo        [WARN] MetaTrader5 -- install MT5 terminal first.
) else (
    echo        [OK] MetaTrader5
)

echo.

if "%VERIFY_OK%"=="0" (
    echo ============================================================
    echo   INSTALLATION INCOMPLETE -- Some packages failed.
    echo ============================================================
    echo.
    echo Try running install.bat again. If the problem persists,
    echo delete the python\ folder and run install.bat again.
    echo.
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
echo   4. Double-click launch.bat to start the dashboard
echo.
pause
