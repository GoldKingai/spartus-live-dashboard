@echo off
title SPARTUS TRADING AI
cd /d "%~dp0"
echo.
echo   SPARTUS TRADING AI
echo   ==================
echo.
echo   Starting TensorBoard (detailed metrics in browser)...
start "Spartus TensorBoard" /min cmd /c "venv\Scripts\tensorboard.exe --logdir storage\logs\tensorboard --port 6006 --reload_interval 5"
timeout /t 2 /nobreak >nul
start "" http://localhost:6006
echo   TensorBoard: http://localhost:6006 (opened in browser)
echo.
echo   Launching Qt dashboard + training...
echo.
venv\Scripts\python.exe scripts\train.py --weeks 700
echo.
echo   Training complete. TensorBoard still running at http://localhost:6006
echo   Close the TensorBoard window to stop it.
pause
