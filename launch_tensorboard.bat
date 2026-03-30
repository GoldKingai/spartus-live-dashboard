@echo off
title SPARTUS - TensorBoard Dashboard
cd /d "%~dp0"
echo.
echo   SPARTUS TRADING AI - TensorBoard Dashboard
echo   =============================================
echo.
echo   Each training run creates its own subfolder under:
echo     storage\logs\tensorboard\run_YYYYMMDD_HHMMSS
echo.
echo   TensorBoard will show all runs side-by-side for comparison.
echo   Metrics are grouped: account/, trading/, sac/, reward/, journal/, etc.
echo.
echo   Opening TensorBoard in your browser...
echo   URL: http://localhost:6006
echo   Press Ctrl+C to stop TensorBoard
echo.
start "" http://localhost:6006
venv\Scripts\tensorboard.exe --logdir storage\logs\tensorboard --port 6006 --reload_interval 5
