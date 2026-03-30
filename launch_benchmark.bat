@echo off
title SPARTUS BENCH - Model Benchmark Tool
cd /d "%~dp0"
echo.
echo   SPARTUS BENCH
echo   =============
echo.
echo   Launching benchmark dashboard...
echo.
venv\Scripts\python.exe scripts\launch_spartusbench.py
echo.
echo   Benchmark closed.
pause
