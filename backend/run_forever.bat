@echo off
title Mortgage AI - Backend Watchdog
cd /d "%~dp0"

:loop
echo ==================================================
echo   [%date% %time%] Starting Backend Service...
echo ==================================================

REM Activate venv if exists
if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
) else if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
)

python -m uvicorn run_server:app --host 0.0.0.0 --port 8001 --reload

echo.
echo ==================================================
echo   [%date% %time%] BACKEND CRASHED! 
echo   Exit Code: %errorlevel%
echo   Restarting in 5 seconds...
echo ==================================================
timeout /t 5 > nul
goto loop
