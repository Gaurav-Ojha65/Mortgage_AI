@echo off
title Mortgage AI - Backend Server
echo ============================================
echo   Mortgage AI Backend - FastAPI Server
echo   Running on: http://localhost:8001
echo ============================================
echo.

cd /d "%~dp0"

REM Try activating virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] No venv found - using system Python
)

echo.
echo [INFO] Starting FastAPI backend on port 8001...
echo [INFO] API Docs available at: http://localhost:8001/docs
echo [INFO] Press CTRL+C to stop the server
echo.

python -m uvicorn run_server:app --host 0.0.0.0 --port 8001 --reload

pause
