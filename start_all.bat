@echo off
title Mortgage AI - Full Stack Launcher
echo ============================================
echo   Mortgage AI - Full Stack Launcher
echo   Backend  : http://localhost:8001
echo   Docs     : http://localhost:8001/docs
echo   Frontend : http://127.0.0.1:5500  (Go Live)
echo ============================================
echo.

cd /d "%~dp0"

REM ── Step 1: Start Backend in a new window ─────────────────────────────────
echo [STEP 1] Launching Backend (FastAPI)...
start "Mortgage AI Backend" cmd /k "cd /d %~dp0 && (if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) else if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat)) && python -m uvicorn run_server:app --host 0.0.0.0 --port 8001 --reload"

echo [INFO] Backend window opened. Waiting 5 seconds for it to initialize...
timeout /t 5 /nobreak > nul

REM ── Step 2: Check if frontend is already built ────────────────────────────
echo.
echo [STEP 2] Checking frontend build...

if not exist "mortgage-frontend\build\index.html" (
    echo [INFO] Build not found - building React app now...
    cd mortgage-frontend
    call npm install
    call npm run build
    cd ..
    echo [INFO] Build complete!
) else (
    echo [INFO] Existing build found at mortgage-frontend\build\
)

REM ── Step 3: Instructions ──────────────────────────────────────────────────
echo.
echo ============================================
echo   EVERYTHING IS READY!
echo ============================================
echo.
echo   Backend API   : http://localhost:8001
echo   API Docs      : http://localhost:8001/docs
echo   Health Check  : http://localhost:8001/health
echo.
echo   HOW TO OPEN FRONTEND WITH GO LIVE:
echo   1. Open VS Code in this folder
echo   2. In the Explorer, navigate to:
echo      mortgage-frontend ^> build ^> index.html
echo   3. Right-click index.html
echo   4. Select "Open with Live Server"
echo   5. App opens at http://127.0.0.1:5500
echo.
echo   OR: Click "Go Live" in the VS Code status bar
echo       (Live Server is pre-configured to serve the build folder)
echo.
echo ============================================
echo   Press any key to open the build folder...
pause > nul

explorer "%~dp0mortgage-frontend\build"
