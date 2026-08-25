@echo off
title Mortgage AI - Full Stack Launcher
echo ============================================
echo   Mortgage AI - Full Stack Launcher
echo ============================================
echo.

cd /d "%~dp0"

REM ── Step 1: Start Backend in a new window ─────────────────────────────────
echo [STEP 1] Launching Backend (FastAPI)...
start "Mortgage AI Backend" cmd /k "cd /d %~dp0\backend && run_forever.bat"


echo [INFO] Backend window opened. Waiting 3 seconds for it to initialize...
timeout /t 3 /nobreak > nul

REM ── Step 2: Start Frontend in a new window ────────────────────────────────
echo.
echo [STEP 2] Launching Frontend (Vite)...
start "Mortgage AI Frontend" cmd /k "cd /d %~dp0\frontend && npm install && npm run dev"

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
echo   Frontend UI   : http://localhost:5173
echo.
echo   Two command prompt windows have been opened for the Backend and Frontend.
echo   Do not close them while using the application.
echo ============================================
echo.
pause
