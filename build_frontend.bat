@echo off
title Mortgage AI - Build Frontend
echo ============================================
echo   Mortgage AI - Building React Frontend
echo ============================================
echo.

cd /d "%~dp0\mortgage-frontend"

echo [INFO] Installing dependencies (if needed)...
call npm install

echo.
echo [INFO] Building React app for production...
echo [INFO] This may take 30-60 seconds...
echo.

call npm run build

echo.
if exist "build\index.html" (
    echo [SUCCESS] Build completed!
    echo [INFO] Built files are in: mortgage-frontend\build\
    echo.
    echo [NEXT STEP] Open VS Code, right-click on mortgage-frontend\build\index.html
    echo             and select "Open with Live Server"
) else (
    echo [ERROR] Build failed - check errors above
)

echo.
pause
