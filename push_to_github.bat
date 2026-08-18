@echo off
setlocal

echo ==========================================
echo   MCM Analytics - Push to GitHub
echo ==========================================
echo.

cd /d "%~dp0"

REM --- Clear a stale index.lock left behind by an interrupted git process ---
if exist ".git\index.lock" (
    echo Found a stale .git\index.lock - removing it...
    del /f /q ".git\index.lock"
)

git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo Adding GitHub remote...
    git remote add origin https://github.com/seanbmcnulty/mcm-analytics.git
) else (
    echo Remote "origin" is already configured.
)

echo.
echo Renaming branch to main...
git branch -M main

echo.
echo Staging and committing local changes...
git add -A
git diff --cached --quiet
if %errorlevel% neq 0 (
    git commit -m "Update before deploy"
) else (
    echo Nothing new to commit.
)

echo.
echo Syncing with GitHub (rebasing on any remote commits)...
git pull --rebase origin main
if %errorlevel% neq 0 (
    echo.
    echo ==========================================
    echo   Could not rebase automatically.
    echo   Run: git rebase --abort
    echo   then ask Claude to help resolve it.
    echo ==========================================
    echo.
    pause
    exit /b 1
)

echo.
echo Pushing to GitHub (a browser or credential prompt may appear)...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo ==========================================
    echo   Push failed - see the error above.
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo   Success! Your code is now on GitHub:
    echo   https://github.com/seanbmcnulty/mcm-analytics
    echo ==========================================
)

echo.
pause
