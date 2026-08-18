@echo off
setlocal

echo ==========================================
echo   MCM Analytics - Push to GitHub
echo ==========================================
echo.

cd /d "%~dp0"

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
echo Staging and committing any local changes...
git add -A
git diff --cached --quiet
if %errorlevel% neq 0 (
    git commit -m "Update before deploy"
) else (
    echo Nothing new to commit.
)

echo.
echo Pushing to GitHub (a browser or credential prompt may appear - sign in if asked)...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo ==========================================
    echo   Push failed - see the error above.
    echo   Common fix: make sure you are logged in
    echo   to GitHub as seanbmcnulty when prompted.
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
