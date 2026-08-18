@echo off
setlocal

echo ==========================================
echo   MCM Analytics - Add snapshot workflow
echo ==========================================
echo.

cd /d "%~dp0"

if not exist "_workflow_staging\record_snapshots.yml" (
    echo ERROR: _workflow_staging\record_snapshots.yml not found.
    echo Make sure this .bat is in the same folder as that file.
    echo.
    pause
    exit /b 1
)

if not exist ".github" (
    echo Creating .github folder...
    mkdir ".github"
)

if not exist ".github\workflows" (
    echo Creating .github\workflows folder...
    mkdir ".github\workflows"
)

echo Moving workflow file into place...
move /Y "_workflow_staging\record_snapshots.yml" ".github\workflows\record_snapshots.yml" >nul

if exist ".github\workflows\record_snapshots.yml" (
    echo Done: .github\workflows\record_snapshots.yml is in place.
) else (
    echo ERROR: move failed - see above.
    pause
    exit /b 1
)

rmdir "_workflow_staging" 2>nul

echo.
echo ==========================================
echo   Now pushing to GitHub...
echo ==========================================
echo.

call push_to_github.bat
