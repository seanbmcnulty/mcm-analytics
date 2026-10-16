@echo off
REM Publishes the recovered/merged snapshot history in data\snapshots to the
REM "snapshots" branch WITHOUT overwriting anything already there:
REM   1. clones the current snapshots branch
REM   2. merges local rows into it (rows already on the branch always win;
REM      only timestamps the branch doesn't have are added)
REM   3. validates that no published row is lost
REM   4. commits and pushes normally (never --force)
REM Safe to run more than once. Avoid :15-:20 and :45-:50 past the hour
REM (the recorder's run times); if the push is rejected just run it again.
setlocal
cd /d "%~dp0"
set TMPD=%TEMP%\mcm_snap_publish
if exist "%TMPD%" rmdir /s /q "%TMPD%"
git clone --depth 1 --branch snapshots --single-branch https://github.com/seanbmcnulty/mcm-analytics.git "%TMPD%" || goto :err
if not exist "%TMPD%\data\snapshots" mkdir "%TMPD%\data\snapshots"
set BASE=%TEMP%\mcm_snap_baseline
if exist "%BASE%" rmdir /s /q "%BASE%"
mkdir "%BASE%"
copy /y "%TMPD%\data\snapshots\*_surface.csv" "%BASE%\" >nul 2>&1
python scripts\snapshot_tools.py tag-sources --dir data\snapshots || goto :err
python scripts\snapshot_tools.py merge --base "%TMPD%\data\snapshots" --incoming data\snapshots || goto :err
python scripts\snapshot_tools.py validate --published "%BASE%" --candidate "%TMPD%\data\snapshots" --allow-insert || goto :err
pushd "%TMPD%"
git add -f data/snapshots
git commit -m "restore merged snapshot history (append-only merge)" || (popd & echo Nothing to commit & goto :end)
git push origin HEAD:snapshots || (popd & goto :err)
popd
echo Done. Check row counts with: python scripts\snapshot_tools.py stale --url-base https://raw.githubusercontent.com/seanbmcnulty/mcm-analytics/snapshots/data/snapshots
goto :end
:err
echo FAILED - nothing was force-pushed; the branch is unchanged. Safe to re-run.
:end
endlocal
pause
