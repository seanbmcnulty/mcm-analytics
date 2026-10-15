@echo off
REM Publishes the recovered/merged snapshot history in data\snapshots to the
REM "snapshots" branch (fast-forward push; no force). Run once, then the hourly
REM Action keeps appending to it. Review the merged CSVs first.
setlocal
set TMPD=%TEMP%\mcm_snap_publish
if exist "%TMPD%" rmdir /s /q "%TMPD%"
git clone --branch snapshots --single-branch https://github.com/seanbmcnulty/mcm-analytics.git "%TMPD%" || goto :err
if not exist "%TMPD%\data\snapshots" mkdir "%TMPD%\data\snapshots"
copy /y data\snapshots\*_surface.csv "%TMPD%\data\snapshots\" || goto :err
pushd "%TMPD%"
git add -f data/snapshots
git commit -m "restore merged snapshot history" || (popd & echo Nothing to commit & goto :end)
git push origin snapshots || (popd & goto :err)
popd
echo Done.
goto :end
:err
echo FAILED - nothing was force-pushed.
:end
endlocal
pause
