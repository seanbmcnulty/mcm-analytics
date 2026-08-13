@echo off
REM MCM Analytics — start script (Windows)
REM Usage: run.bat

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo Starting MCM Analytics on port 8020...
%PYTHON% -m streamlit run Home.py %*
