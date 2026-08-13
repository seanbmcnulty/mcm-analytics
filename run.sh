#!/usr/bin/env bash
# MCM Analytics — start script (macOS/Linux)
# Usage: ./run.sh [--daemon] [streamlit args...]

set -euo pipefail
cd "$(dirname "$0")"

# Find Python (prefer venv, fall back to system)
if [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: Python 3 not found. Install Python 3.10+ and try again."
    exit 1
fi

DAEMON=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--daemon" ]; then
        DAEMON=true
    else
        ARGS+=("$arg")
    fi
done

if [ "$DAEMON" = true ]; then
    echo "Starting MCM Analytics in background (port 8020)..."
    nohup "$PYTHON" -m streamlit run Home.py "${ARGS[@]}" \
        > /tmp/mcm-analytics.log 2>&1 &
    echo "PID: $!"
    echo "Log: /tmp/mcm-analytics.log"
    echo "Stop: pkill -f 'streamlit run Home.py'"
else
    exec "$PYTHON" -m streamlit run Home.py "${ARGS[@]}"
fi
