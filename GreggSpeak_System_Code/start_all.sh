#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m app.web_app &
WEB_PID="$!"

cleanup() {
    kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" -m app.main_rpi "$@" || true
wait "$WEB_PID"
