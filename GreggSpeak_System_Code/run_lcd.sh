#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

LOCK_FILE="${GREGGSPEAK_LCD_LOCK_FILE:-/tmp/greggspeak_lcd.lock}"
if command -v flock >/dev/null 2>&1; then
    exec flock -n "$LOCK_FILE" "$PYTHON_BIN" -m lcd_ui.main "$@"
fi

exec "$PYTHON_BIN" -m lcd_ui.main "$@"
