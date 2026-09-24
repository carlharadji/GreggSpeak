#!/usr/bin/env bash
set -e

CONNECTION_NAME="${GREGGSPEAK_HOTSPOT_NAME:-GreggSpeak-Hotspot}"

if ! command -v nmcli >/dev/null 2>&1; then
    echo "nmcli was not found."
    exit 1
fi

sudo nmcli connection down "$CONNECTION_NAME" || true
echo "GreggSpeak hotspot stopped."
