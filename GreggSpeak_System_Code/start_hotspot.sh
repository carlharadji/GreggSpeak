#!/usr/bin/env bash
set -e

CONNECTION_NAME="${GREGGSPEAK_HOTSPOT_NAME:-GreggSpeak-Hotspot}"

if ! command -v nmcli >/dev/null 2>&1; then
    echo "nmcli was not found. Install/use Raspberry Pi OS with NetworkManager enabled."
    exit 1
fi

if ! nmcli connection show "$CONNECTION_NAME" >/dev/null 2>&1; then
    echo "Hotspot connection '$CONNECTION_NAME' is not configured yet."
    echo "Run: ./setup_hotspot.sh"
    exit 1
fi

sudo nmcli connection up "$CONNECTION_NAME"
echo "GreggSpeak hotspot started."
