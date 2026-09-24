#!/usr/bin/env bash
set -e

SSID="${1:-GreggSpeak-RPi}"
PASSWORD="${GREGGSPEAK_HOTSPOT_PASSWORD:-}"
if [ -z "$PASSWORD" ]; then
    if [ ! -t 0 ]; then
        echo "Set GREGGSPEAK_HOTSPOT_PASSWORD before configuring the hotspot."
        exit 1
    fi
    read -r -s -p "Choose a unique hotspot password: " PASSWORD
    echo
fi
CONNECTION_NAME="${GREGGSPEAK_HOTSPOT_NAME:-GreggSpeak-Hotspot}"
IFACE="${GREGGSPEAK_WIFI_IFACE:-wlan0}"
HOTSPOT_IP="${GREGGSPEAK_HOTSPOT_IP:-10.42.0.1/24}"

if [ "${#PASSWORD}" -lt 8 ]; then
    echo "Hotspot password must be at least 8 characters."
    exit 1
fi

if ! command -v nmcli >/dev/null 2>&1; then
    echo "nmcli was not found. Install/use Raspberry Pi OS with NetworkManager enabled."
    exit 1
fi

if ! nmcli device status | awk '{print $1}' | grep -qx "$IFACE"; then
    echo "Wi-Fi interface '$IFACE' was not found."
    echo "Set another interface with: GREGGSPEAK_WIFI_IFACE=wlan1 ./setup_hotspot.sh"
    exit 1
fi

sudo nmcli connection delete "$CONNECTION_NAME" >/dev/null 2>&1 || true

sudo nmcli connection add \
    type wifi \
    ifname "$IFACE" \
    con-name "$CONNECTION_NAME" \
    autoconnect yes \
    ssid "$SSID"

sudo nmcli connection modify "$CONNECTION_NAME" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    ipv4.addresses "$HOTSPOT_IP" \
    ipv6.method ignore \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASSWORD"

sudo nmcli connection up "$CONNECTION_NAME"

echo
echo "GreggSpeak hotspot is configured."
echo "SSID: $SSID"
echo "Password: configured from your private value"
echo "Web app URL after startup: http://${HOTSPOT_IP%/*}:5000"
