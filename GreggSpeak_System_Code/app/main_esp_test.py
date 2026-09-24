from __future__ import annotations

import argparse

from app.hardware.esp32_serial import DEFAULT_BAUD, list_serial_ports, send_command


def parse_args():
    parser = argparse.ArgumentParser(description="Test Raspberry Pi to ESP32 serial communication.")
    parser.add_argument("--list", action="store_true", help="List detected serial ports and exit.")
    parser.add_argument("--port", default="", help="Serial port, for example /dev/ttyUSB0 or /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate.")
    parser.add_argument("--command", default="PING", help="Command to send to the ESP32.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Read timeout in seconds.")
    parser.add_argument(
        "--read-extra",
        type=float,
        default=None,
        help="Extra seconds to keep reading after the first response. Auto-enabled for FEED commands.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        ports = list_serial_ports()
    except RuntimeError as exc:
        print(exc)
        return 2

    if args.list:
        if not ports:
            print("No serial ports detected.")
            return 1
        for port in ports:
            print(f"{port.device} | {port.description} | {port.hwid}")
        return 0

    command = args.command.strip() or "PING"
    read_extra = args.read_extra
    if read_extra is None:
        read_extra = _auto_extra_read_seconds(command)

    result = send_command(
        command=command,
        port=args.port or None,
        baud=args.baud,
        timeout=args.timeout,
        read_extra_seconds=read_extra,
    )

    print(f"Port: {result.get('port') or 'not found'}")
    print(f"Command: {command}")
    print(f"Status: {result.get('message')}")
    if result.get("response"):
        print(f"Response: {result['response']}")

    return 0 if result.get("success") else 1


def _auto_extra_read_seconds(command: str) -> float:
    parts = command.split()
    if not parts or parts[0].upper() != "FEED":
        return 0.0
    try:
        duration_ms = int(parts[1])
    except (IndexError, ValueError):
        return 3.0
    return max(3.0, (duration_ms / 1000.0) + 3.0)


if __name__ == "__main__":
    raise SystemExit(main())
