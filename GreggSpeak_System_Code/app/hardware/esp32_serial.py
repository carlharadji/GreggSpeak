from __future__ import annotations

from dataclasses import dataclass
import time


DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 2.0


@dataclass
class SerialPortInfo:
    device: str
    description: str
    hwid: str


def require_serial():
    try:
        import serial  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyserial is required. Install it with: pip install pyserial") from exc
    return serial


def list_serial_ports() -> list[SerialPortInfo]:
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyserial is required. Install it with: pip install pyserial") from exc

    ports = []
    for port in list_ports.comports():
        ports.append(
            SerialPortInfo(
                device=str(port.device),
                description=str(port.description or ""),
                hwid=str(port.hwid or ""),
            )
        )
    return ports


def auto_select_port() -> str | None:
    ports = list_serial_ports()
    if not ports:
        return None

    preferred_markers = (
        "ttyUSB",
        "ttyACM",
        "USB Serial",
        "CP210",
        "CH340",
        "ESP",
        "Silicon Labs",
        "wchusbserial",
    )
    for port in ports:
        haystack = f"{port.device} {port.description} {port.hwid}".lower()
        if any(marker.lower() in haystack for marker in preferred_markers):
            return port.device

    return ports[0].device


def send_command(
    command: str,
    port: str | None = None,
    baud: int = DEFAULT_BAUD,
    timeout: float = DEFAULT_TIMEOUT,
    settle_seconds: float = 1.5,
    read_extra_seconds: float = 0.0,
) -> dict:
    serial = require_serial()
    selected_port = port or auto_select_port()
    if not selected_port:
        return {
            "success": False,
            "message": "No serial port found. Connect the ESP32 over USB and try again.",
            "port": None,
            "response": "",
        }

    payload = (command.strip() or "PING") + "\n"

    try:
        serial_kwargs = {
            "port": selected_port,
            "baudrate": baud,
            "timeout": timeout,
            "exclusive": True,
        }
        try:
            connection = serial.Serial(**serial_kwargs)
        except TypeError:
            serial_kwargs.pop("exclusive", None)
            connection = serial.Serial(**serial_kwargs)
        with connection:
            # Many ESP32 boards reset when the serial port opens.
            time.sleep(max(0.0, settle_seconds))
            connection.reset_input_buffer()
            connection.write(payload.encode("utf-8"))
            connection.flush()
            response_lines = []
            response = connection.readline().decode("utf-8", errors="replace").strip()
            if response:
                response_lines.append(response)
            if read_extra_seconds > 0:
                deadline = time.monotonic() + read_extra_seconds
                while time.monotonic() < deadline:
                    line = connection.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        response_lines.append(line)
                        if "FEED_DONE" in line or line.startswith("ERR "):
                            break
            response = "\n".join(response_lines)
    except Exception as exc:
        return {
            "success": False,
            "message": f"Serial communication failed: {exc}",
            "port": selected_port,
            "response": "",
        }

    return {
        "success": bool(response),
        "message": "ESP32 responded." if response else "No response from ESP32 before timeout.",
        "port": selected_port,
        "response": response,
    }
