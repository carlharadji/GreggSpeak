from __future__ import annotations

from dataclasses import dataclass, field
import time
import uuid

from app.hardware.esp32_serial import DEFAULT_BAUD, DEFAULT_TIMEOUT, auto_select_port, require_serial


MAX_FEED_DURATION_MS = 20000
FEED_RESPONSE_GRACE_SECONDS = 6.0


class FeederSerialError(RuntimeError):
    pass


@dataclass
class FeederStatus:
    paper_present: bool = False
    motor_running: bool = False
    state: str = "UNKNOWN"
    raw_response: str = ""


@dataclass
class FeedResult:
    success: bool
    message: str
    job_id: str
    elapsed_ms: int = 0
    paper_present: bool | None = None
    response_lines: list[str] = field(default_factory=list)


class FeederSerialClient:
    def __init__(
        self,
        port: str | None = None,
        baud: int = DEFAULT_BAUD,
        timeout: float = DEFAULT_TIMEOUT,
        open_settle_seconds: float = 1.5,
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.open_settle_seconds = open_settle_seconds
        self.connection = None
        self.selected_port: str | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    @property
    def is_open(self) -> bool:
        return bool(self.connection is not None and getattr(self.connection, "is_open", False))

    def open(self):
        if self.is_open:
            return

        serial = require_serial()
        selected_port = self.port or auto_select_port()
        if not selected_port:
            raise FeederSerialError("No ESP32 serial port found.")

        try:
            serial_kwargs = {
                "port": selected_port,
                "baudrate": self.baud,
                "timeout": self.timeout,
                "write_timeout": self.timeout,
                "exclusive": True,
            }
            try:
                self.connection = serial.Serial(**serial_kwargs)
            except TypeError:
                serial_kwargs.pop("exclusive", None)
                self.connection = serial.Serial(**serial_kwargs)
            self.selected_port = selected_port
            time.sleep(max(0.0, self.open_settle_seconds))
            self.connection.reset_input_buffer()
        except Exception as exc:
            self.connection = None
            raise FeederSerialError(f"Could not open ESP32 serial port: {exc}") from exc

    def close(self):
        if self.connection is None:
            return
        try:
            if self.connection.is_open:
                self.connection.close()
        finally:
            self.connection = None

    def ping(self) -> str:
        return self.command("PING")

    def status(self) -> FeederStatus:
        response = self.command("STATUS")
        return _parse_status_response(response)

    def sensor_status(self) -> bool:
        response = self.command("SENSOR?")
        upper = response.upper()
        if "PAPER_PRESENT" in upper:
            return True
        if "NO_PAPER" in upper:
            return False
        raise FeederSerialError(f"Unexpected sensor response: {response}")

    def stop(self) -> str:
        try:
            return self.command("STOP")
        except FeederSerialError:
            if self.connection is not None:
                try:
                    self._write_line("STOP")
                except Exception:
                    pass
            raise

    def feed(self, duration_ms: int, speed: int, job_id: str | None = None) -> FeedResult:
        try:
            duration_ms = int(duration_ms)
            speed = int(speed)
        except (TypeError, ValueError):
            return FeedResult(False, "Feed duration or speed is invalid.", job_id or "invalid")

        if duration_ms <= 0 or duration_ms > MAX_FEED_DURATION_MS:
            return FeedResult(False, "Feed duration exceeds the safe limit.", job_id or "invalid")

        active_job_id = job_id or f"job-{uuid.uuid4().hex[:8]}"
        command = f"FEED {duration_ms} {speed} {active_job_id}"
        deadline = time.monotonic() + (duration_ms / 1000.0) + FEED_RESPONSE_GRACE_SECONDS
        lines: list[str] = []

        try:
            self.open()
            self.connection.reset_input_buffer()
            self._write_line(command)

            while time.monotonic() < deadline:
                line = self._read_line(deadline=deadline)
                if not line:
                    continue
                lines.append(line)
                upper = line.upper()

                if upper.startswith("ERR "):
                    return FeedResult(False, line, active_job_id, response_lines=lines)

                if "FEED_DONE" in upper and active_job_id.upper() in upper:
                    return FeedResult(
                        True,
                        "Feed completed.",
                        active_job_id,
                        elapsed_ms=_parse_int_token(line, "ELAPSED", duration_ms),
                        paper_present=_parse_bool_token(line, "PAPER"),
                        response_lines=lines,
                    )
        except Exception as exc:
            return FeedResult(False, f"Serial feed failed: {exc}", active_job_id, response_lines=lines)

        try:
            self.stop()
        except Exception:
            pass
        return FeedResult(False, "Motor feed timed out.", active_job_id, response_lines=lines)

    def command(self, command: str, timeout: float | None = None) -> str:
        self.open()
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        self.connection.reset_input_buffer()
        self._write_line(command)

        while time.monotonic() < deadline:
            line = self._read_line(deadline=deadline)
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("OK ") or upper.startswith("ERR "):
                if upper.startswith("ERR "):
                    raise FeederSerialError(line)
                return line

        raise FeederSerialError(f"No response from ESP32 for command: {command}")

    def _write_line(self, command: str):
        if not self.is_open:
            raise FeederSerialError("ESP32 serial port is not open.")
        payload = (command.strip() + "\n").encode("utf-8")
        self.connection.write(payload)
        self.connection.flush()

    def _read_line(self, deadline: float) -> str:
        if not self.is_open:
            raise FeederSerialError("ESP32 serial port is not open.")
        remaining = max(0.05, min(0.5, deadline - time.monotonic()))
        self.connection.timeout = remaining
        data = self.connection.readline()
        if not data:
            return ""
        return data.decode("utf-8", errors="replace").strip()


def _parse_status_response(response: str) -> FeederStatus:
    return FeederStatus(
        paper_present=bool(_parse_bool_token(response, "PAPER")),
        motor_running=bool(_parse_bool_token(response, "MOTOR")),
        state=_parse_str_token(response, "STATE", "UNKNOWN"),
        raw_response=response,
    )


def _parse_bool_token(line: str, key: str) -> bool | None:
    value = _parse_str_token(line, key, "")
    if value == "":
        return None
    return value not in {"0", "false", "FALSE", "NO", "no"}


def _parse_int_token(line: str, key: str, fallback: int = 0) -> int:
    value = _parse_str_token(line, key, "")
    try:
        return int(value)
    except ValueError:
        return fallback


def _parse_str_token(line: str, key: str, fallback: str = "") -> str:
    prefix = f"{key}="
    for token in line.split():
        if token.startswith(prefix):
            return token[len(prefix) :]
    return fallback
