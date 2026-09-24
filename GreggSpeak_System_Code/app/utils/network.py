from __future__ import annotations

import os
import socket
import subprocess

from app.config import WEB_PORT

HOTSPOT_IP_PREFIXES = ("10.42.", "192.168.4.")
DEFAULT_FRIENDLY_HOST = "greggspeak.local"


def get_lan_ips() -> list[str]:
    ips: set[str] = set()

    for ip in _hostname_command_ips():
        if _is_usable_lan_ip(ip):
            ips.add(ip)

    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = result[4][0]
            if _is_usable_lan_ip(ip):
                ips.add(ip)
    except OSError:
        pass

    for target in ("8.8.8.8", "1.1.1.1"):
        ip = _detect_source_ip(target)
        if ip and _is_usable_lan_ip(ip):
            ips.add(ip)

    return sorted(ips, key=_ip_sort_key)


def get_access_urls(port: int = WEB_PORT) -> list[str]:
    urls = []
    friendly_host = get_friendly_access_host()
    if friendly_host:
        urls.append(f"http://{friendly_host}:{port}")
    urls.extend(f"http://{ip}:{port}" for ip in get_lan_ips())
    urls.append(f"http://localhost:{port}")
    return _dedupe(urls)


def get_primary_access_url(port: int = WEB_PORT) -> str:
    urls = get_access_urls(port)
    return urls[0] if urls else f"http://localhost:{port}"


def print_access_urls(port: int = WEB_PORT) -> None:
    print("\nGreggSpeak web app is available at:")
    for url in get_access_urls(port):
        print(f"  {url}")
    print()


def get_friendly_access_host() -> str:
    configured = os.environ.get("GREGGSPEAK_ACCESS_HOST", "").strip()
    if configured:
        return configured

    hostname = socket.gethostname().strip().lower()
    if hostname and hostname not in {"localhost", "raspberrypi"}:
        return f"{hostname}.local"

    return DEFAULT_FRIENDLY_HOST


def _detect_source_ip(target: str) -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.25)
            sock.connect((target, 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _hostname_command_ips() -> list[str]:
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except Exception:
        return []

    return [
        item.strip()
        for item in result.stdout.split()
        if item.strip() and "." in item
    ]


def _is_usable_lan_ip(ip: str) -> bool:
    return not (
        ip.startswith("127.")
        or ip.startswith("169.254.")
        or ip == "0.0.0.0"
    )


def _ip_sort_key(ip: str) -> tuple[int, str]:
    if ip.startswith(HOTSPOT_IP_PREFIXES):
        return (0, ip)
    return (1, ip)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
