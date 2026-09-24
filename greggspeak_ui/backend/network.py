from __future__ import annotations

import socket


def get_lan_ips() -> list[str]:
    ips: set[str] = set()

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

    return sorted(ips)


def get_access_urls(port: int) -> list[str]:
    urls = [f"http://{ip}:{port}" for ip in get_lan_ips()]
    urls.append(f"http://localhost:{port}")
    return _dedupe(urls)


def get_primary_access_url(port: int) -> str:
    urls = get_access_urls(port)
    return urls[0] if urls else f"http://localhost:{port}"


def print_access_urls(port: int) -> None:
    print("\nGreggSpeak web app is available at:")
    for url in get_access_urls(port):
        print(f"  {url}")
    print()


def _detect_source_ip(target: str) -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.25)
            sock.connect((target, 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _is_usable_lan_ip(ip: str) -> bool:
    return not (
        ip.startswith("127.")
        or ip.startswith("169.254.")
        or ip == "0.0.0.0"
    )


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
