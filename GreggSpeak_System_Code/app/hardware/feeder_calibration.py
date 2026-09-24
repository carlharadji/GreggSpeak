from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

from app.config import DATA_DIR


CALIBRATION_PATH = DATA_DIR / "feeder_calibration.json"

MIN_DURATION_MS = 500
MAX_DURATION_MS = 12000
MIN_SPEED = 80
MAX_SPEED = 220
DEFAULT_SPEED = 150
DEFAULT_SETTLE_MS = 2000

PAPER_SIZES = {
    "LONG": {"label": "Long", "default_duration_ms": 12000, "next_duration_ms": 10000, "page_increment_ms": 200},
    "SHORT": {"label": "Short", "default_duration_ms": 9800, "next_duration_ms": 5800, "page_decrement_ms": 50},
    "A4": {"label": "A4", "default_duration_ms": 10100, "next_duration_ms": 7000},
}


@dataclass
class FeederCalibration:
    paper_size: str
    duration_ms: int
    speed: int = DEFAULT_SPEED
    settle_ms: int = DEFAULT_SETTLE_MS
    saved: bool = False
    updated_at: str = ""


def normalize_paper_size(paper_size: str) -> str:
    normalized = str(paper_size or "").strip().upper()
    if normalized not in PAPER_SIZES:
        raise ValueError(f"Unsupported paper size: {paper_size}")
    return normalized


def paper_size_label(paper_size: str) -> str:
    return PAPER_SIZES[normalize_paper_size(paper_size)]["label"]


def feed_duration_for_page(paper_size: str, page_number: int) -> int:
    normalized = normalize_paper_size(paper_size)
    meta = PAPER_SIZES[normalized]
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        page = 1
    if page <= 1:
        return int(meta["default_duration_ms"])

    duration = int(meta.get("next_duration_ms", meta["default_duration_ms"]))
    increment = int(meta.get("page_increment_ms", 0))
    decrement = int(meta.get("page_decrement_ms", 0))
    if increment > 0:
        duration += (page - 2) * increment
    if decrement > 0:
        duration -= (page - 2) * decrement
    return max(MIN_DURATION_MS, min(MAX_DURATION_MS, duration))


def default_calibration(paper_size: str) -> FeederCalibration:
    normalized = normalize_paper_size(paper_size)
    return FeederCalibration(
        paper_size=normalized,
        duration_ms=PAPER_SIZES[normalized]["default_duration_ms"],
        speed=DEFAULT_SPEED,
        settle_ms=DEFAULT_SETTLE_MS,
        saved=False,
    )


def load_calibrations(path: Path = CALIBRATION_PATH) -> dict[str, FeederCalibration]:
    raw = _load_raw(path)
    calibrations: dict[str, FeederCalibration] = {}
    for key, item in raw.items():
        try:
            paper_size = normalize_paper_size(key)
        except ValueError:
            continue
        if not isinstance(item, dict):
            continue
        calibrations[paper_size] = FeederCalibration(
            paper_size=paper_size,
            duration_ms=_clamp_int(
                item.get("duration_ms"),
                MIN_DURATION_MS,
                MAX_DURATION_MS,
                PAPER_SIZES[paper_size]["default_duration_ms"],
            ),
            speed=_clamp_int(item.get("speed"), MIN_SPEED, MAX_SPEED, DEFAULT_SPEED),
            settle_ms=_clamp_int(item.get("settle_ms"), 100, 3000, DEFAULT_SETTLE_MS),
            saved=True,
            updated_at=str(item.get("updated_at") or ""),
        )
    return calibrations


def get_saved_calibration(paper_size: str, path: Path = CALIBRATION_PATH) -> FeederCalibration | None:
    return load_calibrations(path).get(normalize_paper_size(paper_size))


def get_calibration_or_default(paper_size: str, path: Path = CALIBRATION_PATH) -> FeederCalibration:
    return get_saved_calibration(paper_size, path) or default_calibration(paper_size)


def save_calibration(
    paper_size: str,
    duration_ms: int,
    speed: int = DEFAULT_SPEED,
    settle_ms: int = DEFAULT_SETTLE_MS,
    path: Path = CALIBRATION_PATH,
) -> FeederCalibration:
    normalized = normalize_paper_size(paper_size)
    calibration = FeederCalibration(
        paper_size=normalized,
        duration_ms=_clamp_int(duration_ms, MIN_DURATION_MS, MAX_DURATION_MS, PAPER_SIZES[normalized]["default_duration_ms"]),
        speed=_clamp_int(speed, MIN_SPEED, MAX_SPEED, DEFAULT_SPEED),
        settle_ms=_clamp_int(settle_ms, 100, 3000, DEFAULT_SETTLE_MS),
        saved=True,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )

    raw = _load_raw(path)
    item = asdict(calibration)
    item.pop("saved", None)
    raw[normalized] = item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return calibration


def _load_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _clamp_int(value, minimum: int, maximum: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    return max(minimum, min(maximum, number))
