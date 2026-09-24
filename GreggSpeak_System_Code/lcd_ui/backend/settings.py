import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import DATA_DIR


SETTINGS_PATH = DATA_DIR / "lcd_settings.json"


@dataclass
class DeviceSettings:
    voice_type: str = "Female"
    speech_speed: float = 1.0
    volume: int = 70
    save_destination: str = "MicroSD Card"
    auto_upload: bool = False


class SettingsService:
    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path
        self.settings = self.load()

    def load(self):
        if not self.path.exists():
            return DeviceSettings()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return DeviceSettings()

        defaults = asdict(DeviceSettings())
        defaults.update({key: value for key, value in data.items() if key in defaults})
        return DeviceSettings(**defaults)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(self.settings), indent=2),
            encoding="utf-8",
        )

    def update(self, **values):
        for key, value in values.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.save()
        return self.settings
