from dataclasses import dataclass
from pathlib import Path

from backend.hardware import get_camera_index, get_storage_info, probe_camera


STORAGE_WARNING_PERCENT = 10
MIN_SCAN_FREE_GB = 0.1


@dataclass
class SystemStatus:
    camera_ready: bool = False
    camera_label: str = "Checking..."
    feeder_loaded: bool = True
    feeder_pages_loaded: int = 0
    storage_free_percent: int = 0
    storage_free_gb: float = 0.0
    storage_volume_label: str = ""


@dataclass
class BatchSession:
    active: bool = False
    pages_captured: int = 0


class WorkflowService:
    def __init__(self, storage_path=None, camera_index=None):
        self.storage_path = Path(storage_path or Path.cwd())
        self.camera_index = get_camera_index() if camera_index is None else camera_index
        self.simulated_batch_pages = 3
        self.system_status = SystemStatus()
        self.batch_session = BatchSession()
        self.refresh_system_status(check_camera=False)

    def refresh_system_status(self, check_camera=True):
        storage_info = get_storage_info(self.storage_path)
        if check_camera:
            camera_ready, camera_label = probe_camera(self.camera_index)
            self.system_status.camera_ready = camera_ready
            self.system_status.camera_label = camera_label
        else:
            self.system_status.camera_ready = True
            self.system_status.camera_label = f"Selected (Camera {self.camera_index})"
        self.system_status.storage_free_percent = storage_info["free_percent"]
        self.system_status.storage_free_gb = storage_info["free_gb"]
        self.system_status.storage_volume_label = storage_info["volume"]

    def get_home_status(self):
        self.refresh_system_status(check_camera=False)
        return {
            "camera": {
                "label": self.system_status.camera_label,
                "ok": self.system_status.camera_ready,
            },
            "feeder": {
                "label": "Ready" if self.system_status.feeder_loaded else "Unavailable",
                "ok": self.system_status.feeder_loaded,
            },
            "storage": {
                "label": (
                    f"{self.system_status.storage_volume_label} "
                    f"{self.system_status.storage_free_percent}% Free "
                    f"({self.system_status.storage_free_gb:.1f} GB)"
                ),
                "ok": self.system_status.storage_free_percent >= STORAGE_WARNING_PERCENT,
            },
        }

    def start_new_batch(self):
        self.refresh_system_status(check_camera=True)

        if not self.system_status.camera_ready:
            return False, "Camera is not ready."

        if not self.system_status.feeder_loaded:
            return False, "Feeder is not ready."

        if self.system_status.storage_free_gb < MIN_SCAN_FREE_GB:
            return False, "Storage is almost full. Free space before starting a scan batch."

        self.batch_session.active = True
        self.batch_session.pages_captured = 0
        self.system_status.feeder_pages_loaded = self.simulated_batch_pages
        if self.system_status.storage_free_percent < STORAGE_WARNING_PERCENT:
            return True, "Storage is low, but scanning can continue."
        return True, "Scanning workspace ready."

    def consume_feeder_page(self):
        if self.system_status.feeder_pages_loaded <= 0:
            return False
        self.system_status.feeder_pages_loaded -= 1
        return True

    def finish_batch(self, pages_captured):
        self.batch_session.active = False
        self.batch_session.pages_captured = pages_captured
        self.system_status.feeder_pages_loaded = 0
