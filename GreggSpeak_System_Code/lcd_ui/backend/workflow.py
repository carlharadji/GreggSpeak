from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

from app.config import DATA_DIR
from app.hardware.feeder_calibration import (
    DEFAULT_SETTLE_MS,
    FeederCalibration,
    default_calibration,
    get_calibration_or_default,
    get_saved_calibration,
    paper_size_label,
    save_calibration,
)
from app.hardware.feeder_serial import FeedResult, FeederSerialClient

try:
    from lcd_ui.backend.hardware import get_camera_index, get_storage_info, probe_camera
except ImportError:
    from backend.hardware import get_camera_index, get_storage_info, probe_camera


STORAGE_WARNING_PERCENT = 10
MIN_SCAN_FREE_GB = 0.1


@dataclass
class SystemStatus:
    camera_ready: bool = False
    camera_label: str = "Checking..."
    feeder_loaded: bool = False
    feeder_label: str = "Checking..."
    feeder_pages_loaded: int = 0
    storage_free_percent: int = 0
    storage_free_gb: float = 0.0
    storage_volume_label: str = ""


@dataclass
class BatchSession:
    active: bool = False
    pages_captured: int = 0
    state: str = "IDLE"
    paper_size: str = ""
    capture_dir: Path | None = None


class WorkflowService:
    def __init__(self, storage_path=None, camera_index=None):
        self.storage_path = Path(storage_path or Path.cwd())
        self.camera_index = get_camera_index() if camera_index is None else camera_index
        self.simulated_batch_pages = 3
        self.feeder_client: FeederSerialClient | None = None
        self.last_feeder_error = ""
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
        self.refresh_feeder_status()
        return {
            "camera": {
                "label": self.system_status.camera_label,
                "ok": self.system_status.camera_ready,
            },
            "feeder": {
                "label": self.system_status.feeder_label,
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

        if self.system_status.storage_free_gb < MIN_SCAN_FREE_GB:
            return False, "Storage is almost full. Free space before starting a scan batch."

        self.batch_session.active = True
        self.batch_session.pages_captured = 0
        self.batch_session.state = "SELECT_PAPER_SIZE"
        self.batch_session.paper_size = ""
        self.batch_session.capture_dir = None
        if self.system_status.storage_free_percent < STORAGE_WARNING_PERCENT:
            return True, "Storage is low, but scanning can continue."
        return True, "Choose a paper size to start batch scanning."

    def refresh_feeder_status(self):
        try:
            status = self.get_feeder_client().status()
        except Exception as exc:
            self.system_status.feeder_loaded = False
            self.system_status.feeder_label = "ESP32 unavailable"
            self.last_feeder_error = str(exc)
            return False

        self.last_feeder_error = ""
        self.system_status.feeder_loaded = status.paper_present
        if status.motor_running:
            self.system_status.feeder_label = "Motor running"
        elif status.paper_present:
            self.system_status.feeder_label = "Paper loaded"
        else:
            self.system_status.feeder_label = "No paper"
        return status.paper_present

    def get_feeder_client(self):
        if self.feeder_client is None:
            self.feeder_client = FeederSerialClient()
        return self.feeder_client

    def close_feeder(self):
        if self.feeder_client is not None:
            self.feeder_client.close()
            self.feeder_client = None

    def select_paper_size(self, paper_size):
        calibration = default_calibration(paper_size)
        self.batch_session.paper_size = calibration.paper_size
        self.batch_session.state = "CHECK_TRAY"
        return calibration

    def get_saved_feeder_calibration(self, paper_size) -> FeederCalibration | None:
        return get_saved_calibration(paper_size)

    def get_feeder_calibration_or_default(self, paper_size) -> FeederCalibration:
        return get_calibration_or_default(paper_size)

    def get_default_feeder_calibration(self, paper_size) -> FeederCalibration:
        return default_calibration(paper_size)

    def save_feeder_calibration(self, paper_size, duration_ms, speed, settle_ms=DEFAULT_SETTLE_MS):
        calibration = save_calibration(paper_size, duration_ms, speed, settle_ms)
        self.batch_session.paper_size = calibration.paper_size
        self.batch_session.state = "READY_TO_SCAN"
        return calibration

    def check_tray_paper(self):
        try:
            paper_present = self.get_feeder_client().sensor_status()
        except Exception as exc:
            self.system_status.feeder_loaded = False
            self.system_status.feeder_label = "ESP32 unavailable"
            self.last_feeder_error = str(exc)
            return False, f"ESP32 feeder unavailable: {exc}"

        self.last_feeder_error = ""
        self.system_status.feeder_loaded = paper_present
        self.system_status.feeder_label = "Paper loaded" if paper_present else "No paper"
        if paper_present:
            return True, "Paper detected in the feeder."
        return False, "No papers loaded."

    def start_capture_session(self, paper_size):
        label = paper_size_label(paper_size).lower()
        session_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        capture_dir = DATA_DIR / "page_captures" / f"{label}_{session_id}"
        capture_dir.mkdir(parents=True, exist_ok=True)
        self.batch_session.capture_dir = capture_dir
        self.batch_session.paper_size = paper_size
        self.batch_session.state = "READY_TO_SCAN"
        return capture_dir

    def feed_one_page(self, duration_ms, speed, job_id=None) -> FeedResult:
        self.batch_session.state = "FEEDING"
        result = self.get_feeder_client().feed(duration_ms, speed, job_id=job_id)
        self.batch_session.state = "SETTLING" if result.success else "ERROR"
        return result

    def stop_feeder(self):
        try:
            response = self.get_feeder_client().stop()
        except Exception as exc:
            self.last_feeder_error = str(exc)
            return False, str(exc)
        self.batch_session.state = "IDLE" if not self.batch_session.active else "CANCELLED"
        return True, response

    def consume_feeder_page(self):
        if self.system_status.feeder_pages_loaded <= 0:
            return False
        self.system_status.feeder_pages_loaded -= 1
        return True

    def finish_batch(self, pages_captured):
        self.batch_session.active = False
        self.batch_session.pages_captured = pages_captured
        self.batch_session.state = "COMPLETE"
        self.system_status.feeder_pages_loaded = 0
