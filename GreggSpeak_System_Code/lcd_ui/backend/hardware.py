from pathlib import Path
import os
import shutil


DEFAULT_CAMERA_INDEX = 0


def get_camera_index(default=DEFAULT_CAMERA_INDEX):
    raw_value = os.environ.get("GREGGSPEAK_CAMERA_INDEX")
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_storage_free_percent(path=None):
    return get_storage_info(path)["free_percent"]


def get_storage_info(path=None):
    target = Path(path or Path.cwd()).resolve()
    usage = shutil.disk_usage(target)
    free_percent = 0
    if usage.total > 0:
        free_percent = int((usage.free / usage.total) * 100)

    volume = target.anchor.rstrip("\\/") or target.anchor or str(target)
    return {
        "path": str(target),
        "volume": volume,
        "free_percent": free_percent,
        "free_gb": usage.free / (1024**3),
    }


def probe_camera(index=0):
    picamera_ready = _probe_picamera2()
    if picamera_ready:
        return True, "Ready"

    try:
        import cv2  # type: ignore
    except Exception:
        return False, "Unavailable"

    camera = None
    try:
        camera = cv2.VideoCapture(index)
        ready = bool(camera is not None and camera.isOpened())
        return ready, f"Ready (Camera {index})" if ready else f"Offline (Camera {index})"
    except Exception:
        return False, "Offline"
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass


def _probe_picamera2():
    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception:
        return False

    camera = None
    try:
        camera = Picamera2()
        return True
    except Exception:
        return False
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
