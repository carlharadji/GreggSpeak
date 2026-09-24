from pathlib import Path

import cv2
from PIL import Image

from app.config import (
    CAMERA_DOCUMENT_CROP,
    CAMERA_INDEX,
    CAMERA_ROTATE_COUNTERCLOCKWISE,
    INPUT_DIR,
    SUPPORTED_IMAGE_EXTS,
)
from app.preprocessing.document_crop import crop_document_from_camera


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_input_images(input_dir: Path = INPUT_DIR) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


def capture_camera_image(camera_index: int = CAMERA_INDEX) -> Image.Image | None:
    picamera_image = _capture_with_picamera2()
    if picamera_image is not None:
        return prepare_camera_image(picamera_image)

    camera = None
    try:
        camera = cv2.VideoCapture(camera_index)
        if camera is None or not camera.isOpened():
            return None
        ok, frame = camera.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return prepare_camera_image(Image.fromarray(rgb))
    except Exception:
        return None
    finally:
        if camera is not None:
            camera.release()


def _capture_with_picamera2() -> Image.Image | None:
    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception:
        return None

    camera = None
    try:
        camera = Picamera2()
        camera.configure(camera.create_still_configuration(main={"size": (1640, 1232)}))
        camera.start()
        frame = camera.capture_array()
        if frame is None:
            return None
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            rgb = frame[:, :, :3]
            return Image.fromarray(rgb).convert("RGB")
        return Image.fromarray(frame).convert("RGB")
    except Exception:
        return None
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass


def orient_camera_image(image: Image.Image) -> Image.Image:
    if not CAMERA_ROTATE_COUNTERCLOCKWISE:
        return image

    try:
        return image.transpose(Image.Transpose.ROTATE_90)
    except AttributeError:
        return image.transpose(Image.ROTATE_90)


def prepare_camera_image(image: Image.Image) -> Image.Image:
    image = orient_camera_image(image)
    if not CAMERA_DOCUMENT_CROP:
        return image
    try:
        return crop_document_from_camera(image)
    except Exception:
        return image
