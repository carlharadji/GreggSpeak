from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CROP_POINTS_FILE = PROJECT_ROOT / "calibration_crop_points.json"
POINT_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")
DEFAULT_CROP_CONFIG = {
    "image_width": 1232,
    "image_height": 1640,
    "corners": {
        "top_left": [154, 111],
        "top_right": [1063, 119],
        "bottom_right": [1055, 1311],
        "bottom_left": [154, 1308],
    },
}


def crop_document_from_camera(image: Image.Image) -> Image.Image:
    """Apply the fixed 4-point camera crop used by the segmentation GUI test."""
    if image is None:
        return image

    image = ImageOps.exif_transpose(image)
    rgb = np.array(image.convert("RGB"))
    points = _read_fixed_crop_points(rgb.shape[1], rgb.shape[0])
    warped = _perspective_crop_rgb(rgb, points)
    if warped is None:
        return image

    return Image.fromarray(warped).convert("RGB")


def _read_fixed_crop_points(target_width: int, target_height: int) -> np.ndarray:
    config = DEFAULT_CROP_CONFIG
    if CROP_POINTS_FILE.exists():
        try:
            config = json.loads(CROP_POINTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            config = DEFAULT_CROP_CONFIG

    source_width = int(config["image_width"])
    source_height = int(config["image_height"])
    corners = config["corners"]
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    points = []
    for key in POINT_KEYS:
        x, y = corners[key]
        points.append((float(x) * scale_x, float(y) * scale_y))
    return np.float32(points)


def _perspective_crop_rgb(rgb: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    source = np.float32(points)
    width_top = np.linalg.norm(source[1] - source[0])
    width_bottom = np.linalg.norm(source[2] - source[3])
    height_right = np.linalg.norm(source[2] - source[1])
    height_left = np.linalg.norm(source[3] - source[0])

    output_width = max(1, int(round(max(width_top, width_bottom))))
    output_height = max(1, int(round(max(height_right, height_left))))
    if output_width < 80 or output_height < 80:
        return None

    destination = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(rgb, matrix, (output_width, output_height))


def _find_document_corners(rgb: np.ndarray) -> np.ndarray | None:
    height, width = rgb.shape[:2]
    if height < 80 or width < 80:
        return None

    bright_page_points = _find_bright_page_corners(rgb)
    if bright_page_points is not None:
        return bright_page_points

    scale = min(1.0, 1200.0 / float(max(width, height)))
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    resized_h, resized_w = resized.shape[:2]

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 145)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(resized_w * resized_h)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.16:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.018, 0.025, 0.035, 0.05):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype(np.float32)
                points /= scale
                if _valid_document_shape(points, width, height):
                    return points

    return None


def _find_bright_page_corners(rgb: np.ndarray) -> np.ndarray | None:
    height, width = rgb.shape[:2]
    scale = min(1.0, 1200.0 / float(max(width, height)))
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    resized_h, resized_w = resized.shape[:2]

    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 105), (179, 105, 255))
    kernel_size = max(9, int(max(resized_h, resized_w) * 0.018)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(resized_w * resized_h)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18:
            continue

        perimeter = cv2.arcLength(contour, True)
        points = None
        for epsilon_ratio in (0.018, 0.025, 0.035, 0.05, 0.08):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype(np.float32)
                break

        if points is None:
            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect).astype(np.float32)

        points /= scale
        if _valid_document_shape(points, width, height):
            return points

    return None


def _valid_document_shape(points: np.ndarray, width: int, height: int) -> bool:
    ordered = _order_points(points)
    tl, tr, br, bl = ordered
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)

    doc_width = max(top_width, bottom_width)
    doc_height = max(left_height, right_height)
    if doc_width < width * 0.25 or doc_height < height * 0.25:
        return False

    area = cv2.contourArea(ordered.astype(np.float32))
    return area >= float(width * height) * 0.12


def _warp_document(rgb: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    ordered = _order_points(points)
    tl, tr, br, bl = ordered

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 80 or max_height < 80:
        return None

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    return cv2.warpPerspective(rgb, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered
