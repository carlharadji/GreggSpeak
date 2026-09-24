from __future__ import annotations

import csv
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CROP_POINTS_FILE = PROJECT_ROOT / "calibration_crop_points.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.segmentation.line_segmentation import resize_word_crop_for_model

IMAGE_TYPES = (
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("JPEG", "*.jpg *.jpeg"),
    ("PNG", "*.png"),
    ("All files", "*.*"),
)

POINT_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")

MODEL_V52_PATH = PROJECT_ROOT / "models" / "greggspeak_all_words_v5_2.keras"
LABELS_V52_PATH = PROJECT_ROOT / "models" / "labels_all_words_v5_2.txt"
MODEL_V53_PATH = PROJECT_ROOT / "models" / "greggspeak_all_words_v5_3.keras"
BEST_MODEL_V53_PATH = PROJECT_ROOT / "models" / "best_all_words_v5_3.keras"
LABELS_V53_PATH = PROJECT_ROOT / "models" / "labels_all_words_v5_3.txt"

CLEAN_ILLUMINATION_SIGMA_FRACTION = 0.045
CLEAN_ILLUMINATION_MIN_SIGMA = 45
CLEAN_ILLUMINATION_BLEND = 0.52
CLEAN_DENOISE_MEDIAN_SIZE = 0
CLEAN_SHARPEN_AMOUNT = 0.55
CLEAN_SHARPEN_SIGMA = 1.0
CLEAN_CONTRAST_LOW_PERCENTILE = 1.0
CLEAN_CONTRAST_HIGH_PERCENTILE = 99.0
CLEAN_CONTRAST_BLEND = 0.30
RED_INK_LAB_A_WEIGHT = 0.58

RECOGNITION_BACKGROUND_TARGET = 205
RECOGNITION_ILLUMINATION_SIGMA_FRACTION = 0.05
RECOGNITION_ILLUMINATION_MIN_SIGMA = 55
RECOGNITION_ILLUMINATION_BLEND = 0.30
RECOGNITION_CONTRAST_LOW_PERCENTILE = 1.0
RECOGNITION_CONTRAST_HIGH_PERCENTILE = 99.2
RECOGNITION_CONTRAST_BLEND = 0.16
RECOGNITION_SHARPEN_AMOUNT = 0.32
RECOGNITION_SHARPEN_SIGMA = 1.0

# This thresholded image is a segmentation mask only. Recognition crops use the
# recognition-ready page so model input keeps the natural stroke width.
BINARY_THRESHOLD_METHOD = "conservative_otsu"  # "conservative_otsu", "otsu", or "adaptive"
BINARY_OTSU_BLUR_SIZE = 1
# Keep the threshold strict enough that tinted paper/shadows do not become ink.
BINARY_CONSERVATIVE_STD_FACTOR = 2.5
BINARY_CONSERVATIVE_LOW_PERCENTILE = 4.0
BINARY_CONSERVATIVE_PERCENTILE_MARGIN = 0.0
BINARY_MIN_THRESHOLD = 105
BINARY_MAX_THRESHOLD = 195
BINARY_ADAPTIVE_BLOCK_FRACTION = 0.045
BINARY_ADAPTIVE_C = 9
BINARY_MIN_SPECK_AREA = 2
BINARY_PRESERVE_SMALL_COMPONENT_AREA = 8
BINARY_BORDER_ARTIFACT_MARGIN_FRACTION = 0.035
BINARY_BORDER_ARTIFACT_MAX_AREA = 220
BINARY_TOP_EDGE_ARTIFACT_FRACTION = 0.14
BINARY_SIDE_EDGE_ARTIFACT_FRACTION = 0.18
BINARY_TOP_RIGHT_ARTIFACT_X_FRACTION = 0.55

ORIENTATION_ENABLE_AUTO_ROTATE = False
ORIENTATION_ROTATE_TALL_TRAY_CROPS_CLOCKWISE = False
ORIENTATION_SCORE_CLOSE_MARGIN = 12.0
ORIENTATION_SIDEWAYS_TOUCH_FRACTION = 0.30
ORIENTATION_SIDEWAYS_TALL_FRACTION = 0.040

LINE_NOISE_MIN_AREA = 4
LINE_PROFILE_SMOOTH_FRACTION = 0.006
LINE_MERGE_GAP_FRACTION = 0.006
LINE_MIN_HEIGHT_FRACTION = 0.006
LINE_MIN_WIDTH_FRACTION = 0.025
LINE_HORIZONTAL_PAD_FRACTION = 0.009
LINE_VERTICAL_PAD_FRACTION = 0.005
LINE_PEAK_SMOOTH_FRACTION = 0.004
LINE_PEAK_MIN_DISTANCE_FRACTION = 0.03
LINE_PEAK_VALLEY_RATIO = 0.78
LINE_PEAK_EDGE_RATIO = 0.18
LINE_MERGED_HEIGHT_MULTIPLIER = 1.55
LINE_DEBUG_OUTPUT = False
LINE_DEBUG_DIR = PROJECT_ROOT / "reports" / "line_segmentation_debug"

WORD_NOISE_MIN_AREA = 4
WORD_DILATE_WIDTH_LINE_HEIGHT_FRACTION = 0.34
WORD_DILATE_WIDTH_LINE_WIDTH_FRACTION = 0.03
WORD_MIN_WIDTH = 4
WORD_MIN_HEIGHT = 4
WORD_HORIZONTAL_PAD_FRACTION = 0.14
WORD_VERTICAL_PAD_FRACTION = 0.08

LAST_LINE_SEGMENTATION_DEBUG: dict[str, object] = {}
MODEL_CACHE: dict[str, tuple[object, list[str]]] = {}
TF_MODULE: object | None = None


@dataclass
class LineBox:
    index: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class WordBox:
    line_index: int
    word_index: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class WordPrediction:
    line_index: int
    word_index: int
    label: str
    confidence: float


@dataclass
class RecognitionResult:
    model_name: str
    transcript: str
    predictions: list[WordPrediction]


@dataclass
class PageState:
    path: Path
    original_bgr: np.ndarray
    cropped_bgr: np.ndarray | None = None
    enhanced_gray: np.ndarray | None = None
    recognition_ready_image: np.ndarray | None = None
    # White-background ink mask used for line/word segmentation, not recognition.
    segmentation_mask: np.ndarray | None = None
    line_boxes: list[LineBox] | None = None
    line_segmentation_bgr: np.ndarray | None = None
    word_boxes_by_line: list[list[WordBox]] | None = None
    word_segmentation_bgr: np.ndarray | None = None
    recognition_results: dict[str, RecognitionResult] = field(default_factory=dict)
    recognition_overlay_bgr: np.ndarray | None = None
    active_recognition_key: str | None = None


def load_bgr_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Could not save image: {path}")
    encoded.tofile(str(path))


def read_crop_points(path: Path, target_width: int, target_height: int) -> list[tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source_width = int(data["image_width"])
    source_height = int(data["image_height"])
    corners = data["corners"]

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    points: list[tuple[int, int]] = []
    for key in POINT_KEYS:
        x, y = corners[key]
        points.append((int(round(x * scale_x)), int(round(y * scale_y))))
    return points


def perspective_crop(image_bgr: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    source = np.float32(points)
    width_top = np.linalg.norm(source[1] - source[0])
    width_bottom = np.linalg.norm(source[2] - source[3])
    height_right = np.linalg.norm(source[2] - source[1])
    height_left = np.linalg.norm(source[3] - source[0])

    output_width = max(1, int(round(max(width_top, width_bottom))))
    output_height = max(1, int(round(max(height_right, height_left))))
    destination = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image_bgr, matrix, (output_width, output_height))


def normalize_text_orientation(cropped_bgr: np.ndarray) -> np.ndarray:
    if not ORIENTATION_ENABLE_AUTO_ROTATE:
        return cropped_bgr
    if ORIENTATION_ROTATE_TALL_TRAY_CROPS_CLOCKWISE and cropped_bgr.shape[0] > cropped_bgr.shape[1] * 1.12:
        return cv2.rotate(cropped_bgr, cv2.ROTATE_90_CLOCKWISE)

    candidates = [
        ("original", cropped_bgr),
        ("clockwise", cv2.rotate(cropped_bgr, cv2.ROTATE_90_CLOCKWISE)),
        ("counterclockwise", cv2.rotate(cropped_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    scored = [(name, image, *orientation_metrics(image)) for name, image in candidates]
    original_score = scored[0][2]
    sideways = scored[0][3]
    rotated = scored[1:]
    best_rotated = max(rotated, key=lambda item: item[2])

    # When both 90-degree options look similar, prefer clockwise for these
    # tray captures; it keeps the shorthand rows in the expected reading order.
    clockwise = scored[1]
    counterclockwise = scored[2]
    if abs(clockwise[2] - counterclockwise[2]) <= ORIENTATION_SCORE_CLOSE_MARGIN:
        best_rotated = clockwise
    if sideways and counterclockwise[2] <= clockwise[2] + (ORIENTATION_SCORE_CLOSE_MARGIN * 2.5):
        best_rotated = clockwise
    if sideways:
        return best_rotated[1]

    if not sideways and original_score >= best_rotated[2] - ORIENTATION_SCORE_CLOSE_MARGIN:
        return cropped_bgr
    return best_rotated[1] if best_rotated[2] > original_score + 2.0 else cropped_bgr


def orientation_score(image_bgr: np.ndarray) -> float:
    return orientation_metrics(image_bgr)[0]


def orientation_looks_sideways(image_bgr: np.ndarray) -> bool:
    return orientation_metrics(image_bgr)[1]


def orientation_metrics(image_bgr: np.ndarray) -> tuple[float, bool]:
    try:
        enhanced = clean_enhance_image(image_bgr)
        mask = create_segmentation_mask(enhanced)
        lines = segment_lines_from_mask(mask)
    except Exception:
        return -1.0, False

    height, width = mask.shape[:2]
    score = 0.0
    good_lines = 0
    suspicious = 0
    for line in lines:
        aspect = line.w / max(1, line.h)
        touches_left = line.x <= 2
        tall_box = line.h >= height * ORIENTATION_SIDEWAYS_TALL_FRACTION
        wide_enough = line.w >= width * 0.22
        if aspect >= 2.8 and line.w >= width * 0.08:
            good_lines += 1
            score += 3.0 + min(aspect, 12.0) * 0.30 + min(line.w / max(1, width), 1.0) * 2.0
        if line.h > line.w * 0.70:
            score -= 5.0
        if touches_left and tall_box and wide_enough:
            score -= 8.0
            suspicious += 1

    score += good_lines * 1.5
    score -= abs(len(lines) - 14) * 0.15
    sideways = len(lines) >= 3 and suspicious / max(1, len(lines)) >= ORIENTATION_SIDEWAYS_TOUCH_FRACTION
    return score, sideways


def odd_kernel_size(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def percentile_stretch(gray: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low, high = np.percentile(gray, (low_percentile, high_percentile))
    if high <= low:
        return gray.copy()

    stretched = (gray.astype(np.float32) - float(low)) * (255.0 / float(high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def handwriting_luminance(cropped_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2LAB)
    red_channel = lab[:, :, 1].astype(np.float32)
    red_excess = np.maximum(0.0, red_channel - float(np.median(red_channel)))
    adjusted = gray - (red_excess * RED_INK_LAB_A_WEIGHT)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def clean_enhance_image(cropped_bgr: np.ndarray) -> np.ndarray:
    gray = handwriting_luminance(cropped_bgr)
    if CLEAN_DENOISE_MEDIAN_SIZE and CLEAN_DENOISE_MEDIAN_SIZE >= 3:
        gray = cv2.medianBlur(gray, odd_kernel_size(CLEAN_DENOISE_MEDIAN_SIZE, minimum=3))

    min_side = min(gray.shape[:2])
    background_sigma = max(CLEAN_ILLUMINATION_MIN_SIGMA, min_side * CLEAN_ILLUMINATION_SIGMA_FRACTION)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=background_sigma, sigmaY=background_sigma)
    background = np.maximum(background, 1)

    corrected = cv2.divide(gray, background, scale=255)
    corrected = cv2.addWeighted(
        gray,
        1.0 - CLEAN_ILLUMINATION_BLEND,
        corrected,
        CLEAN_ILLUMINATION_BLEND,
        0,
    )

    stretched = percentile_stretch(
        corrected,
        CLEAN_CONTRAST_LOW_PERCENTILE,
        CLEAN_CONTRAST_HIGH_PERCENTILE,
    )
    contrast_balanced = cv2.addWeighted(
        corrected,
        1.0 - CLEAN_CONTRAST_BLEND,
        stretched,
        CLEAN_CONTRAST_BLEND,
        0,
    )

    soft_blur = cv2.GaussianBlur(contrast_balanced, (0, 0), sigmaX=CLEAN_SHARPEN_SIGMA)
    sharpened = cv2.addWeighted(
        contrast_balanced,
        1.0 + CLEAN_SHARPEN_AMOUNT,
        soft_blur,
        -CLEAN_SHARPEN_AMOUNT,
        0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def prepare_recognition_ready_image(cropped_bgr: np.ndarray) -> np.ndarray:
    gray = handwriting_luminance(cropped_bgr)
    min_side = min(gray.shape[:2])
    background_sigma = max(RECOGNITION_ILLUMINATION_MIN_SIGMA, min_side * RECOGNITION_ILLUMINATION_SIGMA_FRACTION)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=background_sigma, sigmaY=background_sigma)
    background = np.maximum(background, 1)

    corrected = cv2.divide(gray, background, scale=RECOGNITION_BACKGROUND_TARGET)
    corrected = cv2.addWeighted(
        gray,
        1.0 - RECOGNITION_ILLUMINATION_BLEND,
        corrected,
        RECOGNITION_ILLUMINATION_BLEND,
        0,
    )

    stretched = percentile_stretch(
        corrected,
        RECOGNITION_CONTRAST_LOW_PERCENTILE,
        RECOGNITION_CONTRAST_HIGH_PERCENTILE,
    )
    balanced = cv2.addWeighted(
        corrected,
        1.0 - RECOGNITION_CONTRAST_BLEND,
        stretched,
        RECOGNITION_CONTRAST_BLEND,
        0,
    )

    soft_blur = cv2.GaussianBlur(balanced, (0, 0), sigmaX=RECOGNITION_SHARPEN_SIGMA)
    sharpened = cv2.addWeighted(
        balanced,
        1.0 + RECOGNITION_SHARPEN_AMOUNT,
        soft_blur,
        -RECOGNITION_SHARPEN_AMOUNT,
        0,
    )
    return as_bgr(np.clip(sharpened, 0, 255).astype(np.uint8))


def create_segmentation_mask(enhanced_gray: np.ndarray) -> np.ndarray:
    if BINARY_OTSU_BLUR_SIZE and BINARY_OTSU_BLUR_SIZE >= 3:
        prepared = cv2.GaussianBlur(enhanced_gray, (odd_kernel_size(BINARY_OTSU_BLUR_SIZE),) * 2, 0)
    else:
        prepared = enhanced_gray.copy()

    if BINARY_THRESHOLD_METHOD == "adaptive":
        block_size = odd_kernel_size(int(min(prepared.shape[:2]) * BINARY_ADAPTIVE_BLOCK_FRACTION), minimum=31)
        ink_mask = cv2.adaptiveThreshold(
            prepared,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            BINARY_ADAPTIVE_C,
        )
    elif BINARY_THRESHOLD_METHOD == "conservative_otsu":
        otsu_value, _otsu_mask = cv2.threshold(
            prepared,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        median = float(np.median(prepared))
        std = float(np.std(prepared))
        percentile_cap = float(np.percentile(prepared, BINARY_CONSERVATIVE_LOW_PERCENTILE))
        conservative_value = min(
            float(otsu_value),
            median - (std * BINARY_CONSERVATIVE_STD_FACTOR),
            percentile_cap + BINARY_CONSERVATIVE_PERCENTILE_MARGIN,
        )
        threshold_value = int(np.clip(conservative_value, BINARY_MIN_THRESHOLD, BINARY_MAX_THRESHOLD))
        _threshold_value, ink_mask = cv2.threshold(
            prepared,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )
    else:
        _threshold_value, ink_mask = cv2.threshold(
            prepared,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    ink_mask = remove_tiny_specks(
        ink_mask,
        min_area=BINARY_MIN_SPECK_AREA,
        preserve_small_area=BINARY_PRESERVE_SMALL_COMPONENT_AREA,
    )
    ink_mask = remove_border_artifacts(
        ink_mask,
        margin_fraction=BINARY_BORDER_ARTIFACT_MARGIN_FRACTION,
        max_area=BINARY_BORDER_ARTIFACT_MAX_AREA,
    )
    ink_mask = remove_top_edge_artifacts(
        ink_mask,
        top_fraction=BINARY_TOP_EDGE_ARTIFACT_FRACTION,
        side_fraction=BINARY_SIDE_EDGE_ARTIFACT_FRACTION,
        right_x_fraction=BINARY_TOP_RIGHT_ARTIFACT_X_FRACTION,
    )
    ink_mask = suppress_sparse_background_rows(ink_mask)
    return 255 - ink_mask


def binarize_image(enhanced_gray: np.ndarray) -> np.ndarray:
    """Backward-compatible name for older scripts; returns a segmentation mask."""
    return create_segmentation_mask(enhanced_gray)


def remove_border_artifacts(
    ink_mask: np.ndarray,
    margin_fraction: float,
    max_area: int,
) -> np.ndarray:
    height, width = ink_mask.shape[:2]
    margin_x = max(2, int(width * margin_fraction))
    margin_y = max(2, int(height * margin_fraction))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    filtered = ink_mask.copy()

    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        touches_border_zone = (
            x <= margin_x
            or y <= margin_y
            or x + w >= width - margin_x
            or y + h >= height - margin_y
        )
        touches_horizontal_edge = y <= margin_y or y + h >= height - margin_y
        touches_vertical_edge = x <= margin_x or x + w >= width - margin_x
        long_thin_edge = touches_horizontal_edge and w >= width * 0.45 and h <= max(18, int(height * 0.055))
        frame_like_edge = (
            touches_border_zone
            and area >= int(width * height * 0.001)
            and (w >= width * 0.16 or h >= height * 0.16)
        )
        corner_or_edge_blob = (
            touches_border_zone
            and area >= max_area
            and area <= int(width * height * 0.025)
            and (
                (touches_horizontal_edge and (x <= width * 0.18 or x + w >= width * 0.82))
                or (touches_vertical_edge and (y <= height * 0.12 or y + h >= height * 0.88))
            )
        )
        if touches_border_zone and (area <= max_area or long_thin_edge or frame_like_edge or corner_or_edge_blob):
            filtered[labels == label] = 0

    return filtered


def remove_top_edge_artifacts(
    ink_mask: np.ndarray,
    top_fraction: float,
    side_fraction: float,
    right_x_fraction: float,
) -> np.ndarray:
    height, width = ink_mask.shape[:2]
    top_limit = max(12, int(height * top_fraction))
    side_limit = max(12, int(width * side_fraction))
    right_start = max(0, int(width * right_x_fraction))
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    filtered = ink_mask.copy()

    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        centroid_x, centroid_y = centroids[label]
        touches_top = y <= max(2, int(height * BINARY_BORDER_ARTIFACT_MARGIN_FRACTION))
        touches_right_zone = x + w >= width - side_limit
        in_upper_band = y < top_limit and centroid_y < top_limit
        in_upper_right = in_upper_band and (x >= right_start or centroid_x >= right_start or touches_right_zone)

        if in_upper_right:
            filtered[labels == label] = 0
        elif in_upper_band and touches_right_zone and area <= max(BINARY_BORDER_ARTIFACT_MAX_AREA, width // 2):
            filtered[labels == label] = 0

    return filtered


def remove_tiny_specks(
    ink_mask: np.ndarray,
    min_area: int,
    preserve_small_area: int | None = None,
) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    filtered = np.zeros_like(ink_mask)
    preserve_small_area = preserve_small_area if preserve_small_area is not None else min_area
    for label in range(1, component_count):
        _x, _y, w, h, area = stats[label]
        if area < min_area:
            continue
        if area < preserve_small_area and w <= 1 and h <= 1:
            continue
        filtered[labels == label] = 255
    return filtered


def suppress_sparse_background_rows(ink_mask: np.ndarray) -> np.ndarray:
    height = ink_mask.shape[0]
    row_counts = np.count_nonzero(ink_mask, axis=1).astype(np.float32)
    if row_counts.max() <= 0:
        return ink_mask

    smooth_window = odd_kernel_size(int(height * LINE_PROFILE_SMOOTH_FRACTION), minimum=15)
    smoothed = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, smooth_window), 0).reshape(-1)
    positive = smoothed[smoothed > 0]
    if positive.size == 0:
        return ink_mask

    threshold = max(
        2.0,
        float(np.percentile(positive, 75)) * 0.30,
        float(smoothed.max()) * 0.012,
    )
    keep_rows = smoothed >= threshold
    keep_kernel = np.ones(max(9, int(height * 0.008)), dtype=np.uint8).reshape(-1, 1)
    keep_rows = cv2.dilate(keep_rows.astype(np.uint8).reshape(-1, 1), keep_kernel, iterations=1).reshape(-1) > 0

    filtered = ink_mask.copy()
    filtered[~keep_rows, :] = 0
    return filtered


def segment_lines_from_mask(segmentation_mask: np.ndarray) -> list[LineBox]:
    gray = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY) if len(segmentation_mask.shape) == 3 else segmentation_mask
    ink_mask = np.where(gray < 128, 255, 0).astype(np.uint8)
    ink_mask = remove_tiny_specks(ink_mask, min_area=LINE_NOISE_MIN_AREA)

    profile_boxes = line_boxes_from_profile_peaks(ink_mask)
    component_boxes = line_boxes_from_component_centers(ink_mask)

    if profile_boxes and (
        not component_boxes
        or len(component_boxes) <= len(profile_boxes) <= len(component_boxes) + 3
    ):
        return [
            LineBox(index=index, x=x, y=y, w=w, h=h)
            for index, (x, y, w, h) in enumerate(profile_boxes, start=1)
        ]

    if component_boxes:
        return [
            LineBox(index=index, x=x, y=y, w=w, h=h)
            for index, (x, y, w, h) in enumerate(component_boxes, start=1)
        ]

    height, width = ink_mask.shape[:2]
    row_counts = np.count_nonzero(ink_mask, axis=1).astype(np.float32)
    if row_counts.max() <= 0:
        return []

    smooth_window = odd_kernel_size(int(height * LINE_PROFILE_SMOOTH_FRACTION), minimum=15)
    smoothed = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, smooth_window), 0).reshape(-1)
    positive = smoothed[smoothed > 0]
    if positive.size == 0:
        return []

    threshold = max(
        2.0,
        float(np.percentile(positive, 60)) * 0.55,
        float(smoothed.max()) * 0.05,
    )
    raw_bands = active_row_bands(smoothed >= threshold)
    if not raw_bands:
        return []

    merge_gap = max(10, int(height * LINE_MERGE_GAP_FRACTION))
    bands = merge_close_bands(raw_bands, merge_gap=merge_gap)
    bands = split_unusually_tall_bands(bands, smoothed, height)

    min_line_height = max(8, int(height * LINE_MIN_HEIGHT_FRACTION))
    min_line_width = max(35, int(width * LINE_MIN_WIDTH_FRACTION))
    horizontal_pad = max(10, int(width * LINE_HORIZONTAL_PAD_FRACTION))
    vertical_pad = max(6, int(height * LINE_VERTICAL_PAD_FRACTION))

    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in bands:
        if bottom - top + 1 < min_line_height:
            continue

        y1 = max(0, top - vertical_pad)
        y2 = min(height - 1, bottom + vertical_pad)
        band_mask = ink_mask[y1 : y2 + 1, :]
        ys, xs = np.where(band_mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue

        ink_pixels = int(xs.size)
        x1 = max(0, int(xs.min()) - horizontal_pad)
        x2 = min(width - 1, int(xs.max()) + horizontal_pad)
        refined_y1 = max(0, y1 + int(ys.min()) - vertical_pad)
        refined_y2 = min(height - 1, y1 + int(ys.max()) + vertical_pad)
        box_width = x2 - x1 + 1
        box_height = refined_y2 - refined_y1 + 1

        if box_width < min_line_width or box_height < min_line_height:
            continue
        if ink_pixels < max(18, box_width * 0.018):
            continue
        if looks_like_centered_header((x1, refined_y1, box_width, box_height), width, height):
            continue

        boxes.append((x1, refined_y1, box_width, box_height))

    boxes = merge_overlapping_line_boxes(boxes, height)
    return [
        LineBox(index=index, x=x, y=y, w=w, h=h)
        for index, (x, y, w, h) in enumerate(sorted(boxes, key=lambda box: (box[1], box[0])), start=1)
    ]


def line_boxes_from_profile_peaks(ink_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    global LAST_LINE_SEGMENTATION_DEBUG

    height, width = ink_mask.shape[:2]
    row_counts = np.count_nonzero(ink_mask, axis=1).astype(np.float32)
    LAST_LINE_SEGMENTATION_DEBUG = {
        "profile": row_counts.copy(),
        "smoothed": row_counts.copy(),
        "peaks": [],
        "valleys": [],
        "bands": [],
    }
    if row_counts.max() <= 0:
        return []

    smooth_window = odd_kernel_size(int(height * LINE_PEAK_SMOOTH_FRACTION), minimum=9)
    smoothed = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, smooth_window), 0).reshape(-1)
    positive = smoothed[smoothed > 0]
    if positive.size == 0:
        return []

    peak_threshold = max(
        1.2,
        float(np.percentile(positive, 35)) * 0.45,
        float(smoothed.max()) * 0.018,
    )
    min_peak_distance = max(14, int(height * LINE_PEAK_MIN_DISTANCE_FRACTION))
    peaks = find_profile_peaks(smoothed, peak_threshold, min_peak_distance)
    peaks = merge_profile_peaks_by_valleys(smoothed, peaks, min_peak_distance)
    if not peaks:
        return []

    valleys = profile_valleys_between_peaks(smoothed, peaks)
    bands = profile_bands_from_peaks(smoothed, peaks, valleys, peak_threshold)

    horizontal_pad = max(10, int(width * LINE_HORIZONTAL_PAD_FRACTION))
    vertical_pad = max(6, int(height * LINE_VERTICAL_PAD_FRACTION))
    min_line_height = max(8, int(height * LINE_MIN_HEIGHT_FRACTION))
    min_line_width = max(35, int(width * LINE_MIN_WIDTH_FRACTION))

    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in bands:
        if bottom - top + 1 < min_line_height:
            continue

        candidate = line_box_from_row_band(ink_mask, top, bottom, horizontal_pad, vertical_pad)
        if candidate is None:
            continue

        x, y, w, h = candidate
        if w < min_line_width or h < min_line_height:
            continue
        if looks_like_centered_header(candidate, width, height):
            continue

        roi = ink_mask[y : y + h, x : x + w]
        ink_pixels = int(np.count_nonzero(roi))
        density = ink_pixels / max(1, roi.size)
        if ink_pixels < max(18, w * 0.015):
            continue
        if density < 0.0035 and ink_pixels < 900:
            continue
        if w > width * 0.65 and density < 0.008 and ink_pixels < 1300:
            continue

        boxes.append(candidate)

    boxes = trim_line_box_horizontal_outliers(boxes, ink_mask)
    boxes = split_merged_component_line_boxes(boxes, ink_mask)
    boxes = separate_vertical_overlaps(sorted(boxes, key=lambda box: (box[1], box[0])))
    boxes = drop_isolated_top_headers(boxes, ink_mask.shape[:2])
    boxes = drop_edge_line_artifacts(boxes, ink_mask.shape[:2])
    LAST_LINE_SEGMENTATION_DEBUG = {
        "profile": row_counts.copy(),
        "smoothed": smoothed.copy(),
        "peaks": peaks,
        "valleys": valleys,
        "bands": [(int(top), int(bottom)) for top, bottom in bands],
    }
    return boxes


def drop_isolated_top_headers(
    boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 2:
        return boxes

    height, width = image_shape
    filtered = list(boxes)

    while len(filtered) >= 2:
        first = filtered[0]
        second = filtered[1]
        x, y, w, h = first
        first_bottom = y + h - 1
        gap_to_next = second[1] - first_bottom
        center_x = x + w / 2.0

        isolated_top_text = (
            y < height * 0.24
            and width * 0.25 <= center_x <= width * 0.75
            and w < width * 0.58
            and h < height * 0.065
            and gap_to_next > max(34, int(height * 0.025))
        )
        tiny_top_artifact = (
            y < height * 0.16
            and w < width * 0.18
            and h < height * 0.05
            and gap_to_next > max(28, int(height * 0.02))
        )
        if not isolated_top_text and not tiny_top_artifact:
            break
        filtered.pop(0)

    return filtered


def drop_edge_line_artifacts(
    boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    height, width = image_shape
    filtered: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        near_right = x + w >= width * 0.92
        near_left = x <= width * 0.025
        upper_band = y < height * 0.35
        compact = w < width * 0.18 and h < height * 0.065
        tiny = w < width * 0.08 and h < height * 0.045
        full_width_edge = w >= width * 0.85 and h <= height * 0.055 and (y < height * 0.18 or y + h > height * 0.82)

        if h < max(4, int(height * 0.003)):
            continue
        if full_width_edge:
            continue
        if compact and upper_band and near_right:
            continue
        if tiny and (near_left or near_right):
            continue

        filtered.append(box)

    return filtered


def find_profile_peaks(
    smoothed: np.ndarray,
    threshold: float,
    min_distance: int,
) -> list[int]:
    candidates: list[tuple[int, float]] = []
    length = len(smoothed)
    y = 1
    while y < length - 1:
        value = float(smoothed[y])
        if value < threshold or value < float(smoothed[y - 1]) or value < float(smoothed[y + 1]):
            y += 1
            continue

        start = y
        end = y
        while end + 1 < length and abs(float(smoothed[end + 1]) - value) < 1e-4:
            end += 1

        peak_y = (start + end) // 2
        candidates.append((peak_y, float(smoothed[peak_y])))
        y = end + 1

    selected: list[int] = []
    for peak_y, _value in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(peak_y - existing) >= min_distance for existing in selected):
            selected.append(peak_y)

    return sorted(selected)


def merge_profile_peaks_by_valleys(
    smoothed: np.ndarray,
    peaks: list[int],
    min_distance: int,
) -> list[int]:
    if len(peaks) < 2:
        return peaks

    merged = list(peaks)
    changed = True
    while changed and len(merged) >= 2:
        changed = False
        result: list[int] = []
        index = 0
        while index < len(merged):
            if index == len(merged) - 1:
                result.append(merged[index])
                break

            first = merged[index]
            second = merged[index + 1]
            valley = find_profile_valley_between(smoothed, first, second)
            weaker_peak = min(float(smoothed[first]), float(smoothed[second]))
            valley_value = float(smoothed[valley])
            clear_gap = (
                second - first >= min_distance
                and valley_value <= max(1.0, weaker_peak * LINE_PEAK_VALLEY_RATIO)
            )

            if clear_gap:
                result.append(first)
                index += 1
                continue

            keep = first if float(smoothed[first]) >= float(smoothed[second]) else second
            result.append(keep)
            index += 2
            changed = True

        merged = sorted(set(result))

    return merged


def profile_valleys_between_peaks(smoothed: np.ndarray, peaks: list[int]) -> list[int]:
    return [
        find_profile_valley_between(smoothed, peaks[index], peaks[index + 1])
        for index in range(len(peaks) - 1)
    ]


def find_profile_valley_between(smoothed: np.ndarray, first_peak: int, second_peak: int) -> int:
    if second_peak <= first_peak + 1:
        return first_peak

    segment = smoothed[first_peak : second_peak + 1]
    return first_peak + int(np.argmin(segment))


def profile_bands_from_peaks(
    smoothed: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    peak_threshold: float,
) -> list[tuple[int, int]]:
    height = len(smoothed)
    bands: list[tuple[int, int]] = []
    for index, peak in enumerate(peaks):
        lower_limit = 0 if index == 0 else valleys[index - 1] + 1
        upper_limit = height - 1 if index == len(peaks) - 1 else valleys[index]
        edge_threshold = max(0.8, min(peak_threshold, float(smoothed[peak]) * LINE_PEAK_EDGE_RATIO))

        top = peak
        while top > lower_limit and float(smoothed[top]) > edge_threshold:
            top -= 1

        bottom = peak
        while bottom < upper_limit and float(smoothed[bottom]) > edge_threshold:
            bottom += 1

        bands.append((max(lower_limit, top), min(upper_limit, bottom)))

    return bands


def line_boxes_from_component_centers(ink_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = ink_mask.shape[:2]
    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    components: list[tuple[int, int, int, int, int, float]] = []

    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < max(8, LINE_NOISE_MIN_AREA * 2) or w < 2 or h < 2:
            continue
        components.append((int(x), int(y), int(w), int(h), int(area), float(centroids[label][1])))

    if not components:
        return []

    component_heights = np.array([item[3] for item in components], dtype=np.float32)
    median_height = float(np.median(component_heights))
    center_gap = max(34.0, min(height * 0.024, median_height * 1.55))

    components.sort(key=lambda item: item[5])
    groups: list[list[tuple[int, int, int, int, int, float]]] = []
    current = [components[0]]
    current_center = components[0][5]

    for component in components[1:]:
        center_y = component[5]
        if center_y - current_center <= center_gap:
            current.append(component)
            current_center = float(np.mean([item[5] for item in current]))
        else:
            groups.append(current)
            current = [component]
            current_center = center_y
    groups.append(current)

    min_line_height = max(8, int(height * LINE_MIN_HEIGHT_FRACTION))
    min_line_width = max(35, int(width * LINE_MIN_WIDTH_FRACTION))
    horizontal_pad = max(10, int(width * LINE_HORIZONTAL_PAD_FRACTION))
    vertical_pad = max(6, int(height * LINE_VERTICAL_PAD_FRACTION))
    boxes: list[tuple[int, int, int, int]] = []

    for group in groups:
        x1 = min(item[0] for item in group)
        y1 = min(item[1] for item in group)
        x2 = max(item[0] + item[2] - 1 for item in group)
        y2 = max(item[1] + item[3] - 1 for item in group)
        ink_area = sum(item[4] for item in group)

        x1 = max(0, x1 - horizontal_pad)
        y1 = max(0, y1 - vertical_pad)
        x2 = min(width - 1, x2 + horizontal_pad)
        y2 = min(height - 1, y2 + vertical_pad)
        box_width = x2 - x1 + 1
        box_height = y2 - y1 + 1
        density = ink_area / max(1, box_width * box_height)

        if box_width < min_line_width or box_height < min_line_height:
            continue
        if ink_area < max(28, box_width * 0.025):
            continue
        if density < 0.0045 and ink_area < 1200:
            continue
        if box_width > width * 0.65 and density < 0.009 and ink_area < 1500:
            continue
        if looks_like_centered_header((x1, y1, box_width, box_height), width, height):
            continue

        boxes.append((x1, y1, box_width, box_height))

    boxes = trim_line_box_horizontal_outliers(boxes, ink_mask)
    boxes = split_merged_component_line_boxes(boxes, ink_mask)
    boxes = separate_vertical_overlaps(sorted(boxes, key=lambda box: (box[1], box[0])))
    boxes = drop_isolated_top_headers(boxes, ink_mask.shape[:2])
    return drop_edge_line_artifacts(boxes, ink_mask.shape[:2])


def split_merged_component_line_boxes(
    boxes: list[tuple[int, int, int, int]],
    ink_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 2:
        return boxes

    height, width = ink_mask.shape[:2]
    box_heights = np.array([box[3] for box in boxes], dtype=np.float32)
    median_height = float(np.median(box_heights))
    tall_threshold = max(median_height * 1.55, height * 0.055)

    row_counts = np.count_nonzero(ink_mask, axis=1).astype(np.float32)
    smooth_window = odd_kernel_size(int(height * LINE_PROFILE_SMOOTH_FRACTION), minimum=15)
    smoothed = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, smooth_window), 0).reshape(-1)

    horizontal_pad = max(10, int(width * LINE_HORIZONTAL_PAD_FRACTION))
    vertical_pad = max(6, int(height * LINE_VERTICAL_PAD_FRACTION))
    min_line_height = max(8, int(height * LINE_MIN_HEIGHT_FRACTION))
    min_line_width = max(35, int(width * LINE_MIN_WIDTH_FRACTION))
    split_boxes: list[tuple[int, int, int, int]] = []

    for box in sorted(boxes, key=lambda item: (item[1], item[0])):
        x, y, w, h = box
        if h < tall_threshold:
            split_boxes.append(box)
            continue

        sub_bands = split_tall_band(
            smoothed_profile=smoothed,
            top=y,
            bottom=y + h - 1,
            expected_height=median_height,
            tall_threshold=tall_threshold,
        )
        if len(sub_bands) < 2:
            split_boxes.append(box)
            continue

        candidates: list[tuple[int, int, int, int]] = []
        for top, bottom in sub_bands:
            candidate = line_box_from_row_band(ink_mask, top, bottom, horizontal_pad, vertical_pad)
            if candidate is None:
                continue

            cx, cy, cw, ch = candidate
            if cw < min_line_width or ch < min_line_height:
                continue
            if looks_like_centered_header(candidate, width, height):
                continue

            roi = ink_mask[cy : cy + ch, cx : cx + cw]
            ink_pixels = int(np.count_nonzero(roi))
            if ink_pixels < max(18, cw * 0.018):
                continue

            candidates.append(candidate)

        split_boxes.extend(candidates if len(candidates) > 1 else [box])

    return split_boxes


def line_box_from_row_band(
    ink_mask: np.ndarray,
    top: int,
    bottom: int,
    horizontal_pad: int,
    vertical_pad: int,
) -> tuple[int, int, int, int] | None:
    height, width = ink_mask.shape[:2]
    y1 = max(0, int(top) - vertical_pad)
    y2 = min(height - 1, int(bottom) + vertical_pad)
    band_mask = ink_mask[y1 : y2 + 1, :]
    ys, xs = np.where(band_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x1 = max(0, int(xs.min()) - horizontal_pad)
    x2 = min(width - 1, int(xs.max()) + horizontal_pad)
    refined_y1 = max(0, y1 + int(ys.min()) - vertical_pad)
    refined_y2 = min(height - 1, y1 + int(ys.max()) + vertical_pad)
    return x1, refined_y1, x2 - x1 + 1, refined_y2 - refined_y1 + 1


def trim_line_box_horizontal_outliers(
    boxes: list[tuple[int, int, int, int]],
    ink_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    _image_height, image_width = ink_mask.shape[:2]
    horizontal_pad = max(10, int(image_width * LINE_HORIZONTAL_PAD_FRACTION))
    min_width = max(35, int(image_width * LINE_MIN_WIDTH_FRACTION))
    trimmed: list[tuple[int, int, int, int]] = []

    for box in boxes:
        x, y, w, h = box
        roi = ink_mask[y : y + h, x : x + w]
        left: int | None = None
        right: int | None = None

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        components: list[tuple[int, int, int]] = []
        for label in range(1, component_count):
            cx, _cy, cw, _ch, area = stats[label]
            if area >= LINE_NOISE_MIN_AREA:
                components.append((int(cx), int(cw), int(area)))

        if components:
            components.sort(key=lambda item: item[0])
            component_areas = np.array([item[2] for item in components], dtype=np.float32)
            median_area = float(np.median(component_areas))
            isolation_gap = max(120, int(image_width * 0.18), int(h * 3.0))
            small_area = max(50.0, median_area * 0.35)
            kept_components: list[tuple[int, int, int]] = []

            for index, component in enumerate(components):
                cx, cw, area = component
                previous_gap = None
                next_gap = None
                if index > 0:
                    previous = components[index - 1]
                    previous_gap = cx - (previous[0] + previous[1])
                if index < len(components) - 1:
                    following = components[index + 1]
                    next_gap = following[0] - (cx + cw)

                isolated_edge = (
                    (previous_gap is None and next_gap is not None and next_gap > isolation_gap)
                    or (next_gap is None and previous_gap is not None and previous_gap > isolation_gap)
                    or (
                        previous_gap is not None
                        and next_gap is not None
                        and previous_gap > isolation_gap
                        and next_gap > isolation_gap
                    )
                )
                if isolated_edge and float(area) < small_area:
                    continue
                kept_components.append(component)

            if kept_components:
                left = min(item[0] for item in kept_components)
                right = max(item[0] + item[1] - 1 for item in kept_components)

        if left is None or right is None:
            _ys, xs = np.where(roi > 0)
            if xs.size < 12:
                trimmed.append(box)
                continue
            left = int(np.percentile(xs, 1.5))
            right = int(np.percentile(xs, 98.5))

        if right <= left or right - left + 1 >= w * 0.92:
            trimmed.append(box)
            continue

        new_x1 = max(0, x + left - horizontal_pad)
        new_x2 = min(image_width - 1, x + right + horizontal_pad)
        if new_x2 - new_x1 + 1 < min_width:
            trimmed.append(box)
            continue

        trimmed.append((new_x1, y, new_x2 - new_x1 + 1, h))

    return trimmed


def separate_vertical_overlaps(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 2:
        return boxes

    adjusted = list(boxes)
    for index in range(len(adjusted) - 1):
        x1, y1, w1, h1 = adjusted[index]
        x2, y2, w2, h2 = adjusted[index + 1]
        bottom1 = y1 + h1 - 1
        bottom2 = y2 + h2 - 1
        if bottom1 < y2:
            continue

        center1 = y1 + h1 / 2.0
        center2 = y2 + h2 / 2.0
        boundary = int(round((center1 + center2) / 2.0))
        new_bottom1 = max(y1, min(bottom1, boundary))
        new_top2 = min(bottom2, max(y2, boundary + 1))

        adjusted[index] = (x1, y1, w1, new_bottom1 - y1 + 1)
        adjusted[index + 1] = (x2, new_top2, w2, bottom2 - new_top2 + 1)

    return [box for box in adjusted if box[2] > 0 and box[3] > 0]


def active_row_bands(active_rows: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, active in enumerate(active_rows):
        if active and start is None:
            start = y
        elif not active and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, len(active_rows) - 1))
    return bands


def merge_close_bands(bands: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not bands:
        return []

    merged = [bands[0]]
    for top, bottom in bands[1:]:
        previous_top, previous_bottom = merged[-1]
        if top - previous_bottom <= merge_gap:
            merged[-1] = (previous_top, bottom)
        else:
            merged.append((top, bottom))
    return merged


def split_unusually_tall_bands(
    bands: list[tuple[int, int]],
    smoothed_profile: np.ndarray,
    image_height: int,
) -> list[tuple[int, int]]:
    if len(bands) < 2:
        return bands

    heights = np.array([bottom - top + 1 for top, bottom in bands], dtype=np.float32)
    median_height = float(np.median(heights))
    tall_threshold = max(median_height * 1.45, image_height * 0.035)
    result: list[tuple[int, int]] = []

    for top, bottom in bands:
        result.extend(split_tall_band(smoothed_profile, top, bottom, median_height, tall_threshold))

    return sorted(result, key=lambda band: band[0])


def split_tall_band(
    smoothed_profile: np.ndarray,
    top: int,
    bottom: int,
    expected_height: float,
    tall_threshold: float,
    depth: int = 0,
) -> list[tuple[int, int]]:
    if depth >= 4 or bottom - top + 1 < tall_threshold:
        return [(top, bottom)]

    split = find_profile_valley(smoothed_profile, top, bottom, expected_height)
    if split is None:
        return [(top, bottom)]

    min_piece_height = max(8, int(expected_height * 0.45))
    left = (top, split - 1)
    right = (split + 1, bottom)
    if left[1] - left[0] + 1 < min_piece_height or right[1] - right[0] + 1 < min_piece_height:
        return [(top, bottom)]

    return (
        split_tall_band(smoothed_profile, left[0], left[1], expected_height, tall_threshold, depth + 1)
        + split_tall_band(smoothed_profile, right[0], right[1], expected_height, tall_threshold, depth + 1)
    )


def find_profile_valley(
    smoothed_profile: np.ndarray,
    top: int,
    bottom: int,
    expected_height: float,
) -> int | None:
    guard = max(8, int(expected_height * 0.45))
    start = top + guard
    end = bottom - guard
    if end <= start:
        return None

    segment = smoothed_profile[start : end + 1]
    valley = start + int(np.argmin(segment))
    left_peak = float(smoothed_profile[top:valley].max()) if valley > top else 0.0
    right_peak = float(smoothed_profile[valley + 1 : bottom + 1].max()) if valley < bottom else 0.0
    peak_floor = min(left_peak, right_peak)
    if peak_floor <= 0:
        return None
    if float(smoothed_profile[valley]) > peak_floor * 0.82:
        return None
    return valley


def looks_like_centered_header(box: tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    x, y, w, h = box
    center_x = x + w / 2.0
    return (
        y < image_height * 0.22
        and image_width * 0.36 <= center_x <= image_width * 0.64
        and w < image_width * 0.26
        and h < image_height * 0.06
    )


def merge_overlapping_line_boxes(
    boxes: list[tuple[int, int, int, int]],
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0])):
        if not merged:
            merged.append(box)
            continue

        x, y, w, h = box
        last_x, last_y, last_w, last_h = merged[-1]
        last_bottom = last_y + last_h - 1
        current_bottom = y + h - 1
        overlap = min(last_bottom, current_bottom) - max(last_y, y) + 1
        close_gap = y - last_bottom <= max(4, int(image_height * 0.004))
        if overlap > min(last_h, h) * 0.25 or close_gap:
            x1 = min(last_x, x)
            y1 = min(last_y, y)
            x2 = max(last_x + last_w - 1, x + w - 1)
            y2 = max(last_bottom, current_bottom)
            merged[-1] = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        else:
            merged.append(box)

    return merged


def draw_line_segmentation_overlay(segmentation_mask: np.ndarray, line_boxes: list[LineBox]) -> np.ndarray:
    overlay = as_bgr(segmentation_mask).copy()
    thickness = max(2, min(5, min(overlay.shape[:2]) // 500))
    font_scale = max(0.55, min(1.2, min(overlay.shape[:2]) / 1800.0))
    label_thickness = max(1, thickness - 1)

    for line in line_boxes:
        cv2.rectangle(
            overlay,
            (line.x, line.y),
            (line.x + line.w - 1, line.y + line.h - 1),
            (0, 160, 0),
            thickness,
        )
        label_y = max(22, line.y - 8)
        cv2.putText(
            overlay,
            f"L{line.index}",
            (line.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 80, 255),
            label_thickness,
            cv2.LINE_AA,
        )

    return overlay


def save_line_segmentation_debug(page: PageState) -> None:
    if not LINE_DEBUG_OUTPUT or page.segmentation_mask is None or page.line_boxes is None:
        return

    LINE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stem = page.path.stem if page.path else datetime.now().strftime("page_%Y%m%d_%H%M%S")
    debug_image = draw_line_segmentation_debug_overlay(page.segmentation_mask, page.line_boxes)
    save_image(LINE_DEBUG_DIR / f"{stem}_line_debug.png", debug_image)
    write_line_segmentation_debug_csv(LINE_DEBUG_DIR / f"{stem}_line_debug.csv", page.line_boxes)


def draw_line_segmentation_debug_overlay(segmentation_mask: np.ndarray, line_boxes: list[LineBox]) -> np.ndarray:
    overlay = draw_line_segmentation_overlay(segmentation_mask, line_boxes)
    height = overlay.shape[0]
    panel_width = 360
    panel = np.full((height, panel_width, 3), 255, dtype=np.uint8)

    smoothed = np.asarray(LAST_LINE_SEGMENTATION_DEBUG.get("smoothed", []), dtype=np.float32)
    peaks = [int(value) for value in LAST_LINE_SEGMENTATION_DEBUG.get("peaks", [])]
    valleys = [int(value) for value in LAST_LINE_SEGMENTATION_DEBUG.get("valleys", [])]
    bands = [
        (int(top), int(bottom))
        for top, bottom in LAST_LINE_SEGMENTATION_DEBUG.get("bands", [])
    ]

    if smoothed.size == height and float(smoothed.max()) > 0:
        scale = (panel_width - 55) / float(smoothed.max())
        previous = (0, 0)
        for y, value in enumerate(smoothed):
            x = int(float(value) * scale)
            point = (x, y)
            if y > 0:
                cv2.line(panel, previous, point, (40, 40, 40), 1)
            previous = point

    for top, bottom in bands:
        cv2.line(panel, (0, top), (panel_width - 1, top), (180, 120, 0), 1)
        cv2.line(panel, (0, bottom), (panel_width - 1, bottom), (180, 120, 0), 1)

    for valley in valleys:
        cv2.line(panel, (0, valley), (panel_width - 1, valley), (0, 0, 220), 1)

    for peak in peaks:
        cv2.line(panel, (0, peak), (panel_width - 1, peak), (0, 150, 0), 1)

    cv2.putText(panel, "profile", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(panel, "green=peaks", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "red=valleys", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "blue=bounds", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 100, 0), 1, cv2.LINE_AA)
    return np.hstack([overlay, panel])


def write_line_segmentation_debug_csv(path: Path, line_boxes: list[LineBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peaks = [int(value) for value in LAST_LINE_SEGMENTATION_DEBUG.get("peaks", [])]
    valleys = [int(value) for value in LAST_LINE_SEGMENTATION_DEBUG.get("valleys", [])]
    bands = [
        (int(top), int(bottom))
        for top, bottom in LAST_LINE_SEGMENTATION_DEBUG.get("bands", [])
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "index", "y", "top", "bottom", "x", "w", "h"])
        for index, peak in enumerate(peaks, start=1):
            writer.writerow(["peak", index, peak, "", "", "", "", ""])
        for index, valley in enumerate(valleys, start=1):
            writer.writerow(["valley", index, valley, "", "", "", "", ""])
        for index, (top, bottom) in enumerate(bands, start=1):
            writer.writerow(["band", index, "", top, bottom, "", "", ""])
        for line in line_boxes:
            writer.writerow(["line_box", line.index, line.y, line.y, line.y + line.h - 1, line.x, line.w, line.h])


def segment_words_from_lines(
    segmentation_mask: np.ndarray,
    line_boxes: list[LineBox],
) -> list[list[WordBox]]:
    gray = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY) if len(segmentation_mask.shape) == 3 else segmentation_mask
    ink_mask = np.where(gray < 128, 255, 0).astype(np.uint8)
    ink_mask = remove_tiny_specks(ink_mask, min_area=WORD_NOISE_MIN_AREA)

    return [segment_words_for_line(ink_mask, line) for line in line_boxes]


def segment_words_for_line(ink_mask: np.ndarray, line: LineBox) -> list[WordBox]:
    image_height, image_width = ink_mask.shape[:2]
    x1 = max(0, line.x)
    y1 = max(0, line.y)
    x2 = min(image_width, line.x + line.w)
    y2 = min(image_height, line.y + line.h)
    if x2 <= x1 or y2 <= y1:
        return []

    roi = ink_mask[y1:y2, x1:x2]
    components = collect_word_components(roi)
    if not components:
        return []

    kernel_width = estimate_word_dilation_width(components, x2 - x1, y2 - y1)
    kernel_height = max(3, min(7, (y2 - y1) // 16))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    word_mask = cv2.dilate(roi, kernel, iterations=1)
    contours, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    word_boxes: list[tuple[int, int, int, int, int]] = []
    for contour in sorted(contours, key=lambda item: cv2.boundingRect(item)[0]):
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw < WORD_MIN_WIDTH or ch < WORD_MIN_HEIGHT:
            continue

        members = [
            component
            for component in components
            if boxes_intersect((component[0], component[1], component[2], component[3]), (cx, cy, cw, ch))
        ]
        if not members:
            continue

        box = word_box_from_components(
            members,
            origin_x=x1,
            origin_y=y1,
            image_width=image_width,
            image_height=image_height,
            line_height=y2 - y1,
        )
        if box is not None:
            word_boxes.append(box)

    if not word_boxes:
        word_boxes = fallback_word_boxes_from_components(
            components,
            origin_x=x1,
            origin_y=y1,
            image_width=image_width,
            image_height=image_height,
            line_height=y2 - y1,
        )

    word_boxes = merge_overlapping_word_boxes(sorted(word_boxes, key=lambda box: (box[0], box[1])))
    return [
        WordBox(line_index=line.index, word_index=index, x=x, y=y, w=w, h=h)
        for index, (x, y, w, h, _area) in enumerate(word_boxes, start=1)
    ]


def collect_word_components(roi: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    components: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < WORD_NOISE_MIN_AREA or (w <= 1 and h <= 1):
            continue
        components.append((int(x), int(y), int(w), int(h), int(area)))
    return sorted(components, key=lambda component: (component[0], component[1]))


def estimate_word_dilation_width(
    components: list[tuple[int, int, int, int, int]],
    line_width: int,
    line_height: int,
) -> int:
    if not components:
        return max(7, int(line_height * WORD_DILATE_WIDTH_LINE_HEIGHT_FRACTION))

    widths = np.array([component[2] for component in components], dtype=np.float32)
    median_width = float(np.median(widths)) if widths.size else 8.0
    by_height = line_height * WORD_DILATE_WIDTH_LINE_HEIGHT_FRACTION
    by_width = line_width * WORD_DILATE_WIDTH_LINE_WIDTH_FRACTION
    by_component = median_width * 1.35
    kernel_width = int(round(max(7.0, min(34.0, by_height, by_width, by_component))))
    return max(3, kernel_width)


def boxes_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    fx, fy, fw, fh = first
    sx, sy, sw, sh = second
    return not (
        fx + fw <= sx
        or sx + sw <= fx
        or fy + fh <= sy
        or sy + sh <= fy
    )


def word_box_from_components(
    components: list[tuple[int, int, int, int, int]],
    origin_x: int,
    origin_y: int,
    image_width: int,
    image_height: int,
    line_height: int,
) -> tuple[int, int, int, int, int] | None:
    if not components:
        return None

    horizontal_pad = max(3, int(line_height * WORD_HORIZONTAL_PAD_FRACTION))
    vertical_pad = max(3, int(line_height * WORD_VERTICAL_PAD_FRACTION))
    x1 = min(component[0] for component in components)
    y1 = min(component[1] for component in components)
    x2 = max(component[0] + component[2] - 1 for component in components)
    y2 = max(component[1] + component[3] - 1 for component in components)
    ink_area = sum(component[4] for component in components)

    gx1 = max(0, origin_x + x1 - horizontal_pad)
    gy1 = max(0, origin_y + y1 - vertical_pad)
    gx2 = min(image_width - 1, origin_x + x2 + horizontal_pad)
    gy2 = min(image_height - 1, origin_y + y2 + vertical_pad)
    width = gx2 - gx1 + 1
    height = gy2 - gy1 + 1

    if width < WORD_MIN_WIDTH or height < WORD_MIN_HEIGHT:
        return None
    if ink_area < max(WORD_NOISE_MIN_AREA, width * 0.015):
        return None

    return gx1, gy1, width, height, ink_area


def fallback_word_boxes_from_components(
    components: list[tuple[int, int, int, int, int]],
    origin_x: int,
    origin_y: int,
    image_width: int,
    image_height: int,
    line_height: int,
) -> list[tuple[int, int, int, int, int]]:
    if not components:
        return []

    groups: list[list[tuple[int, int, int, int, int]]] = []
    current = [components[0]]
    current_right = components[0][0] + components[0][2]
    join_gap = max(8, int(line_height * 0.35))

    for component in components[1:]:
        gap = component[0] - current_right
        if gap <= join_gap:
            current.append(component)
            current_right = max(current_right, component[0] + component[2])
        else:
            groups.append(current)
            current = [component]
            current_right = component[0] + component[2]
    groups.append(current)

    boxes: list[tuple[int, int, int, int, int]] = []
    for group in groups:
        box = word_box_from_components(
            group,
            origin_x=origin_x,
            origin_y=origin_y,
            image_width=image_width,
            image_height=image_height,
            line_height=line_height,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def merge_overlapping_word_boxes(
    boxes: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    if not boxes:
        return []

    merged: list[tuple[int, int, int, int, int]] = [boxes[0]]
    for box in boxes[1:]:
        x, y, w, h, area = box
        last_x, last_y, last_w, last_h, last_area = merged[-1]
        overlap_x = min(last_x + last_w, x + w) - max(last_x, x)
        overlap_y = min(last_y + last_h, y + h) - max(last_y, y)
        substantial_overlap = overlap_x > min(last_w, w) * 0.35 and overlap_y > min(last_h, h) * 0.35
        if substantial_overlap:
            x1 = min(last_x, x)
            y1 = min(last_y, y)
            x2 = max(last_x + last_w - 1, x + w - 1)
            y2 = max(last_y + last_h - 1, y + h - 1)
            merged[-1] = (x1, y1, x2 - x1 + 1, y2 - y1 + 1, last_area + area)
        else:
            merged.append(box)
    return merged


def total_word_count(word_boxes_by_line: list[list[WordBox]] | None) -> int:
    if not word_boxes_by_line:
        return 0
    return sum(len(words) for words in word_boxes_by_line)


def draw_word_segmentation_overlay(
    segmentation_mask: np.ndarray,
    line_boxes: list[LineBox],
    word_boxes_by_line: list[list[WordBox]],
) -> np.ndarray:
    overlay = as_bgr(segmentation_mask).copy()
    line_thickness = max(1, min(3, min(overlay.shape[:2]) // 850))
    word_thickness = max(2, min(4, min(overlay.shape[:2]) // 650))
    font_scale = max(0.36, min(0.75, min(overlay.shape[:2]) / 2300.0))
    label_thickness = max(1, word_thickness - 1)

    for line in line_boxes:
        cv2.rectangle(
            overlay,
            (line.x, line.y),
            (line.x + line.w - 1, line.y + line.h - 1),
            (150, 210, 150),
            line_thickness,
        )

    for words in word_boxes_by_line:
        for word in words:
            cv2.rectangle(
                overlay,
                (word.x, word.y),
                (word.x + word.w - 1, word.y + word.h - 1),
                (255, 90, 0),
                word_thickness,
            )
            label = f"L{word.line_index}-W{word.word_index}"
            cv2.putText(
                overlay,
                label,
                (word.x, max(14, word.y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 70, 255),
                label_thickness,
                cv2.LINE_AA,
            )

    return overlay


def draw_recognition_overlay(
    recognition_source: np.ndarray,
    line_boxes: list[LineBox],
    word_boxes_by_line: list[list[WordBox]],
    result: RecognitionResult,
) -> np.ndarray:
    overlay = as_bgr(recognition_source).copy()
    line_thickness = max(1, min(3, min(overlay.shape[:2]) // 900))
    word_thickness = max(2, min(4, min(overlay.shape[:2]) // 700))
    font_scale = max(0.36, min(0.68, min(overlay.shape[:2]) / 2400.0))
    label_thickness = max(1, word_thickness - 1)
    predictions = {
        (prediction.line_index, prediction.word_index): prediction
        for prediction in result.predictions
    }

    for line in line_boxes:
        cv2.rectangle(
            overlay,
            (line.x, line.y),
            (line.x + line.w - 1, line.y + line.h - 1),
            (170, 220, 170),
            line_thickness,
        )

    for words in word_boxes_by_line:
        for word in words:
            prediction = predictions.get((word.line_index, word.word_index))
            label = ""
            color = (255, 90, 0)
            if prediction is not None:
                label = f"{prediction.label} {prediction.confidence:.0f}%"
                color = (40, 80, 255) if prediction.confidence < 60.0 else (255, 90, 0)

            cv2.rectangle(
                overlay,
                (word.x, word.y),
                (word.x + word.w - 1, word.y + word.h - 1),
                color,
                word_thickness,
            )
            if label:
                cv2.putText(
                    overlay,
                    label[:28],
                    (word.x, max(14, word.y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    label_thickness,
                    cv2.LINE_AA,
                )

    return append_recognition_transcript_panel(overlay, result)


def append_recognition_transcript_panel(overlay: np.ndarray, result: RecognitionResult) -> np.ndarray:
    image_height, image_width = overlay.shape[:2]
    panel_width = max(360, min(620, int(image_width * 0.48)))
    panel = np.full((image_height, panel_width, 3), 255, dtype=np.uint8)
    average_confidence = (
        sum(prediction.confidence for prediction in result.predictions) / len(result.predictions)
        if result.predictions
        else 0.0
    )

    y = 36
    cv2.putText(panel, result.model_name, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    y += 34
    cv2.putText(
        panel,
        f"Words: {len(result.predictions)}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    y += 24
    cv2.putText(
        panel,
        f"Avg confidence: {average_confidence:.1f}%",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    y += 32
    cv2.line(panel, (18, y), (panel_width - 18, y), (210, 210, 210), 1)
    y += 28

    for line in result.transcript.splitlines() or ["No recognized words."]:
        for wrapped in wrap_text_for_panel(line, max_chars=max(18, panel_width // 12)):
            if y > image_height - 22:
                cv2.putText(
                    panel,
                    "...",
                    (18, image_height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (60, 60, 60),
                    1,
                    cv2.LINE_AA,
                )
                return np.hstack([overlay, panel])
            cv2.putText(
                panel,
                wrapped,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )
            y += 23
        y += 10

    return np.hstack([overlay, panel])


def wrap_text_for_panel(text: str, max_chars: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def recognize_segmented_words(
    recognition_source_gray: np.ndarray,
    word_boxes_by_line: list[list[WordBox]],
    model_name: str,
    model_path: Path,
    labels_path: Path,
) -> RecognitionResult:
    model, labels = load_recognition_model(model_name, model_path, labels_path)
    model_inputs: list[np.ndarray] = []
    word_refs: list[tuple[int, int]] = []

    for words in word_boxes_by_line:
        for word in words:
            crop = crop_word_from_recognition_source(recognition_source_gray, word)
            model_crop = resize_word_crop_for_model(crop)
            model_rgb = cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            model_inputs.append(model_rgb)
            word_refs.append((word.line_index, word.word_index))

    predictions: list[WordPrediction] = []
    if model_inputs:
        probabilities = model.predict(np.stack(model_inputs, axis=0), verbose=0)
        for (line_index, word_index), probs in zip(word_refs, probabilities):
            class_index = int(np.argmax(probs))
            label = labels[class_index] if 0 <= class_index < len(labels) else f"class_{class_index}"
            confidence = float(probs[class_index]) * 100.0
            predictions.append(
                WordPrediction(
                    line_index=line_index,
                    word_index=word_index,
                    label=display_prediction_label(label),
                    confidence=confidence,
                )
            )

    transcript = build_transcript_from_predictions(predictions)
    return RecognitionResult(model_name=model_name, transcript=transcript, predictions=predictions)


def load_recognition_model(
    model_name: str,
    model_path: Path,
    labels_path: Path,
) -> tuple[object, list[str]]:
    global TF_MODULE

    if TF_MODULE is None:
        try:
            import tensorflow as tensorflow_module  # type: ignore
        except Exception as exc:
            raise RuntimeError("TensorFlow is not available. Install TensorFlow to run model recognition.") from exc
        TF_MODULE = tensorflow_module

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    cache_key = f"{model_name}:{model_path.resolve()}:{labels_path.resolve()}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    model = TF_MODULE.keras.models.load_model(model_path)
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    MODEL_CACHE[cache_key] = (model, labels)
    return model, labels


def crop_word_from_recognition_source(
    recognition_source_gray: np.ndarray,
    word: WordBox,
    pad: int = 2,
) -> np.ndarray:
    image_bgr = as_bgr(recognition_source_gray)
    height, width = image_bgr.shape[:2]
    dynamic_pad = max(pad, min(12, int(round(max(word.w, word.h) * 0.08))))
    x1 = max(0, word.x - dynamic_pad)
    y1 = max(0, word.y - dynamic_pad)
    x2 = min(width, word.x + word.w + dynamic_pad)
    y2 = min(height, word.y + word.h + dynamic_pad)
    if x2 <= x1 or y2 <= y1:
        return np.full((128, 128, 3), 255, dtype=np.uint8)
    return image_bgr[y1:y2, x1:x2].copy()


def display_prediction_label(label: str) -> str:
    return str(label).replace("_", " ")


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def build_transcript_from_predictions(predictions: list[WordPrediction]) -> str:
    if not predictions:
        return ""

    lines: dict[int, list[WordPrediction]] = {}
    for prediction in predictions:
        lines.setdefault(prediction.line_index, []).append(prediction)

    transcript_lines: list[str] = []
    for line_index in sorted(lines):
        words = sorted(lines[line_index], key=lambda prediction: prediction.word_index)
        transcript_lines.append(" ".join(prediction.label for prediction in words))
    return "\n".join(transcript_lines)


def format_recognition_result(result: RecognitionResult) -> str:
    if not result.predictions:
        return f"{result.model_name}\n\nNo recognized words."

    average_confidence = sum(prediction.confidence for prediction in result.predictions) / len(result.predictions)
    return (
        f"{result.model_name}\n"
        f"Words: {len(result.predictions)}\n"
        f"Average confidence: {average_confidence:.1f}%\n\n"
        f"{result.transcript}"
    )


def as_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def bgr_to_tk_image(image_bgr: np.ndarray, size: tuple[int, int]) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    resized = pil_image.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(resized)


class PreprocessingGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GreggSpeak Image Preprocessing")
        self.root.geometry("1180x760")
        self.root.minsize(960, 620)

        self.pages: list[PageState] = []
        self.canvas_image: ImageTk.PhotoImage | None = None
        self.current_recognition_key: str | None = None
        self.zoom_level = 1.0

        self.preview_mode = tk.StringVar(value="original")
        self.status_var = tk.StringVar(value="Load image(s) to begin.")
        self.image_var = tk.StringVar(value="No image loaded")

        self._build_style()
        self._build_ui()

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", foreground="#374151")

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(12, 10, 12, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(10, weight=1)

        ttk.Button(toolbar, text="Load Image(s)", command=self.load_image).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="Crop Image", command=self.crop_image).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="Clean / Enhance Image", command=self.clean_enhance_image).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Create Segmentation Mask", command=self.create_segmentation_mask).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Segment Lines", command=self.segment_lines).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(toolbar, text="Segment Words", command=self.segment_words).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(toolbar, text="Recognize with Model V5.2", command=self.recognize_with_model_v52).grid(
            row=0, column=6, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Recognize with Model V5.3", command=self.recognize_with_model_v53).grid(
            row=0, column=7, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Process All", command=self.process_all_images).grid(row=0, column=8, padx=(0, 8))
        ttk.Button(toolbar, text="Save Processed Images", command=self.save_processed_images).grid(
            row=0, column=9, padx=(0, 12)
        )
        ttk.Label(toolbar, textvariable=self.image_var, style="Header.TLabel").grid(row=0, column=10, sticky="w")

        main = ttk.Frame(self.root)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(main, padding=(12, 8, 10, 10))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.rowconfigure(13, weight=1)

        ttk.Label(sidebar, text="Preview", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        preview_options = (
            ("Original", "original"),
            ("Cropped", "cropped"),
            ("Cleaned / Enhanced", "cleaned"),
            ("Recognition Ready", "recognition_ready"),
            ("Segmentation Mask", "segmentation_mask"),
            ("Line Segmentation", "line_segmentation"),
            ("Word Segmentation", "word_segmentation"),
            ("Recognition", "recognition"),
        )
        for row, (label, value) in enumerate(preview_options, start=1):
            ttk.Radiobutton(
                sidebar,
                text=label,
                value=value,
                variable=self.preview_mode,
                command=self.update_preview,
            ).grid(row=row, column=0, sticky="w", pady=3)

        ttk.Label(
            sidebar,
            text=(
                "Workflow:\n"
                "1. Load Image(s)\n"
                "2. Crop Image\n"
                "3. Clean / Enhance Image\n"
                "4. Create Segmentation Mask\n"
                "5. Segment Lines\n"
                "6. Segment Words\n"
                "7. Recognize Words\n\n"
                "Use Process All to run every stage on every loaded page."
            ),
            justify="left",
            wraplength=260,
            foreground="#4b5563",
        ).grid(row=9, column=0, sticky="ew", pady=(18, 0))

        ttk.Label(sidebar, text="Loaded Pages", font=("Segoe UI", 11, "bold")).grid(
            row=12, column=0, sticky="w", pady=(18, 6)
        )
        list_frame = ttk.Frame(sidebar)
        list_frame.grid(row=13, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.page_list = tk.Listbox(list_frame, width=36, height=9, exportselection=False)
        self.page_list.grid(row=0, column=0, sticky="nsew")
        self.page_list.bind("<<ListboxSelect>>", self.on_page_select)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.page_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.page_list.configure(yscrollcommand=scroll.set)

        canvas_frame = ttk.Frame(main, padding=(0, 8, 12, 10))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(1, weight=1)

        zoom_bar = ttk.Frame(canvas_frame)
        zoom_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(zoom_bar, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(zoom_bar, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(zoom_bar, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(canvas_frame, background="#111827", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.update_preview())
        self.canvas.bind("<Control-MouseWheel>", self.on_mousewheel_zoom)
        vertical_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vertical_scroll.grid(row=1, column=1, sticky="ns")
        horizontal_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        horizontal_scroll.grid(row=2, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal_scroll.set, yscrollcommand=vertical_scroll.set)

        status = ttk.Frame(self.root, padding=(12, 6))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")

    def load_image(self) -> None:
        selected = filedialog.askopenfilenames(title="Load image(s)", filetypes=IMAGE_TYPES)
        if not selected:
            return

        try:
            self.pages = [PageState(path=Path(path), original_bgr=load_bgr_image(Path(path))) for path in selected]
            self.page_list.delete(0, tk.END)
            for page in self.pages:
                self.page_list.insert(tk.END, self.page_label(page))

            if self.pages:
                self.page_list.selection_set(0)
                self.page_list.activate(0)

            self.preview_mode.set("original")
            self.zoom_level = 1.0
            self.update_image_label()
            self.set_recognition_text("")
            self.status_var.set(f"Loaded {len(self.pages)} image(s). Click Crop Image or Process All.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Load failed", exc)

    def selected_page(self) -> PageState | None:
        if not self.pages:
            return None
        selection = self.page_list.curselection()
        index = selection[0] if selection else 0
        if index >= len(self.pages):
            return None
        return self.pages[index]

    def selected_index(self) -> int | None:
        if not self.pages:
            return None
        selection = self.page_list.curselection()
        return selection[0] if selection else 0

    def on_page_select(self, _event: tk.Event) -> None:
        self.update_image_label()
        self.update_recognition_output()
        self.update_preview()

    def crop_image(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        try:
            self.crop_page(page)
            page.enhanced_gray = None
            page.recognition_ready_image = None
            page.segmentation_mask = None
            page.line_boxes = None
            page.line_segmentation_bgr = None
            page.word_boxes_by_line = None
            page.word_segmentation_bgr = None
            page.recognition_results.clear()
            page.recognition_overlay_bgr = None
            page.active_recognition_key = None
            self.current_recognition_key = None
            self.set_recognition_text("")
            self.refresh_selected_page_label()
            self.preview_mode.set("cropped")
            self.status_var.set("Crop complete. Click Clean / Enhance Image next.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Crop failed", exc)

    def clean_enhance_image(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if page.cropped_bgr is None:
            messagebox.showinfo("No crop", "Click Crop Image before cleaning/enhancing.")
            return

        try:
            page.enhanced_gray = clean_enhance_image(page.cropped_bgr)
            page.recognition_ready_image = prepare_recognition_ready_image(page.cropped_bgr)
            page.segmentation_mask = None
            page.line_boxes = None
            page.line_segmentation_bgr = None
            page.word_boxes_by_line = None
            page.word_segmentation_bgr = None
            page.recognition_results.clear()
            page.recognition_overlay_bgr = None
            page.active_recognition_key = None
            self.current_recognition_key = None
            self.set_recognition_text("")
            self.refresh_selected_page_label()
            self.preview_mode.set("cleaned")
            self.status_var.set("Clean/enhance complete. Click Create Segmentation Mask next.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Clean/enhance failed", exc)

    def create_segmentation_mask(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if page.enhanced_gray is None:
            if page.cropped_bgr is None:
                messagebox.showinfo("No crop", "Click Crop Image before creating a segmentation mask.")
                return
            page.enhanced_gray = clean_enhance_image(page.cropped_bgr)
            page.recognition_ready_image = prepare_recognition_ready_image(page.cropped_bgr)
        elif page.recognition_ready_image is None and page.cropped_bgr is not None:
            page.recognition_ready_image = prepare_recognition_ready_image(page.cropped_bgr)

        try:
            page.segmentation_mask = create_segmentation_mask(page.enhanced_gray)
            page.line_boxes = None
            page.line_segmentation_bgr = None
            page.word_boxes_by_line = None
            page.word_segmentation_bgr = None
            page.recognition_results.clear()
            page.recognition_overlay_bgr = None
            page.active_recognition_key = None
            self.current_recognition_key = None
            self.set_recognition_text("")
            self.refresh_selected_page_label()
            self.preview_mode.set("segmentation_mask")
            self.status_var.set("Segmentation mask created. Click Segment Lines next.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Segmentation mask creation failed", exc)

    def segment_lines(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if page.segmentation_mask is None:
            messagebox.showinfo("No segmentation mask", "Click Create Segmentation Mask before segmenting lines.")
            return

        try:
            page.line_boxes = segment_lines_from_mask(page.segmentation_mask)
            page.line_segmentation_bgr = draw_line_segmentation_overlay(page.segmentation_mask, page.line_boxes)
            page.word_boxes_by_line = None
            page.word_segmentation_bgr = None
            page.recognition_results.clear()
            page.recognition_overlay_bgr = None
            page.active_recognition_key = None
            self.current_recognition_key = None
            self.set_recognition_text("")
            save_line_segmentation_debug(page)
            self.refresh_selected_page_label()
            self.preview_mode.set("line_segmentation")
            self.status_var.set(f"Line segmentation complete. Detected {len(page.line_boxes)} line(s).")
            self.update_preview()
        except Exception as exc:
            self.show_error("Line segmentation failed", exc)

    def segment_words(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if page.segmentation_mask is None:
            messagebox.showinfo("No segmentation mask", "Click Create Segmentation Mask before segmenting words.")
            return
        if not page.line_boxes:
            messagebox.showinfo("No line segmentation", "Please segment lines first.")
            return

        try:
            page.word_boxes_by_line = segment_words_from_lines(page.segmentation_mask, page.line_boxes)
            page.word_segmentation_bgr = draw_word_segmentation_overlay(
                page.segmentation_mask,
                page.line_boxes,
                page.word_boxes_by_line,
            )
            page.recognition_results.clear()
            page.recognition_overlay_bgr = None
            page.active_recognition_key = None
            self.current_recognition_key = None
            self.set_recognition_text("")
            word_count = total_word_count(page.word_boxes_by_line)
            self.refresh_selected_page_label()
            self.preview_mode.set("word_segmentation")
            self.status_var.set(
                f"Detected {len(page.line_boxes)} line(s) and {word_count} word segment(s)."
            )
            self.update_preview()
        except Exception as exc:
            self.show_error("Word segmentation failed", exc)

    def recognize_with_model_v52(self) -> None:
        self.recognize_words("v5.2", "Model V5.2", MODEL_V52_PATH, LABELS_V52_PATH)

    def recognize_with_model_v53(self) -> None:
        model_path = first_existing_path(MODEL_V53_PATH, BEST_MODEL_V53_PATH)
        labels_path = first_existing_path(LABELS_V53_PATH, LABELS_V52_PATH)
        self.recognize_words("v5.3", "Model V5.3", model_path, labels_path)

    def recognize_words(
        self,
        result_key: str,
        model_name: str,
        model_path: Path,
        labels_path: Path,
    ) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if not page.word_boxes_by_line:
            messagebox.showinfo("No word segmentation", "Please segment words first.")
            return
        if page.recognition_ready_image is None:
            if page.cropped_bgr is None:
                messagebox.showinfo("No enhanced image", "Click Clean / Enhance Image before recognition.")
                return
            page.enhanced_gray = clean_enhance_image(page.cropped_bgr)
            page.recognition_ready_image = prepare_recognition_ready_image(page.cropped_bgr)

        try:
            word_count = total_word_count(page.word_boxes_by_line)
            self.status_var.set(f"Loading {model_name} and recognizing {word_count} word crop(s)...")
            self.root.update_idletasks()
            result = recognize_segmented_words(
                page.recognition_ready_image,
                page.word_boxes_by_line,
                model_name=model_name,
                model_path=model_path,
                labels_path=labels_path,
            )
            page.recognition_results.clear()
            page.recognition_results[result_key] = result
            page.recognition_overlay_bgr = draw_recognition_overlay(
                page.recognition_ready_image,
                page.line_boxes or [],
                page.word_boxes_by_line,
                result,
            )
            page.active_recognition_key = result_key
            self.current_recognition_key = result_key
            self.refresh_selected_page_label()
            self.preview_mode.set("recognition")
            self.status_var.set(
                f"{model_name} recognition complete. Recognized {len(result.predictions)} word crop(s)."
            )
            self.update_preview()
        except Exception as exc:
            self.show_error(f"{model_name} recognition failed", exc)

    def process_all_images(self) -> None:
        if not self.pages:
            messagebox.showinfo("No images", "Load images first.")
            return

        try:
            for index, page in enumerate(self.pages, start=1):
                self.status_var.set(f"Processing {index}/{len(self.pages)}: {page.path.name}")
                self.root.update_idletasks()
                self.crop_page(page)
                page.enhanced_gray = clean_enhance_image(page.cropped_bgr)
                page.recognition_ready_image = prepare_recognition_ready_image(page.cropped_bgr)
                page.segmentation_mask = create_segmentation_mask(page.enhanced_gray)
                page.line_boxes = segment_lines_from_mask(page.segmentation_mask)
                page.line_segmentation_bgr = draw_line_segmentation_overlay(page.segmentation_mask, page.line_boxes)
                page.word_boxes_by_line = segment_words_from_lines(page.segmentation_mask, page.line_boxes)
                page.word_segmentation_bgr = draw_word_segmentation_overlay(
                    page.segmentation_mask,
                    page.line_boxes,
                    page.word_boxes_by_line,
                )
                page.recognition_results.clear()
                page.recognition_overlay_bgr = None
                page.active_recognition_key = None
                save_line_segmentation_debug(page)

            self.refresh_page_list()
            self.current_recognition_key = None
            self.set_recognition_text("")
            self.preview_mode.set("word_segmentation")
            self.status_var.set(f"Processed {len(self.pages)} image(s). Use the page list to inspect results.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Process all failed", exc)

    def crop_page(self, page: PageState) -> None:
        if not CROP_POINTS_FILE.exists():
            raise FileNotFoundError(f"Could not find {CROP_POINTS_FILE}")
        height, width = page.original_bgr.shape[:2]
        crop_points = read_crop_points(CROP_POINTS_FILE, width, height)
        page.cropped_bgr = perspective_crop(page.original_bgr, crop_points)

    def save_processed_images(self) -> None:
        pages_with_outputs = [
            page
            for page in self.pages
            if (
                page.cropped_bgr is not None
                or page.enhanced_gray is not None
                or page.recognition_ready_image is not None
                or page.segmentation_mask is not None
                or page.line_segmentation_bgr is not None
                or page.word_segmentation_bgr is not None
                or page.recognition_overlay_bgr is not None
                or bool(page.recognition_results)
            )
        ]
        if not pages_with_outputs:
            messagebox.showinfo(
                "Nothing to save",
                "Crop, clean, create a segmentation mask, or segment at least one image first.",
            )
            return

        initial_dir = PROJECT_ROOT / "reports" / "preprocessing_gui"
        initial_dir.mkdir(parents=True, exist_ok=True)
        output_dir = filedialog.askdirectory(title="Choose output folder", initialdir=str(initial_dir))
        if not output_dir:
            return

        try:
            output_path = Path(output_dir)
            for page in pages_with_outputs:
                stem = page.path.stem if page.path else datetime.now().strftime("page_%Y%m%d_%H%M%S")
                if page.cropped_bgr is not None:
                    save_image(output_path / f"{stem}_01_cropped.png", page.cropped_bgr)
                if page.enhanced_gray is not None:
                    save_image(output_path / f"{stem}_02_enhanced_gray.png", page.enhanced_gray)
                if page.recognition_ready_image is not None:
                    save_image(output_path / f"{stem}_03_recognition_ready.png", page.recognition_ready_image)
                if page.segmentation_mask is not None:
                    save_image(output_path / f"{stem}_04_segmentation_mask.png", page.segmentation_mask)
                if page.line_segmentation_bgr is not None:
                    save_image(output_path / f"{stem}_05_line_segmentation.png", page.line_segmentation_bgr)
                if page.word_segmentation_bgr is not None:
                    save_image(output_path / f"{stem}_06_word_segmentation.png", page.word_segmentation_bgr)
                if page.recognition_overlay_bgr is not None and page.active_recognition_key:
                    safe_key = page.active_recognition_key.replace(".", "_")
                    save_image(output_path / f"{stem}_07_recognition_{safe_key}.png", page.recognition_overlay_bgr)
                if page.line_boxes:
                    with (output_path / f"{stem}_line_boxes.csv").open("w", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(["line_index", "x", "y", "w", "h"])
                        for line in page.line_boxes:
                            writer.writerow([line.index, line.x, line.y, line.w, line.h])
                if page.word_boxes_by_line:
                    with (output_path / f"{stem}_word_boxes.csv").open("w", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(["line_index", "word_index", "x", "y", "w", "h"])
                        for words in page.word_boxes_by_line:
                            for word in words:
                                writer.writerow([word.line_index, word.word_index, word.x, word.y, word.w, word.h])
                    if page.recognition_ready_image is not None:
                        crop_dir = output_path / f"{stem}_recognition_word_crops"
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        for words in page.word_boxes_by_line:
                            for word in words:
                                crop = crop_word_from_recognition_source(page.recognition_ready_image, word)
                                save_image(
                                    crop_dir / f"L{word.line_index:02d}_W{word.word_index:02d}.png",
                                    crop,
                                )
                for result_key, result in page.recognition_results.items():
                    safe_key = result_key.replace(".", "_")
                    (output_path / f"{stem}_recognized_{safe_key}.txt").write_text(
                        format_recognition_result(result),
                        encoding="utf-8",
                    )
                    with (output_path / f"{stem}_recognized_{safe_key}.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(["line_index", "word_index", "label", "confidence"])
                        for prediction in result.predictions:
                            writer.writerow(
                                [
                                    prediction.line_index,
                                    prediction.word_index,
                                    prediction.label,
                                    f"{prediction.confidence:.4f}",
                                ]
                            )

            self.status_var.set(f"Processed images saved to {output_path}")
            messagebox.showinfo("Saved", f"Processed images saved to:\n{output_path}")
        except Exception as exc:
            self.show_error("Save failed", exc)

    def get_preview_image(self) -> np.ndarray | None:
        page = self.selected_page()
        if page is None:
            return None

        mode = self.preview_mode.get()
        if mode == "recognition":
            return page.recognition_overlay_bgr
        if mode == "word_segmentation":
            return page.word_segmentation_bgr
        if mode == "line_segmentation":
            return page.line_segmentation_bgr
        if mode == "segmentation_mask":
            return as_bgr(page.segmentation_mask) if page.segmentation_mask is not None else None
        if mode == "recognition_ready":
            return as_bgr(page.recognition_ready_image) if page.recognition_ready_image is not None else None
        if mode == "cleaned":
            return as_bgr(page.enhanced_gray) if page.enhanced_gray is not None else None
        if mode == "cropped":
            return page.cropped_bgr
        return page.original_bgr

    def update_recognition_output(self) -> None:
        page = self.selected_page()
        if page is None or not page.recognition_results:
            self.current_recognition_key = None
            return

        key = page.active_recognition_key or self.current_recognition_key
        if key not in page.recognition_results:
            key = next(reversed(page.recognition_results))
            page.active_recognition_key = key

        self.current_recognition_key = key

    def set_recognition_text(self, text: str) -> None:
        if not hasattr(self, "recognition_text"):
            return
        self.recognition_text.configure(state="normal")
        self.recognition_text.delete("1.0", tk.END)
        self.recognition_text.insert("1.0", text)
        self.recognition_text.configure(state="disabled")

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom_level * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom_level / 1.25)

    def reset_zoom(self) -> None:
        self.set_zoom(1.0)

    def set_zoom(self, zoom: float) -> None:
        self.zoom_level = max(0.25, min(6.0, float(zoom)))
        self.update_preview()

    def on_mousewheel_zoom(self, event: tk.Event) -> str:
        delta = getattr(event, "delta", 0)
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()
        return "break"

    def update_preview(self) -> None:
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")
        image_bgr = self.get_preview_image()
        if image_bgr is None:
            self.canvas.create_text(
                max(240, self.canvas.winfo_width() // 2),
                max(160, self.canvas.winfo_height() // 2),
                text="No preview for this stage yet",
                fill="#d1d5db",
                font=("Segoe UI", 16),
            )
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_height, image_width = image_bgr.shape[:2]
        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = fit_scale * self.zoom_level
        display_width = max(1, int(round(image_width * scale)))
        display_height = max(1, int(round(image_height * scale)))
        offset_x = max(0, (canvas_width - display_width) // 2)
        offset_y = max(0, (canvas_height - display_height) // 2)

        self.canvas_image = bgr_to_tk_image(image_bgr, (display_width, display_height))
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.canvas_image)
        self.canvas.configure(
            scrollregion=(
                0,
                0,
                max(canvas_width, display_width + offset_x * 2),
                max(canvas_height, display_height + offset_y * 2),
            )
        )

    def update_image_label(self) -> None:
        page = self.selected_page()
        if page is None:
            self.image_var.set("No image loaded")
            return

        height, width = page.original_bgr.shape[:2]
        index = self.selected_index()
        prefix = f"{index + 1}/{len(self.pages)}  " if index is not None else ""
        self.image_var.set(f"{prefix}{page.path.name}  ({width} x {height})")

    def page_label(self, page: PageState) -> str:
        if page.recognition_results:
            stage = f"recognized: {page.active_recognition_key or next(iter(page.recognition_results))}"
        elif page.word_boxes_by_line is not None:
            stage = f"words: {total_word_count(page.word_boxes_by_line)}"
        elif page.line_boxes is not None:
            stage = f"lines: {len(page.line_boxes)}"
        elif page.segmentation_mask is not None:
            stage = "mask"
        elif page.recognition_ready_image is not None:
            stage = "recognition-ready"
        elif page.enhanced_gray is not None:
            stage = "cleaned"
        elif page.cropped_bgr is not None:
            stage = "cropped"
        else:
            stage = "original"
        return f"{page.path.name}  -  {stage}"

    def refresh_selected_page_label(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        self.page_list.delete(index)
        self.page_list.insert(index, self.page_label(self.pages[index]))
        self.page_list.selection_set(index)
        self.page_list.activate(index)
        self.update_image_label()

    def refresh_page_list(self) -> None:
        current = self.selected_index() or 0
        self.page_list.delete(0, tk.END)
        for page in self.pages:
            self.page_list.insert(tk.END, self.page_label(page))
        if self.pages:
            current = min(current, len(self.pages) - 1)
            self.page_list.selection_set(current)
            self.page_list.activate(current)
        self.update_image_label()

    def show_error(self, title: str, exc: Exception) -> None:
        traceback.print_exc()
        self.status_var.set(f"{title}: {exc}")
        messagebox.showerror(title, str(exc))


def main() -> None:
    root = tk.Tk()
    app = PreprocessingGUI(root)
    if len(sys.argv) > 1:
        image_paths = [Path(arg).expanduser() for arg in sys.argv[1:] if Path(arg).expanduser().exists()]
        if image_paths:
            app.pages = [PageState(path=path, original_bgr=load_bgr_image(path)) for path in image_paths]
            for page in app.pages:
                app.page_list.insert(tk.END, app.page_label(page))
            app.page_list.selection_set(0)
            app.update_image_label()
            app.update_preview()
    root.mainloop()


if __name__ == "__main__":
    main()
