from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


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
BINARY_THRESHOLD_METHOD = "conservative_otsu"
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

WORD_NOISE_MIN_AREA = 4
WORD_DILATE_WIDTH_LINE_HEIGHT_FRACTION = 0.34
WORD_DILATE_WIDTH_LINE_WIDTH_FRACTION = 0.03
WORD_MIN_WIDTH = 4
WORD_MIN_HEIGHT = 4
WORD_HORIZONTAL_PAD_FRACTION = 0.14
WORD_VERTICAL_PAD_FRACTION = 0.08


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
    index: int
    x: int
    y: int
    w: int
    h: int
    component_boxes: List[tuple[int, int, int, int]] | None = None


@dataclass
class SegmentationArtifacts:
    grayscale: np.ndarray
    enhanced_gray: np.ndarray
    recognition_bgr: np.ndarray
    binary: np.ndarray
    deskewed_bgr: np.ndarray
    overlay: np.ndarray
    line_boxes: List[LineBox]
    word_boxes_by_line: List[List[WordBox]]
    deskew_angle: float


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def as_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def odd_kernel_size(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def percentile_stretch(gray: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low, high = np.percentile(gray, (low_percentile, high_percentile))
    if high <= low:
        return gray.copy()

    stretched = (gray.astype(np.float32) - float(low)) * (255.0 / float(high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def handwriting_luminance(image_bgr: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image_bgr).astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    red_channel = lab[:, :, 1].astype(np.float32)
    red_excess = np.maximum(0.0, red_channel - float(np.median(red_channel)))
    adjusted = gray - (red_excess * RED_INK_LAB_A_WEIGHT)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def clean_enhance_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = handwriting_luminance(image_bgr)
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


def prepare_recognition_ready_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = handwriting_luminance(image_bgr)
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


def create_segmentation_ink_mask(enhanced_gray: np.ndarray) -> np.ndarray:
    """Create a conservative foreground-ink mask for internal segmentation."""
    if len(enhanced_gray.shape) == 3:
        prepared = to_grayscale(enhanced_gray)
    else:
        prepared = enhanced_gray.copy()

    if BINARY_OTSU_BLUR_SIZE and BINARY_OTSU_BLUR_SIZE >= 3:
        prepared = cv2.GaussianBlur(prepared, (odd_kernel_size(BINARY_OTSU_BLUR_SIZE),) * 2, 0)

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
    return suppress_sparse_background_rows(ink_mask)


def create_segmentation_mask(enhanced_gray: np.ndarray) -> np.ndarray:
    """Match the GUI segmentation test: white page with dark detected ink."""
    return 255 - create_segmentation_ink_mask(enhanced_gray)


def segmentation_mask_to_ink_mask(segmentation_mask: np.ndarray) -> np.ndarray:
    gray = to_grayscale(segmentation_mask) if len(segmentation_mask.shape) == 3 else segmentation_mask
    return np.where(gray < 128, 255, 0).astype(np.uint8)


def binarize_image(enhanced_gray: np.ndarray) -> np.ndarray:
    """Backward-compatible internal foreground mask for older callers."""
    return create_segmentation_ink_mask(enhanced_gray)


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
        touches_right_zone = x + w >= width - side_limit
        in_upper_band = y < top_limit and centroid_y < top_limit
        in_upper_right = in_upper_band and (x >= right_start or centroid_x >= right_start or touches_right_zone)

        if in_upper_right:
            filtered[labels == label] = 0
        elif in_upper_band and touches_right_zone and area <= max(BINARY_BORDER_ARTIFACT_MAX_AREA, width // 2):
            filtered[labels == label] = 0

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


def build_row_mask(binary: np.ndarray) -> np.ndarray:
    kernel_width = max(120, binary.shape[1] // 72)
    kernel_height = max(9, binary.shape[0] // 380)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    return cv2.dilate(binary, kernel, iterations=1)


def estimate_skew_angle(binary: np.ndarray) -> float:
    del binary
    return 0.0


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return image.copy()

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _smoothed_row_bands(row_mask: np.ndarray) -> List[tuple[int, int]]:
    profile = np.sum(row_mask > 0, axis=1).astype(np.float32)
    window_size = max(31, (row_mask.shape[0] // 60) | 1)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(profile, kernel, mode="same")
    threshold = max(20.0, float(smoothed.max()) * 0.18)

    bands = []
    start = None
    for idx, active in enumerate(smoothed > threshold):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            bands.append((start, idx - 1))
            start = None

    if start is not None:
        bands.append((start, len(smoothed) - 1))
    return bands


def _row_profile_and_smoothed(row_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = np.sum(row_mask > 0, axis=1).astype(np.float32)
    window_size = max(31, (row_mask.shape[0] // 60) | 1)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(profile, kernel, mode="same")
    return profile, smoothed


def _find_split_in_band(
    smoothed_profile: np.ndarray,
    top: int,
    bottom: int,
    min_band_height: int,
    valley_ratio: float = 0.72,
) -> int | None:
    band_h = bottom - top + 1
    if band_h < min_band_height * 2:
        return None

    guard = max(20, min_band_height // 2)
    search_start = top + guard
    search_end = bottom - guard
    if search_end <= search_start:
        return None

    segment = smoothed_profile[search_start:search_end + 1]
    valley_idx = int(np.argmin(segment)) + search_start

    left_peak = float(np.max(smoothed_profile[top:valley_idx])) if valley_idx > top else 0.0
    right_peak = float(np.max(smoothed_profile[valley_idx + 1:bottom + 1])) if valley_idx < bottom else 0.0
    valley = float(smoothed_profile[valley_idx])
    peak_floor = min(left_peak, right_peak)

    if peak_floor <= 0.0:
        return None

    if valley > peak_floor * valley_ratio:
        return None

    return valley_idx


def _split_overlapping_row_bands(
    bands: List[tuple[int, int]],
    row_mask: np.ndarray,
    min_line_height: int,
) -> List[tuple[int, int]]:
    if not bands:
        return bands

    heights = np.array([bottom - top + 1 for top, bottom in bands], dtype=np.float32)
    median_height = float(np.median(heights)) if len(heights) else float(min_line_height)
    tall_threshold = int(max(min_line_height * 2.0, median_height * 1.65))
    _, smoothed = _row_profile_and_smoothed(row_mask)

    split_bands: List[tuple[int, int]] = []
    for top, bottom in bands:
        split_bands.extend(
            _split_tall_band_by_valleys(
                smoothed_profile=smoothed,
                top=top,
                bottom=bottom,
                min_line_height=min_line_height,
                tall_threshold=tall_threshold,
            )
        )

    return sorted(split_bands, key=lambda band: band[0])


def _split_tall_band_by_valleys(
    smoothed_profile: np.ndarray,
    top: int,
    bottom: int,
    min_line_height: int,
    tall_threshold: int,
    max_depth: int = 4,
) -> List[tuple[int, int]]:
    band_h = bottom - top + 1
    if max_depth <= 0 or band_h < tall_threshold:
        return [(top, bottom)]

    split_at = _find_split_in_band(
        smoothed_profile,
        top,
        bottom,
        min_line_height,
        valley_ratio=0.88,
    )
    if split_at is None:
        return [(top, bottom)]

    left = (top, split_at - 1)
    right = (split_at + 1, bottom)
    if left[1] - left[0] + 1 < min_line_height:
        return [(top, bottom)]
    if right[1] - right[0] + 1 < min_line_height:
        return [(top, bottom)]

    return (
        _split_tall_band_by_valleys(
            smoothed_profile,
            left[0],
            left[1],
            min_line_height,
            tall_threshold,
            max_depth=max_depth - 1,
        )
        + _split_tall_band_by_valleys(
            smoothed_profile,
            right[0],
            right[1],
            min_line_height,
            tall_threshold,
            max_depth=max_depth - 1,
        )
    )


def _split_bands_to_expected_count(
    bands: List[tuple[int, int]],
    row_mask: np.ndarray,
    expected_rows: int | None,
    min_line_height: int,
) -> List[tuple[int, int]]:
    if not expected_rows or expected_rows <= 0 or len(bands) >= expected_rows:
        return bands

    _, smoothed = _row_profile_and_smoothed(row_mask)
    bands = list(bands)

    while len(bands) < expected_rows:
        split_made = False
        candidates = sorted(
            enumerate(bands),
            key=lambda item: (item[1][1] - item[1][0] + 1),
            reverse=True,
        )

        for idx, (top, bottom) in candidates:
            split_at = _find_split_in_band(smoothed, top, bottom, min_line_height)
            if split_at is None:
                continue

            left = (top, split_at - 1)
            right = (split_at + 1, bottom)
            replacement = []
            if left[1] - left[0] + 1 >= min_line_height:
                replacement.append(left)
            if right[1] - right[0] + 1 >= min_line_height:
                replacement.append(right)
            if len(replacement) < 2:
                continue

            bands = bands[:idx] + replacement + bands[idx + 1:]
            split_made = True
            break

        if not split_made:
            break

    return sorted(bands, key=lambda band: band[0])


def _binary_profile_row_bands(
    binary: np.ndarray,
    min_line_height: int,
) -> List[tuple[int, int]]:
    profile = np.sum(binary > 0, axis=1).astype(np.float32)
    if profile.size == 0 or float(profile.max()) <= 0.0:
        return []

    window_size = max(9, (binary.shape[0] // 35) | 1)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(profile, kernel, mode="same")
    threshold = max(1.5, float(smoothed.max()) * 0.15)
    min_band_height = max(8, min_line_height // 2)

    bands = []
    start = None
    for idx, active in enumerate(smoothed > threshold):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_band_height:
                bands.append((start, idx - 1))
            start = None

    if start is not None and len(smoothed) - start >= min_band_height:
        bands.append((start, len(smoothed) - 1))

    vertical_pad = max(4, min_line_height // 4)
    return [
        (max(0, top - vertical_pad), min(binary.shape[0] - 1, bottom + vertical_pad))
        for top, bottom in bands
    ]


def _should_use_binary_profile_bands(
    row_bands: List[tuple[int, int]],
    profile_bands: List[tuple[int, int]],
    image_height: int,
    min_line_height: int,
) -> bool:
    if not profile_bands:
        return False

    if not row_bands:
        return True

    row_heights = np.array([bottom - top + 1 for top, bottom in row_bands], dtype=np.float32)
    profile_heights = np.array([bottom - top + 1 for top, bottom in profile_bands], dtype=np.float32)
    tallest_row_band = float(row_heights.max())
    median_profile_band = float(np.median(profile_heights)) if len(profile_heights) else 0.0

    collapsed_page = tallest_row_band > max(float(min_line_height * 4), image_height * 0.42)
    profile_found_more_rows = len(profile_bands) > len(row_bands)
    profile_rows_are_reasonable = median_profile_band >= max(8.0, min_line_height * 0.8)

    return collapsed_page and profile_found_more_rows and profile_rows_are_reasonable


def _boxes_from_row_bands(
    binary: np.ndarray,
    row_mask: np.ndarray,
    bands: List[tuple[int, int]],
    min_line_width: int,
    min_line_height: int,
    max_line_height: int,
    horizontal_margin: int,
    limit_vertical_to_band_gutters: bool = False,
) -> List[tuple[int, int, int, int]]:
    boxes = []
    if not bands:
        return boxes

    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    assigned_components: List[List[tuple[int, int, int, int]]] = [[] for _ in bands]
    band_centers = np.array([(top + bottom) / 2.0 for top, bottom in bands], dtype=np.float32)
    gutter_pad = max(6, min_line_height // 2)

    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < 8 or w < 2 or h < 2:
            continue

        component_top = int(y)
        component_bottom = int(y + h - 1)
        overlaps = []
        for top, bottom in bands:
            overlap = max(0, min(component_bottom, bottom) - max(component_top, top) + 1)
            overlaps.append(overlap)

        best_overlap = max(overlaps) if overlaps else 0
        if best_overlap > 0:
            line_idx = int(np.argmax(overlaps))
        else:
            cy = float(centroids[label][1])
            distances = np.abs(band_centers - cy)
            line_idx = int(np.argmin(distances))
            band_top, band_bottom = bands[line_idx]
            band_height = band_bottom - band_top + 1
            max_assignment_distance = max(min_line_height * 1.15, band_height * 0.75)
            if float(distances[line_idx]) > max_assignment_distance:
                continue

        assigned_components[line_idx].append((int(x), int(y), int(w), int(h)))

    vertical_pad = max(28, int(min_line_height * 0.45))
    for band_index, (top, bottom) in enumerate(bands):
        band_h = bottom - top + 1
        if band_h < min_line_height:
            continue

        band_mask = row_mask[top : bottom + 1, :]
        cols = np.where(np.sum(band_mask > 0, axis=0) > 0)[0]
        components = assigned_components[band_index]
        if len(cols) == 0 and not components:
            continue

        if components:
            x1 = min(item[0] for item in components)
            y1 = min(item[1] for item in components)
            x2 = max(item[0] + item[2] - 1 for item in components)
            y2 = max(item[1] + item[3] - 1 for item in components)
        else:
            x1 = int(cols.min())
            x2 = int(cols.max())
            y1 = top
            y2 = bottom

        x1 = max(0, x1 - horizontal_margin)
        x2 = min(binary.shape[1] - 1, x2 + horizontal_margin)
        y1 = max(0, y1 - vertical_pad)
        y2 = min(binary.shape[0] - 1, y2 + vertical_pad)

        if limit_vertical_to_band_gutters:
            upper_limit = 0
            lower_limit = binary.shape[0] - 1
            if band_index > 0:
                prev_center = band_centers[band_index - 1]
                current_center = band_centers[band_index]
                upper_limit = max(0, int((prev_center + current_center) / 2.0) - gutter_pad)
            if band_index < len(bands) - 1:
                current_center = band_centers[band_index]
                next_center = band_centers[band_index + 1]
                lower_limit = min(binary.shape[0] - 1, int((current_center + next_center) / 2.0) + gutter_pad)

            y1 = max(y1, upper_limit)
            y2 = min(y2, lower_limit)

        if components:
            trimmed = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        else:
            trimmed = _trim_box_to_ink(
                binary,
                (x1, y1, x2 - x1 + 1, y2 - y1 + 1),
                horizontal_margin,
                vertical_pad,
            )

        _, _, w, h = trimmed
        if w >= min_line_width and min_line_height <= h <= max_line_height:
            boxes.append(trimmed)

    return boxes


def _trim_box_to_ink(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    horizontal_pad: int,
    vertical_pad: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    roi = binary[y : y + h, x : x + w]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0 or len(ys) == 0:
        return box

    x1 = max(0, x + int(xs.min()) - horizontal_pad)
    y1 = max(0, y + int(ys.min()) - vertical_pad)
    x2 = min(binary.shape[1] - 1, x + int(xs.max()) + horizontal_pad)
    y2 = min(binary.shape[0] - 1, y + int(ys.max()) + vertical_pad)
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _row_looks_like_shorthand(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    min_ink_ratio: float,
    max_ink_ratio: float,
) -> bool:
    x, y, w, h = box
    roi = binary[y : y + h, x : x + w]
    ink_ratio = float(np.count_nonzero(roi)) / max(1, roi.size)
    if not (min_ink_ratio <= ink_ratio <= max_ink_ratio):
        return False

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    usable = []
    for label in range(1, component_count):
        _, _, cw, ch, area = stats[label]
        if area < 10 or cw < 2 or ch < 2:
            continue
        usable.append((int(cw), int(ch), int(area)))

    if len(usable) < 3:
        return False

    widths = np.array([item[0] for item in usable], dtype=np.float32)
    heights = np.array([item[1] for item in usable], dtype=np.float32)
    component_density = len(usable) / max(1.0, w / 100.0)

    if component_density > 22.0 and float(widths.mean()) < 12.0:
        return False

    if float(heights.mean()) < 8.0 and component_density > 14.0:
        return False

    return True


def _looks_like_centered_page_header(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
) -> bool:
    x, y, w, h = box
    image_h, image_w = binary.shape[:2]
    center_x = x + w / 2.0

    return (
        y < image_h * 0.22
        and image_w * 0.35 <= center_x <= image_w * 0.65
        and w < image_w * 0.28
        and h < image_h * 0.055
    )


def _overlap_height(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    _, y1, _, h1 = first
    _, y2, _, h2 = second
    return max(0, min(y1 + h1, y2 + h2) - max(y1, y2))


def _looks_like_short_shorthand_row(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
) -> bool:
    x, y, w, h = box
    roi = binary[y : y + h, x : x + w]
    if roi.size == 0:
        return False

    ink_ratio = float(np.count_nonzero(roi)) / max(1, roi.size)
    if not 0.001 <= ink_ratio <= 0.22:
        return False

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    usable = 0
    total_area = 0
    for label in range(1, component_count):
        _, _, cw, ch, area = stats[label]
        if area < 8 or cw < 2 or ch < 2:
            continue
        usable += 1
        total_area += int(area)

    return usable >= 2 and total_area >= 40


def _find_short_component_line_boxes(
    binary: np.ndarray,
    existing_boxes: List[tuple[int, int, int, int]],
    min_line_width: int,
    min_line_height: int,
    horizontal_margin: int,
) -> List[tuple[int, int, int, int]]:
    uncovered_components = []
    vertical_slack = max(4, min_line_height // 2)

    for x, y, w, h, area in _collect_global_components(binary):
        if area < 8 or w < 2 or h < 2:
            continue

        component_box = (x, y, w, h)
        covered = False
        for line_box in existing_boxes:
            overlap = _overlap_height(component_box, line_box)
            center_y = y + h / 2.0
            inside_with_slack = line_box[1] - vertical_slack <= center_y <= line_box[1] + line_box[3] + vertical_slack
            if overlap >= max(2, h * 0.35) or inside_with_slack:
                covered = True
                break

        if not covered:
            uncovered_components.append((x, y, w, h, area))

    if not uncovered_components:
        return []

    uncovered_components.sort(key=lambda item: item[1] + item[3] / 2.0)
    center_gap = max(18, min_line_height * 2)
    groups: List[List[tuple[int, int, int, int, int]]] = []
    current = [uncovered_components[0]]
    current_center = uncovered_components[0][1] + uncovered_components[0][3] / 2.0

    for component in uncovered_components[1:]:
        center_y = component[1] + component[3] / 2.0
        if abs(center_y - current_center) <= center_gap:
            current.append(component)
            current_center = float(np.mean([item[1] + item[3] / 2.0 for item in current]))
        else:
            groups.append(current)
            current = [component]
            current_center = center_y
    groups.append(current)

    short_boxes = []
    vertical_pad = max(18, min_line_height)
    for group in groups:
        if len(group) < 2:
            continue

        x1 = max(0, min(item[0] for item in group) - horizontal_margin)
        y1 = max(0, min(item[1] for item in group) - vertical_pad)
        x2 = min(binary.shape[1] - 1, max(item[0] + item[2] - 1 for item in group) + horizontal_margin)
        y2 = min(binary.shape[0] - 1, max(item[1] + item[3] - 1 for item in group) + vertical_pad)
        box = _trim_box_to_ink(binary, (x1, y1, x2 - x1 + 1, y2 - y1 + 1), horizontal_margin, vertical_pad)

        if box[2] < min_line_width:
            continue
        if _looks_like_centered_page_header(binary, box):
            continue
        if not _looks_like_short_shorthand_row(binary, box):
            continue

        duplicate = False
        for existing in existing_boxes:
            center_distance = abs((box[1] + box[3] / 2.0) - (existing[1] + existing[3] / 2.0))
            if center_distance < max(18, min(box[3], existing[3]) * 0.35):
                duplicate = True
                break
        if not duplicate:
            short_boxes.append(box)

    return short_boxes


def _collect_word_components(
    roi_binary: np.ndarray,
) -> List[tuple[int, int, int, int, int]]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(roi_binary, connectivity=8)
    components = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < 8 or w < 2 or h < 2:
            continue
        components.append((int(x), int(y), int(w), int(h), int(area)))
    return components


def _collect_global_components(binary: np.ndarray) -> List[tuple[int, int, int, int, int]]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < 8 or w < 2 or h < 2:
            continue
        components.append((int(x), int(y), int(w), int(h), int(area)))
    return components


def _assign_components_to_lines(
    binary: np.ndarray,
    line_boxes: List[LineBox],
) -> List[List[tuple[int, int, int, int, int]]]:
    if not line_boxes:
        return []

    line_centers = np.array([line.y + line.h / 2.0 for line in line_boxes], dtype=np.float32)
    assigned: List[List[tuple[int, int, int, int, int]]] = [[] for _ in line_boxes]

    for component in _collect_global_components(binary):
        x, y, w, h, area = component
        component_center_y = y + h / 2.0
        line_index = int(np.argmin(np.abs(line_centers - component_center_y)))
        assigned[line_index].append(component)

    for components in assigned:
        components.sort(key=lambda item: (item[0], item[1]))

    return assigned


def _word_gap_threshold(components: List[tuple[int, int, int, int, int]], roi_width: int) -> int:
    if not components:
        return max(40, roi_width // 18)

    widths = np.array([item[2] for item in components], dtype=np.float32)
    median_width = float(np.median(widths)) if len(widths) else 12.0
    return int(max(36, min(roi_width // 10, median_width * 2.4)))


def detect_word_boxes_for_line(
    binary: np.ndarray,
    line: LineBox,
    word_gap: int | None = None,
    horizontal_pad: int = 18,
    vertical_pad: int = 14,
    owned_components: List[tuple[int, int, int, int, int]] | None = None,
) -> List[WordBox]:
    roi = np.zeros((line.h, line.w), dtype=np.uint8)
    component_pairs: List[tuple[tuple[int, int, int, int, int], tuple[int, int, int, int]]] = []

    if owned_components is None:
        roi = binary[line.y : line.y + line.h, line.x : line.x + line.w]
        for component in _collect_word_components(roi):
            x, y, w, h, area = component
            component_pairs.append((component, (line.x + x, line.y + y, w, h)))
    else:
        for x, y, w, h, area in owned_components:
            ix1 = max(line.x, x)
            iy1 = max(line.y, y)
            ix2 = min(line.x + line.w, x + w)
            iy2 = min(line.y + line.h, y + h)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            lx1 = ix1 - line.x
            ly1 = iy1 - line.y
            lx2 = ix2 - line.x
            ly2 = iy2 - line.y
            roi[ly1:ly2, lx1:lx2] = binary[iy1:iy2, ix1:ix2]
            component_pairs.append(((lx1, ly1, lx2 - lx1, ly2 - ly1, area), (x, y, w, h)))

    components = [component for component, _ in component_pairs]
    if not components:
        return []

    median_width = float(np.median(np.array([item[2] for item in components], dtype=np.float32)))
    kernel_width = int(max(5, min(42, median_width * 0.35, roi.shape[1] * 0.018)))
    if word_gap is not None and word_gap > 0:
        kernel_width = max(6, min(kernel_width, word_gap // 2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    word_mask = cv2.dilate(roi, kernel, iterations=1)
    contours, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    groups: List[List[int]] = []
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 10:
            continue

        member_indices = []
        for component_index, component in enumerate(components):
            cx, cy, cw, ch, area = component
            c_right = cx + cw
            c_bottom = cy + ch
            overlaps = not (c_right < x or cx > x + w or c_bottom < y or cy > y + h)
            if overlaps:
                member_indices.append(component_index)

        if member_indices:
            groups.append(member_indices)

    if not groups:
        sorted_indices = sorted(range(len(components)), key=lambda idx: components[idx][0])
        gap_threshold = word_gap if word_gap is not None else _word_gap_threshold(components, roi.shape[1])
        current = [sorted_indices[0]]
        first = components[sorted_indices[0]]
        current_right = first[0] + first[2]

        for component_index in sorted_indices[1:]:
            component = components[component_index]
            x, y, w, h, area = component
            gap = x - current_right
            if gap <= gap_threshold:
                current.append(component_index)
                current_right = max(current_right, x + w)
            else:
                groups.append(current)
                current = [component_index]
                current_right = x + w

        groups.append(current)

    words: List[WordBox] = []
    for index, group_indices in enumerate(groups, start=1):
        group = [components[group_index] for group_index in group_indices]
        global_components = [component_pairs[group_index][1] for group_index in group_indices]
        x1 = min(item[0] for item in group)
        y1 = min(item[1] for item in group)
        x2 = max(item[0] + item[2] for item in group)
        y2 = max(item[1] + item[3] for item in group)

        gx1 = max(0, line.x + x1 - horizontal_pad)
        gy1 = max(0, line.y + y1 - vertical_pad)
        gx2 = min(binary.shape[1] - 1, line.x + x2 + horizontal_pad)
        gy2 = min(binary.shape[0] - 1, line.y + y2 + vertical_pad)

        words.append(
            WordBox(
                line_index=line.index,
                index=index,
                x=gx1,
                y=gy1,
                w=gx2 - gx1 + 1,
                h=gy2 - gy1 + 1,
                component_boxes=global_components,
            )
        )

    return words


def detect_word_boxes(
    binary: np.ndarray,
    line_boxes: List[LineBox],
    word_gap: int | None = None,
) -> List[List[WordBox]]:
    owned_components_by_line = _assign_components_to_lines(binary, line_boxes)
    return [
        detect_word_boxes_for_line(
            binary,
            line,
            word_gap=word_gap,
            owned_components=owned_components_by_line[index] if index < len(owned_components_by_line) else None,
        )
        for index, line in enumerate(line_boxes)
    ]


def detect_line_boxes(
    binary: np.ndarray,
    min_line_width: int = 900,
    min_line_height: int = 80,
    max_line_height: int = 1400,
    horizontal_margin: int = 60,
    profile_threshold_ratio: float = 0.0,
    y_gap_tolerance: int = 18,
    min_ink_ratio: float = 0.0012,
    max_ink_ratio: float = 0.08,
    method: str = "components",
    expected_rows: int | None = None,
    split_overlapping_lines: bool = True,
) -> List[LineBox]:
    del profile_threshold_ratio, y_gap_tolerance, method

    row_mask = build_row_mask(binary)
    bands = _smoothed_row_bands(row_mask)
    split_min_line_height = max(min_line_height, min(70, max(18, int(binary.shape[0] * 0.018))))
    if split_overlapping_lines:
        bands = _split_overlapping_row_bands(bands, row_mask, split_min_line_height)

    use_profile_bands = False
    profile_bands = _binary_profile_row_bands(binary, split_min_line_height)
    if _should_use_binary_profile_bands(bands, profile_bands, binary.shape[0], split_min_line_height):
        bands = profile_bands
        use_profile_bands = True

    bands = _split_bands_to_expected_count(bands, row_mask, expected_rows, split_min_line_height)
    merged_boxes = _boxes_from_row_bands(
        binary,
        row_mask,
        bands,
        min_line_width=min_line_width,
        min_line_height=min_line_height,
        max_line_height=max_line_height,
        horizontal_margin=horizontal_margin,
        limit_vertical_to_band_gutters=use_profile_bands,
    )

    final_boxes = []
    for box in merged_boxes:
        trimmed = _trim_box_to_ink(binary, box, horizontal_pad=42, vertical_pad=24)
        if _looks_like_centered_page_header(binary, trimmed):
            continue
        if _row_looks_like_shorthand(binary, trimmed, min_ink_ratio, max_ink_ratio):
            final_boxes.append(trimmed)

    final_boxes.extend(
        _find_short_component_line_boxes(
            binary,
            final_boxes,
            min_line_width=min_line_width,
            min_line_height=min_line_height,
            horizontal_margin=horizontal_margin,
        )
    )
    final_boxes.sort(key=lambda box: (box[1], box[0]))
    return [
        LineBox(index=index, x=x, y=y, w=w, h=h)
        for index, (x, y, w, h) in enumerate(final_boxes, start=1)
    ]


def segment_lines_from_ink_mask(ink_mask: np.ndarray) -> List[LineBox]:
    ink_mask = remove_tiny_specks(ink_mask, min_area=LINE_NOISE_MIN_AREA)

    profile_boxes = line_boxes_from_profile_peaks(ink_mask)
    component_boxes = line_boxes_from_component_centers(ink_mask)

    if profile_boxes and (
        not component_boxes
        or len(component_boxes) <= len(profile_boxes) <= len(component_boxes) + 3
    ):
        boxes = profile_boxes
    elif component_boxes:
        boxes = component_boxes
    else:
        boxes = fallback_line_boxes_from_projection(ink_mask)

    return [
        LineBox(index=index, x=x, y=y, w=w, h=h)
        for index, (x, y, w, h) in enumerate(sorted(boxes, key=lambda box: (box[1], box[0])), start=1)
    ]


def segment_lines_from_mask(segmentation_mask: np.ndarray) -> List[LineBox]:
    return segment_lines_from_ink_mask(segmentation_mask_to_ink_mask(segmentation_mask))


def fallback_line_boxes_from_projection(ink_mask: np.ndarray) -> List[tuple[int, int, int, int]]:
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

    boxes: List[tuple[int, int, int, int]] = []
    for top, bottom in bands:
        if bottom - top + 1 < min_line_height:
            continue

        candidate = line_box_from_row_band(ink_mask, top, bottom, horizontal_pad, vertical_pad)
        if candidate is None:
            continue

        x, y, w, h = candidate
        roi = ink_mask[y : y + h, x : x + w]
        ink_pixels = int(np.count_nonzero(roi))
        if w < min_line_width or h < min_line_height:
            continue
        if ink_pixels < max(18, w * 0.018):
            continue
        if looks_like_centered_header(candidate, width, height):
            continue

        boxes.append(candidate)

    return merge_overlapping_line_boxes(boxes, height)


def line_boxes_from_profile_peaks(ink_mask: np.ndarray) -> List[tuple[int, int, int, int]]:
    height, width = ink_mask.shape[:2]
    row_counts = np.count_nonzero(ink_mask, axis=1).astype(np.float32)
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

    boxes: List[tuple[int, int, int, int]] = []
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
    return drop_edge_line_artifacts(boxes, ink_mask.shape[:2])


def drop_isolated_top_headers(
    boxes: List[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
) -> List[tuple[int, int, int, int]]:
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
    boxes: List[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
) -> List[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    height, width = image_shape
    filtered: List[tuple[int, int, int, int]] = []
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
) -> List[int]:
    candidates: List[tuple[int, float]] = []
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

    selected: List[int] = []
    for peak_y, _value in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(peak_y - existing) >= min_distance for existing in selected):
            selected.append(peak_y)

    return sorted(selected)


def merge_profile_peaks_by_valleys(
    smoothed: np.ndarray,
    peaks: List[int],
    min_distance: int,
) -> List[int]:
    if len(peaks) < 2:
        return peaks

    merged = list(peaks)
    changed = True
    while changed and len(merged) >= 2:
        changed = False
        result: List[int] = []
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


def profile_valleys_between_peaks(smoothed: np.ndarray, peaks: List[int]) -> List[int]:
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
    peaks: List[int],
    valleys: List[int],
    peak_threshold: float,
) -> List[tuple[int, int]]:
    height = len(smoothed)
    bands: List[tuple[int, int]] = []
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


def line_boxes_from_component_centers(ink_mask: np.ndarray) -> List[tuple[int, int, int, int]]:
    height, width = ink_mask.shape[:2]
    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    components: List[tuple[int, int, int, int, int, float]] = []

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
    groups: List[List[tuple[int, int, int, int, int, float]]] = []
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
    boxes: List[tuple[int, int, int, int]] = []

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
    return separate_vertical_overlaps(sorted(boxes, key=lambda box: (box[1], box[0])))


def split_merged_component_line_boxes(
    boxes: List[tuple[int, int, int, int]],
    ink_mask: np.ndarray,
) -> List[tuple[int, int, int, int]]:
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
    split_boxes: List[tuple[int, int, int, int]] = []

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

        candidates: List[tuple[int, int, int, int]] = []
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
    boxes: List[tuple[int, int, int, int]],
    ink_mask: np.ndarray,
) -> List[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    _image_height, image_width = ink_mask.shape[:2]
    horizontal_pad = max(10, int(image_width * LINE_HORIZONTAL_PAD_FRACTION))
    min_width = max(35, int(image_width * LINE_MIN_WIDTH_FRACTION))
    trimmed: List[tuple[int, int, int, int]] = []

    for box in boxes:
        x, y, w, h = box
        roi = ink_mask[y : y + h, x : x + w]
        left: int | None = None
        right: int | None = None

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        components: List[tuple[int, int, int]] = []
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
            kept_components: List[tuple[int, int, int]] = []

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


def separate_vertical_overlaps(boxes: List[tuple[int, int, int, int]]) -> List[tuple[int, int, int, int]]:
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


def active_row_bands(active_rows: np.ndarray) -> List[tuple[int, int]]:
    bands: List[tuple[int, int]] = []
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


def merge_close_bands(bands: List[tuple[int, int]], merge_gap: int) -> List[tuple[int, int]]:
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
    bands: List[tuple[int, int]],
    smoothed_profile: np.ndarray,
    image_height: int,
) -> List[tuple[int, int]]:
    if len(bands) < 2:
        return bands

    heights = np.array([bottom - top + 1 for top, bottom in bands], dtype=np.float32)
    median_height = float(np.median(heights))
    tall_threshold = max(median_height * 1.45, image_height * 0.035)
    result: List[tuple[int, int]] = []

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
) -> List[tuple[int, int]]:
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
    boxes: List[tuple[int, int, int, int]],
    image_height: int,
) -> List[tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged: List[tuple[int, int, int, int]] = []
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


def segment_words_from_ink_mask(
    ink_mask: np.ndarray,
    line_boxes: List[LineBox],
) -> List[List[WordBox]]:
    ink_mask = remove_tiny_specks(ink_mask, min_area=WORD_NOISE_MIN_AREA)
    return [segment_words_for_line(ink_mask, line) for line in line_boxes]


def segment_words_from_mask(
    segmentation_mask: np.ndarray,
    line_boxes: List[LineBox],
) -> List[List[WordBox]]:
    return segment_words_from_ink_mask(segmentation_mask_to_ink_mask(segmentation_mask), line_boxes)


def segment_words_for_line(ink_mask: np.ndarray, line: LineBox) -> List[WordBox]:
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

    word_boxes: List[tuple[int, int, int, int, int]] = []
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
        WordBox(line_index=line.index, index=index, x=x, y=y, w=w, h=h)
        for index, (x, y, w, h, _area) in enumerate(word_boxes, start=1)
    ]


def collect_word_components(roi: np.ndarray) -> List[tuple[int, int, int, int, int]]:
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    components: List[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < WORD_NOISE_MIN_AREA or (w <= 1 and h <= 1):
            continue
        components.append((int(x), int(y), int(w), int(h), int(area)))
    return sorted(components, key=lambda component: (component[0], component[1]))


def estimate_word_dilation_width(
    components: List[tuple[int, int, int, int, int]],
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
    components: List[tuple[int, int, int, int, int]],
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
    components: List[tuple[int, int, int, int, int]],
    origin_x: int,
    origin_y: int,
    image_width: int,
    image_height: int,
    line_height: int,
) -> List[tuple[int, int, int, int, int]]:
    if not components:
        return []

    groups: List[List[tuple[int, int, int, int, int]]] = []
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

    boxes: List[tuple[int, int, int, int, int]] = []
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
    boxes: List[tuple[int, int, int, int, int]],
) -> List[tuple[int, int, int, int, int]]:
    if not boxes:
        return []

    merged: List[tuple[int, int, int, int, int]] = [boxes[0]]
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


def draw_line_overlay(image_bgr: np.ndarray, line_boxes: List[LineBox]) -> np.ndarray:
    overlay = image_bgr.copy()
    for line in line_boxes:
        cv2.rectangle(overlay, (line.x, line.y), (line.x + line.w, line.y + line.h), (0, 180, 0), 4)
        cv2.putText(
            overlay,
            f"L{line.index}",
            (line.x, max(30, line.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 80, 255),
            3,
        )
    return overlay


def draw_word_overlay(image_bgr: np.ndarray, word_boxes_by_line: List[List[WordBox]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for words in word_boxes_by_line:
        for word in words:
            cv2.rectangle(overlay, (word.x, word.y), (word.x + word.w, word.y + word.h), (255, 90, 0), 2)
            cv2.putText(
                overlay,
                f"W{word.index}",
                (word.x, max(24, word.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 90, 0),
                2,
            )
    return overlay


def segment_lines(
    image_bgr: np.ndarray,
    min_line_width: int = 900,
    min_line_height: int = 80,
    max_line_height: int = 1400,
    horizontal_margin: int = 60,
    profile_threshold_ratio: float = 0.0,
    y_gap_tolerance: int = 18,
    min_ink_ratio: float = 0.0012,
    max_ink_ratio: float = 0.08,
    method: str = "components",
    expected_rows: int | None = None,
    word_gap: int | None = None,
    split_overlapping_lines: bool = True,
) -> SegmentationArtifacts:
    enhanced_gray = clean_enhance_image(image_bgr)
    binary = binarize_image(enhanced_gray)
    skew_angle = estimate_skew_angle(binary)

    deskewed_bgr = rotate_image(image_bgr, skew_angle)
    deskewed_enhanced = clean_enhance_image(deskewed_bgr)
    recognition_bgr = prepare_recognition_ready_image(deskewed_bgr)
    deskewed_binary = binarize_image(deskewed_enhanced)

    line_boxes = segment_lines_from_ink_mask(deskewed_binary)
    word_boxes_by_line = segment_words_from_ink_mask(deskewed_binary, line_boxes)
    overlay = draw_line_overlay(recognition_bgr, line_boxes)
    overlay = draw_word_overlay(overlay, word_boxes_by_line)

    return SegmentationArtifacts(
        grayscale=deskewed_enhanced,
        enhanced_gray=deskewed_enhanced,
        recognition_bgr=recognition_bgr,
        binary=deskewed_binary,
        deskewed_bgr=deskewed_bgr,
        overlay=overlay,
        line_boxes=line_boxes,
        word_boxes_by_line=word_boxes_by_line,
        deskew_angle=skew_angle,
    )


def crop_line_images(image_bgr: np.ndarray, line_boxes: List[LineBox], pad: int = 6) -> List[np.ndarray]:
    crops = []
    height, width = image_bgr.shape[:2]
    for line in line_boxes:
        horizontal_pad = max(30, pad + 14)
        vertical_pad = max(22, pad + 10)
        x1 = max(0, line.x - horizontal_pad)
        y1 = max(0, line.y - vertical_pad)
        x2 = min(width, line.x + line.w + horizontal_pad)
        y2 = min(height, line.y + line.h + vertical_pad)
        crops.append(image_bgr[y1:y2, x1:x2].copy())
    return crops


def save_line_crops(
    image_bgr: np.ndarray,
    line_boxes: List[LineBox],
    output_dir: str | Path,
    stem: str = "page",
    pad: int = 6,
) -> List[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for line, crop in zip(line_boxes, crop_line_images(image_bgr, line_boxes, pad=pad)):
        save_path = output_path / f"{stem}_line_{line.index:03d}.png"
        cv2.imwrite(str(save_path), crop)
        saved_paths.append(save_path)

    return saved_paths


def crop_word_images(
    image_bgr: np.ndarray,
    word_boxes: List[WordBox],
    pad: int = 0,
    binary: np.ndarray | None = None,
) -> List[np.ndarray]:
    crops = []
    height, width = image_bgr.shape[:2]
    for word in word_boxes:
        x1 = max(0, word.x - pad)
        y1 = max(0, word.y - pad)
        x2 = min(width, word.x + word.w + pad)
        y2 = min(height, word.y + word.h + pad)

        if binary is not None and word.component_boxes:
            if len(image_bgr.shape) == 2:
                crop = np.full((y2 - y1, x2 - x1), 255, dtype=image_bgr.dtype)
            else:
                crop = np.full((y2 - y1, x2 - x1, image_bgr.shape[2]), 255, dtype=image_bgr.dtype)

            for cx, cy, cw, ch in word.component_boxes:
                ix1 = max(x1, cx - pad)
                iy1 = max(y1, cy - pad)
                ix2 = min(x2, cx + cw + pad)
                iy2 = min(y2, cy + ch + pad)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                src_region = image_bgr[iy1:iy2, ix1:ix2]
                mask_region = binary[iy1:iy2, ix1:ix2] > 0
                local_y1 = iy1 - y1
                local_y2 = iy2 - y1
                local_x1 = ix1 - x1
                local_x2 = ix2 - x1

                if len(crop.shape) == 2:
                    crop_region = crop[local_y1:local_y2, local_x1:local_x2]
                    crop_region[mask_region] = src_region[mask_region]
                else:
                    crop_region = crop[local_y1:local_y2, local_x1:local_x2]
                    crop_region[mask_region] = src_region[mask_region]

            crops.append(crop)
        else:
            crops.append(image_bgr[y1:y2, x1:x2].copy())
    return crops


def normalize_word_crop_for_model(
    word_crop_bgr: np.ndarray,
    canvas_size: tuple[int, int] = (128, 128),
    max_content_size: tuple[int, int] = (116, 116),
    ink_threshold: int = 245,
) -> np.ndarray:
    """Match the V4 training image format: crop ink, preserve ratio, center on white."""
    canvas_w, canvas_h = canvas_size
    max_w, max_h = max_content_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if word_crop_bgr is None or word_crop_bgr.size == 0:
        return canvas

    if len(word_crop_bgr.shape) == 2:
        crop_bgr = cv2.cvtColor(word_crop_bgr, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = word_crop_bgr.copy()

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < ink_threshold
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        crop_bgr = crop_bgr[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]

    crop_h, crop_w = crop_bgr.shape[:2]
    if crop_w <= 0 or crop_h <= 0:
        return canvas

    scale = min(max_w / crop_w, max_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=interpolation)

    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def resize_word_crop_for_model(
    word_crop_bgr: np.ndarray,
    canvas_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Old prediction behavior: directly resize the detected word box to model size."""
    canvas_w, canvas_h = canvas_size
    if word_crop_bgr is None or word_crop_bgr.size == 0:
        return np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if len(word_crop_bgr.shape) == 2:
        crop_bgr = cv2.cvtColor(word_crop_bgr, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = word_crop_bgr.copy()

    crop_h, crop_w = crop_bgr.shape[:2]
    interpolation = cv2.INTER_AREA if crop_w > canvas_w or crop_h > canvas_h else cv2.INTER_CUBIC
    return cv2.resize(crop_bgr, (canvas_w, canvas_h), interpolation=interpolation)
