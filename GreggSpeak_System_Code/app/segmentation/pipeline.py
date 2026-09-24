import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import USE_UNIFORM_CROP
from app.segmentation import laptop_gui_segmentation as gui_segmentation
from app.segmentation.line_segmentation import (
    LineBox,
    WordBox,
    normalize_word_crop_for_model,
    resize_word_crop_for_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CROP_POINTS_FILE = PROJECT_ROOT / "calibration_crop_points.json"
POINT_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass
class WordCrop:
    line_index: int
    word_index: int
    image_rgb: np.ndarray
    raw_bgr: np.ndarray
    box: WordBox


@dataclass
class SegmentedPage:
    words: list[WordCrop]
    rows_detected: int
    words_detected: int
    source_bgr: np.ndarray
    deskewed_bgr: np.ndarray
    binary: np.ndarray
    line_boxes: list[LineBox]
    word_boxes_by_line: list[list[WordBox]]
    enhanced_gray: np.ndarray | None = None
    recognition_bgr: np.ndarray | None = None
    line_overlay_bgr: np.ndarray | None = None
    word_overlay_bgr: np.ndarray | None = None


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def segment_page(
    image: Image.Image,
    use_uniform_crop: bool = USE_UNIFORM_CROP,
    laptop_compatible: bool = False,
    apply_fixed_crop: bool | None = None,
) -> SegmentedPage:
    image_bgr = pil_to_bgr(image)
    if apply_fixed_crop is None:
        apply_fixed_crop = laptop_compatible
    processing_bgr = fixed_calibration_crop(image_bgr) if apply_fixed_crop else image_bgr
    artifacts = _segment_page_with_laptop_gui_logic(processing_bgr)
    word_crops: list[WordCrop] = []
    recognition_bgr = getattr(artifacts, "recognition_bgr", artifacts.deskewed_bgr)

    for line_index, word_boxes in enumerate(artifacts.word_boxes_by_line, start=1):
        for word_box in word_boxes:
            raw_crop = gui_segmentation.crop_word_from_recognition_source(
                recognition_bgr,
                word_box,
            )
            model_crop = _prepare_crop(raw_crop, use_uniform_crop)
            image_rgb = cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            word_crops.append(
                WordCrop(
                    line_index=line_index,
                    word_index=word_box.index,
                    image_rgb=image_rgb,
                    raw_bgr=raw_crop,
                    box=word_box,
                )
            )

    if not word_crops:
        fallback_crop = _prepare_crop(recognition_bgr, use_uniform_crop)
        image_rgb = cv2.cvtColor(fallback_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        fallback_box = WordBox(
            line_index=1,
            index=1,
            x=0,
            y=0,
            w=recognition_bgr.shape[1],
            h=recognition_bgr.shape[0],
        )
        word_crops.append(
            WordCrop(
                line_index=1,
                word_index=1,
                image_rgb=image_rgb,
                raw_bgr=recognition_bgr,
                box=fallback_box,
            )
        )

    return SegmentedPage(
        words=word_crops,
        rows_detected=len(artifacts.line_boxes),
        words_detected=len(word_crops),
        source_bgr=processing_bgr,
        deskewed_bgr=recognition_bgr,
        binary=artifacts.binary,
        line_boxes=artifacts.line_boxes,
        word_boxes_by_line=artifacts.word_boxes_by_line,
        enhanced_gray=getattr(artifacts, "enhanced_gray", None),
        recognition_bgr=recognition_bgr,
        line_overlay_bgr=getattr(artifacts, "line_overlay_bgr", None),
        word_overlay_bgr=getattr(artifacts, "word_overlay_bgr", None),
    )


def _prepare_crop(word_crop_bgr: np.ndarray, use_uniform_crop: bool) -> np.ndarray:
    if use_uniform_crop:
        return normalize_word_crop_for_model(word_crop_bgr)
    return resize_word_crop_for_model(word_crop_bgr)


def fixed_calibration_crop(image_bgr: np.ndarray) -> np.ndarray:
    if not CROP_POINTS_FILE.exists():
        raise FileNotFoundError(f"Could not find {CROP_POINTS_FILE}")
    height, width = image_bgr.shape[:2]
    crop_points = read_crop_points(CROP_POINTS_FILE, width, height)
    return perspective_crop(image_bgr, crop_points)


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


def _segment_page_with_laptop_gui_logic(image_bgr: np.ndarray):
    """Match tools/line_segmentation_gui.py processing for Upload Test parity."""
    enhanced_gray = gui_segmentation.clean_enhance_image(image_bgr)
    recognition_bgr = gui_segmentation.prepare_recognition_ready_image(image_bgr)
    segmentation_mask = gui_segmentation.create_segmentation_mask(enhanced_gray)
    gui_line_boxes = gui_segmentation.segment_lines_from_mask(segmentation_mask)
    gui_word_boxes = gui_segmentation.segment_words_from_lines(segmentation_mask, gui_line_boxes)

    line_boxes = [
        LineBox(index=line.index, x=line.x, y=line.y, w=line.w, h=line.h)
        for line in gui_line_boxes
    ]
    word_boxes_by_line: list[list[WordBox]] = []
    for words in gui_word_boxes:
        converted_words = [
            WordBox(
                line_index=word.line_index,
                index=word.word_index,
                x=word.x,
                y=word.y,
                w=word.w,
                h=word.h,
            )
            for word in words
        ]
        word_boxes_by_line.append(converted_words)

    line_overlay = gui_segmentation.draw_line_segmentation_overlay(segmentation_mask, gui_line_boxes)
    word_overlay = gui_segmentation.draw_word_segmentation_overlay(
        segmentation_mask,
        gui_line_boxes,
        gui_word_boxes,
    )
    return type(
        "LaptopGuiSegmentationArtifacts",
        (),
        {
            "grayscale": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
            "enhanced_gray": enhanced_gray,
            "recognition_bgr": recognition_bgr,
            "binary": segmentation_mask,
            "deskewed_bgr": recognition_bgr,
            "overlay": line_overlay,
            "line_overlay_bgr": line_overlay,
            "word_overlay_bgr": word_overlay,
            "line_boxes": line_boxes,
            "word_boxes_by_line": word_boxes_by_line,
            "deskew_angle": 0.0,
        },
    )()
