from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import line_segmentation_gui as lsg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT.parent / "PINASULAT 1500-20260401T124609Z-3-001"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "demo_v5_3_camera_pages" / "raw"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "reports" / "demo_v5_3_camera_word_exports"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "reports" / "demo_v5_3_camera_debug"
DEFAULT_GLOB = "capture_00*_20260518_*.jpg"


def rotate_bgr(image: np.ndarray, degrees: int) -> np.ndarray:
    normalized = degrees % 360
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def choose_upright_rotation(cropped_bgr: np.ndarray) -> tuple[int, dict[str, float]]:
    scores: dict[int, float] = {}
    for degrees in (0, 90, 270):
        candidate = rotate_bgr(cropped_bgr, degrees)
        scores[degrees] = float(lsg.orientation_score(candidate))

    best_degrees = max(scores, key=scores.get)
    score_report = {str(degrees): round(score, 3) for degrees, score in scores.items()}
    return best_degrees, score_report


def fixed_paper_crop(image_bgr: np.ndarray) -> np.ndarray:
    if not lsg.CROP_POINTS_FILE.exists():
        raise FileNotFoundError(f"Missing fixed crop points: {lsg.CROP_POINTS_FILE}")
    height, width = image_bgr.shape[:2]
    crop_points = lsg.read_crop_points(lsg.CROP_POINTS_FILE, width, height)
    return lsg.perspective_crop(image_bgr, crop_points)


def write_line_boxes(path: Path, lines: list[lsg.LineBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["line_index", "x", "y", "w", "h"])
        for line in lines:
            writer.writerow([line.index, line.x, line.y, line.w, line.h])


def write_word_boxes(path: Path, word_boxes_by_line: list[list[lsg.WordBox]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["line_index", "word_index", "x", "y", "w", "h"])
        for words in word_boxes_by_line:
            for word in words:
                writer.writerow([word.line_index, word.word_index, word.x, word.y, word.w, word.h])


def process_page(
    source_path: Path,
    page_index: int,
    raw_dir: Path,
    export_dir: Path,
    debug_dir: Path,
    force_rotate: str,
) -> dict[str, object]:
    page_name = f"Page_{page_index:04d}"
    page_export_dir = export_dir / page_name
    page_debug_dir = debug_dir / page_name
    word_crop_dir = page_export_dir / "word_crops"
    word_crop_dir.mkdir(parents=True, exist_ok=True)
    page_debug_dir.mkdir(parents=True, exist_ok=True)

    original = lsg.load_bgr_image(source_path)
    raw_copy = raw_dir / f"{page_name}{source_path.suffix.lower()}"
    raw_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, raw_copy)

    paper_crop_unoriented = fixed_paper_crop(original)
    if force_rotate == "auto":
        rotation_degrees, orientation_scores = choose_upright_rotation(paper_crop_unoriented)
    else:
        rotation_degrees = int(force_rotate)
        orientation_scores = {}

    oriented_original = rotate_bgr(original, rotation_degrees)
    paper_crop = rotate_bgr(paper_crop_unoriented, rotation_degrees)
    enhanced = lsg.clean_enhance_image(paper_crop)
    recognition_ready = lsg.prepare_recognition_ready_image(paper_crop)
    segmentation_mask = lsg.create_segmentation_mask(enhanced)
    line_boxes = lsg.segment_lines_from_mask(segmentation_mask)
    line_overlay = lsg.draw_line_segmentation_overlay(segmentation_mask, line_boxes)
    word_boxes_by_line = lsg.segment_words_from_lines(segmentation_mask, line_boxes)
    word_overlay = lsg.draw_word_segmentation_overlay(segmentation_mask, line_boxes, word_boxes_by_line)

    lsg.save_image(page_export_dir / "01_original.png", original)
    lsg.save_image(page_export_dir / "02_oriented.png", oriented_original)
    lsg.save_image(page_export_dir / "03_paper_crop.png", paper_crop)
    lsg.save_image(page_export_dir / "04_clean_enhanced.png", enhanced)
    lsg.save_image(page_export_dir / "05_segmentation_mask.png", segmentation_mask)
    lsg.save_image(page_export_dir / "06_line_overlay.png", line_overlay)
    lsg.save_image(page_export_dir / "07_word_overlay.png", word_overlay)

    lsg.save_image(page_debug_dir / "03_paper_crop.png", paper_crop)
    lsg.save_image(page_debug_dir / "04_clean_enhanced.png", enhanced)
    lsg.save_image(page_debug_dir / "05_segmentation_mask.png", segmentation_mask)
    lsg.save_image(page_debug_dir / "06_line_overlay.png", line_overlay)
    lsg.save_image(page_debug_dir / "07_word_overlay.png", word_overlay)
    write_line_boxes(page_debug_dir / "line_boxes.csv", line_boxes)
    write_word_boxes(page_debug_dir / "word_boxes.csv", word_boxes_by_line)

    for words in word_boxes_by_line:
        for word in words:
            crop = lsg.crop_word_from_recognition_source(recognition_ready, word)
            crop_path = word_crop_dir / f"line_{word.line_index:03d}_word_{word.word_index:03d}.png"
            lsg.save_image(crop_path, crop)

    line_count = len(line_boxes)
    word_count = lsg.total_word_count(word_boxes_by_line)
    page_summary = {
        "page": page_name,
        "source_name": source_path.name,
        "source_path": str(source_path),
        "raw_copy": str(raw_copy),
        "rotation_degrees": rotation_degrees,
        "orientation_scores": orientation_scores,
        "detected_line_count": line_count,
        "detected_word_count": word_count,
        "page_output_dir": str(page_export_dir),
        "word_crops_dir": str(word_crop_dir),
        "debug_dir": str(page_debug_dir),
    }
    (page_debug_dir / "summary.json").write_text(json.dumps(page_summary, indent=2), encoding="utf-8")
    return page_summary


def page_index_from_capture_name(path: Path, fallback: int) -> int:
    match = re.search(r"capture_(\d{3})_", path.name)
    if match:
        return int(match.group(1))
    return fallback


def find_source_pages(source_dir: Path, pattern: str) -> list[Path]:
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No capture images found in {source_dir} using pattern {pattern!r}. "
            "Pass --source-dir if the files are somewhere else."
        )
    return files[:7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RPi camera demo page overlays and word crops for V5.3 review."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--pattern", default=DEFAULT_GLOB)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument(
        "--force-rotate",
        choices=("auto", "0", "90", "180", "270"),
        default="0",
        help="Rotate the cropped paper before segmentation. Default 0 matches the fixed RPi camera captures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pages = find_source_pages(args.source_dir, args.pattern)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(args.source_dir),
        "pattern": args.pattern,
        "raw_dir": str(args.raw_dir),
        "export_dir": str(args.export_dir),
        "debug_dir": str(args.debug_dir),
        "page_count": len(source_pages),
        "pages": [],
    }

    for fallback_index, source_path in enumerate(source_pages, start=1):
        index = page_index_from_capture_name(source_path, fallback_index)
        print(f"[{index}/{len(source_pages)}] Exporting {source_path.name}")
        page_summary = process_page(
            source_path=source_path,
            page_index=index,
            raw_dir=args.raw_dir,
            export_dir=args.export_dir,
            debug_dir=args.debug_dir,
            force_rotate=args.force_rotate,
        )
        summary["pages"].append(page_summary)

    args.export_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.export_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
