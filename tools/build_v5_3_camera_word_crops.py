from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "reports" / "demo_v5_3_new_batch_word_exports"
DEFAULT_TRANSCRIPT = PROJECT_ROOT / "data" / "demo_v5_transcript.json"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "labels_all_words_v5_2.txt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "demo_v5_3_camera_word_crops"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "reports" / "demo_v5_3_new_batch_label_review"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_labels(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def clean_words(line: str) -> list[str]:
    text = line.replace(":", " ")
    text = re.sub(r"[^a-zA-Z0-9_']+", " ", text)
    return [word.lower().strip("'") for word in text.split() if word.strip("'")]


def phrase_labels(words: list[str], label_set: set[str], max_phrase_words: int = 5) -> list[str]:
    labels: list[str] = []
    index = 0
    while index < len(words):
        matched = None
        max_length = min(max_phrase_words, len(words) - index)
        for length in range(max_length, 0, -1):
            candidate = "_".join(words[index:index + length])
            if candidate in label_set:
                matched = candidate
                index += length
                break
        if matched is None:
            matched = words[index]
            index += 1
        labels.append(matched)
    return labels


def load_expected_pages(transcript_path: Path, label_set: set[str]) -> dict[int, list[list[str]]]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    pages: dict[int, list[list[str]]] = {}
    for page in data["pages"]:
        page_number = int(page["page"])
        pages[page_number] = [
            phrase_labels(clean_words(line), label_set)
            for line in page["lines"]
        ]
    return pages


def crop_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"line_(\d+)_word_(\d+)", path.stem)
    if not match:
        return (999999, 999999)
    return (int(match.group(1)), int(match.group(2)))


def find_line_crops(page_dir: Path) -> dict[int, list[Path]]:
    line_to_crops: dict[int, list[Path]] = defaultdict(list)
    crop_dir = page_dir / "word_crops"
    if not crop_dir.exists():
        return line_to_crops
    for crop_path in sorted(crop_dir.iterdir(), key=crop_sort_key):
        if not crop_path.is_file() or crop_path.suffix.lower() not in IMAGE_EXTS:
            continue
        line_index, _word_index = crop_sort_key(crop_path)
        if line_index != 999999:
            line_to_crops[line_index].append(crop_path)
    return line_to_crops


def copy_labeled_crop(crop_path: Path, label: str, page_number: int, line_number: int, label_position: int, out_dir: Path) -> Path:
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing_count = len([path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS])
    destination = label_dir / (
        f"v53_page_{page_number:03d}_line_{line_number:03d}_"
        f"word_{label_position:03d}_{label}_{existing_count + 1:04d}{crop_path.suffix.lower()}"
    )
    shutil.copy2(crop_path, destination)
    return destination


def write_expected_words(path: Path, labels: list[str], transcript_line: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if transcript_line:
        lines.append(f"Transcript: {transcript_line}")
    lines.append(f"Expected label count: {len(labels)}")
    lines.append("Expected labels:")
    lines.append(" ".join(f"{index:02d}_{label}" for index, label in enumerate(labels, start=1)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    export_dir: Path,
    transcript_path: Path,
    labels_path: Path,
    out_dir: Path,
    review_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    if not export_dir.exists():
        raise FileNotFoundError(f"Export folder not found: {export_dir}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if overwrite and review_dir.exists():
        shutil.rmtree(review_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    label_set = load_labels(labels_path)
    transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    expected_pages = load_expected_pages(transcript_path, label_set)
    page_lines = {
        int(page["page"]): list(page["lines"])
        for page in transcript_data["pages"]
    }

    copied = 0
    skipped_lines = []
    unknown_labels = Counter()
    label_counts = Counter()

    for page_number in sorted(expected_pages):
        page_dir = export_dir / f"Page_{page_number:04d}"
        line_to_crops = find_line_crops(page_dir)
        page_review = review_dir / f"Page_{page_number:04d}"
        page_review.mkdir(parents=True, exist_ok=True)

        for line_number, expected_labels in enumerate(expected_pages[page_number], start=1):
            transcript_line = page_lines[page_number][line_number - 1]
            line_review = page_review / f"line_{line_number:03d}"
            write_expected_words(line_review / "_expected_labels.txt", expected_labels, transcript_line)

            for label in expected_labels:
                if label not in label_set:
                    unknown_labels[label] += 1

            crops = line_to_crops.get(line_number, [])
            if len(crops) != len(expected_labels):
                skipped_lines.append(
                    {
                        "page": page_number,
                        "line": line_number,
                        "expected_labels": len(expected_labels),
                        "detected_crops": len(crops),
                        "reason": "crop_count_mismatch",
                    }
                )
                for crop_path in crops:
                    shutil.copy2(crop_path, line_review / crop_path.name)
                continue

            if any(label not in label_set for label in expected_labels):
                skipped_lines.append(
                    {
                        "page": page_number,
                        "line": line_number,
                        "expected_labels": len(expected_labels),
                        "detected_crops": len(crops),
                        "reason": "unknown_label",
                    }
                )
                continue

            for label_position, (crop_path, label) in enumerate(zip(crops, expected_labels), start=1):
                copy_labeled_crop(crop_path, label, page_number, line_number, label_position, out_dir)
                shutil.copy2(crop_path, line_review / f"{label_position:02d}_{label}{crop_path.suffix.lower()}")
                copied += 1
                label_counts[label] += 1

    summary = {
        "export_dir": str(export_dir),
        "transcript": str(transcript_path),
        "labels": str(labels_path),
        "out_dir": str(out_dir),
        "review_dir": str(review_dir),
        "copied_images": copied,
        "unique_labels_copied": len(label_counts),
        "unknown_labels": dict(sorted(unknown_labels.items())),
        "skipped_line_count": len(skipped_lines),
        "skipped_lines": skipped_lines,
        "label_counts": dict(sorted(label_counts.items())),
    }
    (review_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict v5.3 class folders from exported new-batch page word crops."
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        export_dir=args.export_dir,
        transcript_path=args.transcript,
        labels_path=args.labels,
        out_dir=args.out_dir,
        review_dir=args.review_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
