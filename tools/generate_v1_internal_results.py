import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = PROJECT_ROOT / "models" / "greggspeak_all_words.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels_all_words.txt"
TEST_FOLDER = PROJECT_ROOT / "datasets" / "all_words_testing_uniform"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

OUTPUT_CSV = REPORTS_DIR / "testing1_internal_results_v1.csv"
INCORRECT_CSV = REPORTS_DIR / "incorrect_predictions_v1.csv"
OUTPUT_SUMMARY = REPORTS_DIR / "testing1_internal_summary_v1.txt"


def load_labels(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_true_label(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def preprocess_image(img_path: Path) -> np.ndarray:
    img_bytes = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img.numpy()


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    if not TEST_FOLDER.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_FOLDER}")

    class_names = load_labels(LABELS_PATH)
    class_set = set(class_names)
    image_files = sorted(
        p
        for p in TEST_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not image_files:
        raise RuntimeError(f"No image files found in {TEST_FOLDER}")

    missing_labels = sorted({extract_true_label(p.name) for p in image_files} - class_set)
    skipped_files = [p.name for p in image_files if extract_true_label(p.name) not in class_set]
    if skipped_files:
        image_files = [p for p in image_files if extract_true_label(p.name) in class_set]

    print(f"Loading V1 model: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    print(f"Loaded {len(class_names)} labels and {len(image_files)} comparable test images.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files with labels absent from V1: {', '.join(skipped_files[:10])}")

    results = []
    incorrect = []
    misclassified_counts: dict[str, int] = {}
    correct_top1 = 0
    correct_top5 = 0

    for start in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[start : start + BATCH_SIZE]
        batch = np.stack([preprocess_image(path) for path in batch_files], axis=0)
        preds_batch = model.predict(batch, verbose=0)

        for offset, (img_path, preds) in enumerate(zip(batch_files, preds_batch), start=1):
            row_no = start + offset
            true_label = extract_true_label(img_path.name)
            pred_idx = int(np.argmax(preds))
            predicted_label = class_names[pred_idx]
            confidence = float(preds[pred_idx]) * 100.0
            top5_idx = preds.argsort()[-5:][::-1]
            top5_labels = [class_names[i] for i in top5_idx]

            is_top1 = predicted_label == true_label
            is_top5 = true_label in top5_labels
            correct_top1 += int(is_top1)
            correct_top5 += int(is_top5)

            if not is_top1:
                incorrect.append(
                    {
                        "image_file": img_path.name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": f"{confidence:.2f}",
                    }
                )
                misclassified_counts[true_label] = misclassified_counts.get(true_label, 0) + 1

            results.append(
                {
                    "no": row_no,
                    "image_file": img_path.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence_percent": f"{confidence:.2f}",
                    "top1_result": "Yes" if is_top1 else "No",
                    "top5_result": "Yes" if is_top5 else "No",
                    "remarks": "Correct" if is_top1 else f"Misclassified as {predicted_label}",
                }
            )

        print(f"Processed {min(start + BATCH_SIZE, len(image_files))}/{len(image_files)}")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "no",
                "image_file",
                "true_label",
                "predicted_label",
                "confidence_percent",
                "top1_result",
                "top5_result",
                "remarks",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    with INCORRECT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(incorrect)

    total = len(results)
    top1_acc = correct_top1 / total * 100.0 if total else 0.0
    top5_acc = correct_top5 / total * 100.0 if total else 0.0
    most_misclassified = sorted(misclassified_counts.items(), key=lambda item: item[1], reverse=True)[:10]

    summary_lines = [
        "Testing 1 - Internal Testing Summary (V1)",
        "=========================================",
        f"Model Path: {MODEL_PATH.resolve()}",
        f"Labels Path: {LABELS_PATH.resolve()}",
        f"Test Folder: {TEST_FOLDER.resolve()}",
        f"Total Images Tested: {total}",
        f"Skipped Images Not In V1 Labels: {len(skipped_files)}",
        f"Skipped Labels: {', '.join(missing_labels) if missing_labels else 'None'}",
        f"Top-1 Correct: {correct_top1}",
        f"Top-1 Incorrect: {len(incorrect)}",
        f"Top-5 Correct: {correct_top5}",
        f"Top-1 Accuracy: {top1_acc:.2f}%",
        f"Top-5 Accuracy: {top5_acc:.2f}%",
        f"Incorrect Predictions File: {INCORRECT_CSV.resolve()}",
        f"Results CSV: {OUTPUT_CSV.resolve()}",
        "",
        "Most Misclassified Words:",
        "-------------------------",
    ]
    summary_lines.extend(f"{word}: {count} errors" for word, count in most_misclassified)
    OUTPUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
