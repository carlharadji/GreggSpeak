import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# CONFIG (V3)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = PROJECT_ROOT / "models" / "greggspeak_all_words_v3.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels_all_words_v3.txt"
TEST_FOLDER = PROJECT_ROOT / "datasets" / "all_words_testing"
REPORTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)

OUTPUT_CSV = REPORTS_DIR / "testing1_internal_results_v3.csv"
OUTPUT_SUMMARY = REPORTS_DIR / "testing1_internal_summary_v3.txt"
INCORRECT_CSV = REPORTS_DIR / "incorrect_predictions_v3.csv"

# =========================
# LOAD LABELS
# =========================
def load_labels(labels_path: Path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# =========================
# PREPROCESS
# =========================
def preprocess_image_exact(img_path: Path):
    img_bytes = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

# =========================
# EXTRACT LABEL
# =========================
def extract_true_label(filename: str):
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return stem

# =========================
# MAIN
# =========================
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    if not TEST_FOLDER.exists():
        raise FileNotFoundError(f"Testing folder not found: {TEST_FOLDER}")

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading labels...")
    class_names = load_labels(LABELS_PATH)

    print("Model output shape:", model.output_shape)
    print("Number of labels:", len(class_names))

    image_files = sorted([
        p for p in TEST_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ])

    if not image_files:
        raise RuntimeError(f"No image files found in: {TEST_FOLDER}")

    print(f"Found {len(image_files)} testing images.")

    results = []
    incorrect_list = []
    misclassified_counts = {}

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for i, img_path in enumerate(image_files, start=1):
        true_label = extract_true_label(img_path.name)

        try:
            arr = preprocess_image_exact(img_path)
            preds = model.predict(arr, verbose=0)[0]

            # Top-1
            pred_idx = int(np.argmax(preds))
            predicted_label = class_names[pred_idx]
            confidence = float(preds[pred_idx]) * 100.0

            # Top-5
            top5_idx = preds.argsort()[-5:][::-1]
            top5_labels = [class_names[i] for i in top5_idx]

            is_top1 = predicted_label == true_label
            is_top5 = true_label in top5_labels

            if is_top1:
                correct_top1 += 1
            if is_top5:
                correct_top5 += 1

            total += 1

            # Record incorrect
            if not is_top1:
                incorrect_list.append({
                    "image_file": img_path.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": f"{confidence:.2f}"
                })

                misclassified_counts[true_label] = misclassified_counts.get(true_label, 0) + 1

            results.append({
                "no": i,
                "image_file": img_path.name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence_percent": f"{confidence:.2f}",
                "top1_result": "Yes" if is_top1 else "No",
                "top5_result": "Yes" if is_top5 else "No",
                "remarks": "Correct" if is_top1 else f"Misclassified as {predicted_label}"
            })

            if i % 50 == 0:
                print(f"Processed {i}/{len(image_files)}")

        except Exception as e:
            results.append({
                "no": i,
                "image_file": img_path.name,
                "true_label": true_label,
                "predicted_label": "",
                "confidence_percent": "",
                "top1_result": "No",
                "top5_result": "No",
                "remarks": f"Error: {str(e)}"
            })

    # =========================
    # SAVE MAIN RESULTS
    # =========================
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "no",
            "image_file",
            "true_label",
            "predicted_label",
            "confidence_percent",
            "top1_result",
            "top5_result",
            "remarks"
        ])
        writer.writeheader()
        writer.writerows(results)

    # =========================
    # SAVE INCORRECT PREDICTIONS
    # =========================
    with open(INCORRECT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_file",
            "true_label",
            "predicted_label",
            "confidence"
        ])
        writer.writeheader()
        writer.writerows(incorrect_list)

    # =========================
    # SUMMARY
    # =========================
    top1_acc = (correct_top1 / total * 100.0) if total else 0.0
    top5_acc = (correct_top5 / total * 100.0) if total else 0.0

    sorted_misclassified = sorted(
        misclassified_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    summary_text = (
        "Testing 1 - Internal Testing Summary (V3)\n"
        "=========================================\n"
        f"Total Images Tested: {total}\n"
        f"Top-1 Correct: {correct_top1}\n"
        f"Top-5 Correct: {correct_top5}\n"
        f"Top-1 Accuracy: {top1_acc:.2f}%\n"
        f"Top-5 Accuracy: {top5_acc:.2f}%\n"
        f"Incorrect Predictions File: {INCORRECT_CSV.resolve()}\n"
        f"Results CSV: {OUTPUT_CSV.resolve()}\n\n"
        "Most Misclassified Words:\n"
        "-------------------------\n"
    )

    for word, count in sorted_misclassified:
        summary_text += f"{word}: {count} errors\n"

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nDone.")
    print(summary_text)


if __name__ == "__main__":
    main()
