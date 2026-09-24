import argparse
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
DEFAULT_HANDWRITTEN_FOLDER = PROJECT_ROOT / "datasets" / "handwrittens"
REPORTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_labels(labels_path: Path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V3 on a handwritten dataset folder.")
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_HANDWRITTEN_FOLDER,
        help="Folder containing handwritten images. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--output-prefix",
        default="testing2_handwritten",
        help="Prefix for report filenames saved under reports/.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Keras model path. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH,
        help="Label list path. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--title",
        default="Testing 2 - Handwritten Testing Summary",
        help="Summary title without the model suffix.",
    )
    parser.add_argument(
        "--model-name",
        default="V3",
        help="Model name shown in the summary.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path):
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def extract_true_label(img_path: Path, class_names):
    """Use folder labels when possible; otherwise use word label from filename.

    The current handwritten dataset is grouped by alphabet folders (A, B, C...)
    while word labels are stored in filenames such as ability_101.png.
    """
    folder_label = img_path.parent.name.strip()
    if folder_label in class_names:
        return folder_label, "folder"

    stem = img_path.stem
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1]), "filename"
    return stem, "filename"


def preprocess_image_exact(img_path: Path):
    # Training stored raw 0-255 RGB values; MobileNetV2 preprocessing is inside the saved model.
    img_bytes = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img


def main():
    args = parse_args()
    handwritten_folder = resolve_project_path(args.folder)
    model_path = resolve_project_path(args.model)
    labels_path = resolve_project_path(args.labels)
    model_suffix = args.model_name.lower()
    output_csv = REPORTS_DIR / f"{args.output_prefix}_results_{model_suffix}.csv"
    output_summary = REPORTS_DIR / f"{args.output_prefix}_summary_{model_suffix}.txt"
    incorrect_csv = REPORTS_DIR / f"{args.output_prefix}_incorrect_{model_suffix}.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not handwritten_folder.exists():
        raise FileNotFoundError(f"Handwritten folder not found: {handwritten_folder}")

    print("Loading model...")
    model = load_model(model_path)

    print("Loading labels...")
    class_names = load_labels(labels_path)
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    print("Model output shape:", model.output_shape)
    print("Number of labels:", len(class_names))

    image_files = sorted(
        p
        for p in handwritten_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        raise RuntimeError(f"No image files found in: {handwritten_folder}")

    print(f"Found {len(image_files)} handwritten testing images.")

    results = []
    incorrect_list = []
    unknown_labels = []
    x_images = []
    y_indices = []

    for img_path in image_files:
        true_label, label_source = extract_true_label(img_path, class_names)
        rel_path = img_path.relative_to(handwritten_folder)

        if true_label not in class_to_index:
            unknown_labels.append((img_path, true_label, label_source))
            continue

        x_images.append(preprocess_image_exact(img_path))
        y_indices.append(class_to_index[true_label])
        results.append(
            {
                "image_file": str(rel_path),
                "true_label": true_label,
                "label_source": label_source,
                "true_index": class_to_index[true_label],
            }
        )

    if x_images:
        x = tf.stack(x_images, axis=0)
        y = tf.convert_to_tensor(y_indices, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE)

        print("Running predictions...")
        probs = model.predict(ds.map(lambda batch_x, batch_y: batch_x), verbose=1)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        test_loss = float(loss_fn(y, probs).numpy())
    else:
        probs = np.empty((0, len(class_names)), dtype=np.float32)
        test_loss = None

    correct_top1 = 0
    correct_top5 = 0

    for row, true_idx, pred_probs in zip(results, y_indices, probs):
        pred_idx = int(np.argmax(pred_probs))
        top5_idx = pred_probs.argsort()[-5:][::-1]

        predicted_label = class_names[pred_idx]
        confidence = float(pred_probs[pred_idx]) * 100.0
        top5_labels = [class_names[i] for i in top5_idx]
        top5_conf = [float(pred_probs[i]) * 100.0 for i in top5_idx]

        is_top1 = pred_idx == true_idx
        is_top5 = true_idx in top5_idx

        if is_top1:
            correct_top1 += 1
        if is_top5:
            correct_top5 += 1

        row.update(
            {
                "predicted_label": predicted_label,
                "confidence_percent": f"{confidence:.2f}",
                "top5_predictions": "; ".join(
                    f"{label} ({conf:.2f}%)"
                    for label, conf in zip(top5_labels, top5_conf)
                ),
                "top1_result": "Yes" if is_top1 else "No",
                "top5_result": "Yes" if is_top5 else "No",
                "remarks": "Correct" if is_top1 else f"Misclassified as {predicted_label}",
            }
        )

        if not is_top1:
            incorrect_list.append(
                {
                    "image_file": row["image_file"],
                    "true_label": row["true_label"],
                    "predicted_label": predicted_label,
                    "confidence_percent": f"{confidence:.2f}",
                    "top5_predictions": row["top5_predictions"],
                }
            )

    for img_path, true_label, label_source in unknown_labels:
        rel_path = img_path.relative_to(handwritten_folder)
        row = {
            "image_file": str(rel_path),
            "true_label": true_label,
            "label_source": label_source,
            "true_index": "",
            "predicted_label": "",
            "confidence_percent": "",
            "top5_predictions": "",
            "top1_result": "No",
            "top5_result": "No",
            "remarks": "Label not present in V3 class list",
        }
        results.append(row)
        incorrect_list.append(
            {
                "image_file": str(rel_path),
                "true_label": true_label,
                "predicted_label": "",
                "confidence_percent": "",
                "top5_predictions": "",
            }
        )

    results = sorted(results, key=lambda row: row["image_file"].lower())
    incorrect_list = sorted(incorrect_list, key=lambda row: row["image_file"].lower())

    total = len(image_files)
    top1_acc = (correct_top1 / total * 100.0) if total else 0.0
    top5_acc = (correct_top5 / total * 100.0) if total else 0.0
    error_rate = 100.0 - top1_acc

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_file",
                "true_label",
                "label_source",
                "true_index",
                "predicted_label",
                "confidence_percent",
                "top5_predictions",
                "top1_result",
                "top5_result",
                "remarks",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    with open(incorrect_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_file",
                "true_label",
                "predicted_label",
                "confidence_percent",
                "top5_predictions",
            ],
        )
        writer.writeheader()
        writer.writerows(incorrect_list)

    loss_text = f"{test_loss:.4f}" if test_loss is not None else "N/A"
    if unknown_labels:
        loss_note = f" computed on {len(y_indices)} class-matched images"
    else:
        loss_note = ""

    summary_text = (
        f"# {args.title} ({args.model_name})\n\n"
        f"Total Images Tested: {total}\n"
        f"Top-1 Accuracy: {top1_acc:.2f}%\n"
        f"Top-5 Accuracy: {top5_acc:.2f}%\n"
        f"Test Loss: {loss_text}{loss_note}\n"
        f"Remarks: Error rate is {error_rate:.2f}%; performance is lower than Testing 1 due to handwriting "
        "variability, segmentation inconsistencies, and mismatch with the clean training/test distribution.\n"
    )

    with open(output_summary, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nDone.")
    print(summary_text)


if __name__ == "__main__":
    main()
