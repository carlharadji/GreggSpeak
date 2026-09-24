import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "all_words_v4_uniform_clean_only"
DEFAULT_HANDWRITTEN_DIR = PROJECT_ROOT / "datasets" / "allword_handwritten"
DEFAULT_DEMO_DIR = PROJECT_ROOT / "datasets" / "demo_v5_word_crops"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MODEL_OUT = MODELS_DIR / "greggspeak_all_words_v5_demo.keras"
BEST_MODEL_OUT = MODELS_DIR / "best_all_words_v5_demo.keras"
TFLITE_OUT = MODELS_DIR / "greggspeak_all_words_v5_demo.tflite"
LABELS_OUT = MODELS_DIR / "labels_all_words_v5_demo.txt"
HISTORY_OUT = REPORTS_DIR / "training_history_v5_demo.csv"
METRICS_OUT = REPORTS_DIR / "final_metrics_v5_demo.json"
SPLIT_SUMMARY_OUT = REPORTS_DIR / "dataset_split_summary_v5_demo.json"
DEMO_PREDICTIONS_OUT = REPORTS_DIR / "demo_predictions_v5_demo.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GreggSpeak V5 demo model with demo crops strongly oversampled."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--handwritten-dir", type=Path, default=DEFAULT_HANDWRITTEN_DIR)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--include-handwritten", action="store_true", default=True)
    parser.add_argument("--no-handwritten", dest="include_handwritten", action="store_false")
    parser.add_argument("--handwritten-repeat", type=int, default=5)
    parser.add_argument(
        "--demo-repeat",
        type=int,
        default=20,
        help="Repeat verified demo crops in training so the fixed demo pages dominate.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=8)
    parser.add_argument("--epochs-stage2", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tflite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path):
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_environment():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("GPU memory growth warning:", exc)


def collect_class_files(dataset_dir: Path):
    class_to_files = defaultdict(list)
    if not dataset_dir.exists():
        return class_to_files
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                class_to_files[class_dir.name].append(str(path))
    return class_to_files


def split_clean_dataset(clean_files, class_to_index):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    split_summary = {}

    for class_name in sorted(clean_files):
        files = list(clean_files[class_name])
        random.shuffle(files)
        n = len(files)
        if n < 3:
            raise ValueError(f"Clean class '{class_name}' has only {n} image(s). Need at least 3.")

        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_test = max(1, n - n_train - n_val)
        while n_train + n_val + n_test > n:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test -= 1

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        label_idx = class_to_index[class_name]
        train_paths.extend(train_files)
        train_labels.extend([label_idx] * len(train_files))
        val_paths.extend(val_files)
        val_labels.extend([label_idx] * len(val_files))
        test_paths.extend(test_files)
        test_labels.extend([label_idx] * len(test_files))
        split_summary[class_name] = {
            "total_clean": n,
            "train_clean": len(train_files),
            "val_clean": len(val_files),
            "test_clean": len(test_files),
        }

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, split_summary


def add_flat_file_dataset(train_paths, train_labels, folder, class_to_index, repeat):
    if repeat < 1 or not folder.exists():
        return 0, []

    added = 0
    skipped = []
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    for path in files:
        class_name = path.stem
        if class_name not in class_to_index:
            skipped.append(path.name)
            continue
        label_idx = class_to_index[class_name]
        for _ in range(repeat):
            train_paths.append(str(path))
            train_labels.append(label_idx)
            added += 1
    return added, skipped


def add_class_folder_dataset(train_paths, train_labels, class_files, class_to_index, repeat):
    if repeat < 1:
        return 0
    added = 0
    for class_name in sorted(class_files):
        label_idx = class_to_index[class_name]
        for file_path in class_files[class_name]:
            for _ in range(repeat):
                train_paths.append(file_path)
                train_labels.append(label_idx)
                added += 1
    return added


def shuffle_paths_and_labels(paths, labels):
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    if not combined:
        return [], []
    shuffled_paths, shuffled_labels = zip(*combined)
    return list(shuffled_paths), list(shuffled_labels)


def load_image(path, label, num_classes):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def make_dataset(paths, labels, num_classes, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 20000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: load_image(path, label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.003, fill_mode="constant", fill_value=255.0),
            tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="constant", fill_value=255.0),
            tf.keras.layers.RandomZoom(0.04, fill_mode="constant", fill_value=255.0),
            tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.14),
        ],
        name="v5_demo_safe_augment",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
    )


def write_history(history1, history2):
    rows = []
    for stage, history in [("stage1", history1), ("stage2", history2)]:
        for epoch_index in range(len(history.epoch)):
            row = {"stage": stage, "epoch": history.epoch[epoch_index] + 1}
            for key, values in history.history.items():
                row[key] = values[epoch_index]
            rows.append(row)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(HISTORY_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_demo(model, demo_files, class_names, class_to_index, batch_size):
    demo_paths = []
    demo_labels = []
    unknown_demo_labels = []
    for class_name in sorted(demo_files):
        if class_name not in class_to_index:
            unknown_demo_labels.append(class_name)
            continue
        for file_path in demo_files[class_name]:
            demo_paths.append(file_path)
            demo_labels.append(class_to_index[class_name])

    if not demo_paths:
        return {"demo_images": 0, "demo_accuracy": None, "demo_top5_accuracy": None}

    demo_ds = make_dataset(demo_paths, demo_labels, len(class_names), batch_size, training=False)
    demo_loss, demo_acc, demo_top5 = model.evaluate(demo_ds, verbose=1)
    probs = model.predict(demo_ds, verbose=1)
    rows = []
    for path, true_idx, pred_probs in zip(demo_paths, demo_labels, probs):
        pred_idx = int(np.argmax(pred_probs))
        top5_idx = pred_probs.argsort()[-5:][::-1]
        rows.append(
            {
                "image_file": path,
                "true_label": class_names[true_idx],
                "predicted_label": class_names[pred_idx],
                "confidence_percent": f"{float(pred_probs[pred_idx]) * 100.0:.2f}",
                "top1_result": "Yes" if pred_idx == true_idx else "No",
                "top5_result": "Yes" if true_idx in top5_idx else "No",
                "top5_predictions": "; ".join(
                    f"{class_names[i]} ({float(pred_probs[i]) * 100.0:.2f}%)" for i in top5_idx
                ),
            }
        )

    with open(DEMO_PREDICTIONS_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "demo_images": len(demo_paths),
        "demo_loss": float(demo_loss),
        "demo_accuracy": float(demo_acc),
        "demo_top5_accuracy": float(demo_top5),
        "unknown_demo_labels": unknown_demo_labels,
    }


def main():
    args = parse_args()
    dataset_dir = resolve_project_path(args.dataset_dir)
    handwritten_dir = resolve_project_path(args.handwritten_dir)
    demo_dir = resolve_project_path(args.demo_dir)

    setup_environment()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Clean V4 dataset folder not found: {dataset_dir}")
    if args.include_handwritten and not handwritten_dir.exists():
        raise FileNotFoundError(f"Handwritten folder not found: {handwritten_dir}")
    if not demo_dir.exists():
        raise FileNotFoundError(f"Demo crop folder not found: {demo_dir}")

    clean_files = collect_class_files(dataset_dir)
    demo_files = collect_class_files(demo_dir)
    if not clean_files:
        raise ValueError(f"No clean class folders found: {dataset_dir}")
    if not demo_files:
        raise ValueError(f"No demo class folders with images found: {demo_dir}")

    class_names = sorted(set(clean_files) | set(demo_files))
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    demo_only_labels = sorted(set(demo_files) - set(clean_files))

    print("Num classes:", len(class_names))
    print("Demo labels:", len(demo_files))
    print("Demo-only phrase/shortcut labels:", len(demo_only_labels))

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        split_summary,
    ) = split_clean_dataset(clean_files, class_to_index)

    handwritten_added = 0
    skipped_handwritten = []
    if args.include_handwritten:
        handwritten_added, skipped_handwritten = add_flat_file_dataset(
            train_paths,
            train_labels,
            handwritten_dir,
            class_to_index,
            args.handwritten_repeat,
        )

    demo_added = add_class_folder_dataset(
        train_paths,
        train_labels,
        demo_files,
        class_to_index,
        args.demo_repeat,
    )

    train_paths, train_labels = shuffle_paths_and_labels(train_paths, train_labels)
    val_paths, val_labels = shuffle_paths_and_labels(val_paths, val_labels)
    test_paths, test_labels = shuffle_paths_and_labels(test_paths, test_labels)

    split_info = {
        "num_classes": len(class_names),
        "clean_dataset_dir": str(dataset_dir),
        "handwritten_dir": str(handwritten_dir),
        "demo_dir": str(demo_dir),
        "include_handwritten": args.include_handwritten,
        "handwritten_repeat": args.handwritten_repeat if args.include_handwritten else 0,
        "demo_repeat": args.demo_repeat,
        "handwritten_train_paths_added": handwritten_added,
        "demo_train_paths_added": demo_added,
        "skipped_handwritten": skipped_handwritten,
        "demo_only_labels": demo_only_labels,
        "train_total": len(train_paths),
        "val_total": len(val_paths),
        "test_total": len(test_paths),
        "demo_images_unrepeated": sum(len(paths) for paths in demo_files.values()),
        "demo_label_counts": dict(sorted(Counter({k: len(v) for k, v in demo_files.items()}).items())),
        "per_class_clean": split_summary,
    }
    with open(SPLIT_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Train images/paths: {len(train_paths)}")
    print(f"Val images:         {len(val_paths)}")
    print(f"Test images:        {len(test_paths)}")
    print(f"Handwritten added:  {handwritten_added}")
    print(f"Demo added:         {demo_added}")
    print(f"Split summary:      {SPLIT_SUMMARY_OUT}")

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return

    train_ds = make_dataset(train_paths, train_labels, len(class_names), args.batch_size, training=True)
    val_ds = make_dataset(val_paths, val_labels, len(class_names), args.batch_size, training=False)
    test_ds = make_dataset(test_paths, test_labels, len(class_names), args.batch_size, training=False)

    model, base = build_model(len(class_names))
    compile_model(model, 1e-3)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
    ]

    print("\n===== V5 DEMO STAGE 1: TRAINING CLASSIFIER HEAD =====\n")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_stage1, callbacks=callbacks)

    print("\n===== V5 DEMO STAGE 2: FINE-TUNING UPPER LAYERS =====\n")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1 + args.epochs_stage2,
        initial_epoch=history1.epoch[-1] + 1,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(BEST_MODEL_OUT)

    print("\n===== V5 DEMO CLEAN TEST EVALUATION =====\n")
    test_loss, test_acc, test_top5 = best_model.evaluate(test_ds, verbose=1)

    print("\n===== V5 DEMO CROP EVALUATION =====\n")
    demo_metrics = evaluate_demo(best_model, demo_files, class_names, class_to_index, args.batch_size)

    best_model.save(MODEL_OUT)
    print(f"Saved final model: {MODEL_OUT}")

    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Saved labels: {LABELS_OUT}")

    if not args.skip_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        tflite_model = converter.convert()
        with open(TFLITE_OUT, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite: {TFLITE_OUT}")

    write_history(history1, history2)
    final_metrics = {
        "num_classes": len(class_names),
        "train_images_or_repeated_paths": len(train_paths),
        "val_images": len(val_paths),
        "test_images": len(test_paths),
        "handwritten_added": handwritten_added,
        "demo_added_repeated_paths": demo_added,
        "demo_repeat": args.demo_repeat,
        "clean_test_loss": float(test_loss),
        "clean_test_accuracy": float(test_acc),
        "clean_test_top5_accuracy": float(test_top5),
        **demo_metrics,
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics: {METRICS_OUT}")
    print(f"Saved demo predictions: {DEMO_PREDICTIONS_OUT}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
