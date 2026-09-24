import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf


# =========================
# DEFAULT CONFIG (V4)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "all_words_v4_uniform_clean_only"
DEFAULT_HANDWRITTEN_DIR = PROJECT_ROOT / "datasets" / "allword_handwritten"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

MODEL_OUT = MODELS_DIR / "greggspeak_all_words_v4.keras"
BEST_MODEL_OUT = MODELS_DIR / "best_all_words_v4.keras"
TFLITE_OUT = MODELS_DIR / "greggspeak_all_words_v4.tflite"
LABELS_OUT = MODELS_DIR / "labels_all_words_v4.txt"
HISTORY_OUT = REPORTS_DIR / "training_history_v4.csv"
METRICS_OUT = REPORTS_DIR / "final_metrics_v4.json"
SPLIT_SUMMARY_OUT = REPORTS_DIR / "dataset_split_summary_v4.json"
PREDICTIONS_OUT = REPORTS_DIR / "test_predictions_v4.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GreggSpeak V4 with uniform images and on-the-fly augmentation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--handwritten-dir", type=Path, default=DEFAULT_HANDWRITTEN_DIR)
    parser.add_argument(
        "--include-handwritten",
        action="store_true",
        help="Add handwritten word samples to the training split only.",
    )
    parser.add_argument(
        "--handwritten-repeat",
        type=int,
        default=5,
        help="Repeat handwritten paths in the training split without creating extra files.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=10)
    parser.add_argument("--epochs-stage2", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only validate folders and split counts.")
    parser.add_argument("--skip-tflite", action="store_true", help="Skip TFLite export after training.")
    return parser.parse_args()


def resolve_project_path(path: Path):
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                class_to_files[class_dir.name].append(str(path))
    return class_to_files


def add_handwritten_train_files(train_paths, train_labels, handwritten_dir, class_to_index, repeat):
    if repeat < 1:
        return 0, []

    added = 0
    skipped = []
    handwritten_files = sorted(
        path
        for path in handwritten_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )

    for path in handwritten_files:
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


def split_dataset(class_to_files, class_to_index):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    split_summary = {}

    for class_name in sorted(class_to_files):
        files = list(class_to_files[class_name])
        random.shuffle(files)
        n = len(files)
        if n < 3:
            raise ValueError(f"Class '{class_name}' has only {n} image(s). Need at least 3.")

        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_test = max(1, n - n_train - n_val)

        while (n_train + n_val + n_test) > n:
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


def shuffle_paths_and_labels(paths, labels):
    combined = list(zip(paths, labels))
    random.shuffle(combined)
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
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: load_image(path, label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes):
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                0.005,
                fill_mode="constant",
                fill_value=255.0,
                name="random_rotation",
            ),
            tf.keras.layers.RandomTranslation(
                0.04,
                0.04,
                fill_mode="constant",
                fill_value=255.0,
                name="random_translation",
            ),
            tf.keras.layers.RandomZoom(
                0.06,
                fill_mode="constant",
                fill_value=255.0,
                name="random_zoom",
            ),
            tf.keras.layers.RandomBrightness(
                0.12,
                value_range=(0, 255),
                name="random_brightness",
            ),
            tf.keras.layers.RandomContrast(0.18, name="random_contrast"),
        ],
        name="v4_training_augment",
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
    x = tf.keras.layers.GaussianNoise(0.02, name="light_noise")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )


def write_history(history1, history2):
    rows = []
    for stage, history in [("stage1", history1), ("stage2", history2)]:
        keys = list(history.history.keys())
        for i in range(len(history.epoch)):
            row = {"stage": stage, "epoch": history.epoch[i] + 1}
            for key in keys:
                row[key] = history.history[key][i]
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


def main():
    args = parse_args()
    dataset_dir = resolve_project_path(args.dataset_dir)
    handwritten_dir = resolve_project_path(args.handwritten_dir)

    setup_environment()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if args.include_handwritten and not handwritten_dir.exists():
        raise FileNotFoundError(f"Handwritten folder not found: {handwritten_dir}")

    class_to_files = collect_class_files(dataset_dir)
    class_names = sorted(class_to_files.keys())
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("No class folders with images were found.")

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    print("Num classes:", num_classes)

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        split_summary,
    ) = split_dataset(class_to_files, class_to_index)

    handwritten_added = 0
    skipped_handwritten = []
    if args.include_handwritten:
        handwritten_added, skipped_handwritten = add_handwritten_train_files(
            train_paths,
            train_labels,
            handwritten_dir,
            class_to_index,
            args.handwritten_repeat,
        )

    train_paths, train_labels = shuffle_paths_and_labels(train_paths, train_labels)
    val_paths, val_labels = shuffle_paths_and_labels(val_paths, val_labels)
    test_paths, test_labels = shuffle_paths_and_labels(test_paths, test_labels)

    split_info = {
        "num_classes": num_classes,
        "dataset_dir": str(dataset_dir),
        "handwritten_dir": str(handwritten_dir),
        "include_handwritten": args.include_handwritten,
        "handwritten_repeat": args.handwritten_repeat if args.include_handwritten else 0,
        "handwritten_train_paths_added": handwritten_added,
        "skipped_handwritten": skipped_handwritten,
        "train_total": len(train_paths),
        "val_total": len(val_paths),
        "test_total": len(test_paths),
        "per_class": split_summary,
    }
    with open(SPLIT_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Train images: {len(train_paths)}")
    print(f"Val images:   {len(val_paths)}")
    print(f"Test images:  {len(test_paths)}")
    print(f"Handwritten train paths added: {handwritten_added}")
    print(f"Skipped handwritten files: {len(skipped_handwritten)}")
    print(f"Split summary: {SPLIT_SUMMARY_OUT}")

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return

    train_ds = make_dataset(train_paths, train_labels, num_classes, args.batch_size, training=True)
    val_ds = make_dataset(val_paths, val_labels, num_classes, args.batch_size, training=False)
    test_ds = make_dataset(test_paths, test_labels, num_classes, args.batch_size, training=False)

    model, base = build_model(num_classes)
    compile_model(model, 1e-3)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n===== V4 STAGE 1: TRAINING CLASSIFIER HEAD =====\n")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1,
        callbacks=callbacks,
    )

    print("\n===== V4 STAGE 2: FINE-TUNING UPPER LAYERS =====\n")
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

    print("\nLoading best saved V4 model for final evaluation...")
    best_model = tf.keras.models.load_model(BEST_MODEL_OUT)

    print("\n===== V4 FINAL TEST EVALUATION =====\n")
    test_loss, test_acc, test_top5 = best_model.evaluate(test_ds, verbose=1)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Top-5:    {test_top5:.4f}")

    best_model.save(MODEL_OUT)
    print(f"Saved final Keras model: {MODEL_OUT}")

    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Saved labels: {LABELS_OUT}")

    if not args.skip_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        tflite_model = converter.convert()
        with open(TFLITE_OUT, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model: {TFLITE_OUT}")

    write_history(history1, history2)
    print(f"Saved training history: {HISTORY_OUT}")

    final_metrics = {
        "num_classes": num_classes,
        "dataset_dir": str(dataset_dir),
        "include_handwritten": args.include_handwritten,
        "handwritten_repeat": args.handwritten_repeat if args.include_handwritten else 0,
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "test_images": len(test_paths),
        "image_size": list(IMG_SIZE),
        "batch_size": args.batch_size,
        "epochs_stage1": args.epochs_stage1,
        "epochs_stage2": args.epochs_stage2,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_top5_accuracy": float(test_top5),
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved final metrics: {METRICS_OUT}")

    print("\nGenerating V4 clean test predictions log...")
    probs = best_model.predict(test_ds, verbose=1)
    true_indices = []
    for _, batch_labels in test_ds:
        true_indices.extend(tf.argmax(batch_labels, axis=1).numpy().tolist())

    rows = []
    pred_indices = tf.argmax(probs, axis=1).numpy().tolist()
    top1_conf = tf.reduce_max(probs, axis=1).numpy().tolist()
    for i, (true_idx, pred_idx, conf) in enumerate(zip(true_indices, pred_indices, top1_conf)):
        rows.append(
            {
                "sample_index": i,
                "true_label_index": true_idx,
                "true_label": class_names[true_idx],
                "predicted_label_index": pred_idx,
                "predicted_label": class_names[pred_idx],
                "confidence": float(conf),
                "correct": int(true_idx == pred_idx),
            }
        )

    with open(PREDICTIONS_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved test predictions: {PREDICTIONS_OUT}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
