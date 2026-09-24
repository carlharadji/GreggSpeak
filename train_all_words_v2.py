import json
import random
from pathlib import Path
from collections import defaultdict

import tensorflow as tf
import pandas as pd

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "all_words"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 8

MODEL_OUT = MODELS_DIR / "greggspeak_all_words_v3.keras"
BEST_MODEL_OUT = MODELS_DIR / "best_all_words_v3.keras"
TFLITE_OUT = MODELS_DIR / "greggspeak_all_words_v3.tflite"
LABELS_OUT = MODELS_DIR / "labels_all_words_v3.txt"
HISTORY_OUT = REPORTS_DIR / "training_history_v3.csv"
METRICS_OUT = REPORTS_DIR / "final_metrics_v3.json"
SPLIT_SUMMARY_OUT = REPORTS_DIR / "dataset_split_summary_v3.json"
PREDICTIONS_OUT = REPORTS_DIR / "test_predictions_v3.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# REPRODUCIBILITY
# =========================
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# ENV CHECK
# =========================
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth warning:", e)

# =========================
# VALIDATE DATASET DIR
# =========================
if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

# =========================
# COLLECT FILES BY CLASS
# =========================
class_to_files = defaultdict(list)

for class_dir in sorted(DATASET_DIR.iterdir()):
    if class_dir.is_dir():
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                class_to_files[class_dir.name].append(str(f))

class_names = sorted(class_to_files.keys())
num_classes = len(class_names)

if num_classes == 0:
    raise ValueError("No class folders with images were found.")

class_to_index = {name: idx for idx, name in enumerate(class_names)}

print("Num classes:", num_classes)

# =========================
# SPLIT DATA PER CLASS
# =========================
train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

split_summary = {}

for class_name in class_names:
    files = class_to_files[class_name]
    random.shuffle(files)

    n = len(files)
    if n < 3:
        raise ValueError(
            f"Class '{class_name}' has only {n} image(s). Need at least 3 for train/val/test split."
        )

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    # Ensure each split gets at least 1 image
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1

    # Re-adjust if total exceeded
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
        "total": n,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }

# Save split summary
with open(SPLIT_SUMMARY_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "num_classes": num_classes,
        "train_total": len(train_paths),
        "val_total": len(val_paths),
        "test_total": len(test_paths),
        "per_class": split_summary
    }, f, indent=2)

print(f"Train images: {len(train_paths)}")
print(f"Val images:   {len(val_paths)}")
print(f"Test images:  {len(test_paths)}")

# =========================
# DATA LOADER
# =========================
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels, training=False)
test_ds = make_dataset(test_paths, test_labels, training=False)

# =========================
# MODEL
# =========================
augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.01),
        tf.keras.layers.RandomZoom(0.03),
        tf.keras.layers.RandomTranslation(0.02, 0.02),
    ],
    name="augment",
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

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ],
)

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

# =========================
# STAGE 1: TRAIN HEAD
# =========================
print("\n===== STAGE 1: TRAINING CLASSIFIER HEAD =====\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
)

# =========================
# STAGE 2: FINE-TUNE TOP LAYERS
# =========================
print("\n===== STAGE 2: FINE-TUNING UPPER LAYERS =====\n")

base.trainable = True

# Freeze lower layers, unfreeze only upper layers
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
    initial_epoch=history1.epoch[-1] + 1,
    callbacks=callbacks,
)

# =========================
# LOAD BEST MODEL
# =========================
print("\nLoading best saved model for final evaluation...")
best_model = tf.keras.models.load_model(BEST_MODEL_OUT)

# =========================
# FINAL EVALUATION ON TEST SET
# =========================
print("\n===== FINAL TEST EVALUATION =====\n")
test_loss, test_acc, test_top5 = best_model.evaluate(test_ds, verbose=1)

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Top-5:    {test_top5:.4f}")

# =========================
# SAVE FINAL MODEL
# =========================
best_model.save(MODEL_OUT)
print(f"Saved final Keras model: {MODEL_OUT}")

with open(LABELS_OUT, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"Saved labels: {LABELS_OUT}")

# =========================
# EXPORT TFLITE
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model: {TFLITE_OUT}")

# =========================
# SAVE TRAINING HISTORY
# =========================
hist1_df = pd.DataFrame(history1.history)
hist1_df["stage"] = "stage1"

hist2_df = pd.DataFrame(history2.history)
hist2_df["stage"] = "stage2"

history_df = pd.concat([hist1_df, hist2_df], ignore_index=True)
history_df.to_csv(HISTORY_OUT, index=False)
print(f"Saved training history: {HISTORY_OUT}")

# =========================
# SAVE FINAL METRICS
# =========================
final_metrics = {
    "num_classes": num_classes,
    "train_images": len(train_paths),
    "val_images": len(val_paths),
    "test_images": len(test_paths),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_stage1": EPOCHS_STAGE1,
    "epochs_stage2": EPOCHS_STAGE2,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_top5_accuracy": float(test_top5),
}

with open(METRICS_OUT, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)

print(f"Saved final metrics: {METRICS_OUT}")

# =========================
# SAVE TEST PREDICTIONS
# =========================
print("\nGenerating test predictions log...")
probs = best_model.predict(test_ds, verbose=1)

true_indices = []
for _, batch_labels in test_ds:
    true_indices.extend(tf.argmax(batch_labels, axis=1).numpy().tolist())

pred_indices = tf.argmax(probs, axis=1).numpy().tolist()
top1_conf = tf.reduce_max(probs, axis=1).numpy().tolist()

rows = []
for i, (true_idx, pred_idx, conf) in enumerate(zip(true_indices, pred_indices, top1_conf)):
    rows.append({
        "sample_index": i,
        "true_label_index": true_idx,
        "true_label": class_names[true_idx],
        "predicted_label_index": pred_idx,
        "predicted_label": class_names[pred_idx],
        "confidence": float(conf),
        "correct": int(true_idx == pred_idx),
    })

pred_df = pd.DataFrame(rows)
pred_df.to_csv(PREDICTIONS_OUT, index=False)
print(f"Saved test predictions: {PREDICTIONS_OUT}")

print("\nDONE.")
print("Files generated:")
print(f" - {MODEL_OUT}")
print(f" - {BEST_MODEL_OUT}")
print(f" - {TFLITE_OUT}")
print(f" - {LABELS_OUT}")
print(f" - {HISTORY_OUT}")
print(f" - {METRICS_OUT}")
print(f" - {SPLIT_SUMMARY_OUT}")
print(f" - {PREDICTIONS_OUT}")
