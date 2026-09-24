import random
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "all_words"
MODEL_PATH = PROJECT_ROOT / "models" / "best_all_words.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels_all_words.txt"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

random.seed(SEED)
tf.random.set_seed(SEED)

# Load original V1 class list
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

num_classes = len(class_names)
class_to_index = {name: idx for idx, name in enumerate(class_names)}

print("V1 class count from labels file:", num_classes)

# Collect only folders that exist in both labels file and current dataset
paths = []
labels = []

missing_folders = []

for class_name in class_names:
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists() or not class_dir.is_dir():
        missing_folders.append(class_name)
        continue

    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files = sorted(files)
    random.shuffle(files)

    n = len(files)
    if n < 2:
        continue

    # Recreate V1-style validation split to use as baseline test
    n_val = max(1, int(n * VAL_SPLIT))
    val_files = files[-n_val:]

    for f in val_files:
        paths.append(str(f))
        labels.append(class_to_index[class_name])

print("Missing folders from current dataset:", len(missing_folders))
if missing_folders:
    print("Example missing folders:", missing_folders[:10])

print("Evaluation samples collected:", len(paths))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

test_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model(MODEL_PATH)

results = model.evaluate(test_ds, verbose=1)

print("\n=== V1 EVALUATION RESULTS ===")
print(f"Loss: {results[0]:.4f}")
print(f"Top-1 Accuracy: {results[1] * 100:.2f}%")
if len(results) > 2:
    print(f"Top-5 Accuracy: {results[2] * 100:.2f}%")
