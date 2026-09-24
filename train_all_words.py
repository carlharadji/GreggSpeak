# train_all_words.py
# Train + evaluate on laptop, then export .keras + .tflite + labels.

import tensorflow as tf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "all_words"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
VAL_SPLIT = 0.15

MODEL_OUT = MODELS_DIR / "greggspeak_all_words.keras"
BEST_MODEL_OUT = MODELS_DIR / "best_all_words.keras"
TFLITE_OUT = MODELS_DIR / "greggspeak_all_words.tflite"
LABELS_OUT = MODELS_DIR / "labels_all_words.txt"


print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        print("GPU memory growth warning:", exc)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Num classes:", num_classes)

autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(2000, seed=SEED).prefetch(autotune)
val_ds = val_ds.cache().prefetch(autotune)

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomTranslation(0.03, 0.03),
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

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_OUT,
        save_best_only=True,
        monitor="val_accuracy",
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(MODEL_OUT)
print(f"Saved Keras model: {MODEL_OUT}")

with open(LABELS_OUT, "w", encoding="utf-8") as file:
    for name in class_names:
        file.write(name + "\n")
print(f"Saved labels: {LABELS_OUT}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_OUT, "wb") as file:
    file.write(tflite_model)

print(f"Saved TFLite model: {TFLITE_OUT}")
print("DONE")
print("Next laptop test:")
print(" - Use models/best_all_words.keras or models/greggspeak_all_words.keras for prediction")
print(" - models/labels_all_words.txt maps output index to word")
