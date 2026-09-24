from pathlib import Path
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
WEB_DIR = BASE_DIR / "web"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
PAGE_ARTIFACTS_DIR = DATA_DIR / "page_artifacts"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
EXPORTS_DIR = DATA_DIR / "exports"
OUTPUT_DIR = TRANSCRIPTS_DIR
DB_PATH = DATA_DIR / "greggspeak.db"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

MODEL_V53_PATH = MODELS_DIR / "greggspeak_all_words_v5_3.keras"
BEST_MODEL_V53_PATH = MODELS_DIR / "best_all_words_v5_3.keras"
LABELS_V53_PATH = MODELS_DIR / "labels_all_words_v5_3.txt"
MODEL_V52_PATH = MODELS_DIR / "greggspeak_all_words_v5_2.keras"
BEST_MODEL_V52_PATH = MODELS_DIR / "best_all_words_v5_2.keras"
LABELS_V52_PATH = MODELS_DIR / "labels_all_words_v5_2.txt"
MODEL_V5_DEMO_PATH = MODELS_DIR / "greggspeak_all_words_v5_demo.keras"
BEST_MODEL_V5_DEMO_PATH = MODELS_DIR / "best_all_words_v5_demo.keras"
LABELS_V5_DEMO_PATH = MODELS_DIR / "labels_all_words_v5_demo.txt"
MODEL_TFLITE_PATH = MODELS_DIR / "model.tflite"
LABELS_PATH = LABELS_V53_PATH

DEFAULT_MODEL_FILES = (
    (MODEL_V53_PATH, LABELS_V53_PATH),
    (BEST_MODEL_V53_PATH, LABELS_V53_PATH),
    (MODEL_V52_PATH, LABELS_V52_PATH),
    (BEST_MODEL_V52_PATH, LABELS_V52_PATH),
    (MODEL_V5_DEMO_PATH, LABELS_V5_DEMO_PATH),
    (BEST_MODEL_V5_DEMO_PATH, LABELS_V5_DEMO_PATH),
    (MODEL_TFLITE_PATH, MODELS_DIR / "labels.txt"),
)

IMAGE_SIZE = (128, 128)
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

CAMERA_INDEX = _env_int("GREGGSPEAK_CAMERA_INDEX", 0)
CAMERA_ROTATE_COUNTERCLOCKWISE = True
CAMERA_DOCUMENT_CROP = True
USE_UNIFORM_CROP = False

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
