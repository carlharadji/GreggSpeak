#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/7] System info"
cat /etc/os-release || true
python3 --version

echo "[2/7] Installing Raspberry Pi OS packages"
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  python3-opencv \
  python3-picamera2 \
  python3-tk \
  python3-pil \
  python3-numpy \
  espeak-ng \
  libespeak-ng1 \
  git \
  curl \
  wget \
  build-essential \
  network-manager

echo "[3/7] Creating Python virtual environment"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[4/7] Installing Python packages"
python -m pip install \
  numpy \
  Pillow \
  Flask \
  pyttsx3 \
  "qrcode[pil]" \
  customtkinter \
  pyserial

echo "[5/7] Installing TensorFlow Lite runtime when available"
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tflite_runtime") else 1)
PY
then
  echo "tflite_runtime is already available."
else
  if python -m pip install tflite-runtime; then
    echo "Installed tflite-runtime with pip."
  else
    echo "WARNING: tflite-runtime could not be installed with this Python version."
    echo "On Raspberry Pi OS Trixie, Python is commonly 3.13; current tflite-runtime wheels are usually for Python 3.11 or older."
    echo "The web/LCD app can still install, but model inference may need Raspberry Pi OS Legacy Bookworm 64-bit or a Python 3.11 environment."
  fi
fi

echo "[6/7] Marking run scripts executable"
chmod +x \
  run_lcd.sh \
  run_web.sh \
  run_pipeline.sh \
  run_esp_test.sh \
  start_all.sh \
  setup_hotspot.sh \
  start_hotspot.sh \
  stop_hotspot.sh \
  start_device_hotspot.sh

echo "[7/7] Verifying imports"
python - <<'PY'
import importlib.util

required = [
    "cv2",
    "PIL",
    "flask",
    "customtkinter",
    "serial",
    "qrcode",
    "pyttsx3",
]

missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))

has_tflite = importlib.util.find_spec("tflite_runtime") is not None
has_tensorflow = importlib.util.find_spec("tensorflow") is not None
if has_tflite:
    print("Inference runtime: tflite_runtime")
elif has_tensorflow:
    print("Inference runtime: tensorflow")
else:
    print("WARNING: no TFLite inference runtime found yet.")

print("Core project imports look good.")
PY

if [[ "${INSTALL_CODEX:-0}" == "1" ]]; then
  echo "[optional] Installing Codex CLI"
  sudo apt install -y nodejs npm
  mkdir -p "$HOME/.npm-global"
  npm config set prefix "$HOME/.npm-global"
  if ! grep -q 'npm-global/bin' "$HOME/.bashrc"; then
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.bashrc"
  fi
  export PATH="$HOME/.npm-global/bin:$PATH"
  npm install -g @openai/codex
  echo "Codex installed. Run: codex --login"
fi

echo "Done. Activate with: source .venv/bin/activate"
