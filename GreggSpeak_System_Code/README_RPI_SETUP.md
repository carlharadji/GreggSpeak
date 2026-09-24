# GreggSpeak Raspberry Pi 5 Deployment

This folder is the Raspberry Pi runtime for GreggSpeak. It excludes training datasets, training scripts, notebooks, testing scripts, and debug outputs.

## Runtime Structure

- `app/main_rpi.py` - command-line recognition pipeline.
- `app/web_app.py` - full Flask web app adapted from the laptop web app.
- `app/db.py` - shared SQLite batch/page database.
- `app/pipeline/` - recognition orchestration and database saving.
- `app/preprocessing/` - camera and image loading.
- `app/segmentation/` - OpenCV line/word segmentation.
- `app/recognition/` - Keras or TFLite word classification, according to the selected model file.
- `app/translation/` - prediction-to-text formatting.
- `app/storage/` - local transcript text storage.
- `app/tts/` - optional backend text-to-speech.
- `lcd_ui/` - CustomTkinter touchscreen/LCD app adapted from the laptop LCD UI.
- `web/templates/` - adapted laptop Flask templates.
- `web/static/` - adapted laptop web styling.
- `models/` - model weights are required for recognition but are not included in the public source repository. V5.3 labels are included for reference.
- `data/greggspeak.db` - shared database for pipeline and web app.
- `data/page_artifacts/` - scanned page and segmentation review images created at runtime.
- `data/transcripts/` - saved transcript text files.
- `data/exports/` - generated PDF exports.
- `data/input/` - optional fallback folder for CLI image tests.

## Raspberry Pi OS Setup

Use Raspberry Pi OS 64-bit on Raspberry Pi 5.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv python3-picamera2 python3-tk espeak-ng
```

Test the camera:

```bash
libcamera-hello
```

## Python Environment

From inside this folder:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rpi.txt
chmod +x run_lcd.sh run_web.sh run_pipeline.sh run_esp_test.sh start_all.sh
chmod +x setup_hotspot.sh start_hotspot.sh stop_hotspot.sh start_device_hotspot.sh
```

If `tflite-runtime` is unavailable for your Python version:

```bash
pip install tensorflow
```

Model selection prefers the V5.3 Keras model if present, then older Keras models,
then `model.tflite`. A selected Keras model requires TensorFlow/Keras; a selected
TFLite model uses `tflite-runtime` or `tensorflow.lite.Interpreter`. Copy only a
model and matching labels that you are authorized to use. No weights are publicly
distributed with this repository.

The web app requires private `GREGGSPEAK_WEB_PASSWORD` and
`GREGGSPEAK_SECRET_KEY` environment values before startup. Set a unique password
and a random session key in your local environment or a private service environment
file. Do not commit those values. The root `.env.example` lists variable names;
the application does not automatically load `.env` files.

## Run LCD Touchscreen App

This is the main Raspberry Pi device screen. It uses the same CustomTkinter LCD workflow as the laptop version and starts the Flask web app automatically.

```bash
./run_lcd.sh
```

For laptop preview from this deploy folder:

```bash
python -m lcd_ui.main
```

Stop any other app using port `5000` first, otherwise the LCD screen may show the URL for the already-running server.

## Run Full Web App

```bash
./run_web.sh
```

Open on the Pi:

```text
http://localhost:5000
```

Open from another device on the same network:

```bash
hostname -I
```

Then visit:

```text
http://PI_IP_ADDRESS:5000
```

The web app prints the current access URLs in the terminal when it starts. The dashboard and settings page also show the current Pi URL plus a QR code you can scan from a phone or tablet on the same network.

The web app supports the laptop-style dashboard, latest transcript, records, page-level editing, scanned page review, segmentation overlay review, combined transcript regeneration, trash/restore, PDF export, QR/network access, and browser text-to-speech. Device scanning and camera recognition are handled through the LCD app or command-line pipeline, and saved batches appear in this same web app.

## Run As Standalone Hotspot

Use this when there is no school/home Wi-Fi, or when you want phones/tablets to connect directly to the GreggSpeak device.

Set up the hotspot once:

```bash
./setup_hotspot.sh GreggSpeak-RPi
```

The script prompts for a unique hotspot password without echoing it. For unattended
setup, set `GREGGSPEAK_HOTSPOT_PASSWORD` privately before running it. Do not place
the password in a shell command argument or a committed script.

Configured hotspot details:

```text
SSID: GreggSpeak-RPi
Password: the unique password you selected
Web app URL: http://10.42.0.1:5000
```

Start the hotspot later:

```bash
./start_hotspot.sh
```

Stop it:

```bash
./stop_hotspot.sh
```

Start hotspot and LCD UI together:

```bash
./start_device_hotspot.sh
```

On your phone or laptop:

1. Connect to Wi-Fi network `GreggSpeak-RPi`.
2. Enter the unique password you selected during setup.
3. Scan the QR on the LCD home screen or open `http://10.42.0.1:5000`.

Notes:

- This uses Raspberry Pi Wi-Fi, not ESP32 Wi-Fi.
- Internet is not required.
- The web app still runs on the Raspberry Pi.
- The ESP32 can remain wired over USB for serial testing/control.

## Run Pipeline Only

Camera/default fallback:

```bash
./run_pipeline.sh
```

Fallback images:

```bash
./run_pipeline.sh --no-camera
```

Direct file or folder:

```bash
./run_pipeline.sh --input data/input/sample_page.png
./run_pipeline.sh --input data/input/
```

Optional flags:

```bash
./run_pipeline.sh --title "Court Notes Test"
./run_pipeline.sh --uniform-crop
./run_pipeline.sh --speak
```

The pipeline automatically creates a transcript batch in `data/greggspeak.db`, adds page records, regenerates the combined transcript, and writes a text copy to `data/transcripts/`.
It also writes review images to `data/page_artifacts/` so the web workspace can show the original scanned page and the boxed segmentation result.

## Run Web And Pipeline Together

```bash
./start_all.sh
```

This starts the Flask web app, then runs one pipeline pass using the arguments you provide. The web app stays running.

Example:

```bash
./start_all.sh --input data/input/sample_page.png --title "Startup Test"
```

## Test ESP32 Connectivity

Use USB serial for the first ESP32 test. Connect the ESP32 to the Raspberry Pi with a data-capable USB cable, then list ports:

```bash
./run_esp_test.sh --list
```

Send a test command:

```bash
./run_esp_test.sh --port /dev/ttyUSB0 --command PING
```

If the port is obvious, the script can auto-select it:

```bash
./run_esp_test.sh --command PING
```

The ESP32 firmware should read a newline-terminated command and print a newline-terminated response. For the first test, make the ESP32 respond with something simple like `PONG` when it receives `PING`.

After uploading the feeder controller firmware, test the motor/sensor protocol:

```bash
./run_esp_test.sh --command STATUS
./run_esp_test.sh --command "SENSOR?"
./run_esp_test.sh --command "FEED 11500 150 LONGFIRST"
./run_esp_test.sh --command "FEED 7000 150 LONGNEXT"
./run_esp_test.sh --command "FEED 10000 150 A4TEST"
./run_esp_test.sh --command "FEED 9000 150 SHORTTEST"
./run_esp_test.sh --command STOP
```

Starter ESP32 serial test firmware is included in:

```text
esp32_firmware/greggspeak_serial_test/greggspeak_serial_test.ino
```

Full feeder firmware is included in:

```text
esp32_firmware/greggspeak_feeder_controller/greggspeak_feeder_controller.ino
```

ESP32 setup notes are in:

```text
esp32_firmware/README_ESP32_SETUP.md
```

## Shared Storage

One database is used:

```text
data/greggspeak.db
```

Both `app/main_rpi.py` and `app/web_app.py` use `app/db.py`, so batches created by camera/upload/CLI recognition appear in the dashboard and records immediately.

Generated files:

```text
data/transcripts/transcript_YYYYMMDD_HHMMSS.txt
data/exports/<title>.pdf
```

## Optional Autostart For LCD App

Create a systemd service after copying the folder to the Pi. Adjust the path and user if needed.

```bash
sudo nano /etc/systemd/system/greggspeak-lcd.service
```

Example:

```ini
[Unit]
Description=GreggSpeak Raspberry Pi LCD App
After=network.target

[Service]
WorkingDirectory=/home/pi/greggspeak_rpi_deploy
ExecStart=/home/pi/greggspeak_rpi_deploy/.venv/bin/python -m lcd_ui.main
Restart=always
User=pi
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable greggspeak-lcd
sudo systemctl start greggspeak-lcd
sudo systemctl status greggspeak-lcd
```

For standalone hotspot mode, configure the hotspot once before enabling the LCD service:

```bash
./setup_hotspot.sh GreggSpeak-RPi
```

The hotspot connection is configured with `autoconnect yes`, so NetworkManager should bring it back on reboot. If it does not, create a separate root oneshot service to run:

```bash
nmcli connection up GreggSpeak-Hotspot
```

For manual testing from a terminal, you can also use:

```bash
./start_device_hotspot.sh
```

## Hardware Notes

- Camera: Picamera2 is preferred. OpenCV camera index `0` is fallback.
- TTS: browser TTS is available in the transcript workspace; backend TTS is optional.
- LCD: `lcd_ui/` is included as the main touchscreen UI.
- ESP32: USB serial connectivity and feeder motor/sensor commands are available through `run_esp_test.sh`.
- Feeder: LCD batch scanning uses USB serial to command the ESP32. Fixed presets are Short 9s, A4 10s, and Long 11.5s first page then 7s next pages, with a 2s camera-capture pause after each feed.
- Recognition accuracy depends heavily on lighting, page angle, focus, and segmentation quality.

## Troubleshooting

- `No image input found`: connect a camera or add images to `data/input/`.
- `Install tflite-runtime or tensorflow`: install one inference runtime.
- `pyserial is required`: run `pip install -r requirements-rpi.txt` inside the virtual environment.
- ESP32 not detected: use a data USB cable, check `/dev/ttyUSB0` or `/dev/ttyACM0`, and verify permissions with `groups`.
- Camera unavailable: verify `libcamera-hello`, ribbon cable orientation, and camera enablement.
- Web unreachable from another device: verify both devices are on the same network and use `http://PI_IP_ADDRESS:5000`.
