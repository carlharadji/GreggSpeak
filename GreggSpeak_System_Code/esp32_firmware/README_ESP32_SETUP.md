# GreggSpeak ESP32 Firmware Setup

This folder includes two ESP32 sketches:

- `greggspeak_serial_test/` - first wired USB serial smoke test.
- `greggspeak_feeder_controller/` - motor, IR sensor, and batch-feeder command controller.

## Required

- ESP32 board
- USB data cable
- Arduino IDE or Arduino CLI
- ESP32 board support installed in Arduino IDE

## Upload Firmware

1. For the first serial smoke test, open:

```text
greggspeak_serial_test/greggspeak_serial_test.ino
```

For the full feeder controller, open:

```text
greggspeak_feeder_controller/greggspeak_feeder_controller.ino
```

Default feeder pins:

```text
R_EN      -> GPIO25
L_EN      -> GPIO26
RPWM      -> GPIO27
LPWM      -> GPIO12
IR_SENSOR -> GPIO33
```

2. In Arduino IDE, select your ESP32 board.
3. Select the ESP32 serial port.
4. Upload.
5. Open Serial Monitor at `115200`.

Expected serial-test startup message:

```text
GREGGSPEAK_ESP_READY
```

Expected feeder-controller startup message:

```text
GREGGSPEAK_FEEDER_READY
```

## Test From Raspberry Pi

Connect the ESP32 to the Raspberry Pi using USB.

From `greggspeak_rpi_deploy/`:

```bash
source .venv/bin/activate
./run_esp_test.sh --list
./run_esp_test.sh --command PING
```

Expected response:

```text
Response: PONG
```

For the feeder controller:

```bash
./run_esp_test.sh --command STATUS
./run_esp_test.sh --command "SENSOR?"
./run_esp_test.sh --command "FEED 11500 150 LONGFIRST"
./run_esp_test.sh --command "FEED 7000 150 LONGNEXT"
./run_esp_test.sh --command "FEED 10000 150 A4TEST"
./run_esp_test.sh --command "FEED 9000 150 SHORTTEST"
./run_esp_test.sh --command STOP
```

## Test Commands

```text
PING              -> PONG
STATUS            -> OK GREGGSPEAK_ESP32_SERIAL_TEST
ECHO hello        -> ECHO hello
HELP              -> COMMANDS PING STATUS ECHO HELP
```

Feeder controller:

```text
PING                         -> OK PONG
STATUS                       -> OK STATUS PAPER=1 MOTOR=0 STATE=IDLE
SENSOR?                      -> OK PAPER_PRESENT or OK NO_PAPER
FEED <ms> <speed> <job_id>   -> OK FEED_STARTED, then OK FEED_DONE
STOP                         -> OK STOPPED
SET_SPEED <speed>            -> OK SPEED <speed>
```

## Notes

- Use a USB data cable, not a charge-only cable.
- The ESP32 can be powered by the Raspberry Pi USB port for serial control.
- The 24V motor supply must power the IBT-2 motor side separately.
- Raspberry Pi/ESP32 ground, IBT-2 logic ground, and motor supply ground must share common ground.
- The Pi owns the batch loop: Long feeds 11.5s for page 1, 7s for later pages, pauses 2s for camera capture after each feed, then repeats until `SENSOR?` reports `NO_PAPER`.
