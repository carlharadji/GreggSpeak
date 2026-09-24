# GreggSpeak

GreggSpeak is a **Computer Engineering Thesis Project** that helps turn scanned
Gregg shorthand pages into editable transcript drafts. A camera or uploaded
image supplies a page; image processing separates its shorthand outlines; a
word/phrase classifier suggests text; and a person reviews and corrects the
draft before export. The prototype addresses the time-consuming transcription
work of courtroom stenography while keeping a human reviewer in the loop.

This repository presents the software and firmware source. It is **not** a
turnkey recognition download: the research dataset and trained model weights are
intentionally absent. The interface and code can be inspected without them;
inference requires a compatible model and label file obtained with appropriate
permission.

## System overview

The Raspberry Pi prototype combines a touchscreen, camera, and ESP32-controlled
paper feeder. An operator selects a paper size and starts a batch on the LCD.
The feeder moves a page into the camera area, where the Pi captures and
processes it. The operator reviews and saves the draft on the LCD; saved batches
then appear in the Flask web app for detailed editing and export. A laptop
version supports camera capture or image upload without the physical feeder.

### Recognition workflow

```text
Pi feeder / camera or image upload
    -> page preparation and optional calibrated perspective crop
    -> OpenCV ink mask, row detection, and word-box segmentation
    -> 128 x 128 RGB word crops
    -> MobileNetV2-based closed-vocabulary classifier
    -> ordered Top-1 word/phrase predictions and review flags
    -> rule-based transcript formatting
    -> LCD preview and save -> SQLite batch/page records
    -> Flask browser review, correction, and PDF/DOCX export
```

The Pi touchscreen app coordinates the feeder over USB serial. The ESP32
firmware reads a paper sensor and controls the motor; the Pi waits for feeding
to finish, captures a frame, and processes the page. Both the Pi and laptop
versions have a CustomTkinter interface and a companion Flask web app.

OpenCV creates a mask for locating shorthand and crops the original prepared
image at the detected word boxes. The V5.3 training code uses a MobileNetV2
backbone and a softmax output over **1,494 word/phrase classes**. The original
Version 1 dataset had 1,480 verified word classes; 1,494 is the current V5.3
class count. The inspected Pi runtime requests the highest-scoring prediction
for each crop. The laptop path also keeps five candidate scores for review but
uses the highest-scoring label in its transcript. Low scores prompt review;
they do not prove a prediction is correct or incorrect. Editing a transcript
does not retrain the model.

Python assembles predicted labels in row/word order. Rule-based post-processing
applies punctuation, capitalization, common replacements, and courtroom-style
speaker formatting. SQLite stores batches, pages, text edits, confidence
summaries, and paths to image artifacts; the images remain files on disk.
Flask serves saved records to a browser and handles editing and exports. The LCD
app also calls recognition and database functions directly.

## Main features

- LCD-led batch scanning with paper-size presets on the Pi; camera and upload
  workflows on the laptop.
- Page preparation, OpenCV segmentation, and ranked word/phrase predictions.
- Human review of classified words and editable transcript text.
- SQLite-backed batch/page records and browser export to PDF or Word.
- Optional transcript playback through text-to-speech.

## Technology stack

Python, OpenCV, NumPy, Pillow, TensorFlow/Keras, optional TensorFlow Lite runtime,
CustomTkinter, Flask, SQLite, Picamera2 on Raspberry Pi, pyserial for ESP32
communication, and optional text-to-speech through pyttsx3 or the browser.

## Evaluation and limitations

In a **timed evaluation at MTC Sta. Rita**, manual transcription of one
shorthand page took **7 minutes 39 seconds** and the GreggSpeak-assisted process
took **3 minutes 13 seconds**: **4 minutes 26 seconds** less, or **57.95%** for
that measured comparison. This is a project evaluation result, not a claim that
every page or user will see the same reduction. The number of participants is
not stated here because it has not been confirmed for this public summary.

The model is a closed-vocabulary classifier. It cannot directly recognize an
unseen word class, and similar outlines or handwriting different from the
training examples can cause mistakes. Bad row or word segmentation can pass the
wrong crop to an otherwise functioning classifier. The confidence display is a
model score, not a calibrated guarantee of correctness. Human review remains
part of the intended workflow. Reported accuracy from a given test set should
be read with its model version, data source, and split; this README does not
present a universal recognition-accuracy figure.

## Thesis team and contribution

**Carl Eugene S. Haradji served as Software Development Lead.** He led software
development, developed or integrated major software components, contributed to
the computer-vision and recognition pipeline, and integrated application
components. The team worked together on system-level implementation, testing,
and evaluation. Individual authorship of specific files or hardware subsystems
is not inferred from this repository.

The thesis team was:

- Ian Jovic D. Tongol
- Aaron Gabriel D. Bustos
- Mark M. Gonzales
- Carl Eugene S. Haradji
- Raniel N. Tarrosa

Thesis adviser: **Asil Kastle S. Dela Cruz, PCpE, MIT**.

## Project brochure and demonstration

The [GreggSpeak project brochure](assets/GreggSpeak-Brochure.pdf) illustrates the
device, hardware, LCD operation, and web workflow. Its application screenshots
show example content and network details.

**Full GreggSpeak system demonstration video coming soon.** The video will be
hosted separately once an approved public link is available.

## Repository layout

| Path | Purpose |
| --- | --- |
| `greggspeak_ui/` | Laptop LCD-style app, recognition pipeline, Flask companion, and SQLite code |
| `GreggSpeak_System_Code/app/` | Pi pipeline, web app, storage, camera, and feeder integration |
| `GreggSpeak_System_Code/lcd_ui/` | Pi touchscreen workflow |
| `GreggSpeak_System_Code/esp32_firmware/` | Feeder controller firmware |
| `train_all_words*.py` | Versioned model training scripts |
| `run_*testing*.py`, `evaluate_v1.py` | Evaluation entry points |
| `tools/` | Reviewed preprocessing and segmentation utilities |
| `models/*labels*.txt` | Class mappings for reference; weights are excluded |
| `assets/GreggSpeak-Brochure.pdf` | Team brochure and operating overview |

Private `datasets/`, runtime `data/`, model binaries, research reports,
transcripts, browser profiles, and historical deployment archives are ignored
and are not part of the public source distribution.

## Setup and running

Use a Python environment compatible with the installed TensorFlow and platform
packages. The public requirements constrain Flask and Werkzeug to a compatible
2.x pair; other dependency versions are not locked in this historical research
workspace. Reproduce and record a fully tested environment before deployment.

For a laptop, install `requirements-laptop.txt` and run
`python greggspeak_ui/main.py` from the project root. For a Raspberry Pi, see
[`GreggSpeak_System_Code/README_RPI_SETUP.md`](GreggSpeak_System_Code/README_RPI_SETUP.md)
for OS packages, `requirements-rpi.txt`, the LCD launcher, and feeder setup.
The Pi web app can be launched separately through `run_web.sh`.

Before starting either Flask companion, privately set
`GREGGSPEAK_WEB_PASSWORD` and `GREGGSPEAK_SECRET_KEY`; there are no default
values. Set a unique web password and a random session key. The optional
`GREGGSPEAK_WEB_USERNAME` defaults to `admin`. See [`.env.example`](.env.example)
for variable names. Neither application automatically loads `.env`, and actual
environment files are ignored. Change any old deployment that still uses the
prototype's former default credentials. Pi hotspot setup asks for a unique
password or reads `GREGGSPEAK_HOTSPOT_PASSWORD` from a private environment.

Recognition also needs model weights paired with the correct labels. The laptop
runtime searches for V5.3 Keras files first; the Pi searches for V5.3 Keras,
then older Keras, then TFLite files. Installing only `tflite-runtime` will not
load a selected `.keras` model; install TensorFlow/Keras for that path. No
publicly approved model download is provided here. Dataset and training scripts
are included for study, but training is not reproducible from this repository
alone because the underlying images are withheld.

## Project visuals

Screenshots are intentionally omitted until interface captures with **wholly
synthetic text** and approved branding are reviewed. Captures from the local
database, court materials, or research participants are not suitable examples.

## Dataset and model availability

This repository includes training, evaluation, preprocessing, and inference
source plus reviewed label mappings. It does **not** include raw or augmented
shorthand datasets, captured pages, trained `.keras`/`.tflite`/`.h5` weights, or
checkpoints. These materials need separate privacy, provenance, academic-use,
and redistribution clearance. There is currently no approved public model or
dataset download; Git LFS and GitHub Releases would not change those rights.

## Data, provenance, and rights

The local research workspace contains handwritten shorthand, dictionary-derived
references, camera captures, transcripts, and evaluation material associated
with an academic setting and partner court. Those materials, along with SQLite
records and trained weights, are excluded pending privacy, provenance, and
redistribution review. A public privacy statement does not grant permission to
redistribute them. The team has approved public display of this source
repository, but has not selected a reuse license. No open-source license is
asserted here.

GreggSpeak was developed as a Computer Engineering thesis project during
**2025–2026**. This source repository was prepared for public portfolio release
in **September 2026**. Git commit dates reflect publication activity, not the
historical development timeline.
