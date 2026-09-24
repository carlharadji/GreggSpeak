GreggSpeak Laptop Version

Run from this folder after installing the requirements:

    python -m pip install -r requirements-laptop.txt
    python greggspeak_ui/main.py

Before starting the web companion, set unique GREGGSPEAK_WEB_PASSWORD and
GREGGSPEAK_SECRET_KEY environment values. The web app does not start without
them. Keep actual values in private local configuration, never in Git.

Included:
- greggspeak_ui desktop app and local Flask web backend
- V5.3 class labels for reference
- image processing in greggspeak_ui/backend/line_segmentation.py

Not included:
- training datasets and model weights (recognition requires a matching private model)
- reports and evaluation outputs
- historical Raspberry Pi deployment archives (the reviewed Pi source is separate)
- Python __pycache__ files
- previous scan artifacts and local database history

The app will create greggspeak_ui/data/greggspeak.db and page artifacts as needed.
