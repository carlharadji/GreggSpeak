from datetime import datetime
from pathlib import Path

from app.config import TRANSCRIPTS_DIR


def save_transcript(text: str, output_dir: Path = TRANSCRIPTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"transcript_{timestamp}.txt"
    output_path.write_text(text.strip() + "\n", encoding="utf-8")
    return output_path
