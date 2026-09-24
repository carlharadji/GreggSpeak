from pathlib import Path
import re


DATASET_FOLDER = Path(__file__).resolve().parent / "datasets" / "all_words_testing_uniform"
OUTPUT_FILE = Path(__file__).resolve().parent / "datasets" / "all_words_testing_uniform_words.txt"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def word_from_filename(path: Path) -> str:
    """Convert names like ability_000.png into ability."""
    return re.sub(r"_\d+$", "", path.stem)


def main():
    if not DATASET_FOLDER.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_FOLDER}")

    words = sorted(
        {
            word_from_filename(path)
            for path in DATASET_FOLDER.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        }
    )

    OUTPUT_FILE.write_text("\n".join(words) + "\n", encoding="utf-8")

    print(f"Saved {len(words)} words to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
