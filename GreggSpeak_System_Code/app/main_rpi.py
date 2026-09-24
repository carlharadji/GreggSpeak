import argparse
from pathlib import Path

from app.config import USE_UNIFORM_CROP
from app.pipeline.runner import collect_pages, recognize_and_store
from app.tts.speaker import speak_text


def parse_args():
    parser = argparse.ArgumentParser(description="GreggSpeak Raspberry Pi runtime")
    parser.add_argument("--input", type=Path, help="Image file or directory for test fallback.")
    parser.add_argument("--no-camera", action="store_true", help="Skip camera capture and use input files only.")
    parser.add_argument("--uniform-crop", action="store_true", help="Use ink-crop centered 128x128 model input.")
    parser.add_argument("--speak", action="store_true", help="Try text-to-speech after recognition.")
    parser.add_argument("--title", help="Transcript batch title.")
    return parser.parse_args()


def main():
    args = parse_args()
    pages = collect_pages(args.input, no_camera=args.no_camera)

    if not pages:
        print("No image input found. Add images to data/input/ or connect a camera.")
        return 1

    result = recognize_and_store(
        pages,
        title=args.title,
        use_uniform_crop=args.uniform_crop or USE_UNIFORM_CROP,
    )
    if not result.get("success"):
        print(result.get("message", "Recognition failed."))
        return 1

    for page in result["pages"]:
        print(f"Processing page {page.page_number}: {page.source_name}")
        print(f"Rows detected: {page.rows_detected}")
        print(f"Words detected: {page.words_detected}")
        print(f"Average confidence: {page.confidence:.2f}%")
        print(page.transcript_text if page.transcript_text else "[No text recognized]")
        print()

    batch = result["batch"]
    transcript = batch.get("transcript_text") or ""

    print(f"Saved batch: {batch['batch_id']}")
    print(f"Saved transcript: {result['transcript_path']}")
    print(f"Overall confidence: {batch.get('confidence_avg') or 0:.2f}%")

    if args.speak:
        spoken = speak_text(transcript)
        if not spoken:
            print("TTS unavailable; transcript was printed and saved instead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
