from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid

import cv2
from PIL import Image

from app.config import BASE_DIR, DATA_DIR, INPUT_DIR, PAGE_ARTIFACTS_DIR
from app.db import add_page, create_batch, generate_transcript, update_batch_saved_location
from app.preprocessing.input_source import capture_camera_image, list_input_images, load_image
from app.recognition.infer import WordRecognizer
from app.segmentation.pipeline import segment_page
from app.storage.transcript_store import save_transcript
from app.translation.nlp import structure_transcript


WORD_REVIEW_THRESHOLD = 55.0
PAGE_REVIEW_THRESHOLD = 70.0


@dataclass
class PageInput:
    source_name: str
    image: Image.Image
    image_path: str = ""
    apply_fixed_crop: bool | None = None


@dataclass
class PageRecognition:
    page_number: int
    source_name: str
    transcript_text: str
    raw_transcript_text: str
    confidence: float
    rows_detected: int
    words_detected: int
    image_path: str = ""
    segmentation_image_path: str = ""
    low_confidence_words: int = 0
    review_notes: str = ""
    stage_image_paths: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RecognizedWord:
    text: str
    confidence: float
    line_index: int
    word_index: int
    top5: list[tuple[str, float]]
    raw_text: str = ""
    raw_confidence: float = 0.0
    selected_by_context: bool = False
    needs_review: bool = False
    review_reason: str = ""
    skipped: bool = False
    skip_reason: str = ""


def collect_pages(input_path: Path | None = None, no_camera: bool = False) -> list[PageInput]:
    pages: list[PageInput] = []

    if input_path is not None:
        resolved = input_path.expanduser().resolve()
        if resolved.is_file():
            return [PageInput(resolved.name, load_image(resolved), _relative_path(resolved))]
        if resolved.is_dir():
            for image_path in list_input_images(resolved):
                pages.append(PageInput(image_path.name, load_image(image_path), _relative_path(image_path)))
            return pages
        print(f"Input path not found: {resolved}")
        return []

    if not no_camera:
        camera_image = capture_camera_image()
        if camera_image is not None:
            return [PageInput("camera_capture", camera_image, "camera_capture", apply_fixed_crop=False)]
        print("Camera unavailable. Falling back to data/input/.")

    for image_path in list_input_images(INPUT_DIR):
        pages.append(PageInput(image_path.name, load_image(image_path), _relative_path(image_path)))
    return pages


def recognize_pages(
    pages: list[PageInput],
    use_uniform_crop: bool = False,
    recognizer: WordRecognizer | None = None,
    laptop_compatible: bool = False,
) -> list[PageRecognition]:
    active_recognizer = recognizer or WordRecognizer()
    results: list[PageRecognition] = []
    artifact_dir = PAGE_ARTIFACTS_DIR / (
        f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    for page_number, page in enumerate(pages, start=1):
        segmented = segment_page(
            page.image,
            use_uniform_crop=use_uniform_crop,
            laptop_compatible=laptop_compatible,
            apply_fixed_crop=page.apply_fixed_crop,
        )
        model_inputs = []
        word_refs = []
        words: list[RecognizedWord] = []

        for word in segmented.words:
            model_inputs.append(word.image_rgb)
            word_refs.append((word.line_index, word.word_index))

        predictions = active_recognizer.predict_batch(model_inputs, k=1) if model_inputs else []

        for (line_index, word_index), top5 in zip(word_refs, predictions):
            predicted, confidence = top5[0] if top5 else ("", 0.0)
            selected_by_context = False
            selection_reason = ""
            raw_text = top5[0][0] if top5 else ""
            raw_confidence = top5[0][1] if top5 else 0.0
            predicted_text = _display_label(predicted)
            needs_review, review_reason = _word_review_status(
                top5,
                confidence,
                selected_by_context,
                selection_reason,
            )
            words.append(
                RecognizedWord(
                    text=predicted_text,
                    confidence=confidence,
                    line_index=line_index,
                    word_index=word_index,
                    top5=top5,
                    raw_text=raw_text,
                    raw_confidence=raw_confidence,
                    selected_by_context=selected_by_context,
                    needs_review=needs_review,
                    review_reason=review_reason,
                )
            )

        words.sort(key=lambda word: (word.line_index, word.word_index))
        recognized_words = [word for word in words if not word.skipped]
        raw_transcript_text = _build_laptop_style_transcript(recognized_words)
        transcript_text = structure_transcript(raw_transcript_text)
        confidence = sum(word.confidence for word in recognized_words) / max(1, len(recognized_words))
        low_confidence_words = sum(1 for word in recognized_words if word.needs_review)
        review_notes = _page_review_notes(
            page_confidence=confidence,
            rows_detected=len(segmented.line_boxes),
            words_detected=len(recognized_words),
            low_confidence_words=low_confidence_words,
            context_selected_words=0,
            skipped_punctuation_crops=0,
        )
        overlay = _draw_prediction_overlay(segmented, words)
        image_path, segmentation_image_path, stage_image_paths = _save_review_artifacts(
            segmented=segmented,
            overlay_bgr=overlay,
            artifact_dir=artifact_dir,
            page_number=page_number,
        )
        results.append(
            PageRecognition(
                page_number=page_number,
                source_name=page.source_name,
                transcript_text=transcript_text,
                raw_transcript_text=raw_transcript_text,
                confidence=confidence,
                rows_detected=segmented.rows_detected,
                words_detected=len(recognized_words),
                image_path=image_path,
                segmentation_image_path=segmentation_image_path,
                low_confidence_words=low_confidence_words,
                review_notes=review_notes,
                stage_image_paths=stage_image_paths,
            )
        )

    return results


def recognize_and_store(
    pages: list[PageInput],
    title: str | None = None,
    filename: str | None = None,
    use_uniform_crop: bool = False,
    recognizer: WordRecognizer | None = None,
    laptop_compatible: bool = False,
) -> dict:
    page_results = recognize_pages(
        pages,
        use_uniform_crop=use_uniform_crop,
        recognizer=recognizer,
        laptop_compatible=laptop_compatible,
    )
    batch_title = (title or "").strip() or _default_title()
    batch_filename = (filename or "").strip() or _default_filename(page_results)

    batch_result = create_batch(
        title=batch_title,
        filename=batch_filename,
        total_pages=len(page_results),
    )
    if not batch_result.get("success"):
        return {
            "success": False,
            "message": batch_result.get("message", "Could not create transcript batch."),
            "pages": page_results,
        }

    batch = batch_result["data"]
    batch_id = batch["batch_id"]

    for page in page_results:
        add_page(
            batch_id=batch_id,
            page_number=page.page_number,
            raw_text=page.transcript_text,
            raw_transcript_text=page.raw_transcript_text,
            confidence=page.confidence,
            image_path=page.image_path,
            segmentation_image_path=page.segmentation_image_path,
            rows_detected=page.rows_detected,
            words_detected=page.words_detected,
            low_confidence_words=page.low_confidence_words,
            review_notes=page.review_notes,
        )

    transcript_result = generate_transcript(batch_id)
    if not transcript_result.get("success"):
        return {
            "success": False,
            "message": transcript_result.get("message", "Could not generate transcript."),
            "batch_id": batch_id,
            "pages": page_results,
        }

    transcript_text = transcript_result["data"].get("transcript_text") or ""
    transcript_path = save_transcript(transcript_text)
    update_batch_saved_location(batch_id, _relative_path(transcript_path))

    updated_result = generate_transcript(batch_id)
    return {
        "success": True,
        "message": "Recognition batch saved.",
        "batch": updated_result.get("data") or transcript_result.get("data"),
        "batch_id": batch_id,
        "pages": page_results,
        "transcript_path": transcript_path,
    }


def _default_title() -> str:
    return f"GreggSpeak Scan {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def _default_filename(page_results: list[PageRecognition]) -> str:
    if not page_results:
        return "camera_or_upload"
    if len(page_results) == 1:
        return page_results[0].source_name
    return f"{page_results[0].source_name} + {len(page_results) - 1} more"


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _relative_data_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def _save_review_artifacts(
    segmented,
    overlay_bgr,
    artifact_dir: Path,
    page_number: int,
) -> tuple[str, str, list[dict[str, str]]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    original_path = artifact_dir / f"page_{page_number:03d}_scanned.png"
    overlay_path = artifact_dir / f"page_{page_number:03d}_segmentation.png"

    cv2.imwrite(str(original_path), segmented.source_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    recognition_ready = getattr(segmented, "recognition_bgr", None)
    if recognition_ready is None:
        recognition_ready = segmented.deskewed_bgr
    stage_outputs = [
        ("Cropped Image", "01_cropped", segmented.source_bgr),
        ("Clean / Enhanced", "02_clean_enhanced", getattr(segmented, "enhanced_gray", None)),
        ("Recognition Ready", "03_recognition_ready", recognition_ready),
        ("Segmentation Mask", "04_segmentation_mask", segmented.binary),
        ("Line Segmentation", "05_line_segmentation", getattr(segmented, "line_overlay_bgr", None)),
        ("Word Segmentation", "06_word_segmentation", getattr(segmented, "word_overlay_bgr", None)),
        ("Model V5.3 Recognition", "07_recognition", overlay_bgr),
    ]
    stage_image_paths = []
    for label, suffix, image in stage_outputs:
        if image is None:
            continue
        path = artifact_dir / f"page_{page_number:03d}_{suffix}.png"
        cv2.imwrite(str(path), image)
        stage_image_paths.append({"label": label, "path": _relative_data_path(path)})

    return _relative_data_path(original_path), _relative_data_path(overlay_path), stage_image_paths


def _draw_prediction_overlay(segmented, words: list[RecognizedWord]):
    overlay = segmented.deskewed_bgr.copy()
    predictions = {
        (word.line_index, word.word_index): word
        for word in words
    }

    for line in segmented.line_boxes:
        cv2.rectangle(
            overlay,
            (line.x, line.y),
            (line.x + line.w, line.y + line.h),
            (0, 170, 70),
            3,
        )
        cv2.putText(
            overlay,
            f"L{line.index}",
            (line.x, max(26, line.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 115, 255),
            2,
            cv2.LINE_AA,
        )

    for word_boxes in segmented.word_boxes_by_line:
        for word in word_boxes:
            prediction = predictions.get((word.line_index, word.index))
            if prediction is None:
                label = f"W{word.index}"
                color = (255, 90, 0)
            elif prediction.skipped:
                label = f"W{word.index}: skip"
                color = (145, 145, 145)
            else:
                marker = "*" if prediction.selected_by_context else ""
                label = f"W{word.index}: {prediction.text}{marker} {prediction.confidence:.0f}%"
                color = (40, 60, 255) if prediction.needs_review else (255, 90, 0)
            cv2.rectangle(
                overlay,
                (word.x, word.y),
                (word.x + word.w, word.y + word.h),
                color,
                2,
            )
            cv2.putText(
                overlay,
                label[:38],
                (word.x, max(18, word.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    return overlay


def _display_label(label):
    return str(label).replace("_", " ")


def _build_laptop_style_transcript(words: list[RecognizedWord]) -> str:
    if not words:
        return ""

    lines: dict[int, list[RecognizedWord]] = {}
    for word in words:
        if word.text:
            lines.setdefault(word.line_index, []).append(word)

    transcript_lines = []
    for line_index in sorted(lines):
        line_words = sorted(lines[line_index], key=lambda word: word.word_index)
        transcript_lines.append(" ".join(word.text for word in line_words))
    return "\n".join(transcript_lines)


def _word_review_status(top5, confidence, selected_by_context, selection_reason):
    reasons = []
    if confidence < WORD_REVIEW_THRESHOLD:
        reasons.append(f"low confidence {confidence:.0f}%")
    if selected_by_context:
        reasons.append(selection_reason or "context-selected candidate")
    return bool(reasons), "; ".join(reasons)


def _page_review_notes(
    page_confidence,
    rows_detected,
    words_detected,
    low_confidence_words,
    context_selected_words,
    skipped_punctuation_crops=0,
):
    notes = []
    if rows_detected <= 0 or words_detected <= 0:
        notes.append("segmentation needs review")
    if page_confidence < PAGE_REVIEW_THRESHOLD:
        notes.append(f"page confidence below {PAGE_REVIEW_THRESHOLD:.0f}%")
    if low_confidence_words:
        notes.append(f"{low_confidence_words} word(s) need confidence review")
    if context_selected_words:
        notes.append(f"{context_selected_words} word(s) selected from Top-5 context")
    if skipped_punctuation_crops:
        notes.append(f"{skipped_punctuation_crops} punctuation/extra crop(s) skipped")
    return "; ".join(notes)
