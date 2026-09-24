from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


UI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DATA_DIR = UI_ROOT / "data"
PAGE_ARTIFACTS_DIR = UI_DATA_DIR / "page_artifacts"
DEFAULT_MODEL_FILES = (
    (
        PROJECT_ROOT / "models" / "greggspeak_all_words_v5_3.keras",
        PROJECT_ROOT / "models" / "labels_all_words_v5_3.txt",
    ),
    (
        PROJECT_ROOT / "models" / "best_all_words_v5_3.keras",
        PROJECT_ROOT / "models" / "labels_all_words_v5_3.txt",
    ),
    (
        PROJECT_ROOT / "models" / "greggspeak_all_words_v5_demo.keras",
        PROJECT_ROOT / "models" / "labels_all_words_v5_demo.txt",
    ),
    (
        PROJECT_ROOT / "models" / "best_all_words_v5_demo.keras",
        PROJECT_ROOT / "models" / "labels_all_words_v5_demo.txt",
    ),
    (
        PROJECT_ROOT / "models" / "greggspeak_all_words_v5_2.keras",
        PROJECT_ROOT / "models" / "labels_all_words_v5_2.txt",
    ),
    (
        PROJECT_ROOT / "models" / "best_all_words_v5_2.keras",
        PROJECT_ROOT / "models" / "labels_all_words_v5_2.txt",
    ),
)

try:
    from .line_segmentation import (
        SegmentationArtifacts,
        clean_enhance_image,
        create_segmentation_mask,
        crop_word_images,
        draw_line_overlay,
        draw_word_overlay,
        normalize_word_crop_for_model,
        prepare_recognition_ready_image,
        resize_word_crop_for_model,
        segment_lines_from_mask,
        segment_words_from_mask,
    )
except ImportError:
    try:
        from backend.line_segmentation import (  # type: ignore
            SegmentationArtifacts,
            clean_enhance_image,
            create_segmentation_mask,
            crop_word_images,
            draw_line_overlay,
            draw_word_overlay,
            normalize_word_crop_for_model,
            prepare_recognition_ready_image,
            resize_word_crop_for_model,
            segment_lines_from_mask,
            segment_words_from_mask,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"backend", "backend.line_segmentation"}:
            raise
        from line_segmentation import (  # type: ignore
            SegmentationArtifacts,
            clean_enhance_image,
            create_segmentation_mask,
            crop_word_images,
            draw_line_overlay,
            draw_word_overlay,
            normalize_word_crop_for_model,
            prepare_recognition_ready_image,
            resize_word_crop_for_model,
            segment_lines_from_mask,
            segment_words_from_mask,
        )

try:
    from .nlp import structure_transcript
except ImportError:
    try:
        from backend.nlp import structure_transcript  # type: ignore
    except Exception:
        from nlp import structure_transcript  # type: ignore

WORD_REVIEW_THRESHOLD = 55.0
PAGE_REVIEW_THRESHOLD = 70.0
TOP5_CLOSE_GAP = 8.0
RERANK_MAX_TOP1_CONFIDENCE = 68.0

COMMON_CONTEXT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

COURT_CONTEXT_WORDS = {
    "accused",
    "case",
    "court",
    "evidence",
    "judge",
    "justice",
    "law",
    "name",
    "officer",
    "question",
    "record",
    "statement",
    "testified",
    "testimony",
    "witness",
}


@dataclass
class RecognizedWord:
    text: str
    confidence: float
    line_index: int
    word_index: int
    top5: list[tuple[str, float]] = field(default_factory=list)
    raw_text: str = ""
    raw_confidence: float = 0.0
    selected_by_context: bool = False
    needs_review: bool = False
    review_reason: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class RecognizedPage:
    page_number: int
    transcript_text: str
    confidence: float
    words: list[RecognizedWord]
    rows_detected: int
    words_detected: int
    raw_transcript_text: str = ""
    image_path: str = ""
    segmentation_image_path: str = ""
    low_confidence_words: int = 0
    review_notes: str = ""
    error: str | None = None


class RecognitionService:
    def __init__(
        self,
        model_path=None,
        labels_path=None,
        use_uniform_crop=False,
        skip_punctuation_like_crops=False,
    ):
        default_model_path, default_labels_path = self._default_model_and_labels()
        self.model_path = Path(model_path or default_model_path)
        self.labels_path = Path(labels_path or default_labels_path)
        self.use_uniform_crop = use_uniform_crop
        self.skip_punctuation_like_crops = skip_punctuation_like_crops
        self.model = None
        self.labels = []

    @staticmethod
    def _default_model_and_labels():
        for model_path, labels_path in DEFAULT_MODEL_FILES:
            if model_path.exists() and labels_path.exists():
                return model_path, labels_path
        return DEFAULT_MODEL_FILES[-1]

    def is_available(self):
        return (
            cv2 is not None
            and tf is not None
            and Image is not None
            and self.model_path.exists()
            and self.labels_path.exists()
        )

    def load(self):
        if self.model is not None and self.labels:
            return
        if not self.is_available():
            raise RuntimeError("Recognition dependencies or model files are unavailable.")

        self.model = tf.keras.models.load_model(self.model_path)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f if line.strip()]

    def recognize_pages(self, pages):
        self.load()
        results = []
        artifact_dir = PAGE_ARTIFACTS_DIR / (
            f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        for page_number, page_image in enumerate(pages, start=1):
            results.append(self.recognize_page(page_image, page_number, artifact_dir))
        return results

    def recognize_page(self, page_image, page_number=1, artifact_dir=None):
        try:
            self.load()
            image_bgr = self._pil_to_bgr(page_image)
            artifacts = self._segment_page(image_bgr)
            words, raw_transcript_text = self._recognize_words_gui_style(artifacts)
            recognized_words = words

            page_confidence = sum(word.confidence for word in recognized_words) / max(1, len(recognized_words))
            low_confidence_words = sum(1 for word in recognized_words if word.needs_review)
            review_notes = self._page_review_notes(
                page_confidence=page_confidence,
                rows_detected=len(artifacts.line_boxes),
                words_detected=len(recognized_words),
                low_confidence_words=low_confidence_words,
                context_selected_words=0,
                skipped_punctuation_crops=0,
            )
            overlay = self._draw_prediction_overlay(artifacts, words)
            image_path, segmentation_image_path = self._save_review_artifacts(
                image_bgr=image_bgr,
                overlay_bgr=overlay,
                artifact_dir=artifact_dir,
                page_number=page_number,
            )
            return RecognizedPage(
                page_number=page_number,
                transcript_text=structure_transcript(raw_transcript_text),
                confidence=page_confidence,
                words=words,
                rows_detected=len(artifacts.line_boxes),
                words_detected=len(recognized_words),
                raw_transcript_text=raw_transcript_text,
                image_path=image_path,
                segmentation_image_path=segmentation_image_path,
                low_confidence_words=low_confidence_words,
                review_notes=review_notes,
            )
        except Exception as exc:
            return RecognizedPage(
                page_number=page_number,
                transcript_text="",
                confidence=0.0,
                words=[],
                rows_detected=0,
                words_detected=0,
                low_confidence_words=0,
                review_notes=str(exc),
                error=str(exc),
            )

    def _prepare_word_crop(self, word_crop_bgr):
        if self.use_uniform_crop:
            return normalize_word_crop_for_model(word_crop_bgr)
        return resize_word_crop_for_model(word_crop_bgr)

    def _recognize_words_gui_style(self, artifacts):
        recognition_source = getattr(artifacts, "recognition_bgr", artifacts.deskewed_bgr)
        model_inputs = []
        word_refs = []

        for word_boxes in artifacts.word_boxes_by_line:
            for word_box in word_boxes:
                word_crop = self._crop_word_from_recognition_source(recognition_source, word_box)
                model_crop = resize_word_crop_for_model(word_crop)
                model_rgb = cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
                model_inputs.append(model_rgb)
                word_refs.append((word_box.line_index, word_box.index))

        words = []
        if model_inputs:
            probabilities = self.model.predict(np.stack(model_inputs, axis=0), verbose=0)
            for (line_idx, word_idx), probs in zip(word_refs, probabilities):
                class_index = int(np.argmax(probs))
                label = self.labels[class_index] if 0 <= class_index < len(self.labels) else f"class_{class_index}"
                confidence = float(probs[class_index]) * 100.0
                top5_idx = probs.argsort()[-5:][::-1]
                top5 = [
                    (self.labels[i] if 0 <= i < len(self.labels) else f"class_{i}", float(probs[i]) * 100.0)
                    for i in top5_idx
                ]
                display_label = self._display_label(label)
                words.append(
                    RecognizedWord(
                        text=display_label,
                        confidence=confidence,
                        line_index=line_idx,
                        word_index=word_idx,
                        top5=top5,
                        raw_text=label,
                        raw_confidence=confidence,
                        needs_review=confidence < WORD_REVIEW_THRESHOLD,
                        review_reason=f"low confidence {confidence:.0f}%" if confidence < WORD_REVIEW_THRESHOLD else "",
                    )
                )

        words.sort(key=lambda word: (word.line_index, word.word_index))
        line_texts = []
        for line in artifacts.line_boxes:
            current_words = [
                word.text
                for word in words
                if word.line_index == line.index and word.text
            ]
            if current_words:
                line_texts.append(" ".join(current_words))
        return words, "\n".join(line_texts)

    @staticmethod
    def _crop_word_from_recognition_source(recognition_source, word, pad=2):
        if len(recognition_source.shape) == 2:
            image_bgr = cv2.cvtColor(recognition_source, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = recognition_source.copy()

        height, width = image_bgr.shape[:2]
        dynamic_pad = max(pad, min(12, int(round(max(word.w, word.h) * 0.08))))
        x1 = max(0, word.x - dynamic_pad)
        y1 = max(0, word.y - dynamic_pad)
        x2 = min(width, word.x + word.w + dynamic_pad)
        y2 = min(height, word.y + word.h + dynamic_pad)
        if x2 <= x1 or y2 <= y1:
            return np.full((128, 128, 3), 255, dtype=np.uint8)
        return image_bgr[y1:y2, x1:x2].copy()

    @staticmethod
    def _segment_page(image_bgr):
        enhanced_gray = clean_enhance_image(image_bgr)
        recognition_bgr = prepare_recognition_ready_image(image_bgr)
        segmentation_mask = create_segmentation_mask(enhanced_gray)
        line_boxes = segment_lines_from_mask(segmentation_mask)
        word_boxes_by_line = segment_words_from_mask(segmentation_mask, line_boxes)
        overlay = draw_word_overlay(
            draw_line_overlay(recognition_bgr, line_boxes),
            word_boxes_by_line,
        )
        return SegmentationArtifacts(
            grayscale=enhanced_gray,
            enhanced_gray=enhanced_gray,
            recognition_bgr=recognition_bgr,
            binary=segmentation_mask,
            deskewed_bgr=recognition_bgr,
            overlay=overlay,
            line_boxes=line_boxes,
            word_boxes_by_line=word_boxes_by_line,
            deskew_angle=0.0,
        )

    @staticmethod
    def _draw_prediction_overlay(artifacts, words):
        overlay = artifacts.deskewed_bgr.copy()
        predictions = {
            (word.line_index, word.word_index): word
            for word in words
        }

        for line in artifacts.line_boxes:
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

        for word_boxes in artifacts.word_boxes_by_line:
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
                label_y = max(18, word.y - 6)
                cv2.putText(
                    overlay,
                    label[:38],
                    (word.x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return overlay

    @staticmethod
    def _display_label(label):
        return str(label).replace("_", " ")

    @staticmethod
    def _looks_like_punctuation_crop(word_crop, word_box):
        if word_crop is None or word_crop.size == 0:
            return True, "empty crop"

        if len(word_crop.shape) == 2:
            gray = word_crop
        else:
            gray = cv2.cvtColor(word_crop, cv2.COLOR_BGR2GRAY)

        ink_mask = gray < 245
        ink_area = int(np.count_nonzero(ink_mask))
        if ink_area < 8:
            return True, "almost no ink"

        ys, xs = np.where(ink_mask)
        if len(xs) == 0 or len(ys) == 0:
            return True, "no ink"

        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_area = max(1, bbox_w * bbox_h)
        density = ink_area / bbox_area

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            ink_mask.astype(np.uint8),
            connectivity=8,
        )
        usable_components = 0
        for label in range(1, component_count):
            _, _, cw, ch, area = stats[label]
            if area >= 6 and cw >= 2 and ch >= 2:
                usable_components += 1

        compact_box = word_box.w <= 42 and word_box.h <= 42
        tiny_ink = bbox_w <= 12 and bbox_h <= 12 and ink_area <= 70
        dot_or_comma = bbox_w <= 18 and bbox_h <= 22 and ink_area <= 130 and usable_components <= 2
        small_cross_or_extra = (
            compact_box
            and bbox_w <= 28
            and bbox_h <= 28
            and ink_area <= 180
            and usable_components <= 2
            and density <= 0.45
        )

        if tiny_ink:
            return True, "tiny dot-like mark"
        if dot_or_comma:
            return True, "small punctuation-like mark"
        if small_cross_or_extra:
            return True, "small x/extra-like mark"

        return False, ""

    @staticmethod
    def _select_top5_candidate(top5, previous_words):
        if not top5:
            return "", 0.0, False, ""

        top_word, top_confidence = top5[0]
        selected_word = top_word
        selected_confidence = top_confidence
        selected_score = RecognitionService._candidate_score(top_word, previous_words)

        if top_confidence > RERANK_MAX_TOP1_CONFIDENCE:
            return selected_word, selected_confidence, False, ""

        for candidate, confidence in top5[1:]:
            if top_confidence - confidence > TOP5_CLOSE_GAP:
                continue
            candidate_score = RecognitionService._candidate_score(candidate, previous_words)
            if candidate_score >= selected_score + 6:
                return candidate, confidence, True, f"Top-5 context selected over {top_word}"

        return selected_word, selected_confidence, False, ""

    @staticmethod
    def _candidate_score(candidate, previous_words):
        word = RecognitionService._display_label(candidate).lower()
        previous = [RecognitionService._display_label(item).lower() for item in previous_words[-3:]]
        score = 0

        if word in COMMON_CONTEXT_WORDS:
            score += 4
        if word in COURT_CONTEXT_WORDS:
            score += 6
        if len(word) == 1 and word not in {"a", "i"}:
            score -= 5
        if previous and previous[-1] in {"the", "a", "an"} and word in COURT_CONTEXT_WORDS:
            score += 3
        if previous and previous[-1] in {"mr", "mrs", "ms", "dr"}:
            score += 2
        return score

    @staticmethod
    def _word_review_status(top5, confidence, selected_by_context, selection_reason):
        reasons = []
        if confidence < WORD_REVIEW_THRESHOLD:
            reasons.append(f"low confidence {confidence:.0f}%")
        if len(top5) > 1 and abs(top5[0][1] - top5[1][1]) <= TOP5_CLOSE_GAP:
            reasons.append("close Top-5 candidates")
        if selected_by_context:
            reasons.append(selection_reason or "context-selected candidate")
        return bool(reasons), "; ".join(reasons)

    @staticmethod
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

    @staticmethod
    def _save_review_artifacts(image_bgr, overlay_bgr, artifact_dir, page_number):
        if artifact_dir is None:
            artifact_dir = PAGE_ARTIFACTS_DIR / (
                f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        original_path = artifact_dir / f"page_{page_number:03d}_scanned.png"
        overlay_path = artifact_dir / f"page_{page_number:03d}_segmentation.png"

        cv2.imwrite(str(original_path), image_bgr)
        cv2.imwrite(str(overlay_path), overlay_bgr)

        return (
            RecognitionService._relative_artifact_path(original_path),
            RecognitionService._relative_artifact_path(overlay_path),
        )

    @staticmethod
    def _relative_artifact_path(path):
        try:
            return Path(path).resolve().relative_to(UI_DATA_DIR.resolve()).as_posix()
        except Exception:
            return str(path)

    @staticmethod
    def _pil_to_bgr(page_image):
        if Image is not None and isinstance(page_image, Image.Image):
            rgb = np.array(page_image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(page_image, np.ndarray):
            if len(page_image.shape) == 2:
                return cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)
            return page_image.copy()
        raise TypeError("Unsupported page image type.")
