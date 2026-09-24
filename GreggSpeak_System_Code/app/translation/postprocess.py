import re

from app.translation.nlp import structure_transcript


def build_transcript(word_predictions: list[tuple[int, int, list[tuple[str, float]]]]) -> tuple[str, float]:
    lines: list[str] = []
    current_line = None
    current_words: list[str] = []
    confidences: list[float] = []

    for line_index, _word_index, top5 in word_predictions:
        if not top5:
            continue
        word, confidence = top5[0]
        if current_line is None:
            current_line = line_index
        if line_index != current_line:
            lines.append(" ".join(current_words))
            current_words = []
            current_line = line_index
        current_words.append(word)
        confidences.append(confidence)

    if current_words:
        lines.append(" ".join(current_words))

    text = structure_transcript("\n".join(_clean_line(line) for line in lines if line.strip()))
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_confidence


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    return line
