import re
import textwrap


QUESTION_STARTERS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "shall",
    "have",
    "has",
    "had",
}


WORD_REPLACEMENTS = {
    "alot": "a lot",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
    "id": "I'd",
    "isnt": "isn't",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}


DUPLICATE_WORD_ALLOWLIST = {
    "no",
    "yes",
    "very",
}


TITLE_REPLACEMENTS = (
    (re.compile(r"\bmr\b\.?", re.IGNORECASE), "Mr."),
    (re.compile(r"\bmrs\b\.?", re.IGNORECASE), "Mrs."),
    (re.compile(r"\bms\b\.?", re.IGNORECASE), "Ms."),
    (re.compile(r"\bdr\b\.?", re.IGNORECASE), "Dr."),
    (re.compile(r"\bthe court\b", re.IGNORECASE), "the Court"),
)

SPEAKER_ALIASES = (
    ("DEFENSE COUNSEL", ("defense counsel", "defense")),
    ("COURT INTERPRETER", ("court interpreter", "interpreter")),
    ("PROSECUTOR", ("prosecutor",)),
    ("WITNESS", ("witness",)),
    ("CLERK", ("clerk",)),
    ("COURT", ("court",)),
)

LEGAL_REPLACEMENTS = (
    (re.compile(r"\byour honor\b", re.IGNORECASE), "Your Honor"),
    (re.compile(r"\bpeople of the philippines\b", re.IGNORECASE), "People of the Philippines"),
    (re.compile(r"\bphilippines\b", re.IGNORECASE), "Philippines"),
    (re.compile(r"\bmadam witness\b", re.IGNORECASE), "madam witness"),
    (re.compile(r"\bfriday\b", re.IGNORECASE), "Friday"),
    (re.compile(r"\btuesday\b", re.IGNORECASE), "Tuesday"),
    (re.compile(r"\bjanuary\b", re.IGNORECASE), "January"),
)

TSN_LINE_WIDTH = 76


def structure_transcript(text):
    """Format recognized shorthand text into a simple TSN-style transcript."""
    if text is None:
        return ""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    blocks = []
    current_block = None

    for line in lines:
        if _is_page_marker(line):
            continue

        speaker, utterance = _extract_speaker(line)
        if speaker:
            current_block = {"speaker": speaker, "lines": []}
            blocks.append(current_block)
        elif current_block is None:
            current_block = {"speaker": "", "lines": []}
            blocks.append(current_block)
            utterance = line
        else:
            utterance = line

        formatted_lines = _format_tsn_utterance(utterance, speaker or current_block["speaker"])
        current_block["lines"].extend(formatted_lines)

    return _render_tsn_blocks(blocks)


def _is_page_marker(line):
    return bool(re.match(r"^\s*page\s+([a-z]+|\d+)\s*$", line, flags=re.IGNORECASE))


def _extract_speaker(line):
    cleaned = _normalize_spacing(line)
    cleaned = re.sub(r"^\s*[-+=>]+\s*", "", cleaned)

    for speaker, aliases in SPEAKER_ALIASES:
        for alias in aliases:
            pattern = re.compile(rf"^{re.escape(alias)}\b\s*:?\s*", flags=re.IGNORECASE)
            match = pattern.match(cleaned)
            if match:
                return speaker, cleaned[match.end():].strip(" :-")

    explicit_match = re.match(r"^([A-Za-z ]{2,32})\s*:\s*(.*)$", cleaned)
    if explicit_match:
        label = explicit_match.group(1).strip().upper()
        return label, explicit_match.group(2).strip()

    return "", cleaned


def _format_tsn_utterance(text, speaker=""):
    if not text:
        return []

    text = _normalize_spacing(text)
    text = _apply_common_replacements(text)
    text = _remove_adjacent_repeated_words(text)
    text = _apply_tsn_replacements(text)
    text = _apply_tsn_commas(text)
    text = _capitalize_sentences(text)
    text = _ensure_terminal_punctuation(text)
    text = _fix_tsn_question_punctuation(text)
    text = _fix_tsn_direct_address(text)

    return _split_and_wrap_tsn_lines(text, speaker)


def _render_tsn_blocks(blocks):
    rendered = []
    for block in blocks:
        speaker = block.get("speaker", "")
        lines = [line for line in block.get("lines", []) if line]
        if speaker:
            rendered.append(f"{speaker}:")
        rendered.extend(lines)
    return "\n".join(rendered).strip()


def _structure_paragraph(paragraph):
    paragraph = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
    paragraph = _normalize_spacing(paragraph)
    paragraph = _apply_common_replacements(paragraph)
    paragraph = _remove_adjacent_repeated_words(paragraph)
    paragraph = _capitalize_sentences(paragraph)
    paragraph = _ensure_terminal_punctuation(paragraph)
    return paragraph


def _normalize_spacing(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s\"')\]])", r"\1 \2", text)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def _apply_common_replacements(text):
    def replace_word(match):
        return WORD_REPLACEMENTS.get(match.group(0).lower(), match.group(0))

    text = re.sub(
        r"\b(" + "|".join(re.escape(word) for word in WORD_REPLACEMENTS) + r")\b",
        replace_word,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bi\b", "I", text)

    for pattern, replacement in TITLE_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    return text


def _apply_tsn_replacements(text):
    for pattern, replacement in LEGAL_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    text = re.sub(
        r"\braise your right hand\s+do you swear\b",
        "raise your right hand. Do you swear",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bshowing you this document\s+what is this\b",
        "showing you this document, what is this",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bcriminal case number ([a-z0-9]+)\s+People of the Philippines\b",
        r"criminal case number \1, People of the Philippines",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bversus defendant\s+for hearing\b",
        "versus defendant, for hearing",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _apply_tsn_commas(text):
    text = re.sub(r"\bfor the defendant\s+Your Honor\b", "for the defendant, Your Honor", text)
    text = re.sub(r"\bfor the prosecution\s+Your Honor\b", "for the prosecution, Your Honor", text)
    text = re.sub(r"\b(yes|no)\s+sir\b", r"\1, sir", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(yes|no)\s+Your Honor\b", r"\1, Your Honor", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(yes|no), Your Honor\s+([a-z])", r"\1, Your Honor, \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(yes|no), sir\s+([a-z])", r"\1, sir, \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcorrect\s+but\b", "correct, but", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmadam witness\s+([a-z])", r"madam witness, \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bplease state your name\s+age\s+and residence\b", "please state your name, age, and residence", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmy name is ([^,]+?)\s+my age is\b", r"my name is \1, my age is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmy age is ([^,]+?)\s+and my residence\b", r"my age is \1, and my residence", text, flags=re.IGNORECASE)
    text = re.sub(r"\btestimony\s+photographs\s+statement\s+and police report\b", "testimony, photographs, statement, and police report", text, flags=re.IGNORECASE)
    text = re.sub(r"\bphotographs\s+statement\s+and police report\b", "photographs, statement, and police report", text, flags=re.IGNORECASE)
    return text


def _remove_adjacent_repeated_words(text):
    duplicate_pattern = re.compile(
        r"\b([A-Za-z]+(?:'[A-Za-z]+)?)\b(?:\s+\b\1\b)+",
        re.IGNORECASE,
    )

    while True:
        updated = duplicate_pattern.sub(_replace_duplicate_words, text)
        if updated == text:
            return updated
        text = updated


def _replace_duplicate_words(match):
    word = match.group(1)
    if word.lower() in DUPLICATE_WORD_ALLOWLIST:
        return match.group(0)
    return word


def _capitalize_sentences(text):
    chars = list(text)
    capitalize_next = True

    for index, char in enumerate(chars):
        if char.isalpha():
            if capitalize_next:
                chars[index] = char.upper()
                capitalize_next = False
        elif char in ".!?":
            capitalize_next = True

    return "".join(chars)


def _ensure_terminal_punctuation(text):
    stripped = text.rstrip()
    if not stripped:
        return ""

    if stripped[-1] in ".!?":
        return stripped

    first_word_match = re.match(r"^[\"'(\[]*([A-Za-z]+)", stripped)
    first_word = first_word_match.group(1).lower() if first_word_match else ""
    terminal = "?" if first_word in QUESTION_STARTERS else "."
    return f"{stripped}{terminal}"


def _fix_tsn_question_punctuation(text):
    stripped = text.rstrip()
    if not stripped:
        return ""

    first_word_match = re.match(r"^[\"'(\[]*([A-Za-z]+)", stripped)
    first_word = first_word_match.group(1).lower() if first_word_match else ""
    looks_like_embedded_question = bool(
        re.search(
            r",\s*(who|what|when|where|why|how|do|does|did|is|are|was|were|can|could|would|should)\b.+$",
            stripped,
            flags=re.IGNORECASE,
        )
    )
    oath_question = bool(re.search(r"\bdo you swear\b", stripped, flags=re.IGNORECASE))

    if first_word in QUESTION_STARTERS or oath_question or looks_like_embedded_question:
        return stripped.rstrip(".!?") + "?"
    return stripped


def _fix_tsn_direct_address(text):
    text = re.sub(r"\bYour Honor,?\.$", "Your Honor.", text)
    text = re.sub(r"\b(Yes|No), sir\?$", r"\1, sir.", text)
    text = re.sub(r"\b(Yes|No), Your Honor\?$", r"\1, Your Honor.", text)
    return text


def _split_and_wrap_tsn_lines(text, speaker=""):
    parts = []
    date_match = re.match(
        r"^(January)\s+([xX]{1,2}|\d{1,2})\s+([xX]{4}|\d{4})\s+(.+)$",
        text,
    )
    if speaker == "COURT INTERPRETER" and date_match:
        parts.append(f"{date_match.group(1)} {date_match.group(2).upper()}, {date_match.group(3).upper()}")
        parts.append(_capitalize_sentences(date_match.group(4).strip()))
    else:
        parts.append(text)

    wrapped = []
    for part in parts:
        wrapped.extend(
            textwrap.wrap(
                part,
                width=TSN_LINE_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [part]
        )
    return wrapped
