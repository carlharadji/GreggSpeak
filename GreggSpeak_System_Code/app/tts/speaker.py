def speak_text(text: str) -> bool:
    if not text.strip():
        return False

    try:
        import pyttsx3  # type: ignore
    except Exception:
        return False

    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False
