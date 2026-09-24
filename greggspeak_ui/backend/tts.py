import threading


class TTSService:
    def __init__(self):
        self._lock = threading.Lock()
        self._engine = None
        self._thread = None
        self.available = self._probe_available()

    def _probe_available(self):
        try:
            import pyttsx3  # type: ignore
            return pyttsx3 is not None
        except Exception:
            return False

    def speak(self, text, rate_multiplier=1.0):
        if not self.available or not text.strip():
            return False

        self.stop()

        def runner():
            try:
                import pyttsx3  # type: ignore

                engine = pyttsx3.init()
                default_rate = engine.getProperty("rate")
                engine.setProperty("rate", max(80, int(default_rate * rate_multiplier)))

                with self._lock:
                    self._engine = engine

                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._engine = None

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
