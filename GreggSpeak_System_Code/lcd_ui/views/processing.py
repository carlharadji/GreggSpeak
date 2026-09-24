import tkinter as tk
import threading
import time

import customtkinter as ctk

from lcd_ui.assets.theme import *
from lcd_ui.views.ui_helpers import build_header


class ProcessingPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.spinner_angle = 0
        self.spinner_job = None
        self.text_job = None
        self.auto_next_job = None
        self.recognition_job = None
        self.timer_job = None
        self.worker_thread = None
        self.loading_step = 0
        self.is_active = False
        self.cancel_requested = False
        self.started_at = None

        build_header(self, "PROCESSING", WARNING)

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(fill="both", expand=True)

        center = ctk.CTkFrame(content, fg_color="transparent")
        center.place(relx=0.5, rely=0.54, anchor="center")

        self.spinner_canvas = tk.Canvas(
            center,
            width=86,
            height=86,
            bg=BG_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.spinner_canvas.pack(pady=(0, 18))

        self.spinner_canvas.create_arc(
            12,
            12,
            74,
            74,
            start=self.spinner_angle,
            extent=280,
            style="arc",
            outline="#d0ae59",
            width=6,
            tags="spinner",
        )

        self.loading_label = ctk.CTkLabel(
            center,
            text="Processing...",
            text_color=TEXT,
            font=("Arial", 20),
        )
        self.loading_label.pack()

        self.time_label = ctk.CTkLabel(
            center,
            text="Processing time: 0.0s",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        )
        self.time_label.pack(pady=(10, 0))

        self.cancel_button = ctk.CTkButton(
            center,
            text="CANCEL",
            width=150,
            height=44,
            corner_radius=12,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 13, "bold"),
            command=self.cancel_processing,
        )
        self.cancel_button.pack(pady=(22, 0))

        self.bind("<Destroy>", self._stop_jobs)

    def on_show(self):
        if self.is_active:
            return
        self.is_active = True
        self.cancel_requested = False
        self.started_at = time.perf_counter()
        self.spinner_angle = 0
        self.loading_step = 0
        self._set_processing_time(0.0)
        self.spinner_canvas.itemconfigure("spinner", start=self.spinner_angle)
        self.loading_label.configure(text="Processing...")
        self.time_label.configure(text="Processing time: 0.0s")
        self.cancel_button.configure(state="normal", text="CANCEL")
        self._animate_spinner()
        self._animate_text()
        self._update_timer()
        self._schedule_recognition()

    def on_hide(self):
        self.is_active = False
        self._cancel_jobs()

    def _animate_spinner(self):
        if not self.is_active:
            self.spinner_job = None
            return
        self.spinner_angle = (self.spinner_angle - 12) % 360
        self.spinner_canvas.itemconfigure("spinner", start=self.spinner_angle)
        self.spinner_job = self.after(50, self._animate_spinner)

    def _animate_text(self):
        if not self.is_active:
            self.text_job = None
            return
        dots = "." * ((self.loading_step % 3) + 1)
        self.loading_label.configure(text=f"Processing{dots}")
        self.loading_step += 1
        self.text_job = self.after(450, self._animate_text)

    def _update_timer(self):
        if not self.is_active or self.started_at is None:
            self.timer_job = None
            return
        elapsed = time.perf_counter() - self.started_at
        self._set_processing_time(elapsed)
        self.time_label.configure(text=self.controller.processing_time_text)
        self.timer_job = self.after(250, self._update_timer)

    def _schedule_recognition(self):
        if not self.is_active:
            self.recognition_job = None
            return
        self.recognition_job = self.after(200, self._start_recognition)

    def _start_recognition(self):
        self.recognition_job = None
        pages = list(
            getattr(self.controller, "recognition_pages", None)
            or getattr(self.controller, "captured_pages", [])
            or []
        )
        if not pages:
            self.controller.transcript_results = []
            self._finish_processing(0.0)
            return

        self.loading_label.configure(text="Recognizing shorthand words...")
        self.worker_thread = threading.Thread(
            target=self._recognize_pages_worker,
            args=(pages,),
            daemon=True,
        )
        self.worker_thread.start()

    def _recognize_pages_worker(self, pages):
        start_time = self.started_at or time.perf_counter()
        error_text = None
        try:
            apply_fixed_crop = bool(getattr(self.controller, "upload_test_mode", False))
            results = self.controller.recognition.recognize_pages(
                pages,
                laptop_compatible=True,
                apply_fixed_crop=apply_fixed_crop,
            )
        except Exception as exc:
            results = []
            error_text = str(exc)

        if self.cancel_requested:
            return

        self.controller.recognition_error = error_text
        elapsed = time.perf_counter() - start_time
        self._set_processing_time(elapsed)
        self.controller.transcript_results = results
        if self.is_active and not self.cancel_requested:
            self.after(0, lambda: self._finish_processing(elapsed))

    def _finish_processing(self, elapsed=None):
        if not self.is_active or self.cancel_requested:
            return
        if elapsed is None and self.started_at is not None:
            elapsed = time.perf_counter() - self.started_at
        elapsed = float(elapsed or 0.0)
        self._set_processing_time(elapsed)
        self.time_label.configure(text=self.controller.processing_time_text)
        self.loading_label.configure(text="Processing complete")
        self.cancel_button.configure(state="disabled", text="DONE")
        for job_name in ("text_job", "timer_job"):
            job = getattr(self, job_name, None)
            if job is None:
                continue
            try:
                self.after_cancel(job)
            except Exception:
                pass
            setattr(self, job_name, None)
        self.auto_next_job = self.after(750, self._go_to_transcript)

    def _go_to_transcript(self):
        self.auto_next_job = None
        if self.is_active and not self.cancel_requested:
            if getattr(self.controller, "upload_test_mode", False) and getattr(
                self.controller,
                "transcript_results",
                None,
            ):
                self.controller.show_frame("ProcessingResultsPage")
            else:
                self.controller.show_frame("TranscriptPage")

    def cancel_processing(self):
        self.cancel_requested = True
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = time.perf_counter() - self.started_at
        self._set_processing_time(elapsed)
        self.is_active = False
        self._cancel_jobs()
        self.controller.transcript_results = []
        self.controller.recognition_error = "Processing cancelled."
        self.controller.show_frame("ScanningPage")

    @staticmethod
    def _format_elapsed(seconds):
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining = int(seconds % 60)
        return f"{minutes}m {remaining:02d}s"

    def _set_processing_time(self, seconds):
        text = f"Processing time: {self._format_elapsed(seconds)}"
        self.controller.processing_time_seconds = float(seconds)
        self.controller.processing_time_text = text

    def _cancel_jobs(self):
        for job_name in ("spinner_job", "text_job", "auto_next_job", "recognition_job", "timer_job"):
            job = getattr(self, job_name, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, job_name, None)

    def _stop_jobs(self, _event=None):
        self.is_active = False
        self._cancel_jobs()
