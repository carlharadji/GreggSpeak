import time

import customtkinter as ctk

from assets.theme import *
from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation


class AudioPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.speed_var = ctk.StringVar(value="1.0x")
        self.is_playing = False
        self.playback_job = None
        self.playback_start = None
        self.elapsed_before_stop = 0.0
        self.is_paused = False
        self.current_duration = 1.0
        self.current_text = ""

        build_header(self, "AUDIO", "#19c28e")

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(fill="both", expand=True)

        info_row = ctk.CTkFrame(content, fg_color="transparent")
        info_row.pack(fill="x", padx=24, pady=(18, 8))

        self.audio_title = ctk.CTkLabel(
            info_row,
            text="Audio Playback",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
        )
        self.audio_title.pack(side="left")

        self.page_indicator = ctk.CTkLabel(
            info_row,
            text="Page 1 of 1",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        )
        self.page_indicator.pack(side="right")

        self.play_button = ctk.CTkButton(
            content,
            text="▶",
            width=82,
            height=82,
            corner_radius=41,
            fg_color="#d0ae59",
            hover_color="#bc9b48",
            text_color=BG_COLOR,
            font=("Arial", 30, "bold"),
            command=self.toggle_playback,
        )
        self.play_button.pack(pady=(8, 12))

        self.audio_label = ctk.CTkLabel(
            content,
            text="Ready to speak transcript text",
            text_color="#e5eef9",
            font=("Georgia", 14),
        )
        self.audio_label.pack(pady=(0, 6))

        self.transcript_box = ctk.CTkTextbox(
            content,
            fg_color=CARD_COLOR,
            text_color=TEXT,
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            font=("Georgia", 14),
            wrap="word",
            activate_scrollbars=True,
            scrollbar_button_color="#7b7b7b",
            scrollbar_button_hover_color="#9a9a9a",
            height=100,
        )
        self.transcript_box.pack(padx=24, pady=(0, 12), fill="x")

        self.time_label = ctk.CTkLabel(
            content,
            text="00:00 / 00:00",
            text_color=TEXT,
            font=("Arial", 22),
        )
        self.time_label.pack(pady=(0, 14))

        self.progress = ctk.CTkProgressBar(
            content,
            width=450,
            height=8,
            corner_radius=8,
            progress_color="#d0ae59",
            fg_color="#172c5a",
        )
        self.progress.set(0)
        self.progress.pack(pady=(0, 18))

        speed_frame = ctk.CTkFrame(content, fg_color="transparent")
        speed_frame.pack(pady=(0, 8))

        self.create_speed_button(speed_frame, "0.75x", 0)
        self.create_speed_button(speed_frame, "1.0x", 1)
        self.create_speed_button(speed_frame, "1.25x", 2)

        page_nav_frame = ctk.CTkFrame(content, fg_color="transparent")
        page_nav_frame.pack(pady=(2, 8))

        self.prev_button = ctk.CTkButton(
            page_nav_frame,
            text="PREV PAGE",
            width=140,
            height=42,
            corner_radius=12,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 13, "bold"),
            command=self.show_prev_audio_page,
        )
        self.prev_button.grid(row=0, column=0, padx=8)

        self.next_button = ctk.CTkButton(
            page_nav_frame,
            text="NEXT PAGE",
            width=140,
            height=42,
            corner_radius=12,
            fg_color="#2f6fa3",
            hover_color="#285f8d",
            font=("Arial", 13, "bold"),
            command=self.show_next_audio_page,
        )
        self.next_button.grid(row=0, column=1, padx=8)

        ctk.CTkFrame(self, height=1, fg_color="#d0ae59").pack(fill="x")

        footer = ctk.CTkFrame(self, fg_color=BG_COLOR, height=82)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        btn_frame = ctk.CTkFrame(footer, fg_color="transparent")
        btn_frame.pack(expand=True)

        ctk.CTkButton(
            btn_frame,
            text="STOP",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 14, "bold"),
            command=self.stop_playback,
        ).grid(row=0, column=0, padx=8, pady=12)

        ctk.CTkButton(
            btn_frame,
            text="BACK",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 14, "bold"),
            command=lambda: confirm_navigation(
                self,
                controller,
                "TranscriptPage",
                "Are you sure you want to go back to the transcript?",
            ),
        ).grid(row=0, column=1, padx=8, pady=12)

        ctk.CTkButton(
            btn_frame,
            text="SAVE",
            width=160,
            height=50,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#2857c8",
            font=("Arial", 14, "bold"),
            command=lambda: controller.show_frame("SaveOptionsPage"),
        ).grid(row=0, column=2, padx=8, pady=12)

    def on_show(self):
        self.stop_playback()
        self.refresh_audio_page()

    def create_speed_button(self, parent, label, column):
        is_selected = self.speed_var.get() == label

        ctk.CTkButton(
            parent,
            text=label,
            width=90,
            height=48,
            corner_radius=14,
            fg_color="#d0ae59" if is_selected else "#172c5a",
            hover_color="#bc9b48" if is_selected else "#20376d",
            text_color=BG_COLOR if is_selected else TEXT,
            font=("Arial", 14, "bold"),
            command=lambda value=label: self.select_speed(value, parent),
        ).grid(row=0, column=column, padx=8)

    def select_speed(self, value, parent):
        self.speed_var.set(value)
        for widget in parent.winfo_children():
            widget.destroy()
        self.create_speed_button(parent, "0.75x", 0)
        self.create_speed_button(parent, "1.0x", 1)
        self.create_speed_button(parent, "1.25x", 2)
        self.refresh_audio_page()

    def toggle_playback(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if not self.current_text.strip():
            self.audio_label.configure(text="No transcript text available for speech.", text_color=WARNING)
            return

        speed = float(self.speed_var.get().replace("x", ""))
        text_to_speak = self.get_remaining_text()
        started = self.controller.tts.speak(text_to_speak, rate_multiplier=speed)
        if not started:
            self.audio_label.configure(text="Text-to-speech is unavailable on this device.", text_color=WARNING)
            return

        self.is_playing = True
        self.is_paused = False
        self.playback_start = time.time()
        self.update_play_button()
        self.schedule_progress_tick()

    def pause_playback(self):
        self.controller.tts.stop()
        if self.playback_start is not None:
            self.elapsed_before_stop = min(
                self.current_duration,
                self.elapsed_before_stop + (time.time() - self.playback_start),
            )
        self.is_playing = False
        self.is_paused = True
        self.playback_start = None
        if self.playback_job is not None:
            try:
                self.after_cancel(self.playback_job)
            except Exception:
                pass
            self.playback_job = None
        self.update_play_button()
        self.progress.set(min(1.0, self.elapsed_before_stop / max(self.current_duration, 1)))
        self.time_label.configure(
            text=f"{self.format_seconds(self.elapsed_before_stop)} / {self.format_seconds(self.current_duration)}"
        )

    def stop_playback(self):
        self.controller.tts.stop()
        self.is_playing = False
        self.is_paused = False
        self.elapsed_before_stop = 0.0
        self.playback_start = None
        if self.playback_job is not None:
            try:
                self.after_cancel(self.playback_job)
            except Exception:
                pass
            self.playback_job = None
        self.update_play_button()
        self.progress.set(0)
        self.time_label.configure(text=f"00:00 / {self.format_seconds(self.current_duration)}")

    def schedule_progress_tick(self):
        if self.playback_job is not None:
            try:
                self.after_cancel(self.playback_job)
            except Exception:
                pass
        self.playback_job = self.after(250, self.update_progress_tick)

    def update_progress_tick(self):
        if not self.is_playing or self.playback_start is None:
            self.playback_job = None
            return

        elapsed = self.elapsed_before_stop + (time.time() - self.playback_start)
        capped = min(elapsed, self.current_duration)
        progress_value = min(1.0, capped / max(self.current_duration, 1))

        self.progress.set(progress_value)
        self.time_label.configure(
            text=f"{self.format_seconds(capped)} / {self.format_seconds(self.current_duration)}"
        )

        if capped >= self.current_duration:
            self.is_playing = False
            self.is_paused = False
            self.elapsed_before_stop = 0.0
            self.playback_start = None
            self.playback_job = None
            self.update_play_button()
            return

        self.schedule_progress_tick()

    def update_play_button(self):
        self.play_button.configure(text="⏸" if self.is_playing else "▶")

    def show_prev_audio_page(self):
        total_pages = self.get_total_pages()
        current_index = getattr(self.controller, "current_transcript_page", 0)
        self.controller.current_transcript_page = (current_index - 1) % total_pages
        self.stop_playback()
        self.refresh_audio_page()

    def show_next_audio_page(self):
        total_pages = self.get_total_pages()
        current_index = getattr(self.controller, "current_transcript_page", 0)
        self.controller.current_transcript_page = (current_index + 1) % total_pages
        self.stop_playback()
        self.refresh_audio_page()

    def get_remaining_text(self):
        words = self.current_text.split()
        if not words or self.elapsed_before_stop <= 0:
            return self.current_text

        progress_ratio = min(1.0, self.elapsed_before_stop / max(self.current_duration, 1))
        start_index = min(len(words) - 1, int(progress_ratio * len(words)))
        return " ".join(words[start_index:])

    def get_current_transcript_text(self):
        transcript_page = self.controller.frames.get("TranscriptPage")
        if transcript_page is None:
            return ""
        getter = getattr(transcript_page, "get_current_transcript_text", None)
        if callable(getter):
            return getter()
        return ""

    def get_total_pages(self):
        transcript_page = self.controller.frames.get("TranscriptPage")
        if transcript_page is not None:
            getter = getattr(transcript_page, "get_total_pages", None)
            if callable(getter):
                return getter()
        return max(1, getattr(self.controller, "batch_pages", 1))

    def refresh_audio_page(self):
        total_pages = self.get_total_pages()
        current_index = min(
            getattr(self.controller, "current_transcript_page", 0),
            max(total_pages - 1, 0),
        )
        self.controller.current_transcript_page = current_index

        self.current_text = self.get_current_transcript_text()
        speed = float(self.speed_var.get().replace("x", ""))
        self.current_duration = self.estimate_duration_seconds(self.current_text, speed)
        self.elapsed_before_stop = 0.0
        self.is_paused = False

        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", self.current_text if self.current_text else "No transcript text available.")
        self.transcript_box.configure(state="disabled")

        self.page_indicator.configure(text=f"Page {current_index + 1} of {total_pages}")
        self.audio_title.configure(
            text="Audio Playback"

        )
        self.audio_label.configure(
            text=f"Speaking transcript page {current_index + 1}",
            text_color="#e5eef9",
        )
        self.progress.set(0)
        self.time_label.configure(text=f"00:00 / {self.format_seconds(self.current_duration)}")

        if total_pages > 1:
            self.prev_button.grid()
            self.next_button.grid()
        else:
            self.prev_button.grid_remove()
            self.next_button.grid_remove()

    def estimate_duration_seconds(self, text, speed):
        words = max(1, len(text.split()))
        base_wpm = 150
        seconds = (words / max(base_wpm * speed, 1)) * 60
        return max(3.0, seconds)

    def format_seconds(self, seconds_value):
        total_seconds = int(seconds_value)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def on_hide(self):
        self.stop_playback()
        hide_navigation_confirmation(self)
