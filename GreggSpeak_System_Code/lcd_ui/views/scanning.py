import threading
from tkinter import filedialog

import customtkinter as ctk

try:
    from lcd_ui.assets.theme import *
    from lcd_ui.views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation
except ImportError:
    from assets.theme import *
    from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation

from app.hardware.feeder_calibration import (
    feed_duration_for_page,
    MAX_DURATION_MS,
    MAX_SPEED,
    MIN_DURATION_MS,
    MIN_SPEED,
    PAPER_SIZES,
)

try:
    from lcd_ui.backend.document_crop import crop_document_from_camera
except Exception:
    try:
        from backend.document_crop import crop_document_from_camera
    except Exception:
        crop_document_from_camera = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from picamera2 import Picamera2  # type: ignore
except Exception:
    Picamera2 = None

try:
    from PIL import Image, ImageDraw, ImageOps
except Exception:
    Image = None
    ImageDraw = None
    ImageOps = None


CAMERA_ROTATE_COUNTERCLOCKWISE = True
CAMERA_DOCUMENT_CROP = True
FINAL_PAGE_RUNOUT_MS = 10000


def format_duration_ms(duration_ms):
    seconds = duration_ms / 1000
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:.1f}s"


class ScanningPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.camera = None
        self.camera_backend = None
        self.preview_job = None
        self.camera_retry_job = None
        self.camera_retry_attempts = 0
        self.camera_frame_failures = 0
        self.capture_popup_job = None
        self.batch_scan_job = None
        self.current_frame = None
        self.current_camera_frame = None
        self.preview_image = None
        self.preview_modal_image = None
        self.captured_pages = []
        self.captured_page_paths = []
        self.recognition_pages = []
        self.pages_scanned = 0
        self.preview_index = 0
        self.selected_paper_size = ""
        self.current_calibration = None
        self.calibration_saved = False
        self.batch_scan_active = False
        self.scan_cancel_requested = False
        self.calibration_test_active = False
        self.feed_worker = None
        self.sensor_ready = False

        build_header(self, "SCANNING", WARNING, divider_color="#1f3b57")

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(fill="both", expand=True)

        ctk.CTkLabel(
            content,
            text="Current captured page",
            text_color=TEXT,
            font=("Arial", 18),
        ).pack(pady=(8, 6))

        preview_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#7f6d49",
            height=140,
        )
        preview_card.pack(padx=24, pady=(0, 8), fill="x")
        preview_card.pack_propagate(False)

        self.preview_label = ctk.CTkLabel(
            preview_card,
            text="Starting camera preview...",
            text_color="#d3bac0",
            font=("Arial", 16, "bold"),
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        summary_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#1f3b57",
        )
        summary_card.pack(padx=24, pady=(0, 8), fill="x")

        summary_card.grid_columnconfigure(0, weight=1)
        summary_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            summary_card,
            text="Batch Summary",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 3))

        ctk.CTkLabel(
            summary_card,
            text="Pages captured:",
            text_color=TEXT,
            font=("Arial", 13),
        ).grid(row=1, column=0, sticky="w", padx=16, pady=2)

        self.pages_value = ctk.CTkLabel(
            summary_card,
            text="0",
            text_color=ACCENT,
            font=("Arial", 16, "bold"),
        )
        self.pages_value.grid(row=1, column=1, sticky="e", padx=16, pady=2)

        ctk.CTkLabel(
            summary_card,
            text="Paper size:",
            text_color=TEXT,
            font=("Arial", 13),
        ).grid(row=2, column=0, sticky="w", padx=16, pady=2)

        self.paper_size_value = ctk.CTkLabel(
            summary_card,
            text="Not selected",
            text_color=SUBTEXT,
            font=("Arial", 13, "bold"),
        )
        self.paper_size_value.grid(row=2, column=1, sticky="e", padx=16, pady=2)

        ctk.CTkLabel(
            summary_card,
            text="Feeder timing:",
            text_color=TEXT,
            font=("Arial", 13),
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(2, 8))

        self.calibration_value = ctk.CTkLabel(
            summary_card,
            text="Needs setup",
            text_color=SUBTEXT,
            font=("Arial", 13, "bold"),
        )
        self.calibration_value.grid(row=3, column=1, sticky="e", padx=16, pady=(2, 8))

        self.prompt_label = ctk.CTkLabel(
            content,
            text="Camera preview ready. Scan the loaded feeder batch.",
            text_color="#e3c2c8",
            font=("Georgia", 15),
        )
        self.prompt_label.pack(pady=(4, 4))

        self.helper_label = ctk.CTkLabel(
            content,
            text="Batch scan uses the feeder IR sensor and stops when no paper is detected.",
            text_color=SUBTEXT,
            font=("Arial", 12),
        )
        self.helper_label.pack(pady=(0, 4))

        button_row = ctk.CTkFrame(content, fg_color="transparent")
        button_row.pack(side="bottom", pady=(0, 8), padx=14, fill="x")
        button_row.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        self.scan_button = ctk.CTkButton(
            button_row,
            text="SCAN BATCH",
            width=118,
            height=44,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#163bb5",
            font=("Arial", 13, "bold"),
            command=self.start_batch_scan,
        )
        self.scan_button.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        ctk.CTkButton(
            button_row,
            text="UPLOAD",
            width=100,
            height=44,
            corner_radius=12,
            fg_color="#2f8f68",
            hover_color="#246f51",
            font=("Arial", 13, "bold"),
            command=self.upload_pages,
        ).grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        ctk.CTkButton(
            button_row,
            text="SETUP",
            width=100,
            height=44,
            corner_radius=12,
            fg_color="#7f6d49",
            hover_color="#6b5b3d",
            font=("Arial", 13, "bold"),
            command=self.show_feeder_setup_overlay,
        ).grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        ctk.CTkButton(
            button_row,
            text="PREVIEW",
            width=100,
            height=44,
            corner_radius=12,
            fg_color="#254265",
            hover_color="#1c3150",
            font=("Arial", 13, "bold"),
            command=self.open_preview_modal,
        ).grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        ctk.CTkButton(
            button_row,
            text="DONE",
            width=100,
            height=44,
            corner_radius=12,
            fg_color="#2f8f68",
            hover_color="#246f51",
            font=("Arial", 13, "bold"),
            command=self.finish_batch,
        ).grid(row=0, column=4, padx=4, pady=4, sticky="ew")

        ctk.CTkButton(
            button_row,
            text="CANCEL",
            width=100,
            height=44,
            corner_radius=12,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 13, "bold"),
            command=self.cancel_batch_scan,
        ).grid(row=0, column=5, padx=4, pady=4, sticky="ew")

        self.capture_overlay = ctk.CTkFrame(self, fg_color="#050b12")
        self.capture_card = ctk.CTkFrame(
            self.capture_overlay,
            fg_color="#2c4597",
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            width=320,
            height=180,
        )
        self.capture_card.place(relx=0.5, rely=0.5, anchor="center")
        self.capture_card.pack_propagate(False)

        ctk.CTkLabel(
            self.capture_card,
            text="Page Captured",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
        ).pack(pady=(28, 10))

        self.capture_message = ctk.CTkLabel(
            self.capture_card,
            text="Transcript page 1 added to the batch.",
            text_color=TEXT,
            font=("Arial", 14),
            wraplength=240,
            justify="center",
        )
        self.capture_message.pack(padx=20, pady=(0, 10))

        self.preview_overlay = ctk.CTkFrame(self, fg_color="#050b12")
        self.preview_card = ctk.CTkFrame(
            self.preview_overlay,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            width=520,
            height=360,
        )
        self.preview_card.place(relx=0.5, rely=0.5, anchor="center")
        self.preview_card.pack_propagate(False)

        self.preview_title = ctk.CTkLabel(
            self.preview_card,
            text="Captured Page 1 of 1",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
        )
        self.preview_title.pack(pady=(16, 10))

        self.preview_display_frame = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        self.preview_display_frame.pack(pady=(0, 10))
        self.preview_modal_label = None
        self.reset_preview_modal_label("No captured pages yet.")

        self.delete_confirm_overlay = ctk.CTkFrame(self.preview_overlay, fg_color="#050b12")
        self.delete_confirm_card = ctk.CTkFrame(
            self.delete_confirm_overlay,
            fg_color="#203557",
            corner_radius=14,
            border_width=1,
            border_color="#d0ae59",
            width=300,
            height=150,
        )
        self.delete_confirm_card.place(relx=0.5, rely=0.5, anchor="center")
        self.delete_confirm_card.pack_propagate(False)

        self.delete_confirm_message = ctk.CTkLabel(
            self.delete_confirm_card,
            text="Delete this captured page from the batch?",
            text_color="#f3c6c6",
            font=("Arial", 13),
            wraplength=220,
            justify="center",
        )
        self.delete_confirm_message.pack(pady=(24, 12), padx=20)

        delete_buttons = ctk.CTkFrame(self.delete_confirm_card, fg_color="transparent")
        delete_buttons.pack()

        ctk.CTkButton(
            delete_buttons,
            text="YES, DELETE",
            width=120,
            height=38,
            corner_radius=10,
            fg_color="#8b3a3a",
            hover_color="#6f2d2d",
            font=("Arial", 12, "bold"),
            command=self.confirm_delete_current_preview_page,
        ).grid(row=0, column=0, padx=6)

        ctk.CTkButton(
            delete_buttons,
            text="CANCEL",
            width=120,
            height=38,
            corner_radius=10,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 12, "bold"),
            command=self.hide_delete_confirmation,
        ).grid(row=0, column=1, padx=6)

        preview_nav = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_nav.pack(side="bottom", pady=14)

        ctk.CTkButton(
            preview_nav,
            text="PREV",
            width=100,
            height=42,
            corner_radius=10,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 13, "bold"),
            command=lambda: self.change_preview_page(-1),
        ).grid(row=0, column=0, padx=6)

        ctk.CTkButton(
            preview_nav,
            text="DELETE",
            width=100,
            height=42,
            corner_radius=10,
            fg_color="#8b3a3a",
            hover_color="#6f2d2d",
            font=("Arial", 13, "bold"),
            command=self.delete_current_preview_page,
        ).grid(row=0, column=1, padx=6)

        ctk.CTkButton(
            preview_nav,
            text="CLOSE",
            width=100,
            height=42,
            corner_radius=10,
            fg_color=PRIMARY,
            hover_color="#163bb5",
            font=("Arial", 13, "bold"),
            command=self.close_preview_modal,
        ).grid(row=0, column=2, padx=6)

        ctk.CTkButton(
            preview_nav,
            text="NEXT",
            width=100,
            height=42,
            corner_radius=10,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 13, "bold"),
            command=lambda: self.change_preview_page(1),
        ).grid(row=0, column=3, padx=6)

        self.build_feeder_setup_overlay()

    def build_feeder_setup_overlay(self):
        self.paper_overlay = ctk.CTkFrame(self, fg_color="#050b12")
        setup_card = ctk.CTkFrame(
            self.paper_overlay,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            width=620,
            height=430,
        )
        setup_card.place(relx=0.5, rely=0.5, anchor="center")
        setup_card.pack_propagate(False)

        ctk.CTkLabel(
            setup_card,
            text="Batch Feeder Setup",
            text_color=TEXT,
            font=("Arial", 20, "bold"),
        ).pack(pady=(20, 6))

        ctk.CTkLabel(
            setup_card,
            text="Select paper size. Feed timing is already preset for each page size.",
            text_color=SUBTEXT,
            font=("Arial", 13),
            wraplength=540,
        ).pack(pady=(0, 12))

        size_row = ctk.CTkFrame(setup_card, fg_color="transparent")
        size_row.pack(pady=(0, 12))

        self.paper_size_buttons = {}
        for index, (paper_size, meta) in enumerate(PAPER_SIZES.items()):
            first_duration = int(meta["default_duration_ms"])
            next_seconds = meta.get("next_duration_ms")
            timing_label = format_duration_ms(first_duration)
            if next_seconds and int(next_seconds) != first_duration:
                timing_label = f"{format_duration_ms(first_duration)} / {format_duration_ms(int(next_seconds))}"
            button = ctk.CTkButton(
                size_row,
                text=f"{meta['label'].upper()}\n{timing_label}",
                width=130,
                height=52,
                corner_radius=10,
                fg_color="#254265",
                hover_color="#1c3150",
                font=("Arial", 13, "bold"),
                command=lambda value=paper_size: self.select_paper_size(value),
            )
            button.grid(row=0, column=index, padx=6)
            self.paper_size_buttons[paper_size] = button

        self.setup_status_label = ctk.CTkLabel(
            setup_card,
            text="Choose a paper size to continue.",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
            wraplength=540,
            justify="center",
        )
        self.setup_status_label.pack(pady=(0, 8))

        self.setup_sensor_label = ctk.CTkLabel(
            setup_card,
            text="Sensor: waiting",
            text_color=SUBTEXT,
            font=("Arial", 13),
        )
        self.setup_sensor_label.pack(pady=(0, 12))

        timing_frame = ctk.CTkFrame(setup_card, fg_color="#0d2236", corner_radius=12)
        timing_frame.pack(padx=28, pady=(0, 14), fill="x")
        timing_frame.grid_columnconfigure(0, weight=1)
        timing_frame.grid_columnconfigure(1, weight=1)

        self.duration_value_label = ctk.CTkLabel(
            timing_frame,
            text="Duration: -- ms",
            text_color=TEXT,
            font=("Arial", 14, "bold"),
        )
        self.duration_value_label.grid(row=0, column=0, padx=16, pady=(14, 8), sticky="w")

        self.speed_value_label = ctk.CTkLabel(
            timing_frame,
            text="Speed: --",
            text_color=TEXT,
            font=("Arial", 14, "bold"),
        )
        self.speed_value_label.grid(row=0, column=1, padx=16, pady=(14, 8), sticky="e")

        ctk.CTkLabel(
            timing_frame,
            text="Fixed presets: Short 9s, A4 10s, Long 11.5s first page then 7s next pages. Capture pause: 2s. IR sensor controls batch stop.",
            text_color=SUBTEXT,
            font=("Arial", 12),
        ).grid(row=1, column=0, columnspan=2, padx=16, pady=(0, 14), sticky="w")

        action_row = ctk.CTkFrame(setup_card, fg_color="transparent")
        action_row.pack(pady=(0, 16))

        ctk.CTkButton(
            action_row,
            text="TEST FEED",
            width=140,
            height=42,
            corner_radius=10,
            fg_color="#7f6d49",
            hover_color="#6b5b3d",
            font=("Arial", 13, "bold"),
            command=self.test_calibration_feed,
        ).grid(row=0, column=0, padx=6)

        ctk.CTkButton(
            action_row,
            text="USE PRESET",
            width=150,
            height=42,
            corner_radius=10,
            fg_color=PRIMARY,
            hover_color="#163bb5",
            font=("Arial", 13, "bold"),
            command=self.use_current_setup,
        ).grid(row=0, column=1, padx=6)

        ctk.CTkButton(
            action_row,
            text="CLOSE",
            width=110,
            height=42,
            corner_radius=10,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 13, "bold"),
            command=self.close_feeder_setup_overlay,
        ).grid(row=0, column=2, padx=6)

    def on_show(self):
        self.pages_scanned = 0
        self.captured_pages = []
        self.captured_page_paths = []
        self.recognition_pages = []
        self.preview_index = 0
        self.current_camera_frame = None
        self.selected_paper_size = ""
        self.current_calibration = None
        self.calibration_saved = False
        self.batch_scan_active = False
        self.scan_cancel_requested = False
        self.calibration_test_active = False
        self.sensor_ready = False
        self.controller.captured_pages = []
        self.controller.captured_page_paths = []
        self.controller.recognition_pages = []
        self.controller.transcript_results = []
        self.camera_retry_attempts = 0
        self.refresh_setup_labels()
        append_title = getattr(self.controller, "append_batch_title", "")
        if getattr(self.controller, "upload_test_mode", False):
            self.refresh_batch_state(
                "Upload test mode ready.",
                "Press UPLOAD to load captured page images, then press DONE to transcribe.",
            )
            self.current_frame = self.create_placeholder_image("Upload test mode")
            self.update_preview_widget(self.current_frame)
        elif getattr(self.controller, "append_batch_id", ""):
            self.refresh_batch_state(
                "Add Pages mode ready.",
                f"New scanned pages will be appended to {append_title or 'the selected transcript batch'}.",
            )
        else:
            self.refresh_batch_state(
                "Choose the paper size before scanning.",
                "Press SETUP or select a paper size in the feeder setup window.",
            )
            self.start_camera_preview()
        self.hide_feeder_setup_overlay()

    def on_hide(self):
        if self.batch_scan_active or self.calibration_test_active:
            try:
                self.controller.workflow.stop_feeder()
            except Exception:
                pass
        self.stop_camera_preview()
        self.cancel_pending_jobs()
        self.close_preview_modal()
        self.hide_feeder_setup_overlay()
        self.hide_capture_popup()
        hide_navigation_confirmation(self)

    def show_feeder_setup_overlay(self):
        self.refresh_setup_labels()
        self.paper_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.paper_overlay.lift()

    def hide_feeder_setup_overlay(self):
        self.paper_overlay.place_forget()

    def close_feeder_setup_overlay(self):
        if self.calibration_test_active:
            self.setup_status_label.configure(
                text="Wait for the test feed to finish before closing.",
                text_color=WARNING,
            )
            return

        self.hide_feeder_setup_overlay()
        if not self.selected_paper_size:
            self.refresh_batch_state(
                "Batch setup closed.",
                "Press SETUP when you are ready to choose Long, Short, or A4.",
            )

    def select_paper_size(self, paper_size):
        try:
            self.current_calibration = self.controller.workflow.select_paper_size(paper_size)
        except Exception as exc:
            self.setup_status_label.configure(text=f"Paper size error: {exc}", text_color=WARNING)
            return

        self.calibration_saved = True
        self.selected_paper_size = self.current_calibration.paper_size
        self.refresh_setup_labels()
        self.check_feeder_sensor_for_setup()

        self.setup_status_label.configure(
            text="Preset timing loaded. Use preset to start batch scanning.",
            text_color=SUCCESS,
        )

    def check_feeder_sensor_for_setup(self):
        paper_present, message = self.controller.workflow.check_tray_paper()
        self.sensor_ready = paper_present
        self.setup_sensor_label.configure(
            text=f"Sensor: {message}",
            text_color=SUCCESS if paper_present else WARNING,
        )
        return paper_present

    def refresh_setup_labels(self):
        for paper_size, button in getattr(self, "paper_size_buttons", {}).items():
            selected = paper_size == self.selected_paper_size
            button.configure(fg_color=PRIMARY if selected else "#254265")

        if self.current_calibration is None:
            self.paper_size_value.configure(text="Not selected", text_color=SUBTEXT)
            self.calibration_value.configure(text="Needs setup", text_color=SUBTEXT)
            if hasattr(self, "duration_value_label"):
                self.duration_value_label.configure(text="Duration: -- ms")
                self.speed_value_label.configure(text="Speed: --")
            return

        label = PAPER_SIZES[self.current_calibration.paper_size]["label"]
        meta = PAPER_SIZES[self.current_calibration.paper_size]
        first_duration = int(meta["default_duration_ms"])
        next_duration = int(meta.get("next_duration_ms", first_duration))
        pause_duration = int(getattr(self.current_calibration, "settle_ms", 2000))
        if next_duration != first_duration:
            duration_text = (
                f"{format_duration_ms(first_duration)} first, "
                f"{format_duration_ms(next_duration)} next"
            )
            duration_label = (
                f"Duration: {first_duration} ms first, {next_duration} ms next; "
                f"capture pause: {pause_duration} ms"
            )
        else:
            duration_text = f"{format_duration_ms(first_duration)} preset"
            duration_label = f"Duration: {first_duration} ms; capture pause: {pause_duration} ms"
        self.paper_size_value.configure(text=label, text_color=ACCENT)
        self.calibration_value.configure(
            text=f"{duration_text} @ speed {self.current_calibration.speed}",
            text_color=SUCCESS,
        )
        self.duration_value_label.configure(text=duration_label)
        self.speed_value_label.configure(text=f"Speed: {self.current_calibration.speed}")

    def adjust_calibration_duration(self, delta):
        if self.current_calibration is None:
            self.setup_status_label.configure(text="Select a paper size first.", text_color=WARNING)
            return
        self.current_calibration.duration_ms = max(
            MIN_DURATION_MS,
            min(MAX_DURATION_MS, self.current_calibration.duration_ms + delta),
        )
        self.calibration_saved = False
        self.setup_status_label.configure(text="Timing changed. Test and save when ready.", text_color=ACCENT)
        self.refresh_setup_labels()

    def adjust_calibration_speed(self, delta):
        if self.current_calibration is None:
            self.setup_status_label.configure(text="Select a paper size first.", text_color=WARNING)
            return
        self.current_calibration.speed = max(
            MIN_SPEED,
            min(MAX_SPEED, self.current_calibration.speed + delta),
        )
        self.calibration_saved = False
        self.setup_status_label.configure(text="Speed changed. Test and save when ready.", text_color=ACCENT)
        self.refresh_setup_labels()

    def save_current_calibration(self):
        if self.current_calibration is None:
            self.setup_status_label.configure(text="Select a paper size first.", text_color=WARNING)
            return
        try:
            self.current_calibration = self.controller.workflow.save_feeder_calibration(
                self.current_calibration.paper_size,
                self.current_calibration.duration_ms,
                self.current_calibration.speed,
                self.current_calibration.settle_ms,
            )
        except Exception as exc:
            self.setup_status_label.configure(text=f"Could not save calibration: {exc}", text_color=WARNING)
            return
        self.calibration_saved = True
        self.refresh_setup_labels()
        self.setup_status_label.configure(text="Calibration saved and ready for scanning.", text_color=SUCCESS)

    def use_current_setup(self):
        if self.current_calibration is None:
            self.setup_status_label.configure(text="Select a paper size first.", text_color=WARNING)
            return
        if not self.check_feeder_sensor_for_setup():
            self.refresh_batch_state("No papers loaded.", "Load papers into the feeder and check the sensor again.")
            return
        self.hide_feeder_setup_overlay()
        self.refresh_batch_state(
            "Feeder setup ready.",
            "It will feed, pause 2 seconds, capture, and stop when no paper is detected.",
        )

    def test_calibration_feed(self):
        if self.current_calibration is None:
            self.setup_status_label.configure(text="Select a paper size first.", text_color=WARNING)
            return
        if self.calibration_test_active:
            return

        self.calibration_test_active = True
        self.setup_status_label.configure(text="Testing feeder timing...", text_color=ACCENT)

        duration_ms = self.current_calibration.duration_ms
        speed = self.current_calibration.speed

        def worker():
            try:
                self.controller.workflow.stop_feeder()
            except Exception:
                pass
            result = self.controller.workflow.feed_one_page(duration_ms, speed, job_id="CAL")
            try:
                self.controller.workflow.stop_feeder()
            except Exception:
                pass
            self.after(0, lambda: self.finish_calibration_test(result))

        threading.Thread(target=worker, daemon=True).start()

    def finish_calibration_test(self, result):
        self.calibration_test_active = False
        if result.success:
            self.setup_status_label.configure(
                text="Test feed complete. Preset timing is ready.",
                text_color=SUCCESS,
            )
            if result.paper_present is not None:
                self.sensor_ready = result.paper_present
                sensor_text = "Paper loaded" if result.paper_present else "No paper"
                self.setup_sensor_label.configure(text=f"Sensor: {sensor_text}", text_color=SUCCESS if result.paper_present else WARNING)
        else:
            self.setup_status_label.configure(text=result.message, text_color=WARNING)

    def start_camera_preview(self, from_retry=False):
        if not from_retry:
            self.camera_retry_attempts = 0
        self.stop_camera_preview(cancel_retry=False)
        if Image is None:
            self.current_frame = self.create_placeholder_image("Camera preview unavailable")
            self.update_preview_widget(self.current_frame)
            return

        if self.start_picamera2_preview():
            self.update_camera_preview()
            return

        if self.start_opencv_preview():
            self.update_camera_preview()
            return

        camera_index = getattr(self.controller.workflow, "camera_index", 0)
        self.current_frame = self.create_placeholder_image(self.camera_offline_message(camera_index))
        self.update_preview_widget(self.current_frame)
        self.schedule_camera_retry()

    def camera_offline_message(self, camera_index):
        if Picamera2 is not None:
            try:
                if not Picamera2.global_camera_info():
                    return "No camera detected"
            except Exception:
                pass
        return f"Camera {camera_index} warming up"

    def schedule_camera_retry(self):
        if self.camera_retry_job is not None:
            return
        self.camera_retry_attempts += 1
        delay_ms = min(5000, 1000 + self.camera_retry_attempts * 500)
        self.camera_retry_job = self.after(delay_ms, self.retry_camera_preview)

    def retry_camera_preview(self):
        self.camera_retry_job = None
        self.start_camera_preview(from_retry=True)

    def start_picamera2_preview(self):
        if Picamera2 is None:
            return False

        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            self.camera_backend = "picamera2"
            self.camera_frame_failures = 0
            return True
        except Exception:
            self.close_camera()
            return False

    def start_opencv_preview(self):
        if cv2 is None:
            return False

        try:
            camera_index = getattr(self.controller.workflow, "camera_index", 0)
            backend = getattr(cv2, "CAP_V4L2", 0)
            self.camera = cv2.VideoCapture(camera_index, backend)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera_backend = "opencv"
            self.camera_frame_failures = 0
        except Exception:
            self.close_camera()
            return False

        if self.camera is None or not self.camera.isOpened():
            self.close_camera()
            return False
        return True

    def stop_camera_preview(self, cancel_retry=True):
        if cancel_retry and self.camera_retry_job is not None:
            try:
                self.after_cancel(self.camera_retry_job)
            except Exception:
                pass
            self.camera_retry_job = None
        if self.preview_job is not None:
            try:
                self.after_cancel(self.preview_job)
            except Exception:
                pass
            self.preview_job = None
        self.close_camera()

    def close_camera(self):
        if self.camera is not None:
            try:
                if self.camera_backend == "picamera2":
                    self.camera.stop()
                    self.camera.close()
                else:
                    self.camera.release()
            except Exception:
                pass
            self.camera = None
        self.camera_backend = None

    def cancel_pending_jobs(self):
        for job_name in ("capture_popup_job", "batch_scan_job"):
            job = getattr(self, job_name, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, job_name, None)

    def update_camera_preview(self):
        if self.camera is None or Image is None:
            self.preview_job = None
            return

        ok = False
        frame_rgb = None
        if self.camera_backend == "picamera2":
            try:
                frame_rgb = self.camera.capture_array()
                ok = frame_rgb is not None
            except Exception:
                ok = False
        elif cv2 is not None:
            ok, frame = self.camera.read()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ok:
            self.camera_frame_failures = 0
            if getattr(frame_rgb, "ndim", 0) == 2 and cv2 is not None:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
            elif len(getattr(frame_rgb, "shape", ())) >= 3 and frame_rgb.shape[2] == 4:
                frame_rgb = frame_rgb[:, :, :3]
            oriented_frame = self.orient_camera_frame(Image.fromarray(frame_rgb))
            self.current_camera_frame = oriented_frame
            self.current_frame = oriented_frame
            self.update_preview_widget(self.current_frame)
        else:
            self.camera_frame_failures += 1
            self.current_frame = self.create_placeholder_image("Waiting for camera frame")
            self.update_preview_widget(self.current_frame)
            if self.camera_frame_failures >= 20:
                self.close_camera()
                self.schedule_camera_retry()
                return

        self.preview_job = self.after(80, self.update_camera_preview)

    def orient_camera_frame(self, image):
        if not CAMERA_ROTATE_COUNTERCLOCKWISE or Image is None or image is None:
            return image

        try:
            return image.transpose(Image.Transpose.ROTATE_90)
        except AttributeError:
            return image.transpose(Image.ROTATE_90)

    def update_preview_widget(self, image):
        if Image is None:
            self.preview_label.configure(text="Camera preview unavailable")
            return

        display = image.copy()
        display.thumbnail((620, 120))
        self.preview_image = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
        self.preview_label.configure(text="", image=self.preview_image)

    def create_placeholder_image(self, text):
        if Image is None:
            return None

        image = Image.new("RGB", (640, 360), "#112a44")
        if ImageDraw is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 628, 348), outline="#7f6d49", width=2)
            draw.text((180, 165), text, fill="#d3bac0")
        return image

    def start_batch_scan(self):
        if self.batch_scan_active or self.batch_scan_job is not None:
            return

        if self.current_calibration is None or not self.selected_paper_size:
            self.show_feeder_setup_overlay()
            self.refresh_batch_state(
                "Choose a paper size before scanning.",
                "Select Short, A4, or Long to load the fixed feed preset.",
            )
            return

        if not self.calibration_saved:
            self.show_feeder_setup_overlay()
            self.refresh_batch_state(
                "Paper preset required before scanning.",
                "Select Short, A4, or Long to load the fixed feed preset.",
            )
            return

        paper_present, message = self.controller.workflow.check_tray_paper()
        if not paper_present:
            self.refresh_batch_state("No papers loaded.", message)
            self.show_feeder_setup_overlay()
            return

        self.controller.workflow.start_capture_session(self.selected_paper_size)
        self.batch_scan_active = True
        self.scan_cancel_requested = False
        self.hide_feeder_setup_overlay()
        self.capture_batch_page()

    def capture_batch_page(self):
        if not self.batch_scan_active or self.scan_cancel_requested:
            self.batch_scan_job = None
            return

        paper_present, message = self.controller.workflow.check_tray_paper()
        if not paper_present:
            self.batch_scan_active = False
            if self.pages_scanned <= 0:
                self.refresh_batch_state(
                    "No papers loaded.",
                    "Load papers into the feeder and start the batch again.",
                )
                self.batch_scan_job = None
                return
            self.refresh_batch_state(
                "Batch scanning complete.",
                "The feeder is empty. Press DONE to start transcription.",
            )
            self.batch_scan_job = None
            return

        page_number = self.pages_scanned + 1
        duration_ms = feed_duration_for_page(self.current_calibration.paper_size, page_number)
        speed = self.current_calibration.speed
        job_id = f"PAGE{page_number}"

        self.refresh_batch_state(
            f"Feeding page {page_number}...",
            f"Motor running for {duration_ms} ms at speed {speed}.",
        )

        self.feed_worker = threading.Thread(
            target=self._feed_page_worker,
            args=(duration_ms, speed, job_id, page_number),
            daemon=True,
        )
        self.feed_worker.start()

    def _feed_page_worker(self, duration_ms, speed, job_id, page_number):
        result = self.controller.workflow.feed_one_page(duration_ms, speed, job_id=job_id)
        self.after(0, lambda: self.handle_feed_result(result, page_number))

    def handle_feed_result(self, result, page_number):
        self.feed_worker = None
        if self.scan_cancel_requested:
            return

        if not result.success:
            self.batch_scan_active = False
            try:
                self.controller.workflow.stop_feeder()
            except Exception:
                pass
            self.refresh_batch_state(
                "Feed failed.",
                f"{result.message} Clear the feeder, then try again.",
            )
            return

        settle_ms = getattr(self.current_calibration, "settle_ms", 2000)
        self.refresh_batch_state(
            f"Page {page_number} fed. Capture pause...",
            f"Pausing {format_duration_ms(settle_ms)} for camera capture.",
        )
        self.batch_scan_job = self.after(settle_ms, lambda: self.capture_after_feed(page_number))

    def capture_after_feed(self, page_number):
        self.batch_scan_job = None
        if not self.batch_scan_active or self.scan_cancel_requested:
            return

        self.capture_current_page(
            prompt=f"Page {page_number} captured.",
            helper="Checking the feeder tray for more papers.",
        )

        paper_present, message = self.controller.workflow.check_tray_paper()
        if paper_present:
            self.refresh_batch_state(
                f"Page {page_number} captured.",
                "Paper still detected. Feeding the next page.",
            )
            self.batch_scan_job = self.after(0, self.capture_batch_page)
            return

        self.run_final_page_out(page_number)

    def run_final_page_out(self, page_number):
        speed = self.current_calibration.speed if self.current_calibration is not None else 150
        self.refresh_batch_state(
            "Final page detected.",
            f"Running feeder {format_duration_ms(FINAL_PAGE_RUNOUT_MS)} more seconds to clear the last sheet.",
        )
        self.feed_worker = threading.Thread(
            target=self._final_runout_worker,
            args=(speed, page_number),
            daemon=True,
        )
        self.feed_worker.start()

    def _final_runout_worker(self, speed, page_number):
        result = self.controller.workflow.feed_one_page(
            FINAL_PAGE_RUNOUT_MS,
            speed,
            job_id=f"RUNOUT{page_number}",
        )
        try:
            self.controller.workflow.stop_feeder()
        except Exception:
            pass
        self.after(0, lambda: self.finish_final_runout(result))

    def finish_final_runout(self, result):
        self.feed_worker = None
        self.batch_scan_active = False
        self.batch_scan_job = None
        if result.success:
            self.refresh_batch_state(
                "Batch scanning complete.",
                "Final page cleared. Press DONE to start transcription.",
            )
        else:
            self.refresh_batch_state(
                "Final page runout stopped.",
                f"{result.message} Press DONE if the captured pages are complete.",
            )

    def capture_current_page(self, prompt, helper):
        frame = self.current_camera_frame or self.current_frame
        if frame is None:
            frame = self.create_placeholder_image("No preview available")
        elif self.current_camera_frame is not None:
            frame = self.prepare_camera_capture(frame)

        page_number = len(self.captured_pages) + 1
        saved_path = self.save_captured_page_image(frame, page_number)
        self.captured_pages.append(frame.copy() if hasattr(frame, "copy") else frame)
        self.recognition_pages.append(frame.copy() if hasattr(frame, "copy") else frame)
        if saved_path:
            self.captured_page_paths.append(saved_path)
        self.controller.captured_pages = list(self.captured_pages)
        self.controller.captured_page_paths = list(self.captured_page_paths)
        self.controller.recognition_pages = list(self.recognition_pages)
        self.pages_scanned = len(self.captured_pages)
        self.refresh_batch_state(prompt, helper)
        self.show_capture_popup(self.pages_scanned)

    def save_captured_page_image(self, image, page_number):
        if image is None or not hasattr(image, "save"):
            return ""
        capture_dir = self.controller.workflow.batch_session.capture_dir
        if capture_dir is None and self.selected_paper_size:
            capture_dir = self.controller.workflow.start_capture_session(self.selected_paper_size)
        if capture_dir is None:
            return ""
        try:
            capture_dir.mkdir(parents=True, exist_ok=True)
            path = capture_dir / f"page_{page_number:03d}_capture.png"
            image.save(path)
            return str(path)
        except Exception:
            return ""

    def save_uploaded_preview_image(self, image, page_number):
        if image is None or not hasattr(image, "save"):
            return ""
        capture_dir = self.controller.workflow.batch_session.capture_dir
        if capture_dir is None:
            return ""
        try:
            capture_dir.mkdir(parents=True, exist_ok=True)
            path = capture_dir / f"page_{page_number:03d}_preview.png"
            image.save(path)
            return str(path)
        except Exception:
            return ""

    def prepare_camera_capture(self, image):
        if not CAMERA_DOCUMENT_CROP or crop_document_from_camera is None:
            return image
        try:
            return crop_document_from_camera(image)
        except Exception:
            return image

    def prepare_uploaded_page_for_recognition(self, image):
        if crop_document_from_camera is None:
            return image.copy() if hasattr(image, "copy") else image
        try:
            return crop_document_from_camera(image)
        except Exception:
            return image.copy() if hasattr(image, "copy") else image

    def upload_pages(self):
        if Image is None:
            self.refresh_batch_state(
                "Upload unavailable.",
                "Pillow is not installed, so image files cannot be loaded.",
            )
            return

        file_paths = filedialog.askopenfilenames(
            title="Select GreggSpeak test pages",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        if not file_paths:
            return

        if self.controller.workflow.batch_session.capture_dir is None:
            self.controller.workflow.start_capture_session(self.selected_paper_size or "A4")

        loaded = 0
        for file_path in file_paths:
            try:
                image = Image.open(file_path)
                if ImageOps is not None:
                    image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
            except Exception:
                continue

            page_number = len(self.captured_pages) + 1
            saved_path = self.save_captured_page_image(image, page_number)
            preview_page = self.prepare_uploaded_page_for_recognition(image)
            self.save_uploaded_preview_image(preview_page, page_number)
            self.captured_pages.append(preview_page)
            self.recognition_pages.append(image.copy() if hasattr(image, "copy") else image)
            if saved_path:
                self.captured_page_paths.append(saved_path)
            loaded += 1

        if loaded <= 0:
            self.refresh_batch_state(
                "No images were uploaded.",
                "Choose PNG or JPG page images captured from the RPi camera.",
            )
            return

        self.controller.captured_pages = list(self.captured_pages)
        self.controller.captured_page_paths = list(self.captured_page_paths)
        self.controller.recognition_pages = list(self.recognition_pages)
        self.pages_scanned = len(self.captured_pages)
        self.current_frame = self.captured_pages[-1].copy()
        self.update_preview_widget(self.current_frame)
        self.refresh_batch_state(
            f"Uploaded {loaded} page(s) for recognition testing.",
            "Preview the batch or press DONE to run segmentation and recognition.",
        )
        self.show_capture_popup(self.pages_scanned)

    def show_capture_popup(self, page_number):
        self.hide_capture_popup()
        self.capture_message.configure(text=f"Transcript page {page_number} added to the batch.")
        self.capture_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.capture_overlay.lift()
        self.capture_popup_job = self.after(1200, self.hide_capture_popup)

    def hide_capture_popup(self):
        self.capture_overlay.place_forget()
        if self.capture_popup_job is not None:
            try:
                self.after_cancel(self.capture_popup_job)
            except Exception:
                pass
            self.capture_popup_job = None

    def open_preview_modal(self):
        if not self.captured_pages:
            self.preview_title.configure(text="Captured Pages")
            self.preview_modal_label.configure(text="No captured pages yet.")
        else:
            self.preview_index = min(self.preview_index, len(self.captured_pages) - 1)
            self.render_preview_modal()
        self.preview_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.preview_overlay.lift()

    def close_preview_modal(self):
        self.preview_overlay.place_forget()
        self.hide_delete_confirmation()

    def change_preview_page(self, delta):
        if not self.captured_pages:
            return
        self.preview_index = (self.preview_index + delta) % len(self.captured_pages)
        self.render_preview_modal()

    def render_preview_modal(self):
        if not self.captured_pages:
            self.preview_title.configure(text="Captured Pages")
            self.preview_modal_image = None
            self.reset_preview_modal_label("No captured pages yet.")
            self.hide_delete_confirmation()
            return

        image = self.captured_pages[self.preview_index].copy()
        if Image is not None:
            image.thumbnail((440, 220))
            self.preview_modal_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.reset_preview_modal_label("", self.preview_modal_image)
        else:
            self.preview_modal_image = None
            self.reset_preview_modal_label("Preview unavailable.")
        self.preview_title.configure(
            text=f"Captured Page {self.preview_index + 1} of {len(self.captured_pages)}"
        )
        self.hide_delete_confirmation()

    def reset_preview_modal_label(self, text, image=None):
        if self.preview_modal_label is not None:
            self.preview_modal_label.destroy()

        self.preview_modal_label = ctk.CTkLabel(
            self.preview_display_frame,
            text=text,
            text_color=TEXT,
            font=("Arial", 14),
            image=image,
        )
        self.preview_modal_label.pack()

    def delete_current_preview_page(self):
        if not self.captured_pages:
            return
        self.delete_confirm_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.delete_confirm_overlay.lift()

    def hide_delete_confirmation(self):
        self.delete_confirm_overlay.place_forget()

    def confirm_delete_current_preview_page(self):
        if not self.captured_pages:
            return

        self.hide_delete_confirmation()
        del self.captured_pages[self.preview_index]
        if self.preview_index < len(self.recognition_pages):
            del self.recognition_pages[self.preview_index]
        if self.preview_index < len(self.captured_page_paths):
            del self.captured_page_paths[self.preview_index]
        self.pages_scanned = len(self.captured_pages)
        self.controller.captured_pages = list(self.captured_pages)
        self.controller.captured_page_paths = list(self.captured_page_paths)
        self.controller.recognition_pages = list(self.recognition_pages)

        if self.pages_scanned == 0:
            self.preview_index = 0
            self.refresh_batch_state(
                "No captured pages in this batch yet.",
                "Wait for the camera preview, then capture the transcript page.",
            )
            self.render_preview_modal()
            return

        if self.preview_index >= self.pages_scanned:
            self.preview_index = self.pages_scanned - 1

        self.refresh_batch_state(
            "Captured pages updated. Review the batch or scan another page.",
            "Press SCAN BATCH to capture more loaded pages, or DONE to start transcription.",
        )
        self.render_preview_modal()

    def finish_batch(self):
        if self.batch_scan_active:
            self.prompt_label.configure(text="Wait for the current feed/capture cycle to finish.")
            self.helper_label.configure(text="Use CANCEL to stop the motor and pause the batch.")
            return
        self.batch_scan_job = None
        if self.pages_scanned <= 0:
            self.prompt_label.configure(text="Capture at least one page before transcribing.")
            self.helper_label.configure(text="Wait for the camera preview, then capture the transcript page.")
            return
        self.controller.workflow.finish_batch(self.pages_scanned)
        self.controller.batch_pages = self.pages_scanned
        self.controller.current_transcript_page = 0
        self.controller.captured_pages = list(self.captured_pages)
        self.controller.captured_page_paths = list(self.captured_page_paths)
        self.controller.recognition_pages = list(self.recognition_pages)
        self.controller.transcript_results = []
        self.controller.show_frame("ProcessingPage")

    def cancel_batch_scan(self):
        if self.batch_scan_active or self.calibration_test_active:
            self.scan_cancel_requested = True
            self.batch_scan_active = False
            self.calibration_test_active = False
            self.cancel_pending_jobs()
            try:
                self.controller.workflow.stop_feeder()
            except Exception:
                pass
            self.controller.workflow.batch_session.active = False
            if self.pages_scanned > 0:
                self.refresh_batch_state(
                    "Batch cancelled. Captured pages were kept.",
                    "Preview the partial batch or press DONE to transcribe the captured pages.",
                )
            else:
                self.refresh_batch_state(
                    "Batch cancelled.",
                    "No pages were captured. Return Home or start again.",
                )
            return

        confirm_navigation(
            self,
            self.controller,
            "HomePage",
            "Are you sure you want to cancel this batch and go back to Home?",
        )

    def refresh_batch_state(self, prompt, helper):
        self.pages_value.configure(text=str(self.pages_scanned))
        self.prompt_label.configure(text=prompt)
        self.helper_label.configure(text=helper)
