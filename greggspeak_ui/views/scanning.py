import customtkinter as ctk
from tkinter import filedialog

from assets.theme import *
from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation

try:
    from backend.document_crop import crop_document_from_camera
except Exception:
    crop_document_from_camera = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageOps
except Exception:
    Image = None
    ImageDraw = None
    ImageOps = None


CAMERA_ROTATE_COUNTERCLOCKWISE = True
CAMERA_DOCUMENT_CROP = True


class ScanningPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.camera = None
        self.preview_job = None
        self.capture_popup_job = None
        self.batch_scan_job = None
        self.current_frame = None
        self.current_camera_frame = None
        self.preview_image = None
        self.preview_modal_image = None
        self.captured_pages = []
        self.pages_scanned = 0
        self.preview_index = 0
        self.preview_zoom = 1.0
        self.preview_focus_ratio = None
        self.preview_zoom_label = None

        build_header(self, "SCANNING", WARNING, divider_color="#53779d")

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(fill="both", expand=True)

        ctk.CTkLabel(
            content,
            text="Current captured page",
            text_color=TEXT,
            font=("Arial", 20),
        ).pack(pady=(24, 12))

        preview_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            height=190,
        )
        preview_card.pack(padx=28, pady=(0, 24), fill="x")
        preview_card.pack_propagate(False)

        self.preview_label = ctk.CTkLabel(
            preview_card,
            text="Starting camera preview...",
            text_color="#e0ebf6",
            font=("Arial", 18, "bold"),
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        summary_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#53779d",
        )
        summary_card.pack(padx=28, fill="x")

        summary_card.grid_columnconfigure(0, weight=1)
        summary_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            summary_card,
            text="Batch Summary",
            text_color=ACCENT,
            font=("Arial", 15, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        ctk.CTkLabel(
            summary_card,
            text="Pages captured:",
            text_color=TEXT,
            font=("Arial", 14),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=6)

        self.pages_value = ctk.CTkLabel(
            summary_card,
            text="0",
            text_color=ACCENT,
            font=("Arial", 18, "bold"),
        )
        self.pages_value.grid(row=1, column=1, sticky="e", padx=18, pady=6)

        self.prompt_label = ctk.CTkLabel(
            content,
            text="Camera preview ready. Scan the loaded feeder batch.",
            text_color="#e6eef8",
            font=("Georgia", 17),
        )
        self.prompt_label.pack(pady=(26, 10))

        self.helper_label = ctk.CTkLabel(
            content,
            text="Press SCAN to capture all loaded pages. Scanning stops automatically when the feeder is empty.",
            text_color=SUBTEXT,
            font=("Arial", 14),
        )
        self.helper_label.pack(pady=(0, 14))

        button_row = ctk.CTkFrame(content, fg_color="transparent")
        button_row.pack(pady=(0, 8))

        ctk.CTkButton(
            button_row,
            text="SCAN BATCH",
            width=132,
            height=50,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#2857c8",
            font=("Arial", 14, "bold"),
            command=self.start_batch_scan,
        ).grid(row=0, column=0, padx=6, pady=8)

        ctk.CTkButton(
            button_row,
            text="UPLOAD PAGE",
            width=132,
            height=50,
            corner_radius=12,
            fg_color="#2f6fa3",
            hover_color="#285f8d",
            font=("Arial", 13, "bold"),
            command=self.upload_pages,
        ).grid(row=0, column=1, padx=6, pady=8)

        ctk.CTkButton(
            button_row,
            text="PREVIEW",
            width=132,
            height=50,
            corner_radius=12,
            fg_color="#2b5788",
            hover_color="#244971",
            font=("Arial", 14, "bold"),
            command=self.open_preview_modal,
        ).grid(row=0, column=2, padx=6, pady=8)

        ctk.CTkButton(
            button_row,
            text="DONE",
            width=132,
            height=50,
            corner_radius=12,
            fg_color="#2f6fa3",
            hover_color="#285f8d",
            font=("Arial", 14, "bold"),
            command=self.finish_batch,
        ).grid(row=0, column=3, padx=6, pady=8)

        ctk.CTkButton(
            button_row,
            text="CANCEL",
            width=132,
            height=50,
            corner_radius=12,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 14, "bold"),
            command=lambda: confirm_navigation(
                self,
                controller,
                "HomePage",
                "Are you sure you want to cancel this batch and go back to Home?",
            ),
        ).grid(row=0, column=4, padx=6, pady=8)

        self.capture_overlay = ctk.CTkFrame(self, fg_color="#12365a")
        self.capture_card = ctk.CTkFrame(
            self.capture_overlay,
            fg_color="#2e5f9c",
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

        self.preview_overlay = ctk.CTkFrame(self, fg_color="#12365a")
        self.preview_card = ctk.CTkFrame(
            self.preview_overlay,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            width=720,
            height=500,
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

        self.preview_display_frame = ctk.CTkFrame(
            self.preview_card,
            fg_color="#173e66",
            width=660,
            height=285,
            corner_radius=12,
        )
        self.preview_display_frame.pack(pady=(0, 8), padx=18, fill="x")
        self.preview_display_frame.pack_propagate(False)
        self.preview_modal_label = None
        self.reset_preview_modal_label("No captured pages yet.")

        preview_zoom_row = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_zoom_row.pack(pady=(0, 6))

        ctk.CTkButton(
            preview_zoom_row,
            text="ZOOM -",
            width=96,
            height=34,
            corner_radius=9,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 12, "bold"),
            command=lambda: self.change_preview_zoom(-0.25),
        ).grid(row=0, column=0, padx=5)

        self.preview_zoom_label = ctk.CTkLabel(
            preview_zoom_row,
            text="100%",
            text_color=ACCENT,
            font=("Arial", 13, "bold"),
            width=70,
        )
        self.preview_zoom_label.grid(row=0, column=1, padx=5)

        ctk.CTkButton(
            preview_zoom_row,
            text="ZOOM +",
            width=96,
            height=34,
            corner_radius=9,
            fg_color="#2b5788",
            hover_color="#244971",
            font=("Arial", 12, "bold"),
            command=lambda: self.change_preview_zoom(0.25),
        ).grid(row=0, column=2, padx=5)

        ctk.CTkButton(
            preview_zoom_row,
            text="RESET",
            width=96,
            height=34,
            corner_radius=9,
            fg_color=PRIMARY,
            hover_color="#2857c8",
            font=("Arial", 12, "bold"),
            command=self.reset_preview_zoom,
        ).grid(row=0, column=3, padx=5)

        preview_pan_row = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_pan_row.pack(pady=(0, 4))

        ctk.CTkButton(
            preview_pan_row,
            text="↑",
            width=72,
            height=30,
            corner_radius=8,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 11, "bold"),
            command=lambda: self.pan_preview(0, -1),
        ).grid(row=0, column=0, padx=4, pady=2)

        ctk.CTkButton(
            preview_pan_row,
            text="←",
            width=72,
            height=30,
            corner_radius=8,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 11, "bold"),
            command=lambda: self.pan_preview(-1, 0),
        ).grid(row=0, column=1, padx=4, pady=2)

        ctk.CTkButton(
            preview_pan_row,
            text="→",
            width=72,
            height=30,
            corner_radius=8,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 11, "bold"),
            command=lambda: self.pan_preview(1, 0),
        ).grid(row=0, column=2, padx=4, pady=2)

        ctk.CTkButton(
            preview_pan_row,
            text="↓",
            width=72,
            height=30,
            corner_radius=8,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 11, "bold"),
            command=lambda: self.pan_preview(0, 1),
        ).grid(row=0, column=3, padx=4, pady=2)

        self.delete_confirm_overlay = ctk.CTkFrame(self.preview_overlay, fg_color="#12365a")
        self.delete_confirm_card = ctk.CTkFrame(
            self.delete_confirm_overlay,
            fg_color="#244d78",
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
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 12, "bold"),
            command=self.hide_delete_confirmation,
        ).grid(row=0, column=1, padx=6)

        preview_nav = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_nav.pack(side="bottom", pady=12)

        ctk.CTkButton(
            preview_nav,
            text="←",
            width=100,
            height=42,
            corner_radius=10,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
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
            hover_color="#2857c8",
            font=("Arial", 13, "bold"),
            command=self.close_preview_modal,
        ).grid(row=0, column=2, padx=6)

        ctk.CTkButton(
            preview_nav,
            text="→",
            width=100,
            height=42,
            corner_radius=10,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 13, "bold"),
            command=lambda: self.change_preview_page(1),
        ).grid(row=0, column=3, padx=6)

    def on_show(self):
        self.pages_scanned = 0
        self.captured_pages = []
        self.preview_index = 0
        self.current_camera_frame = None
        self.controller.captured_pages = []
        self.controller.transcript_results = []
        append_title = getattr(self.controller, "append_batch_title", "")
        if getattr(self.controller, "upload_test_mode", False):
            self.refresh_batch_state(
                "Upload test mode ready.",
                "Press UPLOAD PAGE to load scanned images, then press DONE to transcribe.",
            )
        elif getattr(self.controller, "append_batch_id", ""):
            self.refresh_batch_state(
                "Add Pages mode ready.",
                f"New scanned pages will be appended to {append_title or 'the selected transcript batch'}.",
            )
        else:
            self.refresh_batch_state(
                "Camera preview ready. Scan the loaded feeder batch.",
                "Press SCAN BATCH to capture all loaded pages from the feeder.",
            )
        self.start_camera_preview()

    def on_hide(self):
        self.stop_camera_preview()
        self.cancel_pending_jobs()
        self.close_preview_modal()
        self.hide_capture_popup()
        hide_navigation_confirmation(self)

    def start_camera_preview(self):
        self.stop_camera_preview()
        if cv2 is None or Image is None:
            self.current_frame = self.create_placeholder_image("Camera preview unavailable")
            self.update_preview_widget(self.current_frame)
            return

        try:
            camera_index = getattr(self.controller.workflow, "camera_index", 1)
            self.camera = cv2.VideoCapture(camera_index)
        except Exception:
            self.camera = None

        if self.camera is None or not self.camera.isOpened():
            camera_index = getattr(self.controller.workflow, "camera_index", 1)
            self.current_frame = self.create_placeholder_image(f"Camera {camera_index} offline")
            self.update_preview_widget(self.current_frame)
            return

        self.update_camera_preview()

    def stop_camera_preview(self):
        if self.preview_job is not None:
            try:
                self.after_cancel(self.preview_job)
            except Exception:
                pass
            self.preview_job = None
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

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
        if self.camera is None or cv2 is None or Image is None:
            self.preview_job = None
            return

        ok, frame = self.camera.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            oriented_frame = self.orient_camera_frame(Image.fromarray(frame_rgb))
            self.current_camera_frame = oriented_frame
            self.current_frame = oriented_frame
            self.update_preview_widget(self.current_frame)
        else:
            self.current_frame = self.create_placeholder_image("Waiting for camera frame")
            self.update_preview_widget(self.current_frame)

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
        display.thumbnail((620, 170))
        self.preview_image = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
        self.preview_label.configure(text="", image=self.preview_image)

    def create_placeholder_image(self, text):
        if Image is None:
            return None

        image = Image.new("RGB", (640, 360), "#1f527e")
        if ImageDraw is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 628, 348), outline="#d0ae59", width=2)
            draw.text((180, 165), text, fill="#e0ebf6")
        return image

    def start_batch_scan(self):
        if self.batch_scan_job is not None:
            return

        if self.controller.workflow.system_status.feeder_pages_loaded <= 0:
            self.refresh_batch_state(
                "No papers detected in the feeder.",
                "Return to Home and reload the feeder before starting a scan batch.",
            )
            return
        self.capture_batch_page()

    def capture_batch_page(self):
        if not self.controller.workflow.consume_feeder_page():
            self.refresh_batch_state(
                "Batch capture complete. The feeder is now empty.",
                "Preview the captured batch or press DONE to start transcription.",
            )
            self.batch_scan_job = None
            return

        remaining_pages = self.controller.workflow.system_status.feeder_pages_loaded

        if self.pages_scanned == 0:
            prompt = "Batch scanning started. Capturing loaded feeder pages..."
        else:
            prompt = "Capturing the next page from the feeder..."

        self.capture_current_page(
            prompt=prompt,
            helper=f"{remaining_pages} page(s) remaining in the feeder after this capture."
            if remaining_pages > 0
            else "This is the last loaded page. Scanning will stop after this capture.",
        )
        self.batch_scan_job = self.after(700, self.capture_batch_page)

    def capture_current_page(self, prompt, helper):
        frame = self.current_camera_frame or self.current_frame
        if frame is None:
            frame = self.create_placeholder_image("No preview available")
        elif self.current_camera_frame is not None:
            frame = self.prepare_page_capture(frame)

        self.captured_pages.append(frame.copy() if hasattr(frame, "copy") else frame)
        self.controller.captured_pages = list(self.captured_pages)
        self.pages_scanned = len(self.captured_pages)
        self.refresh_batch_state(prompt, helper)
        self.show_capture_popup(self.pages_scanned)

    def prepare_page_capture(self, image):
        if not CAMERA_DOCUMENT_CROP or crop_document_from_camera is None:
            return image
        try:
            return crop_document_from_camera(image)
        except Exception:
            return image

    def prepare_camera_capture(self, image):
        return self.prepare_page_capture(image)

    def upload_pages(self):
        if Image is None:
            self.prompt_label.configure(text="Image support is unavailable on this device.")
            return

        paths = filedialog.askopenfilenames(
            title="Select transcript page image(s)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not paths:
            return

        loaded = 0
        for path in paths:
            try:
                image = Image.open(path)
                if ImageOps is not None:
                    image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                image = self.prepare_page_capture(image)
            except Exception:
                continue
            self.captured_pages.append(image)
            loaded += 1

        self.pages_scanned = len(self.captured_pages)
        self.controller.captured_pages = list(self.captured_pages)
        if self.captured_pages:
            self.current_frame = self.captured_pages[-1].copy()
            self.update_preview_widget(self.current_frame)

        self.refresh_batch_state(
            f"Uploaded {loaded} page(s) for recognition testing.",
            "Preview the batch or press DONE to run segmentation and recognition.",
        )

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
        self.preview_zoom = 1.0
        self.preview_focus_ratio = None
        if not self.captured_pages:
            self.preview_title.configure(text="Captured Pages")
            self.preview_modal_image = None
            self.reset_preview_modal_label("No captured pages yet.")
            self.update_preview_zoom_label()
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
        self.preview_zoom = 1.0
        self.preview_focus_ratio = None
        self.render_preview_modal()

    def change_preview_zoom(self, delta):
        if not self.captured_pages:
            return
        self.preview_zoom = max(1.0, min(3.0, self.preview_zoom + delta))
        self.render_preview_modal()

    def reset_preview_zoom(self):
        self.preview_zoom = 1.0
        self.preview_focus_ratio = None
        self.render_preview_modal()

    def pan_preview(self, dx, dy):
        if not self.captured_pages or Image is None:
            return

        image = self.captured_pages[self.preview_index]
        width, height = image.size
        if width <= 0 or height <= 0:
            return

        if self.preview_focus_ratio is None:
            focus_x, focus_y = self.find_preview_focus_point(image)
            self.preview_focus_ratio = (focus_x / width, focus_y / height)

        step = 0.12
        focus_x = max(0.0, min(1.0, self.preview_focus_ratio[0] + (dx * step)))
        focus_y = max(0.0, min(1.0, self.preview_focus_ratio[1] + (dy * step)))
        self.preview_focus_ratio = (focus_x, focus_y)
        self.render_preview_modal()

    def update_preview_zoom_label(self):
        if self.preview_zoom_label is not None:
            self.preview_zoom_label.configure(text=f"{int(self.preview_zoom * 100)}%")

    def render_preview_modal(self):
        if not self.captured_pages:
            self.preview_title.configure(text="Captured Pages")
            self.preview_modal_image = None
            self.reset_preview_modal_label("No captured pages yet.")
            self.update_preview_zoom_label()
            self.hide_delete_confirmation()
            return

        image = self.captured_pages[self.preview_index].copy()
        if Image is not None:
            image = self.build_zoomed_preview_image(image)
            self.preview_modal_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.reset_preview_modal_label("", self.preview_modal_image)
        else:
            self.preview_modal_image = None
            self.reset_preview_modal_label("Preview unavailable.")
        self.preview_title.configure(
            text=f"Captured Page {self.preview_index + 1} of {len(self.captured_pages)}"
        )
        self.update_preview_zoom_label()
        self.hide_delete_confirmation()

    def build_zoomed_preview_image(self, image):
        zoom = max(1.0, self.preview_zoom)
        if zoom > 1.0:
            width, height = image.size
            crop_width = max(1, int(width / zoom))
            crop_height = max(1, int(height / zoom))
            if self.preview_focus_ratio is None:
                focus_x, focus_y = self.find_preview_focus_point(image)
            else:
                focus_x = width * self.preview_focus_ratio[0]
                focus_y = height * self.preview_focus_ratio[1]
            left = int(max(0, min(width - crop_width, focus_x - (crop_width / 2))))
            top = int(max(0, min(height - crop_height, focus_y - (crop_height / 2))))
            image = image.crop((left, top, left + crop_width, top + crop_height))

        image.thumbnail((620, 275))
        return image

    def find_preview_focus_point(self, image):
        width, height = image.size
        try:
            gray = image.convert("L")
            ink_mask = gray.point(lambda pixel: 255 if pixel < 205 else 0)
            bbox = ink_mask.getbbox()
        except Exception:
            bbox = None

        if bbox is None:
            return width / 2, height / 2

        left, top, right, bottom = bbox
        return (left + right) / 2, (top + bottom) / 2

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
        self.preview_modal_label.place(relx=0.5, rely=0.5, anchor="center")

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
        self.pages_scanned = len(self.captured_pages)
        self.controller.captured_pages = list(self.captured_pages)

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
        if self.pages_scanned <= 0:
            self.prompt_label.configure(text="Capture at least one page before transcribing.")
            self.helper_label.configure(text="Wait for the camera preview, then capture the transcript page.")
            return
        self.controller.workflow.finish_batch(self.pages_scanned)
        self.controller.batch_pages = self.pages_scanned
        self.controller.current_transcript_page = 0
        self.controller.captured_pages = list(self.captured_pages)
        self.controller.transcript_results = []
        self.controller.show_frame("ProcessingPage")

    def refresh_batch_state(self, prompt, helper):
        self.pages_value.configure(text=str(self.pages_scanned))
        self.prompt_label.configure(text=prompt)
        self.helper_label.configure(text=helper)
