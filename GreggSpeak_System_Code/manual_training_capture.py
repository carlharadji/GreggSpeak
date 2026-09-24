from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageDraw


APP_DIR = Path(__file__).resolve().parent
SAVE_DIR = APP_DIR / "data" / "training_captures"
PREVIEW_SIZE = (900, 315)
CAMERA_INDEX = int(os.environ.get("GREGGSPEAK_CAMERA_INDEX", "0"))

BG_COLOR = "#0b1e2d"
CARD_COLOR = "#112a44"
PRIMARY = "#1f4ed8"
ACCENT = "#c9a85d"
TEXT = "#ffffff"
SUBTEXT = "#a0aec0"
WARNING = "#f59e0b"


class CameraBackend:
    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.mode = "none"
        self.picamera = None
        self.cv2 = None
        self.cv_camera = None

    def start(self) -> str:
        picamera_status = self._start_picamera2()
        if self.picamera is not None:
            self.mode = "picamera2"
            return picamera_status

        opencv_status = self._start_opencv()
        if self.cv_camera is not None:
            self.mode = "opencv"
            return opencv_status

        self.mode = "none"
        return f"Camera unavailable. {picamera_status} {opencv_status}"

    def _start_picamera2(self) -> str:
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as exc:
            return f"Picamera2 not available: {exc}"

        try:
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"size": (1640, 1232), "format": "RGB888"}
            )
            camera.configure(config)
            camera.start()
            self.picamera = camera
            return "Camera ready: Picamera2"
        except Exception as exc:
            self.picamera = None
            return f"Picamera2 failed: {exc}"

    def _start_opencv(self) -> str:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            return f"OpenCV not available: {exc}"

        try:
            camera = cv2.VideoCapture(self.camera_index)
            if camera is None or not camera.isOpened():
                return f"OpenCV camera {self.camera_index} unavailable"
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cv2 = cv2
            self.cv_camera = camera
            return f"Camera ready: OpenCV index {self.camera_index}"
        except Exception as exc:
            self.cv_camera = None
            return f"OpenCV failed: {exc}"

    def read(self) -> Image.Image | None:
        if self.picamera is not None:
            try:
                frame = self.picamera.capture_array()
                return Image.fromarray(frame[:, :, :3]).convert("RGB")
            except Exception:
                return None

        if self.cv_camera is not None and self.cv2 is not None:
            try:
                ok, frame = self.cv_camera.read()
                if not ok:
                    return None
                rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb).convert("RGB")
            except Exception:
                return None

        return None

    def close(self):
        if self.picamera is not None:
            try:
                self.picamera.stop()
            except Exception:
                pass
            self.picamera = None

        if self.cv_camera is not None:
            try:
                self.cv_camera.release()
            except Exception:
                pass
            self.cv_camera = None


class ManualTrainingCaptureApp(ctk.CTk):
    DESIGN_WIDTH = 1024
    DESIGN_HEIGHT = 600

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("GreggSpeak Manual Training Capture")
        self.configure_display()
        self.configure(fg_color=BG_COLOR)

        self.camera = CameraBackend()
        self.current_frame: Image.Image | None = None
        self.preview_image = None
        self.capture_count = 0
        self.preview_job = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        status = self.camera.start()
        self.status_label.configure(text=status)
        self.update_preview()

    def configure_display(self):
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        scale = min(
            screen_width / self.DESIGN_WIDTH,
            screen_height / self.DESIGN_HEIGHT,
            1.0,
        )
        ctk.set_widget_scaling(max(scale, 0.72))

        windowed = os.environ.get("GREGGSPEAK_LCD_WINDOWED", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if windowed:
            width = min(self.DESIGN_WIDTH, screen_width)
            height = min(self.DESIGN_HEIGHT, screen_height)
            self.geometry(f"{width}x{height}+0+0")
            self.minsize(min(800, screen_width), min(480, screen_height))
            self.resizable(True, True)
            return

        self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.minsize(1, 1)
        self.resizable(True, True)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda _event: self.attributes("-fullscreen", False))
        self.bind(
            "<F11>",
            lambda _event: self.attributes(
                "-fullscreen",
                not bool(self.attributes("-fullscreen")),
            ),
        )

    def _build_ui(self):
        ctk.CTkLabel(
            self,
            text="Manual Training Capture",
            text_color=TEXT,
            font=("Arial", 24, "bold"),
        ).pack(pady=(10, 2))

        ctk.CTkLabel(
            self,
            text="Capture raw camera images for future model retraining.",
            text_color=SUBTEXT,
            font=("Arial", 13),
        ).pack(pady=(0, 8))

        preview_card = ctk.CTkFrame(
            self,
            fg_color=CARD_COLOR,
            corner_radius=18,
            border_width=1,
            border_color="#2d5b86",
            width=940,
            height=335,
        )
        preview_card.pack(padx=22, pady=(0, 10), fill="x")
        preview_card.pack_propagate(False)

        self.preview_label = ctk.CTkLabel(
            preview_card,
            text="Starting camera...",
            text_color=SUBTEXT,
            font=("Arial", 18, "bold"),
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        action_row = ctk.CTkFrame(self, fg_color="transparent")
        action_row.pack(pady=(0, 8), fill="x", padx=22)
        action_row.grid_columnconfigure(0, weight=1)
        action_row.grid_columnconfigure(1, weight=1)

        self.capture_button = ctk.CTkButton(
            action_row,
            text="Capture Image",
            height=64,
            corner_radius=16,
            fg_color=PRIMARY,
            hover_color="#163bb5",
            font=("Arial", 22, "bold"),
            command=self.capture_image,
        )
        self.capture_button.grid(row=0, column=0, columnspan=2, sticky="ew")

        info_row = ctk.CTkFrame(self, fg_color="transparent")
        info_row.pack(fill="x", padx=22, pady=(0, 8))
        info_row.grid_columnconfigure(0, weight=1)
        info_row.grid_columnconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(
            info_row,
            text="Camera status: starting",
            text_color=ACCENT,
            font=("Arial", 13, "bold"),
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.count_label = ctk.CTkLabel(
            info_row,
            text="Session captures: 0",
            text_color=TEXT,
            font=("Arial", 13, "bold"),
        )
        self.count_label.grid(row=0, column=1, sticky="e")

        self.saved_label = ctk.CTkLabel(
            self,
            text=f"Save folder: {SAVE_DIR}",
            text_color=SUBTEXT,
            font=("Arial", 12),
            wraplength=900,
        )
        self.saved_label.pack(padx=22, pady=(0, 8))

    def update_preview(self):
        frame = self.camera.read()
        if frame is not None:
            self.current_frame = frame
            display = frame.copy()
            display.thumbnail(PREVIEW_SIZE)
            self.preview_image = ctk.CTkImage(
                light_image=display,
                dark_image=display,
                size=display.size,
            )
            self.preview_label.configure(text="", image=self.preview_image)
        elif self.current_frame is None:
            placeholder = self.placeholder_image("Waiting for camera frame")
            self.preview_image = ctk.CTkImage(
                light_image=placeholder,
                dark_image=placeholder,
                size=placeholder.size,
            )
            self.preview_label.configure(text="", image=self.preview_image)

        self.preview_job = self.after(120, self.update_preview)

    def capture_image(self):
        frame = self.current_frame or self.camera.read()
        if frame is None:
            self.saved_label.configure(
                text="Capture failed: no camera frame available.",
                text_color=WARNING,
            )
            return

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = SAVE_DIR / filename
        frame.save(path, "JPEG", quality=95)

        self.capture_count += 1
        self.count_label.configure(text=f"Session captures: {self.capture_count}")
        self.saved_label.configure(text=f"Saved: {filename}", text_color=ACCENT)

    def placeholder_image(self, text: str) -> Image.Image:
        image = Image.new("RGB", PREVIEW_SIZE, "#0d2236")
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, PREVIEW_SIZE[0] - 12, PREVIEW_SIZE[1] - 12), outline="#c9a85d", width=2)
        draw.text((PREVIEW_SIZE[0] // 2 - 120, PREVIEW_SIZE[1] // 2), text, fill="#ffffff")
        return image

    def on_close(self):
        if self.preview_job is not None:
            try:
                self.after_cancel(self.preview_job)
            except Exception:
                pass
        self.camera.close()
        self.destroy()


if __name__ == "__main__":
    app = ManualTrainingCaptureApp()
    app.mainloop()
