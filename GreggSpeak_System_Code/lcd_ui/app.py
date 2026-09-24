import customtkinter as ctk
import os
import socket
import subprocess
import sys
from pathlib import Path


LCD_DIR = Path(__file__).resolve().parent
if str(LCD_DIR) not in sys.path:
    sys.path.insert(1, str(LCD_DIR))

from backend.recognition import RecognitionService
from backend.workflow import WorkflowService
from backend.settings import SettingsService
from backend.tts import TTSService
from backend.db import clear_pending_append_batch, get_pending_append_batch
from views.home import HomePage
from views.settings import SettingsPage
from views.scanning import ScanningPage
from views.processing import ProcessingPage
from views.processing_results import ProcessingResultsPage
from views.transcript import TranscriptPage
from views.audio import AudioPage
from views.save_options import SaveOptionsPage
from lcd_ui.touch_keyboard import TouchKeyboard

class App(ctk.CTk):
    DESIGN_WIDTH = 1024
    DESIGN_HEIGHT = 600

    def __init__(self):
        super().__init__()

        self.web_server_process = None
        self.web_server_host = "127.0.0.1"
        self.web_server_port = 5000
        self.start_web_server()

        self.title("GreggSpeak")
        self.configure_display()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.workflow = WorkflowService()
        self.recognition = RecognitionService()
        self.tts = TTSService()
        self.settings_service = SettingsService()
        self.settings = self.settings_service.settings
        self.batch_pages = 1
        self.current_transcript_page = 0
        self.captured_pages = []
        self.captured_page_paths = []
        self.recognition_pages = []
        self.transcript_results = []
        self.processing_time_seconds = 0.0
        self.processing_time_text = ""
        self.upload_test_mode = False
        self.append_batch_id = ""
        self.append_batch_title = ""
        self._last_append_request = ""

        self.frames = {}
        self.current_frame = None

        for F in (
            HomePage,
            SettingsPage,
            ScanningPage,
            ProcessingPage,
            ProcessingResultsPage,
            TranscriptPage,
            AudioPage,
            SaveOptionsPage,
        ):
            frame = F(container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.touch_keyboard = TouchKeyboard(self)
        self.register_touch_inputs()
        self.show_frame("HomePage")
        clear_pending_append_batch()
        self.after(1000, self.check_pending_append_request)

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

    def show_frame(self, name):
        if hasattr(self, "touch_keyboard"):
            self.touch_keyboard.hide()
        frame = self.frames[name]
        if self.current_frame is not None and self.current_frame is not frame:
            on_hide = getattr(self.current_frame, "on_hide", None)
            if callable(on_hide):
                on_hide()
        frame.tkraise()
        on_show = getattr(frame, "on_show", None)
        if callable(on_show):
            on_show()
        self.current_frame = frame

    def register_touch_inputs(self):
        save_options = self.frames.get("SaveOptionsPage")
        title_entry = getattr(save_options, "title_entry", None)
        if title_entry is not None:
            self.touch_keyboard.register(title_entry)

        home_page = self.frames.get("HomePage")
        wifi_password_entry = getattr(home_page, "wifi_password_entry", None)
        if wifi_password_entry is not None:
            self.touch_keyboard.register(wifi_password_entry)

    def check_pending_append_request(self):
        try:
            pending = get_pending_append_batch()
        except Exception:
            pending = None

        request_key = ""
        if pending:
            request_key = f"{pending.get('batch_id', '')}:{pending.get('created_at', '')}"
            self.append_batch_id = pending.get("batch_id", "")
            self.append_batch_title = pending.get("title", "")
            if request_key and request_key != self._last_append_request:
                self._last_append_request = request_key
                current_name = type(self.current_frame).__name__ if self.current_frame else ""
                if current_name in {"HomePage", "SettingsPage"}:
                    self.upload_test_mode = False
                    self.workflow.start_new_batch()
                    self.show_frame("ScanningPage")
        else:
            self.append_batch_id = ""
            self.append_batch_title = ""
            self._last_append_request = ""

        self.after(1000, self.check_pending_append_request)

    def start_web_server(self):
        if self.is_web_server_running():
            return

        deploy_dir = Path(__file__).resolve().parents[1]
        startupinfo = None
        creationflags = 0

        if sys.platform.startswith("win"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self.web_server_process = subprocess.Popen(
            [sys.executable, "-m", "app.web_app"],
            cwd=deploy_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def is_web_server_running(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((self.web_server_host, self.web_server_port)) == 0

    def stop_web_server(self):
        if self.web_server_process is None:
            return

        if self.web_server_process.poll() is None:
            self.web_server_process.terminate()
            try:
                self.web_server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.web_server_process.kill()

        self.web_server_process = None

    def on_close(self):
        try:
            self.workflow.close_feeder()
        except Exception:
            pass
        self.stop_web_server()
        self.destroy()
