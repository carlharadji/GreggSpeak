import customtkinter as ctk
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

from backend.recognition import RecognitionService
from backend.workflow import WorkflowService
from backend.db import clear_pending_append_batch, get_pending_append_batch
from backend.network import get_lan_ips
from views.home import HomePage
from views.settings import SettingsPage
from views.scanning import ScanningPage
from views.processing import ProcessingPage
from views.transcript import TranscriptPage
from views.save_options import SaveOptionsPage

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.web_server_process = None
        self.web_server_bind_host = "0.0.0.0"
        self.web_server_host = "127.0.0.1"
        self.web_server_port = 5000
        self.start_web_server()

        self.title("GreggSpeak")
        self.geometry("1024x600")
        self.minsize(960, 540)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.workflow = WorkflowService()
        self.recognition = RecognitionService()
        self.batch_pages = 1
        self.current_transcript_page = 0
        self.captured_pages = []
        self.transcript_results = []
        self.processing_time_seconds = 0.0
        self.processing_time_text = ""
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
            TranscriptPage,
            SaveOptionsPage,
        ):
            frame = F(container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")
        clear_pending_append_batch()
        self.after(1000, self.check_pending_append_request)

    def show_frame(self, name):
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
                    self.workflow.start_new_batch()
                    self.show_frame("ScanningPage")
        else:
            self.append_batch_id = ""
            self.append_batch_title = ""
            self._last_append_request = ""

        self.after(1000, self.check_pending_append_request)

    def start_web_server(self):
        self.web_server_port = self.resolve_web_server_port(self.web_server_port)
        if self.is_web_server_running():
            return

        backend_dir = Path(__file__).resolve().parent / "backend"
        startupinfo = None
        creationflags = 0
        env = os.environ.copy()
        env["GREGGSPEAK_WEB_HOST"] = self.web_server_bind_host
        env["GREGGSPEAK_WEB_PORT"] = str(self.web_server_port)

        if sys.platform.startswith("win"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self.web_server_process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=backend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def resolve_web_server_port(self, preferred_port):
        lan_ips = get_lan_ips()
        if self.is_lan_web_server_running(preferred_port, lan_ips):
            return preferred_port
        if not lan_ips and self.is_greggspeak_web_server(preferred_port):
            return preferred_port

        for port in range(preferred_port, preferred_port + 20):
            if self.can_bind_web_port(port):
                return port

        return preferred_port

    def can_bind_web_port(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((self.web_server_bind_host, port))
            except OSError:
                return False
        return True

    def is_web_server_running(self):
        lan_ips = get_lan_ips()
        if lan_ips:
            return self.is_lan_web_server_running(self.web_server_port, lan_ips)
        return self.is_greggspeak_web_server(self.web_server_port)

    def is_lan_web_server_running(self, port, lan_ips=None):
        for ip in lan_ips or get_lan_ips():
            if self.is_greggspeak_web_server(port, host=ip):
                return True
        return False

    def is_greggspeak_web_server(self, port, host=None):
        host = host or self.web_server_host
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/login", timeout=0.75) as response:
                body = response.read(4096).decode("utf-8", errors="ignore")
            return "GreggSpeak" in body
        except Exception:
            return False

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
        self.stop_web_server()
        self.destroy()
