import customtkinter as ctk
from PIL import Image, ImageDraw

from assets.theme import *
from backend.network import get_primary_access_url
from views.ui_helpers import build_header


class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.status_labels = {}
        self.refresh_job = None
        self.qr_image = None

        build_header(self, "HOME", SUCCESS, divider_color="#53779d")

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, pady=(6, 0))

        ctk.CTkLabel(
            content,
            text="Load Documents into the Feeder",
            text_color=TEXT,
            font=("Arial", 20),
        ).pack(pady=(10, 8))

        ctk.CTkLabel(
            content,
            text="Press START when the first transcript page is ready to be fed.",
            text_color=SUBTEXT,
            font=("Arial", 14),
        ).pack(pady=(0, 14))

        top_cards = ctk.CTkFrame(content, fg_color="transparent")
        top_cards.pack(padx=20, pady=8, fill="x")
        top_cards.grid_columnconfigure(0, weight=1)
        top_cards.grid_columnconfigure(1, weight=1)

        card = ctk.CTkFrame(
            top_cards,
            fg_color=CARD_COLOR,
            corner_radius=15,
            border_width=1,
            border_color="#53779d",
        )
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        access_card = ctk.CTkFrame(
            top_cards,
            fg_color=CARD_COLOR,
            corner_radius=15,
            border_width=1,
            border_color="#53779d",
        )
        access_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(
            card,
            text="System Status",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 5))

        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)

        self.status_row(card, "Camera:", "camera", 1)
        self.status_row(card, "Feeder:", "feeder", 2)
        self.status_row(card, "Storage:", "storage", 3)

        ctk.CTkLabel(
            access_card,
            text="Web Access",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 5))

        self.qr_label = ctk.CTkLabel(access_card, text="")
        self.qr_label.grid(row=1, column=0, rowspan=3, padx=(15, 10), pady=(0, 12))

        ctk.CTkLabel(
            access_card,
            text="Scan QR or open:",
            text_color=SUBTEXT,
            font=("Arial", 12),
        ).grid(row=1, column=1, sticky="w", padx=(0, 15), pady=(0, 2))

        self.web_url_label = ctk.CTkLabel(
            access_card,
            text="Checking network...",
            text_color=TEXT,
            font=("Arial", 12, "bold"),
            wraplength=260,
            justify="left",
        )
        self.web_url_label.grid(row=2, column=1, sticky="w", padx=(0, 15), pady=(0, 4))

        ctk.CTkButton(
            access_card,
            text="COPY LINK",
            width=110,
            height=32,
            corner_radius=8,
            fg_color="#2f6fa3",
            hover_color="#285f8d",
            font=("Arial", 12, "bold"),
            command=self.copy_web_link,
        ).grid(row=3, column=1, sticky="w", padx=(0, 15), pady=(0, 12))

        feeder_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=15,
            border_width=1,
            border_color="#d0ae59",
        )
        feeder_card.pack(padx=20, pady=(6, 8), fill="x")

        ctk.CTkLabel(
            feeder_card,
            text="Feeder Workflow",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", padx=15, pady=(12, 6))

        ctk.CTkLabel(
            feeder_card,
            text="1. Load the transcript pages into the feeder",
            text_color=TEXT,
            font=("Arial", 13),
        ).pack(anchor="w", padx=15, pady=3)

        ctk.CTkLabel(
            feeder_card,
            text="2. Start the scan batch",
            text_color=TEXT,
            font=("Arial", 13),
        ).pack(anchor="w", padx=15, pady=3)

        ctk.CTkLabel(
            feeder_card,
            text="3. Scanning stops automatically when the feeder is empty",
            text_color=TEXT,
            font=("Arial", 13),
        ).pack(anchor="w", padx=15, pady=(3, 12))

        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=14)

        ctk.CTkButton(
            btn_frame,
            text="START",
            width=160,
            height=50,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#2857c8",
            font=("Arial", 14, "bold"),
            command=self.start_batch,
        ).grid(row=0, column=0, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="UPLOAD TEST",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#2f6fa3",
            hover_color="#285f8d",
            font=("Arial", 14, "bold"),
            command=self.start_upload_test,
        ).grid(row=0, column=1, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="SETTINGS",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#607086",
            hover_color="#4f5f73",
            font=("Arial", 14, "bold"),
            command=lambda: controller.show_frame("SettingsPage"),
        ).grid(row=0, column=2, padx=10)

        self.feedback_label = ctk.CTkLabel(
            content,
            text="",
            text_color=SUBTEXT,
            font=("Arial", 13),
        )
        self.feedback_label.pack(pady=(2, 0))

    def on_show(self):
        self.refresh_status()
        self.refresh_web_access()
        self.schedule_refresh()
        self.feedback_label.configure(text="")

    def on_hide(self):
        self.cancel_refresh()

    def status_row(self, parent, label, key, row):
        ctk.CTkLabel(
            parent,
            text=label,
            text_color=TEXT,
            font=("Arial", 13),
        ).grid(row=row, column=0, sticky="w", padx=15, pady=5)

        value_label = ctk.CTkLabel(
            parent,
            text="Checking...",
            text_color=SUBTEXT,
            font=("Arial", 13, "bold"),
        )
        value_label.grid(row=row, column=1, sticky="e", padx=15, pady=5)
        self.status_labels[key] = value_label

    def schedule_refresh(self):
        self.cancel_refresh()
        self.refresh_job = self.after(3000, self._auto_refresh)

    def cancel_refresh(self):
        if self.refresh_job is not None:
            try:
                self.after_cancel(self.refresh_job)
            except Exception:
                pass
            self.refresh_job = None

    def _auto_refresh(self):
        self.refresh_job = None
        self.refresh_status()
        self.schedule_refresh()

    def refresh_status(self):
        status = self.controller.workflow.get_home_status()
        for key, label in self.status_labels.items():
            item = status[key]
            label.configure(
                text=item["label"],
                text_color=SUCCESS if item["ok"] else WARNING,
            )

    def refresh_web_access(self):
        self.web_url = get_primary_access_url(self.controller.web_server_port)
        self.web_url_label.configure(text=self.web_url)
        image = self._build_qr_image(self.web_url)
        self.qr_image = ctk.CTkImage(light_image=image, dark_image=image, size=(96, 96))
        self.qr_label.configure(image=self.qr_image)

    def copy_web_link(self):
        if not hasattr(self, "web_url"):
            self.refresh_web_access()
        self.clipboard_clear()
        self.clipboard_append(self.web_url)
        self.feedback_label.configure(text="Web link copied.", text_color=SUCCESS)

    @staticmethod
    def _build_qr_image(url):
        try:
            import qrcode  # type: ignore

            qr = qrcode.QRCode(border=1, box_size=4)
            qr.add_data(url)
            qr.make(fit=True)
            return qr.make_image(fill_color="black", back_color="white").convert("RGB")
        except Exception:
            image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((6, 6, 122, 122), outline="black", width=2)
            draw.text((16, 48), "QR unavailable", fill="black")
            draw.text((16, 68), "Install qrcode", fill="black")
            return image

    def start_batch(self):
        success, message = self.controller.workflow.start_new_batch()
        self.refresh_status()

        if success:
            self.controller.upload_test_mode = False
            self.controller.batch_pages = 0
            self.controller.current_transcript_page = 0
            self.feedback_label.configure(text=message, text_color=SUCCESS)
            self.controller.show_frame("ScanningPage")
        else:
            self.feedback_label.configure(text=message, text_color=WARNING)

    def start_upload_test(self):
        self.controller.upload_test_mode = True
        self.controller.batch_pages = 0
        self.controller.current_transcript_page = 0
        self.controller.captured_pages = []
        self.controller.transcript_results = []
        self.feedback_label.configure(text="Upload test mode ready.", text_color=SUCCESS)
        self.controller.show_frame("ScanningPage")
