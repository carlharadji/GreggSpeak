import threading

import customtkinter as ctk
from PIL import Image, ImageDraw

from lcd_ui.assets.theme import *
from lcd_ui.backend.network import (
    connect_to_wifi,
    get_connected_wifi_ssid,
    get_primary_access_url,
    scan_wifi_networks,
)
from lcd_ui.views.ui_helpers import build_header


class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.status_labels = {}
        self.refresh_job = None
        self.qr_image = None
        self.web_url = ""
        self.wifi_networks = []
        self.wifi_scan_loaded = False
        self.wifi_scan_in_progress = False
        self.selected_wifi = ctk.StringVar(value="Scanning...")

        build_header(self, "HOME", SUCCESS, divider_color="#1f3b57")

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
            border_color="#1f3b57",
        )
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        access_card = ctk.CTkFrame(
            top_cards,
            fg_color=CARD_COLOR,
            corner_radius=15,
            border_width=1,
            border_color="#1f3b57",
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

        access_card.grid_columnconfigure(0, weight=0)
        access_card.grid_columnconfigure(1, weight=1)

        self.connected_access_frame = ctk.CTkFrame(access_card, fg_color="transparent")
        self.connected_access_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.connected_access_frame.grid_columnconfigure(1, weight=1)

        self.qr_label = ctk.CTkLabel(self.connected_access_frame, text="")
        self.qr_label.grid(row=0, column=0, rowspan=3, padx=(15, 10), pady=(0, 12))

        self.web_hint_label = ctk.CTkLabel(
            self.connected_access_frame,
            text="Scan QR or open:",
            text_color=SUBTEXT,
            font=("Arial", 12),
        )
        self.web_hint_label.grid(row=0, column=1, sticky="w", padx=(0, 15), pady=(0, 2))

        self.web_url_label = ctk.CTkLabel(
            self.connected_access_frame,
            text="Checking network...",
            text_color=TEXT,
            font=("Arial", 12, "bold"),
            wraplength=260,
            justify="left",
        )
        self.web_url_label.grid(row=1, column=1, sticky="w", padx=(0, 15), pady=(0, 4))

        ctk.CTkButton(
            self.connected_access_frame,
            text="COPY LINK",
            width=110,
            height=32,
            corner_radius=8,
            fg_color="#2f8f68",
            hover_color="#246f51",
            font=("Arial", 12, "bold"),
            command=self.copy_web_link,
        ).grid(row=2, column=1, sticky="w", padx=(0, 15), pady=(0, 12))

        self.wifi_setup_frame = ctk.CTkFrame(access_card, fg_color="transparent")
        self.wifi_setup_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.wifi_setup_frame.grid_columnconfigure(0, weight=1)

        self.wifi_status_label = ctk.CTkLabel(
            self.wifi_setup_frame,
            text="Connect Raspberry Pi to Wi-Fi to enable web access.",
            text_color=SUBTEXT,
            font=("Arial", 12),
            wraplength=390,
            justify="left",
        )
        self.wifi_status_label.grid(row=0, column=0, sticky="w", padx=15, pady=(0, 6))

        self.wifi_menu = ctk.CTkOptionMenu(
            self.wifi_setup_frame,
            values=["Scanning..."],
            variable=self.selected_wifi,
            fg_color=CARD_COLOR,
            button_color=PRIMARY,
            button_hover_color="#163bb5",
            text_color=TEXT,
            dropdown_fg_color=CARD_COLOR,
            dropdown_hover_color="#20376d",
            dropdown_text_color=TEXT,
        )
        self.wifi_menu.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 6))

        self.wifi_password_entry = ctk.CTkEntry(
            self.wifi_setup_frame,
            placeholder_text="Wi-Fi password",
            show="*",
            height=34,
            fg_color="#0d2238",
            border_color="#1f3b57",
            text_color=TEXT,
            placeholder_text_color=SUBTEXT,
        )
        self.wifi_password_entry.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 8))

        wifi_buttons = ctk.CTkFrame(self.wifi_setup_frame, fg_color="transparent")
        wifi_buttons.grid(row=3, column=0, sticky="w", padx=15, pady=(0, 12))

        self.connect_wifi_button = ctk.CTkButton(
            wifi_buttons,
            text="CONNECT",
            width=110,
            height=32,
            corner_radius=8,
            fg_color="#2f8f68",
            hover_color="#246f51",
            font=("Arial", 12, "bold"),
            command=self.connect_selected_wifi,
        )
        self.connect_wifi_button.grid(row=0, column=0, padx=(0, 8))

        self.refresh_wifi_button = ctk.CTkButton(
            wifi_buttons,
            text="REFRESH",
            width=100,
            height=32,
            corner_radius=8,
            fg_color="#4b5563",
            hover_color="#374151",
            font=("Arial", 12, "bold"),
            command=self.refresh_wifi_networks,
        )
        self.refresh_wifi_button.grid(row=0, column=1)

        self.wifi_setup_frame.grid_remove()

        feeder_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=15,
            border_width=1,
            border_color="#7f6d49",
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
            hover_color="#163bb5",
            font=("Arial", 14, "bold"),
            command=self.start_batch,
        ).grid(row=0, column=0, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="UPLOAD TEST",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#2f8f68",
            hover_color="#246f51",
            font=("Arial", 14, "bold"),
            command=self.start_upload_test,
        ).grid(row=0, column=1, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="SETTINGS",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#4b5563",
            hover_color="#374151",
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
        self.refresh_web_access()
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
        connected_ssid = get_connected_wifi_ssid()
        if not connected_ssid:
            self.connected_access_frame.grid_remove()
            self.wifi_setup_frame.grid()
            if not self.wifi_scan_loaded and not self.wifi_scan_in_progress:
                self.refresh_wifi_networks()
            return

        self.wifi_setup_frame.grid_remove()
        self.connected_access_frame.grid()
        self.web_hint_label.configure(text=f"{connected_ssid} connected. Scan QR or open:")
        self.web_url = get_primary_access_url(self.controller.web_server_port)
        self.web_url_label.configure(text=self.web_url)
        image = self._build_qr_image(self.web_url)
        self.qr_image = ctk.CTkImage(light_image=image, dark_image=image, size=(96, 96))
        self.qr_label.configure(image=self.qr_image)

    def copy_web_link(self):
        if not self.web_url:
            self.refresh_web_access()
        if not self.web_url:
            self.feedback_label.configure(text="Connect to Wi-Fi first.", text_color=WARNING)
            return
        self.clipboard_clear()
        self.clipboard_append(self.web_url)
        self.feedback_label.configure(text="Web link copied.", text_color=SUCCESS)

    def refresh_wifi_networks(self):
        if self.wifi_scan_in_progress:
            return
        self.wifi_scan_in_progress = True
        self.wifi_scan_loaded = False
        self.wifi_status_label.configure(text="Scanning available Wi-Fi networks...", text_color=SUBTEXT)
        self.refresh_wifi_button.configure(state="disabled")
        self.connect_wifi_button.configure(state="disabled")
        threading.Thread(target=self._scan_wifi_worker, daemon=True).start()

    def _scan_wifi_worker(self):
        networks, message = scan_wifi_networks()
        self.after(0, lambda: self._apply_wifi_networks(networks, message))

    def _apply_wifi_networks(self, networks, message):
        self.wifi_scan_in_progress = False
        self.wifi_scan_loaded = True
        self.wifi_networks = networks
        labels = [self._wifi_label(item) for item in networks]
        if labels:
            self.wifi_menu.configure(values=labels)
            self.selected_wifi.set(labels[0])
            self.wifi_status_label.configure(
                text="Choose a Wi-Fi network, enter the password, then tap CONNECT.",
                text_color=SUBTEXT,
            )
            self.connect_wifi_button.configure(state="normal")
        else:
            self.wifi_menu.configure(values=["No networks found"])
            self.selected_wifi.set("No networks found")
            self.wifi_status_label.configure(
                text=message or "No Wi-Fi networks found. Tap REFRESH to scan again.",
                text_color=WARNING,
            )
        self.refresh_wifi_button.configure(state="normal")

    def connect_selected_wifi(self):
        network = self._selected_wifi_network()
        ssid = network["ssid"] if network else None
        password = self.wifi_password_entry.get()
        if not ssid:
            self.wifi_status_label.configure(text="Select a Wi-Fi network first.", text_color=WARNING)
            return
        if network.get("requires_password") and not password:
            self.wifi_status_label.configure(text="Enter the Wi-Fi password first.", text_color=WARNING)
            return

        self.wifi_status_label.configure(text=f"Connecting to {ssid}...", text_color=SUBTEXT)
        self.connect_wifi_button.configure(state="disabled")
        self.refresh_wifi_button.configure(state="disabled")
        threading.Thread(
            target=self._connect_wifi_worker,
            args=(ssid, password),
            daemon=True,
        ).start()

    def _connect_wifi_worker(self, ssid, password):
        success, message = connect_to_wifi(ssid, password)
        self.after(0, lambda: self._apply_wifi_connection_result(success, message))

    def _apply_wifi_connection_result(self, success, message):
        self.connect_wifi_button.configure(state="normal")
        self.refresh_wifi_button.configure(state="normal")
        if success:
            self.wifi_password_entry.delete(0, "end")
            self.wifi_status_label.configure(text=message, text_color=SUCCESS)
            self.wifi_networks = []
            self.wifi_scan_loaded = False
            self.after(1500, self.refresh_web_access)
        else:
            self.wifi_status_label.configure(text=message, text_color=WARNING)

    def _selected_wifi_network(self):
        label = self.selected_wifi.get()
        for network in self.wifi_networks:
            if self._wifi_label(network) == label:
                return network
        return None

    @staticmethod
    def _wifi_label(network):
        lock = "secured" if network.get("requires_password") else "open"
        return f"{network['ssid']} ({network['signal']}%, {lock})"

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
            self.controller.append_batch_id = ""
            self.controller.append_batch_title = ""
            self.controller.batch_pages = 0
            self.controller.current_transcript_page = 0
            self.feedback_label.configure(text=message, text_color=SUCCESS)
            self.controller.show_frame("ScanningPage")
        else:
            self.feedback_label.configure(text=message, text_color=WARNING)

    def start_upload_test(self):
        self.controller.upload_test_mode = True
        self.controller.append_batch_id = ""
        self.controller.append_batch_title = ""
        self.controller.batch_pages = 0
        self.controller.current_transcript_page = 0
        self.controller.captured_pages = []
        self.controller.recognition_pages = []
        self.controller.transcript_results = []
        self.feedback_label.configure(text="Upload test mode ready.", text_color=SUCCESS)
        self.controller.show_frame("ScanningPage")
