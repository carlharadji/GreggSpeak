import customtkinter as ctk

from assets.theme import *
from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        build_header(self, "SETTINGS", SUCCESS, divider_color="#1f3b57")
        ctk.CTkFrame(self, fg_color="transparent", height=5).pack(fill="x")

        scroll = ctk.CTkScrollableFrame(self, fg_color=BG_COLOR)
        scroll.pack(fill="both", expand=True)

        self.section_title(scroll, "Audio Settings")
        ctk.CTkLabel(scroll, text="Voice Type").pack(anchor="w", padx=20)

        self.voice_var = ctk.StringVar(value="Female")
        ctk.CTkSegmentedButton(
            scroll,
            values=["Male", "Female"],
            variable=self.voice_var,
            selected_color=ACCENT,
            selected_hover_color="#b8954f",
        ).pack(padx=20, pady=5, fill="x")

        ctk.CTkLabel(scroll, text="Speech Speed").pack(anchor="w", padx=20)
        speed_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        speed_frame.pack(fill="x", padx=20)

        self.speed_value = ctk.CTkLabel(speed_frame, text="1.0x")
        self.speed_value.pack(anchor="e")

        self.speed_slider = ctk.CTkSlider(
            scroll,
            from_=0.5,
            to=2.0,
            progress_color=ACCENT,
            button_color="#9333ea",
            command=self.update_speed,
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(padx=20, fill="x")

        ctk.CTkLabel(scroll, text="Volume").pack(anchor="w", padx=20)
        vol_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        vol_frame.pack(fill="x", padx=20)

        self.vol_value = ctk.CTkLabel(vol_frame, text="70%")
        self.vol_value.pack(anchor="e")

        self.vol_slider = ctk.CTkSlider(
            scroll,
            from_=0,
            to=100,
            progress_color=ACCENT,
            button_color="#9333ea",
            command=self.update_volume,
        )
        self.vol_slider.set(70)
        self.vol_slider.pack(padx=20, fill="x")

        self.section_title(scroll, "Storage Settings")
        status_card = ctk.CTkFrame(
            scroll,
            fg_color=CARD_COLOR,
            corner_radius=12,
            border_width=1,
            border_color="#24415f",
        )
        status_card.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(
            status_card,
            text="Database Storage Enabled",
            text_color=SUCCESS,
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 4))

        ctk.CTkLabel(
            status_card,
            text=(
                "All saved transcripts are stored in the GreggSpeak database "
                "and shown automatically in the web app records."
            ),
            text_color=SUBTEXT,
            justify="left",
            anchor="w",
            wraplength=640,
        ).pack(anchor="w", padx=10, pady=10)

        self.section_title(scroll, "About GreggSpeak")
        about_card = ctk.CTkFrame(scroll, fg_color=CARD_COLOR, corner_radius=12)
        about_card.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(
            about_card,
            text="Thesis Title",
            text_color=ACCENT,
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 4))

        ctk.CTkLabel(
            about_card,
            text=(
                "GreggSpeak: Vision-Based Translator and Reader for Gregg "
                "Shorthand in Municipal Trial Courts"
            ),
            justify="left",
            anchor="w",
            wraplength=640,
        ).pack(fill="x", padx=14)

        ctk.CTkLabel(
            about_card,
            text="System Information",
            text_color=ACCENT,
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))

        ctk.CTkLabel(
            about_card,
            text="* Device: Raspberry Pi 5\n* Display: 7-inch Touchscreen LCD",
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=14)

        ctk.CTkLabel(
            about_card,
            text="Development Team - Group 4C9",
            text_color=ACCENT,
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))

        ctk.CTkLabel(
            about_card,
            text=(
                "* Tongol, Ian Jovic D.\n"
                "* Bustos, Aaron Gabriel D.\n"
                "* Gonzales, Mark M.\n"
                "* Haradji, Carl Eugene S.\n"
                "* Tarrosa, Raniel N."
            ),
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 14))

        ctk.CTkButton(
            self,
            text="BACK",
            width=160,
            height=50,
            corner_radius=12,
            fg_color="#4b5563",
            hover_color="#374151",
            font=("Arial", 14, "bold"),
            command=lambda: confirm_navigation(
                self,
                controller,
                "HomePage",
                "Are you sure you want to go back to Home?",
            ),
        ).pack(pady=10)

    def section_title(self, parent, text):
        ctk.CTkLabel(
            parent,
            text=text,
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", padx=20, pady=(15, 5))

    def update_speed(self, value):
        self.speed_value.configure(text=f"{value:.1f}x")

    def update_volume(self, value):
        self.vol_value.configure(text=f"{int(value)}%")

    def on_hide(self):
        hide_navigation_confirmation(self)
