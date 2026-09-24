import customtkinter as ctk

from assets.theme import *
from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        build_header(self, "SETTINGS", SUCCESS, divider_color="#53779d")
        ctk.CTkFrame(self, fg_color="transparent", height=5).pack(fill="x")

        scroll = ctk.CTkScrollableFrame(self, fg_color=BG_COLOR)
        scroll.pack(fill="both", expand=True)

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
            fg_color="#607086",
            hover_color="#4f5f73",
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

    def on_hide(self):
        hide_navigation_confirmation(self)
