from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageOps

from app.config import DATA_DIR
from lcd_ui.assets.theme import *
from lcd_ui.views.ui_helpers import build_header


THUMB_SIZE = (198, 104)


class ProcessingResultsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.current_page_index = 0
        self.image_refs = []

        build_header(self, "PROCESSING RESULTS", SUCCESS)

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(fill="both", expand=True)

        top_bar = ctk.CTkFrame(content, fg_color="transparent")
        top_bar.pack(fill="x", padx=24, pady=(12, 6))

        self.title_label = ctk.CTkLabel(
            top_bar,
            text="Generated Pipeline Outputs",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
        )
        self.title_label.pack(side="left")

        self.page_indicator = ctk.CTkLabel(
            top_bar,
            text="Page 1 of 1",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        )
        self.page_indicator.pack(side="right")

        self.stage_grid = ctk.CTkFrame(content, fg_color="transparent")
        self.stage_grid.pack(fill="both", expand=True, padx=22, pady=(2, 8))

        ctk.CTkFrame(self, height=1, fg_color="#7f6d49").pack(fill="x")

        footer = ctk.CTkFrame(self, fg_color=BG_COLOR, height=78)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        buttons = ctk.CTkFrame(footer, fg_color="transparent")
        buttons.pack(expand=True)

        self.prev_button = ctk.CTkButton(
            buttons,
            text="PREV PAGE",
            width=140,
            height=48,
            corner_radius=12,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 13, "bold"),
            command=self.show_prev_page,
        )
        self.prev_button.grid(row=0, column=0, padx=8, pady=10)

        self.next_button = ctk.CTkButton(
            buttons,
            text="NEXT PAGE",
            width=140,
            height=48,
            corner_radius=12,
            fg_color="#2f8f68",
            hover_color="#246f51",
            font=("Arial", 13, "bold"),
            command=self.show_next_page,
        )
        self.next_button.grid(row=0, column=1, padx=8, pady=10)

        ctk.CTkButton(
            buttons,
            text="CONTINUE",
            width=160,
            height=48,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#163bb5",
            font=("Arial", 14, "bold"),
            command=self.continue_to_transcript,
        ).grid(row=0, column=2, padx=8, pady=10)

    def on_show(self):
        total_pages = self.get_total_pages()
        self.current_page_index = min(self.current_page_index, total_pages - 1)
        self.refresh_results()

    def get_total_pages(self):
        return max(1, len(getattr(self.controller, "transcript_results", []) or []))

    def show_next_page(self):
        total_pages = self.get_total_pages()
        self.current_page_index = (self.current_page_index + 1) % total_pages
        self.refresh_results()

    def show_prev_page(self):
        total_pages = self.get_total_pages()
        self.current_page_index = (self.current_page_index - 1) % total_pages
        self.refresh_results()

    def continue_to_transcript(self):
        self.controller.current_transcript_page = self.current_page_index
        self.controller.show_frame("TranscriptPage")

    def refresh_results(self):
        for child in self.stage_grid.winfo_children():
            child.destroy()
        self.image_refs = []

        results = getattr(self.controller, "transcript_results", []) or []
        total_pages = self.get_total_pages()
        self.page_indicator.configure(text=f"Page {self.current_page_index + 1} of {total_pages}")
        if total_pages > 1:
            self.prev_button.grid()
            self.next_button.grid()
        else:
            self.prev_button.grid_remove()
            self.next_button.grid_remove()

        if not results:
            self._show_empty_state("No processing outputs are available yet.")
            return

        page = results[self.current_page_index % len(results)]
        self.title_label.configure(text=f"Generated Pipeline Outputs - Page {self.current_page_index + 1}")
        stages = list(getattr(page, "stage_image_paths", []) or [])
        if not stages:
            self._show_empty_state("This page did not return saved processing outputs.")
            return

        for index, stage in enumerate(stages[:8]):
            row = index // 4
            column = index % 4
            self._add_stage_card(row, column, stage.get("label", "Stage"), stage.get("path", ""))

    def _show_empty_state(self, message):
        ctk.CTkLabel(
            self.stage_grid,
            text=message,
            text_color=SUBTEXT,
            font=("Arial", 18),
        ).place(relx=0.5, rely=0.45, anchor="center")

    def _add_stage_card(self, row, column, label, path):
        card = ctk.CTkFrame(
            self.stage_grid,
            fg_color=CARD_COLOR,
            corner_radius=14,
            width=232,
            height=166,
        )
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_propagate(False)

        ctk.CTkLabel(
            card,
            text=label,
            text_color=TEXT,
            font=("Arial", 12, "bold"),
        ).pack(pady=(8, 5))

        thumbnail = self._load_thumbnail(path)
        if thumbnail is None:
            ctk.CTkLabel(
                card,
                text="Preview unavailable",
                text_color=SUBTEXT,
                width=THUMB_SIZE[0],
                height=THUMB_SIZE[1],
                fg_color="#0f2033",
                corner_radius=10,
            ).pack()
            return

        image = ctk.CTkImage(light_image=thumbnail, dark_image=thumbnail, size=thumbnail.size)
        self.image_refs.append(image)
        ctk.CTkLabel(card, text="", image=image).pack()

    def _load_thumbnail(self, relative_path):
        if not relative_path:
            return None

        path = Path(relative_path)
        if not path.is_absolute():
            path = DATA_DIR / relative_path
        if not path.is_file():
            return None

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None
        return ImageOps.contain(image, THUMB_SIZE)
