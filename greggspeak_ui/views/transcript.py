import customtkinter as ctk

from assets.theme import *
from backend.nlp import structure_transcript
from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation


class TranscriptPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.transcript_pages = [
            (
                "SYNTHETIC DEMO — no real testimony or personal information. "
                "The Court asked the witness to confirm that the demonstration "
                "page was received. The witness confirmed receipt.\n\n"
                "The Court asked whether the sample was legible. The witness "
                "answered yes."
            ),
            (
                "SYNTHETIC DEMO — this page illustrates transcript layout. "
                "The Court asked whether the sample image was clear. The witness "
                "answered that the demonstration image was readable.\n\n"
                "The clerk marked the sample page for review."
            ),
            (
                "SYNTHETIC DEMO — this is fictional interface text. The Court "
                "asked whether any correction was needed. The witness answered "
                "that the sample could be reviewed.\n\n"
                "The clerk marked the end of the demonstration."
            ),
        ]

        build_header(self, "TRANSCRIPT", "#19c28e")

        content = ctk.CTkFrame(self, fg_color=BG_COLOR)
        content.pack(fill="both", expand=True)

        page_info = ctk.CTkFrame(content, fg_color="transparent")
        page_info.pack(fill="x", padx=24, pady=(18, 8))

        self.page_title = ctk.CTkLabel(
            page_info,
            text="Transcript Output",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
        )
        self.page_title.pack(side="left")

        self.page_indicator = ctk.CTkLabel(
            page_info,
            text="Page 1 of 1",
            text_color=ACCENT,
            font=("Arial", 14, "bold"),
        )
        self.page_indicator.pack(side="right")

        self.transcript_box = ctk.CTkTextbox(
            content,
            fg_color=CARD_COLOR,
            text_color=TEXT,
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            font=("Georgia", 15),
            wrap="word",
            activate_scrollbars=True,
            scrollbar_button_color="#7b7b7b",
            scrollbar_button_hover_color="#9a9a9a",
        )
        self.transcript_box.pack(padx=24, pady=(8, 14), fill="both", expand=True)

        confidence_frame = ctk.CTkFrame(content, fg_color="transparent")
        confidence_frame.pack(pady=(0, 18))

        ctk.CTkLabel(
            confidence_frame,
            text="Confidence Level:",
            text_color="#e5eef9",
            font=("Georgia", 14),
        ).pack(side="left")

        self.confidence_label = ctk.CTkLabel(
            confidence_frame,
            text=" 92%",
            text_color="#10d39d",
            font=("Georgia", 14),
        )
        self.confidence_label.pack(side="left")

        self.processing_time_label = ctk.CTkLabel(
            confidence_frame,
            text="",
            text_color=ACCENT,
            font=("Georgia", 14),
        )
        self.processing_time_label.pack(side="left", padx=(18, 0))

        self.page_nav_frame = ctk.CTkFrame(content, fg_color="transparent")
        self.page_nav_frame.pack(pady=(0, 18))

        self.prev_button = ctk.CTkButton(
            self.page_nav_frame,
            text="PREV PAGE",
            width=140,
            height=42,
            corner_radius=12,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 13, "bold"),
            command=self.show_prev_page,
        )
        self.prev_button.grid(row=0, column=0, padx=8)

        self.next_button = ctk.CTkButton(
            self.page_nav_frame,
            text="NEXT PAGE",
            width=140,
            height=42,
            corner_radius=12,
            fg_color="#2f6fa3",
            hover_color="#285f8d",
            font=("Arial", 13, "bold"),
            command=self.show_next_page,
        )
        self.next_button.grid(row=0, column=1, padx=8)

        ctk.CTkFrame(self, height=1, fg_color="#d0ae59").pack(fill="x")

        footer = ctk.CTkFrame(self, fg_color=BG_COLOR, height=82)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        btn_frame = ctk.CTkFrame(footer, fg_color="transparent")
        btn_frame.pack(expand=True)

        ctk.CTkButton(
            btn_frame,
            text="SAVE",
            width=140,
            height=50,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#2857c8",
            font=("Arial", 14, "bold"),
            command=lambda: controller.show_frame("SaveOptionsPage"),
        ).grid(row=0, column=0, padx=8, pady=12)

        ctk.CTkButton(
            btn_frame,
            text="BACK",
            width=140,
            height=50,
            corner_radius=12,
            fg_color="#6f7f95",
            hover_color="#5f6e82",
            font=("Arial", 14, "bold"),
            command=lambda: confirm_navigation(
                self,
                controller,
                "HomePage",
                "Are you sure you want to go back to Home?",
            ),
        ).grid(row=0, column=1, padx=8, pady=12)

    def on_show(self):
        total_pages = self.get_total_pages()
        current_index = min(
            getattr(self.controller, "current_transcript_page", 0),
            total_pages - 1,
        )
        self.controller.current_transcript_page = current_index
        self.refresh_transcript_page()

    def on_hide(self):
        hide_navigation_confirmation(self)

    def show_next_page(self):
        total_pages = self.get_total_pages()
        current_index = getattr(self.controller, "current_transcript_page", 0)
        self.controller.current_transcript_page = (current_index + 1) % total_pages
        self.refresh_transcript_page()

    def show_prev_page(self):
        total_pages = self.get_total_pages()
        current_index = getattr(self.controller, "current_transcript_page", 0)
        self.controller.current_transcript_page = (current_index - 1) % total_pages
        self.refresh_transcript_page()

    def refresh_transcript_page(self):
        total_pages = self.get_total_pages()
        current_index = getattr(self.controller, "current_transcript_page", 0)

        page_text = self.get_current_transcript_text()
        confidence = self.get_current_confidence()

        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", page_text)
        self.transcript_box.configure(state="disabled")

        self.page_indicator.configure(text=f"Page {current_index + 1} of {total_pages}")
        self.page_title.configure(
            text="Transcript Output"

        )
        self.confidence_label.configure(text=f" {confidence:.2f}%")
        processing_time_text = getattr(self.controller, "processing_time_text", "")
        self.processing_time_label.configure(
            text=f"|  {processing_time_text}" if processing_time_text else ""
        )

        if total_pages > 1:
            self.prev_button.grid()
            self.next_button.grid()
        else:
            self.prev_button.grid_remove()
            self.next_button.grid_remove()

    def get_current_transcript_text(self):
        current_index = getattr(self.controller, "current_transcript_page", 0)
        results = getattr(self.controller, "transcript_results", []) or []
        if results:
            page = results[current_index % len(results)]
            if getattr(page, "error", None):
                return f"Recognition issue: {page.error}"
            if getattr(page, "transcript_text", "").strip():
                return page.transcript_text
            raw_text = getattr(page, "raw_transcript_text", "")
            formatted_text = structure_transcript(raw_text)
            if formatted_text.strip():
                return formatted_text
            return "No shorthand words were detected on this page."
        return self.transcript_pages[current_index % len(self.transcript_pages)]

    def get_current_confidence(self):
        current_index = getattr(self.controller, "current_transcript_page", 0)
        results = getattr(self.controller, "transcript_results", []) or []
        if results:
            page = results[current_index % len(results)]
            return float(getattr(page, "confidence", 0.0) or 0.0)
        return float(92 - (current_index * 2))

    def get_total_pages(self):
        results = getattr(self.controller, "transcript_results", []) or []
        if results:
            return max(1, len(results))
        return max(1, getattr(self.controller, "batch_pages", 1))

    def get_all_transcript_pages(self):
        total_pages = self.get_total_pages()
        pages = []
        original_index = getattr(self.controller, "current_transcript_page", 0)
        results = getattr(self.controller, "transcript_results", []) or []

        for page_index in range(total_pages):
            self.controller.current_transcript_page = page_index
            result_page = results[page_index] if page_index < len(results) else None
            pages.append(
                {
                    "page_number": page_index + 1,
                    "transcript_text": self.get_current_transcript_text(),
                    "confidence": self.get_current_confidence(),
                    "image_path": getattr(result_page, "image_path", "") if result_page else "",
                    "segmentation_image_path": getattr(result_page, "segmentation_image_path", "") if result_page else "",
                    "rows_detected": getattr(result_page, "rows_detected", 0) if result_page else 0,
                    "words_detected": getattr(result_page, "words_detected", 0) if result_page else 0,
                    "low_confidence_words": getattr(result_page, "low_confidence_words", 0) if result_page else 0,
                    "review_notes": getattr(result_page, "review_notes", "") if result_page else "",
                }
            )

        self.controller.current_transcript_page = original_index
        return pages
