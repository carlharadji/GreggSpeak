import customtkinter as ctk
from datetime import datetime

from assets.theme import *
from backend.db import (
    add_page,
    clear_pending_append_batch,
    create_batch,
    generate_transcript,
    get_pages,
    get_pending_append_batch,
)
from views.ui_helpers import build_header, confirm_navigation, hide_navigation_confirmation


class SaveOptionsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_COLOR)

        self.controller = controller
        self.save_job = None
        self.confirm_button = None
        self.cancel_button = None
        self.overlay = None
        self.modal_card = None
        self.title_entry = None
        self.title_error_label = None
        self.append_target = None

        build_header(self, "SAVE TRANSCRIPT", "#19c28e")

        content = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color="#7b7b7b",
            scrollbar_button_hover_color="#9a9a9a",
        )
        content.pack(fill="both", expand=True)

        ctk.CTkLabel(
            content,
            text="Transcript Title",
            text_color=ACCENT,
            font=("Arial", 16),
        ).pack(anchor="w", padx=24, pady=(0, 8))

        self.title_entry = ctk.CTkEntry(
            content,
            height=46,
            corner_radius=12,
            fg_color=CARD_COLOR,
            text_color=TEXT,
            border_color="#7f6d49",
            border_width=1,
            font=("Arial", 15),
            placeholder_text="Enter transcript title",
            placeholder_text_color="#8892a5",
        )
        self.title_entry.pack(fill="x", padx=24)

        self.title_error_label = ctk.CTkLabel(
            content,
            text="",
            text_color="#ff7b7b",
            font=("Arial", 12),
        )
        self.title_error_label.pack(anchor="w", padx=24, pady=(6, 18))

        save_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color="#7f6d49",
        )
        save_card.pack(fill="x", padx=24, pady=(0, 24))

        ctk.CTkLabel(
            save_card,
            text="Database Save",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 6))

        ctk.CTkLabel(
            save_card,
            text=(
                "This transcript will be saved once to the GreggSpeak database "
                "and will automatically appear in the web app saved records."
            ),
            text_color=SUBTEXT,
            font=("Arial", 14),
            justify="left",
            anchor="w",
            wraplength=760,
        ).pack(fill="x", padx=20, pady=(0, 18))

        ctk.CTkFrame(content, fg_color="transparent", height=150).pack(fill="x")

        ctk.CTkFrame(self, height=1, fg_color="#7f6d49").pack(fill="x")

        footer = ctk.CTkFrame(self, fg_color=BG_COLOR, height=82)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        btn_frame = ctk.CTkFrame(footer, fg_color="transparent")
        btn_frame.pack(expand=True)

        self.confirm_button = ctk.CTkButton(
            btn_frame,
            text="SAVE TRANSCRIPT",
            width=180,
            height=50,
            corner_radius=12,
            fg_color=PRIMARY,
            hover_color="#163bb5",
            font=("Arial", 14, "bold"),
            command=self.confirm_save,
        )
        self.confirm_button.grid(row=0, column=0, padx=8, pady=12)

        self.cancel_button = ctk.CTkButton(
            btn_frame,
            text="CANCEL",
            width=150,
            height=50,
            corner_radius=12,
            fg_color="#5b6678",
            hover_color="#4a5362",
            font=("Arial", 14, "bold"),
            command=lambda: confirm_navigation(
                self,
                controller,
                "TranscriptPage",
                "Are you sure you want to cancel and go back to the transcript?",
            ),
        )
        self.cancel_button.grid(row=0, column=1, padx=8, pady=12)

        self.overlay = ctk.CTkFrame(self, fg_color="#050b12")
        self.save_error = False

        self.modal_card = ctk.CTkFrame(
            self.overlay,
            fg_color="#2c4597",
            corner_radius=16,
            border_width=1,
            border_color="#d0ae59",
            width=360,
            height=230,
        )
        self.modal_card.place(relx=0.5, rely=0.5, anchor="center")
        self.modal_card.pack_propagate(False)

        ctk.CTkLabel(
            self.modal_card,
            text="OK",
            text_color="#18cf96",
            font=("Arial", 26, "bold"),
        ).pack(pady=(30, 8))

        ctk.CTkLabel(
            self.modal_card,
            text="Save Successful!",
            text_color=TEXT,
            font=("Arial", 18, "bold"),
        ).pack(pady=(0, 10))

        self.modal_message = ctk.CTkLabel(
            self.modal_card,
            text="Transcript saved to database",
            text_color=TEXT,
            font=("Arial", 14),
            wraplength=300,
        )
        self.modal_message.pack()

        self.bind("<Destroy>", self._stop_jobs)

    def confirm_save(self):
        self._cancel_jobs()

        self.append_target = get_pending_append_batch()
        title = self.title_entry.get().strip()
        if not title and not self.append_target:
            self.title_error_label.configure(text="Please enter a transcript title before saving.")
            self.title_entry.focus()
            return

        self.title_error_label.configure(text="")
        self.confirm_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")

        self.save_error = False
        try:
            batch_id = self.save_batch_to_database(title)
            if self.append_target:
                modal_text = f"Pages added to \"{self.append_target.get('title', 'selected batch')}\""
            else:
                modal_text = f"\"{title}\" saved to the database ({batch_id})"
        except Exception as error:
            self.save_error = True
            modal_text = f"Save failed: {error}"

        self.modal_message.configure(text=modal_text)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay.lift()
        self.save_job = self.after(1800, self.finish_save)

    def save_batch_to_database(self, title):
        transcript_page = self.controller.frames.get("TranscriptPage")
        if transcript_page is None or not hasattr(transcript_page, "get_all_transcript_pages"):
            raise ValueError("Transcript data is unavailable.")

        transcript_pages = transcript_page.get_all_transcript_pages()
        if not transcript_pages:
            raise ValueError("No transcript pages found.")

        append_target = self.append_target or get_pending_append_batch()
        if append_target:
            batch_id = append_target.get("batch_id")
            pages_result = get_pages(batch_id)
            if not pages_result.get("success"):
                raise ValueError(pages_result.get("message", "Could not open target batch."))
            next_page_number = len(pages_result.get("data", [])) + 1
            self.save_pages_to_existing_batch(batch_id, transcript_pages, next_page_number)
            clear_pending_append_batch(batch_id)
            return batch_id

        filename = self.build_filename(title)
        batch_result = create_batch(title, filename, len(transcript_pages))
        if not batch_result.get("success"):
            raise ValueError(batch_result.get("message", "Could not create batch."))

        batch_data = batch_result.get("data") or {}
        batch_id = batch_data.get("batch_id")
        if not batch_id:
            raise ValueError("Batch ID was not returned.")

        self.save_pages_to_existing_batch(batch_id, transcript_pages, 1)
        return batch_id

    def save_pages_to_existing_batch(self, batch_id, transcript_pages, start_page_number):
        for offset, page in enumerate(transcript_pages):
            page_result = add_page(
                batch_id=batch_id,
                page_number=start_page_number + offset,
                raw_text=page.get("transcript_text", ""),
                raw_transcript_text=page.get("raw_transcript_text", ""),
                confidence=page.get("confidence", 0),
                image_path=page.get("image_path", ""),
                segmentation_image_path=page.get("segmentation_image_path", ""),
                rows_detected=page.get("rows_detected", 0),
                words_detected=page.get("words_detected", 0),
                low_confidence_words=page.get("low_confidence_words", 0),
                review_notes=page.get("review_notes", ""),
            )
            if not page_result.get("success"):
                raise ValueError(page_result.get("message", "Could not save page."))

        transcript_result = generate_transcript(batch_id)
        if not transcript_result.get("success"):
            raise ValueError(transcript_result.get("message", "Could not generate transcript."))

    def build_filename(self, title):
        safe_title = "".join(
            char.lower() if char.isalnum() else "_" for char in title.strip()
        )
        safe_title = "_".join(filter(None, safe_title.split("_"))) or "transcript"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_title}_{timestamp}"

    def finish_save(self):
        self.save_job = None
        self.hide_modal()
        if not self.save_error:
            self.controller.show_frame("HomePage")

    def hide_modal(self):
        self.overlay.place_forget()
        self.confirm_button.configure(state="normal")
        self.cancel_button.configure(state="normal")

    def on_show(self):
        self.append_target = get_pending_append_batch()
        if self.append_target:
            self.title_entry.configure(state="normal")
            self.title_entry.delete(0, "end")
            self.title_entry.insert(0, f"Adding pages to: {self.append_target.get('title', 'selected batch')}")
            self.title_entry.configure(state="disabled")
            self.confirm_button.configure(text="ADD PAGES")
            self.title_error_label.configure(text="")
        else:
            self.title_entry.configure(state="normal")
            self.title_entry.delete(0, "end")
            self.confirm_button.configure(text="SAVE TRANSCRIPT")
            self.title_error_label.configure(text="")

    def on_hide(self):
        self._cancel_jobs()
        self.hide_modal()
        if self.title_error_label is not None:
            self.title_error_label.configure(text="")
        hide_navigation_confirmation(self)

    def _cancel_jobs(self):
        if self.save_job is not None:
            try:
                self.after_cancel(self.save_job)
            except Exception:
                pass
            self.save_job = None

    def _stop_jobs(self, _event=None):
        self._cancel_jobs()
