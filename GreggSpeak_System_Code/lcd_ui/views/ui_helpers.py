import customtkinter as ctk

from assets.theme import ACCENT, BG_COLOR, PRIMARY, TEXT


def build_header(parent, title, status_color, divider_color="#7f6d49"):
    header = ctk.CTkFrame(parent, fg_color=BG_COLOR, height=50)
    header.pack(fill="x", pady=(5, 0))
    header.pack_propagate(False)

    ctk.CTkLabel(
        header,
        text="GREGGSPEAK",
        text_color=ACCENT,
        font=("Arial", 14, "bold"),
    ).pack(side="left", padx=20)

    ctk.CTkLabel(
        header,
        text=title,
        text_color=TEXT,
        font=("Arial", 16, "bold"),
    ).place(relx=0.5, rely=0.5, anchor="center")

    ctk.CTkFrame(parent, height=1, fg_color=divider_color).pack(fill="x")


def _ensure_nav_overlay(frame):
    if hasattr(frame, "_nav_overlay"):
        return

    frame._nav_target = None
    frame._nav_overlay = ctk.CTkFrame(frame, fg_color="#050b12")

    card = ctk.CTkFrame(
        frame._nav_overlay,
        fg_color="#2c4597",
        corner_radius=16,
        border_width=1,
        border_color="#d0ae59",
        width=360,
        height=220,
    )
    card.place(relx=0.5, rely=0.5, anchor="center")
    card.pack_propagate(False)

    ctk.CTkLabel(
        card,
        text="Confirm",
        text_color=TEXT,
        font=("Arial", 18, "bold"),
    ).pack(pady=(28, 12))

    frame._nav_message_label = ctk.CTkLabel(
        card,
        text="Are you sure?",
        text_color=TEXT,
        font=("Arial", 14),
        wraplength=290,
        justify="center",
    )
    frame._nav_message_label.pack(padx=24, pady=(0, 20))

    btn_row = ctk.CTkFrame(card, fg_color="transparent")
    btn_row.pack(pady=(0, 20))

    ctk.CTkButton(
        btn_row,
        text="YES",
        width=120,
        height=44,
        corner_radius=12,
        fg_color=PRIMARY,
        hover_color="#163bb5",
        font=("Arial", 14, "bold"),
        command=lambda: _confirm_nav_yes(frame),
    ).grid(row=0, column=0, padx=8)

    ctk.CTkButton(
        btn_row,
        text="NO",
        width=120,
        height=44,
        corner_radius=12,
        fg_color="#5b6678",
        hover_color="#4a5362",
        font=("Arial", 14, "bold"),
        command=lambda: hide_navigation_confirmation(frame),
    ).grid(row=0, column=1, padx=8)


def confirm_navigation(frame, controller, target, prompt):
    _ensure_nav_overlay(frame)
    frame._nav_target = (controller, target)
    frame._nav_message_label.configure(text=prompt)
    frame._nav_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
    frame._nav_overlay.lift()


def hide_navigation_confirmation(frame):
    overlay = getattr(frame, "_nav_overlay", None)
    if overlay is not None:
        overlay.place_forget()
    frame._nav_target = None


def _confirm_nav_yes(frame):
    nav_target = getattr(frame, "_nav_target", None)
    hide_navigation_confirmation(frame)
    if nav_target is not None:
        controller, target = nav_target
        controller.show_frame(target)
