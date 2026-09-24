import customtkinter as ctk

from lcd_ui.assets.theme import ACCENT, BG_COLOR, CARD_COLOR, PRIMARY, TEXT


class TouchKeyboard:
    def __init__(self, root):
        self.root = root
        self.target = None
        self.shift = False

        self.frame = ctk.CTkFrame(
            root,
            fg_color=BG_COLOR,
            border_width=1,
            border_color="#1f3b57",
        )

        self.key_buttons = []
        self._build()

    def _build(self):
        rows = (
            ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"),
            ("q", "w", "e", "r", "t", "y", "u", "i", "o", "p"),
            ("a", "s", "d", "f", "g", "h", "j", "k", "l"),
            ("shift", "z", "x", "c", "v", "b", "n", "m", "back"),
            ("clear", "space", "done"),
        )

        for row_index, keys in enumerate(rows):
            row = ctk.CTkFrame(self.frame, fg_color="transparent")
            row.pack(pady=(6 if row_index == 0 else 2, 2))

            for key in keys:
                label = self._label_for_key(key)
                width = 56
                if key == "space":
                    width = 260
                elif key in {"shift", "clear", "done", "back"}:
                    width = 88

                button = ctk.CTkButton(
                    row,
                    text=label,
                    width=width,
                    height=34,
                    corner_radius=8,
                    fg_color=PRIMARY if key == "done" else CARD_COLOR,
                    hover_color="#163bb5" if key == "done" else "#1b3958",
                    text_color=TEXT,
                    font=("Arial", 12, "bold"),
                    command=lambda value=key: self.press(value),
                )
                button.grid(row=0, column=len(row.grid_slaves()), padx=3)
                self.key_buttons.append((key, button))

    def _label_for_key(self, key):
        labels = {
            "shift": "SHIFT",
            "back": "DEL",
            "space": "SPACE",
            "clear": "CLEAR",
            "done": "DONE",
        }
        if key in labels:
            return labels[key]
        return key.upper() if self.shift else key

    def register(self, entry):
        entry.bind("<FocusIn>", lambda _event, target=entry: self.show(target), add="+")
        entry.bind("<Button-1>", lambda _event, target=entry: self.show(target), add="+")

    def show(self, target):
        self.target = target
        self.frame.place(relx=0, rely=1, relwidth=1, anchor="sw")
        self.frame.lift()

    def hide(self):
        self.frame.place_forget()
        self.target = None

    def press(self, key):
        if key == "done":
            self.hide()
            if self.root.focus_get() is not None:
                self.root.focus_set()
            return

        if self.target is None:
            return

        if key == "shift":
            self.shift = not self.shift
            self._refresh_labels()
            return
        if key == "back":
            self._backspace()
            return
        if key == "clear":
            self.target.delete(0, "end")
            return
        if key == "space":
            self.target.insert("insert", " ")
            return

        value = key.upper() if self.shift else key
        self.target.insert("insert", value)
        if self.shift:
            self.shift = False
            self._refresh_labels()

    def _backspace(self):
        try:
            start = self.target.index("sel.first")
            end = self.target.index("sel.last")
            self.target.delete(start, end)
            return
        except Exception:
            pass

        cursor = self.target.index("insert")
        if cursor > 0:
            self.target.delete(cursor - 1, cursor)

    def _refresh_labels(self):
        for key, button in self.key_buttons:
            button.configure(
                text=self._label_for_key(key),
                border_width=1 if key == "shift" and self.shift else 0,
                border_color=ACCENT,
            )
