from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "reports" / "demo_v5_next_original_orientation_word_exports"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "reports" / "demo_v5_next_original_orientation_debug"
DEFAULT_TRANSCRIPT = PROJECT_ROOT / "data" / "demo_v5_transcript.json"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "labels_all_words_v5_2.txt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "demo_v5_next_manual_word_crops"
POINT_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass
class LineTask:
    page: int
    line: int
    transcript: str
    labels: list[str]


@dataclass
class SavedCrop:
    path: Path
    label: str
    page: int
    line: int
    word_position: int
    box: tuple[int, int, int, int]
    source_image: Path
    transcript: str


def clean_words(line: str) -> list[str]:
    text = line.replace(":", " ")
    text = re.sub(r"[^a-zA-Z0-9_']+", " ", text)
    return [word.lower().strip("'") for word in text.split() if word.strip("'")]


def phrase_labels(words: list[str], label_set: set[str], max_phrase_words: int = 5) -> list[str]:
    labels: list[str] = []
    index = 0
    while index < len(words):
        matched = None
        max_length = min(max_phrase_words, len(words) - index)
        for length in range(max_length, 0, -1):
            candidate = "_".join(words[index:index + length])
            if candidate in label_set:
                matched = candidate
                index += length
                break
        if matched is None:
            matched = words[index]
            index += 1
        labels.append(matched)
    return labels


def load_labels(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_tasks(transcript_path: Path, label_set: set[str]) -> list[LineTask]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    tasks: list[LineTask] = []
    for page in data["pages"]:
        page_number = int(page["page"])
        for line_index, line in enumerate(page["lines"], start=1):
            tasks.append(
                LineTask(
                    page=page_number,
                    line=line_index,
                    transcript=line,
                    labels=phrase_labels(clean_words(line), label_set),
                )
            )
    return tasks


def read_line_boxes(debug_dir: Path) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for csv_path in sorted(debug_dir.glob("Page_*/line_boxes.csv")):
        page_match = re.search(r"Page_(\d+)", str(csv_path.parent.name))
        if not page_match:
            continue
        page_number = int(page_match.group(1))
        try:
            with csv_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    line_number = int(row["line_index"])
                    boxes[(page_number, line_number)] = (
                        int(row["x"]),
                        int(row["y"]),
                        int(row["w"]),
                        int(row["h"]),
                    )
        except Exception:
            continue
    return boxes


def safe_name(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_").lower() or "unknown"


class ManualWordCropper:
    def __init__(
        self,
        root: tk.Tk,
        export_dir: Path,
        debug_dir: Path,
        transcript_path: Path,
        labels_path: Path,
        out_dir: Path,
    ) -> None:
        self.root = root
        self.export_dir = export_dir
        self.debug_dir = debug_dir
        self.transcript_path = transcript_path
        self.labels_path = labels_path
        self.out_dir = out_dir
        self.manifest_path = out_dir / "_manual_crops_manifest.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.label_set = load_labels(labels_path)
        self.tasks = load_tasks(transcript_path, self.label_set)
        self.line_boxes = read_line_boxes(debug_dir)
        self.task_index = 0
        self.word_index = 0
        self.zoom = 1.0
        self.pad_px = tk.IntVar(value=8)
        self.current_image_path: Path | None = None
        self.current_image: Image.Image | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.image_canvas_id: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.current_box: tuple[int, int, int, int] | None = None
        self.box_id: int | None = None
        self.line_box_id: int | None = None
        self.saved: list[SavedCrop] = []

        self.root.title("GreggSpeak Manual Word Cropper")
        self.root.geometry("1220x780")
        self.build_ui()
        self.bind_keys()
        self.load_current_task()

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#1f2933", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        side = ttk.Frame(self.root, padding=12)
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)

        self.progress_var = tk.StringVar()
        self.label_var = tk.StringVar()
        self.transcript_var = tk.StringVar()
        self.expected_var = tk.StringVar()
        self.status_var = tk.StringVar()

        ttk.Label(side, text="Current Line", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(side, textvariable=self.progress_var, wraplength=330).grid(row=1, column=0, sticky="w", pady=(2, 10))

        ttk.Label(side, text="Crop This Label", font=("Segoe UI", 11, "bold")).grid(row=2, column=0, sticky="w")
        ttk.Label(side, textvariable=self.label_var, font=("Segoe UI", 16, "bold"), foreground="#065f46").grid(
            row=3, column=0, sticky="w", pady=(2, 10)
        )

        ttk.Label(side, text="Transcript").grid(row=4, column=0, sticky="w")
        ttk.Label(side, textvariable=self.transcript_var, wraplength=330).grid(row=5, column=0, sticky="w", pady=(2, 10))

        ttk.Label(side, text="Expected Labels").grid(row=6, column=0, sticky="w")
        ttk.Label(side, textvariable=self.expected_var, wraplength=330).grid(row=7, column=0, sticky="w", pady=(2, 12))

        ttk.Button(side, text="Save Crop + Next (Enter)", command=self.save_crop_and_next).grid(
            row=8, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(side, text="Skip Label", command=self.skip_label).grid(row=9, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(side, text="Undo Last Crop (Backspace)", command=self.undo_last).grid(
            row=10, column=0, sticky="ew", pady=(0, 12)
        )

        nav = ttk.Frame(side)
        nav.grid(row=11, column=0, sticky="ew", pady=(0, 12))
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        ttk.Button(nav, text="Prev Line", command=self.previous_line).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(nav, text="Next Line", command=self.next_line).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        zoom_bar = ttk.Frame(side)
        zoom_bar.grid(row=12, column=0, sticky="ew", pady=(0, 12))
        zoom_bar.columnconfigure(0, weight=1)
        zoom_bar.columnconfigure(1, weight=1)
        ttk.Button(zoom_bar, text="Zoom -", command=lambda: self.set_zoom(self.zoom / 1.25)).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(zoom_bar, text="Zoom +", command=lambda: self.set_zoom(self.zoom * 1.25)).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        pad_frame = ttk.Frame(side)
        pad_frame.grid(row=13, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(pad_frame, text="Crop padding px").pack(side=tk.LEFT)
        ttk.Spinbox(pad_frame, from_=0, to=40, textvariable=self.pad_px, width=5).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(side, text="Status").grid(row=14, column=0, sticky="w")
        ttk.Label(side, textvariable=self.status_var, wraplength=330).grid(row=15, column=0, sticky="w", pady=(2, 0))

    def bind_keys(self) -> None:
        self.root.bind("<Return>", lambda _event: self.save_crop_and_next())
        self.root.bind("<BackSpace>", lambda _event: self.undo_last())
        self.root.bind("<Right>", lambda _event: self.skip_label())
        self.root.bind("<Down>", lambda _event: self.next_line())
        self.root.bind("<Up>", lambda _event: self.previous_line())
        self.root.bind("<plus>", lambda _event: self.set_zoom(self.zoom * 1.25))
        self.root.bind("<minus>", lambda _event: self.set_zoom(self.zoom / 1.25))

    def task(self) -> LineTask:
        return self.tasks[self.task_index]

    def page_image_path(self, page_number: int) -> Path:
        page_dir = self.export_dir / f"Page_{page_number:04d}"
        for name in ("03_paper_crop.png", "04_clean_enhanced.png", "02_original_copy.png", "02_oriented.png"):
            path = page_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"No usable page image found in {page_dir}")

    def load_current_task(self) -> None:
        task = self.task()
        image_path = self.page_image_path(task.page)
        if image_path != self.current_image_path:
            self.current_image_path = image_path
            self.current_image = Image.open(image_path).convert("RGB")
            self.render_image()
        self.word_index = min(self.word_index, max(0, len(task.labels) - 1))
        self.update_text()
        self.draw_line_box()
        self.clear_current_box()
        self.scroll_to_current_line()

    def render_image(self) -> None:
        if self.current_image is None:
            return
        width = max(1, int(self.current_image.width * self.zoom))
        height = max(1, int(self.current_image.height * self.zoom))
        resized = self.current_image.resize((width, height), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.image_canvas_id = self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.box_id = None
        self.line_box_id = None
        self.draw_line_box()

    def update_text(self) -> None:
        task = self.task()
        total_words = len(task.labels)
        current_label = task.labels[self.word_index] if task.labels else "(no labels)"
        self.progress_var.set(
            f"Page {task.page:03d}, line {task.line:03d}  |  "
            f"Line {self.task_index + 1}/{len(self.tasks)}  |  "
            f"Word {self.word_index + 1}/{max(1, total_words)}"
        )
        self.label_var.set(current_label)
        self.transcript_var.set(task.transcript)
        self.expected_var.set("  ".join(f"{index + 1}:{label}" for index, label in enumerate(task.labels)))
        if current_label not in self.label_set:
            self.status_var.set(f"Warning: label '{current_label}' is not in the labels file.")
        else:
            self.status_var.set(f"Draw a box around '{current_label}', then press Enter.")

    def draw_line_box(self) -> None:
        if self.line_box_id is not None:
            self.canvas.delete(self.line_box_id)
            self.line_box_id = None
        task = self.task()
        box = self.line_boxes.get((task.page, task.line))
        if box is None:
            return
        x, y, w, h = box
        self.line_box_id = self.canvas.create_rectangle(
            x * self.zoom,
            y * self.zoom,
            (x + w) * self.zoom,
            (y + h) * self.zoom,
            outline="#16a34a",
            width=3,
        )

    def scroll_to_current_line(self) -> None:
        task = self.task()
        box = self.line_boxes.get((task.page, task.line))
        if box is None or self.current_image is None:
            return
        _x, y, _w, _h = box
        full_height = max(1, int(self.current_image.height * self.zoom))
        self.canvas.yview_moveto(max(0.0, min(1.0, (y * self.zoom - 80) / full_height)))

    def canvas_to_image(self, event: tk.Event) -> tuple[int, int]:
        x = self.canvas.canvasx(event.x) / self.zoom
        y = self.canvas.canvasy(event.y) / self.zoom
        if self.current_image is None:
            return (0, 0)
        return (
            int(max(0, min(self.current_image.width - 1, x))),
            int(max(0, min(self.current_image.height - 1, y))),
        )

    def on_mouse_down(self, event: tk.Event) -> None:
        self.drag_start = self.canvas_to_image(event)
        self.clear_current_box()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = self.canvas_to_image(event)
        self.draw_current_box((x1, y1, x2, y2))

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = self.canvas_to_image(event)
        self.drag_start = None
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        if x_max - x_min < 4 or y_max - y_min < 4:
            self.clear_current_box()
            self.status_var.set("Box is too small. Draw the word again.")
            return
        self.current_box = (x_min, y_min, x_max, y_max)
        self.draw_current_box(self.current_box)
        self.status_var.set("Box ready. Press Enter to save crop.")

    def draw_current_box(self, box: tuple[int, int, int, int]) -> None:
        if self.box_id is not None:
            self.canvas.delete(self.box_id)
        x1, y1, x2, y2 = box
        self.box_id = self.canvas.create_rectangle(
            x1 * self.zoom,
            y1 * self.zoom,
            x2 * self.zoom,
            y2 * self.zoom,
            outline="#dc2626",
            width=3,
        )

    def clear_current_box(self) -> None:
        self.current_box = None
        if self.box_id is not None:
            self.canvas.delete(self.box_id)
            self.box_id = None

    def next_label(self) -> None:
        task = self.task()
        if self.word_index + 1 < len(task.labels):
            self.word_index += 1
            self.update_text()
            self.clear_current_box()
            return
        self.next_line()

    def skip_label(self) -> None:
        self.status_var.set("Skipped current label.")
        self.next_label()

    def next_line(self) -> None:
        if self.task_index + 1 >= len(self.tasks):
            self.status_var.set("Reached the last line.")
            return
        self.task_index += 1
        self.word_index = 0
        self.load_current_task()

    def previous_line(self) -> None:
        if self.task_index <= 0:
            self.status_var.set("Already at the first line.")
            return
        self.task_index -= 1
        self.word_index = 0
        self.load_current_task()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = max(0.2, min(5.0, float(zoom)))
        self.render_image()
        self.scroll_to_current_line()
        self.status_var.set(f"Zoom: {self.zoom:.2f}x")

    def next_crop_path(self, label: str, task: LineTask, word_position: int) -> Path:
        label_dir = self.out_dir / safe_name(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(label_dir.glob("*.png"))
        return label_dir / (
            f"manual_page_{task.page:03d}_line_{task.line:03d}_"
            f"word_{word_position:03d}_{safe_name(label)}_{len(existing) + 1:04d}.png"
        )

    def save_crop_and_next(self) -> None:
        if self.current_image is None or self.current_image_path is None:
            return
        task = self.task()
        if not task.labels:
            self.status_var.set("This line has no expected labels.")
            return
        label = task.labels[self.word_index]
        if label not in self.label_set:
            messagebox.showwarning("Unknown label", f"Label is not in the labels file:\n{label}")
            return
        if self.current_box is None:
            self.status_var.set("Draw a box first.")
            return

        pad = max(0, int(self.pad_px.get()))
        x1, y1, x2, y2 = self.current_box
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(self.current_image.width, x2 + pad)
        y2 = min(self.current_image.height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            self.status_var.set("Invalid crop box.")
            return

        crop = self.current_image.crop((x1, y1, x2, y2))
        path = self.next_crop_path(label, task, self.word_index + 1)
        crop.save(path)
        record = SavedCrop(
            path=path,
            label=label,
            page=task.page,
            line=task.line,
            word_position=self.word_index + 1,
            box=(x1, y1, x2, y2),
            source_image=self.current_image_path,
            transcript=task.transcript,
        )
        self.saved.append(record)
        self.write_manifest()
        self.status_var.set(f"Saved {label}: {path.name}")
        self.next_label()

    def undo_last(self) -> None:
        if not self.saved:
            self.status_var.set("Nothing to undo in this session.")
            return
        record = self.saved.pop()
        try:
            record.path.unlink(missing_ok=True)
        except Exception:
            pass
        self.write_manifest()
        self.status_var.set(f"Removed {record.path.name}")

    def write_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "image_file",
                    "label",
                    "page",
                    "line",
                    "word_position",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "source_image",
                    "transcript",
                ]
            )
            for record in self.saved:
                x1, y1, x2, y2 = record.box
                writer.writerow(
                    [
                        str(record.path),
                        record.label,
                        record.page,
                        record.line,
                        record.word_position,
                        x1,
                        y1,
                        x2,
                        y2,
                        str(record.source_image),
                        record.transcript,
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual page/line/word cropper for GreggSpeak training data.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        path
        for path in (args.export_dir, args.transcript, args.labels)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(str(path) for path in missing))

    root = tk.Tk()
    ManualWordCropper(
        root=root,
        export_dir=args.export_dir,
        debug_dir=args.debug_dir,
        transcript_path=args.transcript,
        labels_path=args.labels,
        out_dir=args.out_dir,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
