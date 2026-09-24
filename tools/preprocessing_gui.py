from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROP_POINTS_FILE = PROJECT_ROOT / "calibration_crop_points.json"

IMAGE_TYPES = (
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("JPEG", "*.jpg *.jpeg"),
    ("PNG", "*.png"),
    ("All files", "*.*"),
)

POINT_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass
class PageState:
    path: Path
    original_bgr: np.ndarray
    cropped_bgr: np.ndarray | None = None
    cleaned_gray: np.ndarray | None = None
    binary_image: np.ndarray | None = None


def load_bgr_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Could not save image: {path}")
    encoded.tofile(str(path))


def read_crop_points(path: Path, target_width: int, target_height: int) -> list[tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source_width = int(data["image_width"])
    source_height = int(data["image_height"])
    corners = data["corners"]

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    points: list[tuple[int, int]] = []
    for key in POINT_KEYS:
        x, y = corners[key]
        points.append((int(round(x * scale_x)), int(round(y * scale_y))))
    return points


def perspective_crop(image_bgr: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    source = np.float32(points)
    width_top = np.linalg.norm(source[1] - source[0])
    width_bottom = np.linalg.norm(source[2] - source[3])
    height_right = np.linalg.norm(source[2] - source[1])
    height_left = np.linalg.norm(source[3] - source[0])

    output_width = max(1, int(round(max(width_top, width_bottom))))
    output_height = max(1, int(round(max(height_right, height_left))))
    destination = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image_bgr, matrix, (output_width, output_height))


def clean_enhance_image(cropped_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    flattened = cv2.divide(gray, background, scale=255)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

    filtered = cv2.bilateralFilter(flattened, d=7, sigmaColor=35, sigmaSpace=35)
    darkness = 255 - filtered.astype(np.float32)
    darkness = cv2.GaussianBlur(darkness, (3, 3), 0)

    enhanced = 255 - np.clip(darkness * 2.25, 0, 255).astype(np.uint8)
    enhanced[enhanced > 235] = 255
    enhanced = cv2.medianBlur(enhanced, 3)
    return enhanced


def binarize_image(cleaned_gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(cleaned_gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12,
    )
    ink_mask = cv2.bitwise_or(otsu, adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return 255 - ink_mask


def as_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def bgr_to_tk_image(image_bgr: np.ndarray, size: tuple[int, int]) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    resized = pil_image.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(resized)


class PreprocessingGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GreggSpeak Image Preprocessing")
        self.root.geometry("1180x760")
        self.root.minsize(960, 620)

        self.pages: list[PageState] = []
        self.canvas_image: ImageTk.PhotoImage | None = None

        self.preview_mode = tk.StringVar(value="original")
        self.status_var = tk.StringVar(value="Load image(s) to begin.")
        self.image_var = tk.StringVar(value="No image loaded")

        self._build_style()
        self._build_ui()

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", foreground="#374151")

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(12, 10, 12, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(6, weight=1)

        ttk.Button(toolbar, text="Load Image(s)", command=self.load_image).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="Crop Image", command=self.crop_image).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="Clean / Enhance Image", command=self.clean_enhance_image).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Binarize Image", command=self.binarize_image).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(toolbar, text="Process All", command=self.process_all_images).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(toolbar, text="Save Processed Images", command=self.save_processed_images).grid(
            row=0, column=5, padx=(0, 12)
        )
        ttk.Label(toolbar, textvariable=self.image_var, style="Header.TLabel").grid(row=0, column=6, sticky="w")

        main = ttk.Frame(self.root)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(main, padding=(12, 8, 10, 10))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.rowconfigure(11, weight=1)

        ttk.Label(sidebar, text="Preview", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        preview_options = (
            ("Original", "original"),
            ("Cropped", "cropped"),
            ("Cleaned / Enhanced", "cleaned"),
            ("Binary", "binary"),
        )
        for row, (label, value) in enumerate(preview_options, start=1):
            ttk.Radiobutton(
                sidebar,
                text=label,
                value=value,
                variable=self.preview_mode,
                command=self.update_preview,
            ).grid(row=row, column=0, sticky="w", pady=3)

        ttk.Label(
            sidebar,
            text=(
                "Workflow:\n"
                "1. Load Image(s)\n"
                "2. Crop Image\n"
                "3. Clean / Enhance Image\n"
                "4. Binarize Image\n\n"
                "Use Process All to run crop, clean, and binary on every loaded page."
            ),
            justify="left",
            wraplength=260,
            foreground="#4b5563",
        ).grid(row=6, column=0, sticky="ew", pady=(18, 0))

        ttk.Label(sidebar, text="Loaded Pages", font=("Segoe UI", 11, "bold")).grid(
            row=10, column=0, sticky="w", pady=(18, 6)
        )
        list_frame = ttk.Frame(sidebar)
        list_frame.grid(row=11, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.page_list = tk.Listbox(list_frame, width=36, height=9, exportselection=False)
        self.page_list.grid(row=0, column=0, sticky="nsew")
        self.page_list.bind("<<ListboxSelect>>", self.on_page_select)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.page_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.page_list.configure(yscrollcommand=scroll.set)

        canvas_frame = ttk.Frame(main, padding=(0, 8, 12, 10))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, background="#111827", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.update_preview())

        status = ttk.Frame(self.root, padding=(12, 6))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")

    def load_image(self) -> None:
        selected = filedialog.askopenfilenames(title="Load image(s)", filetypes=IMAGE_TYPES)
        if not selected:
            return

        try:
            self.pages = [PageState(path=Path(path), original_bgr=load_bgr_image(Path(path))) for path in selected]
            self.page_list.delete(0, tk.END)
            for page in self.pages:
                self.page_list.insert(tk.END, self.page_label(page))

            if self.pages:
                self.page_list.selection_set(0)
                self.page_list.activate(0)

            self.preview_mode.set("original")
            self.update_image_label()
            self.status_var.set(f"Loaded {len(self.pages)} image(s). Click Crop Image or Process All.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Load failed", exc)

    def selected_page(self) -> PageState | None:
        if not self.pages:
            return None
        selection = self.page_list.curselection()
        index = selection[0] if selection else 0
        if index >= len(self.pages):
            return None
        return self.pages[index]

    def selected_index(self) -> int | None:
        if not self.pages:
            return None
        selection = self.page_list.curselection()
        return selection[0] if selection else 0

    def on_page_select(self, _event: tk.Event) -> None:
        self.update_image_label()
        self.update_preview()

    def crop_image(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        try:
            self.crop_page(page)
            page.cleaned_gray = None
            page.binary_image = None
            self.refresh_selected_page_label()
            self.preview_mode.set("cropped")
            self.status_var.set("Crop complete. Click Clean / Enhance Image next.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Crop failed", exc)

    def clean_enhance_image(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if page.cropped_bgr is None:
            messagebox.showinfo("No crop", "Click Crop Image before cleaning/enhancing.")
            return

        try:
            page.cleaned_gray = clean_enhance_image(page.cropped_bgr)
            page.binary_image = None
            self.refresh_selected_page_label()
            self.preview_mode.set("cleaned")
            self.status_var.set("Clean/enhance complete. Click Binarize Image next.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Clean/enhance failed", exc)

    def binarize_image(self) -> None:
        page = self.selected_page()
        if page is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if page.cleaned_gray is None:
            if page.cropped_bgr is None:
                messagebox.showinfo("No crop", "Click Crop Image before binarizing.")
                return
            page.cleaned_gray = clean_enhance_image(page.cropped_bgr)

        try:
            page.binary_image = binarize_image(page.cleaned_gray)
            self.refresh_selected_page_label()
            self.preview_mode.set("binary")
            self.status_var.set("Binarization complete.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Binarization failed", exc)

    def process_all_images(self) -> None:
        if not self.pages:
            messagebox.showinfo("No images", "Load images first.")
            return

        try:
            for index, page in enumerate(self.pages, start=1):
                self.status_var.set(f"Processing {index}/{len(self.pages)}: {page.path.name}")
                self.root.update_idletasks()
                self.crop_page(page)
                page.cleaned_gray = clean_enhance_image(page.cropped_bgr)
                page.binary_image = binarize_image(page.cleaned_gray)

            self.refresh_page_list()
            self.preview_mode.set("cropped")
            self.status_var.set(f"Processed {len(self.pages)} image(s). Use the page list to inspect results.")
            self.update_preview()
        except Exception as exc:
            self.show_error("Process all failed", exc)

    def crop_page(self, page: PageState) -> None:
        if not CROP_POINTS_FILE.exists():
            raise FileNotFoundError(f"Could not find {CROP_POINTS_FILE}")
        height, width = page.original_bgr.shape[:2]
        crop_points = read_crop_points(CROP_POINTS_FILE, width, height)
        page.cropped_bgr = perspective_crop(page.original_bgr, crop_points)

    def save_processed_images(self) -> None:
        pages_with_outputs = [
            page
            for page in self.pages
            if page.cropped_bgr is not None or page.cleaned_gray is not None or page.binary_image is not None
        ]
        if not pages_with_outputs:
            messagebox.showinfo("Nothing to save", "Crop, clean, or binarize at least one image first.")
            return

        initial_dir = PROJECT_ROOT / "reports" / "preprocessing_gui"
        initial_dir.mkdir(parents=True, exist_ok=True)
        output_dir = filedialog.askdirectory(title="Choose output folder", initialdir=str(initial_dir))
        if not output_dir:
            return

        try:
            output_path = Path(output_dir)
            for page in pages_with_outputs:
                stem = page.path.stem if page.path else datetime.now().strftime("page_%Y%m%d_%H%M%S")
                if page.cropped_bgr is not None:
                    save_image(output_path / f"{stem}_01_cropped.png", page.cropped_bgr)
                if page.cleaned_gray is not None:
                    save_image(output_path / f"{stem}_02_cleaned.png", page.cleaned_gray)
                if page.binary_image is not None:
                    save_image(output_path / f"{stem}_03_binary.png", page.binary_image)

            self.status_var.set(f"Processed images saved to {output_path}")
            messagebox.showinfo("Saved", f"Processed images saved to:\n{output_path}")
        except Exception as exc:
            self.show_error("Save failed", exc)

    def get_preview_image(self) -> np.ndarray | None:
        page = self.selected_page()
        if page is None:
            return None

        mode = self.preview_mode.get()
        if mode == "binary":
            return as_bgr(page.binary_image) if page.binary_image is not None else None
        if mode == "cleaned":
            return as_bgr(page.cleaned_gray) if page.cleaned_gray is not None else None
        if mode == "cropped":
            return page.cropped_bgr
        return page.original_bgr

    def update_preview(self) -> None:
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")
        image_bgr = self.get_preview_image()
        if image_bgr is None:
            self.canvas.create_text(
                max(240, self.canvas.winfo_width() // 2),
                max(160, self.canvas.winfo_height() // 2),
                text="No preview for this stage yet",
                fill="#d1d5db",
                font=("Segoe UI", 16),
            )
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_height, image_width = image_bgr.shape[:2]
        scale = min(canvas_width / image_width, canvas_height / image_height)
        display_width = max(1, int(round(image_width * scale)))
        display_height = max(1, int(round(image_height * scale)))
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        self.canvas_image = bgr_to_tk_image(image_bgr, (display_width, display_height))
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.canvas_image)

    def update_image_label(self) -> None:
        page = self.selected_page()
        if page is None:
            self.image_var.set("No image loaded")
            return

        height, width = page.original_bgr.shape[:2]
        index = self.selected_index()
        prefix = f"{index + 1}/{len(self.pages)}  " if index is not None else ""
        self.image_var.set(f"{prefix}{page.path.name}  ({width} x {height})")

    def page_label(self, page: PageState) -> str:
        if page.binary_image is not None:
            stage = "binary"
        elif page.cleaned_gray is not None:
            stage = "cleaned"
        elif page.cropped_bgr is not None:
            stage = "cropped"
        else:
            stage = "original"
        return f"{page.path.name}  -  {stage}"

    def refresh_selected_page_label(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        self.page_list.delete(index)
        self.page_list.insert(index, self.page_label(self.pages[index]))
        self.page_list.selection_set(index)
        self.page_list.activate(index)
        self.update_image_label()

    def refresh_page_list(self) -> None:
        current = self.selected_index() or 0
        self.page_list.delete(0, tk.END)
        for page in self.pages:
            self.page_list.insert(tk.END, self.page_label(page))
        if self.pages:
            current = min(current, len(self.pages) - 1)
            self.page_list.selection_set(current)
            self.page_list.activate(current)
        self.update_image_label()

    def show_error(self, title: str, exc: Exception) -> None:
        traceback.print_exc()
        self.status_var.set(f"{title}: {exc}")
        messagebox.showerror(title, str(exc))


def main() -> None:
    root = tk.Tk()
    app = PreprocessingGUI(root)
    if len(sys.argv) > 1:
        image_paths = [Path(arg).expanduser() for arg in sys.argv[1:] if Path(arg).expanduser().exists()]
        if image_paths:
            app.pages = [PageState(path=path, original_bgr=load_bgr_image(path)) for path in image_paths]
            for page in app.pages:
                app.page_list.insert(tk.END, app.page_label(page))
            app.page_list.selection_set(0)
            app.update_image_label()
            app.update_preview()
    root.mainloop()


if __name__ == "__main__":
    main()
