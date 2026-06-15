"""
FaceMorph — portable desktop app
================================

Morph a face along the female <-> male axis with a StarGAN, live, via a slider.
Runs natively on Windows (x64 & ARM64), macOS (Intel & Apple Silicon) and Linux
using ONNX Runtime — no TensorFlow, no WSL.

Run:  pip install -r requirements.txt   then   python FaceMorph.py
The model files (generator.onnx, discriminator.onnx) and assets/ must sit next
to this script.
"""
import os
import queue
import random
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

if getattr(sys, "frozen", False):          # bundled into a one-click .exe by PyInstaller
    APP_DIR = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from morph_onnx import OnnxMorpher, center_crop

GENERATOR = os.path.join(APP_DIR, "generator.onnx")
DISCRIMINATOR = os.path.join(APP_DIR, "discriminator.onnx")
SAMPLES_DIR = os.path.join(APP_DIR, "assets", "samples")

PREVIEW_SIZE = 320
SLIDER_DEBOUNCE_MS = 60
ACCENT = "#7c5cff"
ACCENT_HOVER = "#6847f0"
CARD_COLOR = ("#f2f3f7", "#23252e")
SIDEBAR_COLOR = ("#e8eaf1", "#1b1d24")
PLACEHOLDER_COLOR = (34, 36, 44)
ABOUT_TEXT = (
    "A StarGAN trained on 202,599 CelebA faces. The target gender is a continuous "
    "input, so dragging the slider morphs the face smoothly while cycle and identity "
    "losses keep the person recognizable. Portable build (ONNX Runtime)."
)


class FaceMorphApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceMorph")
        self.geometry("1020x640")
        self.minsize(920, 600)
        ctk.set_appearance_mode("dark")

        self.morpher = None
        self._aligned = None
        self._morphed = None
        self._debounce_id = None
        self._work_queue: "queue.Queue[float]" = queue.Queue(maxsize=1)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_content()
        self._set_controls_enabled(False)
        self._load_model_async()
        threading.Thread(target=self._morph_worker, daemon=True).start()

    # ---------------- layout ----------------
    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color=SIDEBAR_COLOR)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(8, weight=1)
        ctk.CTkLabel(sb, text="FaceMorph", font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, padx=24, pady=(28, 0), sticky="w")
        ctk.CTkLabel(sb, text="CelebA Gender GAN · ONNX", font=ctk.CTkFont(size=13),
                     text_color="gray60").grid(row=1, column=0, padx=24, pady=(0, 24), sticky="w")

        self.load_button = ctk.CTkButton(sb, text="Load photo", height=42, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, font=ctk.CTkFont(size=14, weight="bold"),
            command=self._choose_image)
        self.load_button.grid(row=2, column=0, padx=24, pady=(0, 10), sticky="ew")
        self.random_button = ctk.CTkButton(sb, text="Surprise me", height=42, corner_radius=10,
            fg_color="transparent", border_width=2, border_color=ACCENT,
            hover_color=("gray85", "#2a2438"), text_color=("gray10", "gray90"),
            font=ctk.CTkFont(size=14, weight="bold"), command=self._load_random)
        self.random_button.grid(row=3, column=0, padx=24, pady=(0, 24), sticky="ew")

        for r, (txt, cmd, attr) in enumerate([
            ("Save result", self._save_result, "save_button"),
            ("Export strip", self._export_strip, "strip_button"),
            ("Export GIF", self._export_gif, "gif_button"),
        ], start=4):
            b = ctk.CTkButton(sb, text=txt, height=36, corner_radius=10,
                fg_color=("gray80", "#2c2e38"), hover_color=("gray70", "#363946"),
                text_color=("gray10", "gray90"), command=cmd)
            b.grid(row=r, column=0, padx=24, pady=(0, 10), sticky="ew")
            setattr(self, attr, b)

        ctk.CTkLabel(sb, text=ABOUT_TEXT, font=ctk.CTkFont(size=12), text_color="gray55",
                     wraplength=212, justify="left").grid(row=7, column=0, padx=24, pady=(14, 0), sticky="w")
        status = ctk.CTkFrame(sb, fg_color="transparent")
        status.grid(row=9, column=0, padx=24, pady=(0, 22), sticky="sew")
        status.grid_columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Loading model…")
        ctk.CTkLabel(status, textvariable=self.status_var, font=ctk.CTkFont(size=12),
                     text_color="gray60", wraplength=212, justify="left", anchor="w").grid(
            row=0, column=0, sticky="ew")
        self.progress = ctk.CTkProgressBar(status, mode="indeterminate", height=6, progress_color=ACCENT)
        self.progress.grid(row=1, column=0, pady=(8, 0), sticky="ew")
        self.progress.start()

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        content.grid_columnconfigure((0, 1), weight=1, uniform="cards")
        content.grid_rowconfigure(0, weight=1)
        ph = Image.new("RGB", (PREVIEW_SIZE, PREVIEW_SIZE), PLACEHOLDER_COLOR)
        self.original_image_label = self._build_card(content, "ORIGINAL  ·  ALIGNED", 0, ph)
        self.morphed_image_label = self._build_card(content, "MORPHED", 1, ph)

        card = ctk.CTkFrame(content, corner_radius=16, fg_color=CARD_COLOR)
        card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        card.grid_columnconfigure(1, weight=1)
        self.value_label = ctk.CTkLabel(card, text="drag the slider",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=ACCENT)
        self.value_label.grid(row=0, column=0, columnspan=3, pady=(14, 2))
        ctk.CTkLabel(card, text="Female", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=1, column=0, padx=(24, 12), pady=(0, 18))
        self.slider = ctk.CTkSlider(card, from_=0.0, to=1.0, number_of_steps=200, height=18,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER, progress_color=ACCENT,
            command=self._on_slider_moved)
        self.slider.set(0.5)
        self.slider.grid(row=1, column=1, sticky="ew", pady=(0, 18))
        ctk.CTkLabel(card, text="Male", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=1, column=2, padx=(12, 24), pady=(0, 18))

    def _build_card(self, parent, title, column, placeholder):
        card = ctk.CTkFrame(parent, corner_radius=16, fg_color=CARD_COLOR)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 20) if column == 0 else 0)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="gray55").grid(row=0, column=0, pady=(16, 6))
        image = ctk.CTkImage(placeholder, size=(PREVIEW_SIZE, PREVIEW_SIZE))
        label = ctk.CTkLabel(card, image=image, text="")
        label.grid(row=1, column=0, padx=18, pady=(0, 18))
        label._placeholder = image
        return label

    # ---------------- model ----------------
    def _load_model_async(self) -> None:
        def loader():
            try:
                if not os.path.exists(GENERATOR):
                    raise FileNotFoundError("generator.onnx not found next to FaceMorph.py")
                morpher = OnnxMorpher(GENERATOR, DISCRIMINATOR)
                self.after(0, lambda: self._on_model_loaded(morpher))
            except Exception as exc:
                self.after(0, lambda exc=exc: self._on_model_failed(exc))
        threading.Thread(target=loader, daemon=True).start()

    def _on_model_loaded(self, morpher):
        self.morpher = morpher
        self.progress.stop()
        self.progress.grid_remove()
        self._set_controls_enabled(True)
        self.status_var.set(f"Model ready · {morpher.image_size}×{morpher.image_size} · ONNX (CPU)")

    def _on_model_failed(self, exc):
        self.progress.stop()
        self.status_var.set("Model failed to load.")
        messagebox.showerror("Startup Error", f"Could not load the model: {exc}")

    def _set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for w in (self.load_button, self.random_button, self.save_button,
                  self.strip_button, self.gif_button, self.slider):
            w.configure(state=state)

    # ---------------- photo loading ----------------
    def _choose_image(self):
        path = filedialog.askopenfilename(title="Choose a photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self._load_image(path, already_aligned=False)

    def _load_random(self):
        try:
            files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".png")]
        except FileNotFoundError:
            files = []
        if not files:
            messagebox.showwarning("No samples", "No bundled samples in assets/samples.")
            return
        self._load_image(os.path.join(SAMPLES_DIR, random.choice(files)), already_aligned=True)

    def _load_image(self, path, *, already_aligned):
        self.status_var.set("Preparing the face…")

        def task():
            try:
                image = Image.open(path)
                size = self.morpher.image_size
                if already_aligned:
                    aligned = image.convert("RGB")
                    if aligned.size != (size, size):
                        aligned = aligned.resize((size, size), Image.BICUBIC)
                    note = "Sample loaded"
                else:
                    aligned = center_crop(image, size)
                    note = "Photo loaded (center crop)"
                estimated = self.morpher.estimate_maleness(aligned)

                def apply():
                    self._aligned = aligned
                    self._show_image(self.original_image_label, aligned)
                    if estimated is not None:
                        self.slider.set(round(estimated, 2))
                    self.status_var.set(note + " · drag the slider")
                    self._request_morph()
                self.after(0, apply)
            except Exception as exc:
                self.after(0, lambda exc=exc: messagebox.showerror("Error", f"An error occurred: {exc}"))
        threading.Thread(target=task, daemon=True).start()

    # ---------------- morphing ----------------
    def _on_slider_moved(self, value):
        self.value_label.configure(text=f"{value * 100:.0f}% male")
        if self._aligned is None:
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(SLIDER_DEBOUNCE_MS, self._request_morph)

    def _request_morph(self):
        self._debounce_id = None
        if self._aligned is None or self.morpher is None:
            return
        value = float(self.slider.get())
        try:
            self._work_queue.put_nowait(value)
        except queue.Full:
            try:
                self._work_queue.get_nowait()
            except queue.Empty:
                pass
            self._work_queue.put_nowait(value)

    def _morph_worker(self):
        while True:
            value = self._work_queue.get()
            aligned, morpher = self._aligned, self.morpher
            if aligned is None or morpher is None:
                continue
            try:
                result = morpher.morph(aligned, value)
                self.after(0, lambda img=result: self._show_morph(img))
            except Exception as exc:
                self.after(0, lambda exc=exc: self.status_var.set(f"Morph failed: {exc}"))

    def _show_morph(self, image):
        self._morphed = image
        self._show_image(self.morphed_image_label, image)

    def _show_image(self, label, image):
        photo = ctk.CTkImage(image, size=(PREVIEW_SIZE, PREVIEW_SIZE))
        label.configure(image=photo)
        label._current = photo

    # ---------------- export ----------------
    def _save_result(self):
        if self._morphed is None:
            messagebox.showwarning("Nothing to save", "Load a photo and morph it first.")
            return
        path = filedialog.asksaveasfilename(title="Save morphed image",
            defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if path:
            self._morphed.save(path)
            self.status_var.set(f"Saved {os.path.basename(path)}")

    def _export_strip(self):
        if self._aligned is None:
            messagebox.showwarning("Nothing to export", "Load a photo first.")
            return
        path = filedialog.asksaveasfilename(title="Save interpolation strip",
            defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not path:
            return

        def task():
            try:
                strip = self.morpher.morph_strip(self._aligned, [i / 6 for i in range(7)])
                strip.save(path)
                self.after(0, lambda: self.status_var.set(f"Strip saved · {os.path.basename(path)}"))
            except Exception as exc:
                self.after(0, lambda exc=exc: messagebox.showerror("Error", str(exc)))
        threading.Thread(target=task, daemon=True).start()

    def _export_gif(self):
        """Smooth Female<->Male<->Female looping GIF (PIL only, cross-platform)."""
        if self._aligned is None:
            messagebox.showwarning("Nothing to export", "Load a photo first.")
            return
        path = filedialog.asksaveasfilename(title="Save morph GIF",
            defaultextension=".gif", filetypes=[("GIF animation", "*.gif")])
        if not path:
            return
        self.status_var.set("Rendering GIF…")

        def task():
            try:
                self.morpher.save_morph_gif(self._aligned, path)
                self.after(0, lambda: self.status_var.set(f"GIF saved · {os.path.basename(path)}"))
            except Exception as exc:
                self.after(0, lambda exc=exc: messagebox.showerror("Error", str(exc)))
        threading.Thread(target=task, daemon=True).start()


def main():
    import traceback
    from multiprocessing import freeze_support
    freeze_support()
    try:
        FaceMorphApp().mainloop()
    except Exception as exc:
        traceback.print_exc()
        try:
            messagebox.showerror("Startup Error", f"Error starting the app: {exc}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
