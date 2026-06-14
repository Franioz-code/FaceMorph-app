"""Portable ONNX-Runtime inference for FaceMorph — runs on Windows (x64 & ARM64),
macOS (Intel & Apple Silicon) and Linux, with no TensorFlow.

Mirrors the training preprocessing: images are normalised to [-1, 1], the target
gender enters as a constant 4th-channel "label plane" (2*maleness - 1), and the
generator output in [-1, 1] is mapped back to a uint8 image.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

DEFAULT_STRIP_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def center_crop(image: "Image.Image", size: int) -> "Image.Image":
    """Square center-crop (slight upward bias for faces) + resize. Portable
    fallback alignment that works on every platform (no MediaPipe needed)."""
    img = image.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = max(0, min(int((h - s) * 0.4), h - s))
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)


class OnnxMorpher:
    """Load the ONNX generator (and optional discriminator) and morph faces."""

    def __init__(self, generator_path: str, discriminator_path: Optional[str] = None) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.gen = ort.InferenceSession(
            generator_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        ins = self.gen.get_inputs()
        # Identify inputs by channel count: 3 = image, 1 = label plane.
        self._img_name = next(i.name for i in ins if i.shape[-1] == 3)
        self._plane_name = next(i.name for i in ins if i.shape[-1] == 1)
        self._gen_out = self.gen.get_outputs()[0].name
        img_shape = next(i.shape for i in ins if i.shape[-1] == 3)
        self.image_size = int(img_shape[1]) if isinstance(img_shape[1], int) else 128

        self.disc = None
        if discriminator_path and os.path.exists(discriminator_path):
            self.disc = ort.InferenceSession(
                discriminator_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._disc_in = self.disc.get_inputs()[0].name
            # The gender logit is the rank-2 output (N, 1); the patch map is rank-4.
            self._disc_cls = next(o.name for o in self.disc.get_outputs() if len(o.shape) == 2)

    # ------------------------------------------------------------------
    def _batch(self, image: Image.Image) -> np.ndarray:
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return arr[np.newaxis]

    def _plane(self, maleness: float) -> np.ndarray:
        v = float(maleness) * 2.0 - 1.0
        return np.full((1, self.image_size, self.image_size, 1), v, dtype=np.float32)

    # ------------------------------------------------------------------
    def morph(self, image: Image.Image, maleness: float) -> Image.Image:
        """Translate an already-aligned crop to the target gender (0=female, 1=male)."""
        out = self.gen.run(
            [self._gen_out],
            {self._img_name: self._batch(image), self._plane_name: self._plane(maleness)},
        )[0]
        arr = np.clip((out[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def morph_strip(
        self, image: Image.Image, values: Sequence[float] = DEFAULT_STRIP_VALUES
    ) -> Image.Image:
        crop = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        panels = [np.asarray(crop, dtype=np.uint8)]
        panels += [np.asarray(self.morph(crop, v)) for v in values]
        return Image.fromarray(np.concatenate(panels, axis=1))

    def save_morph_gif(self, image: Image.Image, path: str, *, size: int = 256,
                       n_steps: int = 20, step_ms: int = 130, hold_ms: int = 1200) -> str:
        """Render a slow, looping Female<->Male<->Female GIF that LINGERS on the
        two extremes so the end results are clearly visible (fixes 'too fast')."""
        crop = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        sweep = [i / n_steps for i in range(n_steps + 1)]    # 0 -> 1
        values = sweep + sweep[::-1][1:-1]                    # 0 -> 1 -> back; loops to 0
        frames, durations = [], []
        for v in values:
            frames.append(self.morph(crop, v).resize((size, size), Image.LANCZOS))
            durations.append(hold_ms if (v <= 1e-3 or v >= 1 - 1e-3) else step_ms)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=2)
        return path

    def estimate_maleness(self, image: Image.Image) -> Optional[float]:
        if self.disc is None:
            return None
        cls = self.disc.run([self._disc_cls], {self._disc_in: self._batch(image)})[0]
        return float(1.0 / (1.0 + np.exp(-np.asarray(cls)[0, 0])))
