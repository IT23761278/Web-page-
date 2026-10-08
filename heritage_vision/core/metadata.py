"""
heritage_vision.core.metadata
=============================

Image information required by Part B: "display dimensions, channels, bit
depth, file size, and selected region information".

The metadata object is intentionally independent of the GUI so that the CLI
batch tool can write the same information into ``metadata.csv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # PIL is optional: it exposes the true stored bit depth
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None


@dataclass
class ImageMetadata:
    path: str = ""
    filename: str = ""
    format: str = ""
    file_size_bytes: int = 0
    width: int = 0
    height: int = 0
    channels: int = 0
    dtype: str = "uint8"
    bits_per_sample: int = 8
    bits_per_pixel: int = 8
    is_grayscale_source: bool = True
    color_ratio: float = 0.0
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    dynamic_range: int = 0
    contrast_constant: float = 0.0      # Michelson contrast
    entropy_bits: float = 0.0
    ink_area_pct: float = 0.0
    resolution_dpi: Tuple[int, int] = (0, 0)
    physical_size_mm: Tuple[float, float] = (0.0, 0.0)
    aspect_ratio: float = 0.0
    megapixels: float = 0.0
    region: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- ctor
    @classmethod
    def from_file(cls, path: str, image: Optional[np.ndarray] = None,
                  target_dpi: int = 300) -> "ImageMetadata":
        if image is None:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot read image: {path}")
        m = cls(path=path, filename=os.path.basename(path))
        m.format = (os.path.splitext(path)[1].lstrip(".").upper() or "UNKNOWN")
        m.file_size_bytes = os.path.getsize(path)
        m.height, m.width = int(image.shape[0]), int(image.shape[1])
        m.channels = 1 if image.ndim == 2 else int(image.shape[2])
        m.dtype = str(image.dtype)
        m.bits_per_sample = np.iinfo(image.dtype).bits if np.issubdtype(
            image.dtype, np.integer) else 32
        m.bits_per_pixel = m.bits_per_sample * m.channels
        m.megapixels = round(m.width * m.height / 1e6, 3)
        m.aspect_ratio = round(m.width / float(m.height), 3) if m.height else 0.0

        gray = (image if image.ndim == 2
                else cv2.cvtColor(_three(image), cv2.COLOR_BGR2GRAY))
        g8 = _to8(gray)
        m.mean_intensity = round(float(np.mean(g8)), 2)
        m.std_intensity = round(float(np.std(g8)), 2)
        m.dynamic_range = int(np.max(g8) - np.min(g8))
        lo, hi = float(np.min(g8)), float(np.max(g8))
        m.contrast_constant = round((hi - lo) / max(hi + lo, 1.0), 3)
        hist = cv2.calcHist([g8], [0], None, [256], [0, 256]).ravel()
        p = hist / max(hist.sum(), 1)
        nz = p[p > 0]
        m.entropy_bits = round(float(-(nz * np.log2(nz)).sum()), 2)
        m.is_grayscale_source = _is_effectively_gray(image)
        if not m.is_grayscale_source:
            hsv = cv2.cvtColor(_three(image).astype(np.uint8), cv2.COLOR_BGR2HSV)
            m.color_ratio = round(float(np.mean(hsv[..., 1] > 32)), 4)
        thr = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        m.ink_area_pct = round(float(np.count_nonzero(thr)) / thr.size * 100.0, 2)
        m.resolution_dpi = _read_dpi(path, target_dpi)
        if m.resolution_dpi[0] > 0:
            m.physical_size_mm = (round(m.width * 25.4 / m.resolution_dpi[0], 1),
                                  round(m.height * 25.4 / m.resolution_dpi[1], 1))
        m.warnings += _quality_warnings(m, g8)
        return m

    # --------------------------------------------------------------- helpers
    def set_region(self, mask: np.ndarray, name: str = "selection",
                   target_dpi: int = 300) -> Dict[str, Any]:
        """Selected-region information: bbox, area, % of image, ink ratio, mm."""
        x, y, w, h = cv2.boundingRect(mask)
        area = int(np.count_nonzero(mask))
        total = float(mask.shape[0] * mask.shape[1])
        ink = 0
        self.region = {
            "name": name,
            "bbox_px": [int(x), int(y), int(w), int(h)],
            "area_px": area,
            "area_pct_of_image": round(area / max(total, 1.0) * 100.0, 3),
            "area_mm2": round(area * (25.4 / max(target_dpi, 1)) ** 2, 1),
            "bbox_aspect": round(w / float(h), 3) if h else 0.0,
        }
        return self.region

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolution_dpi"] = "x".join(str(v) for v in self.resolution_dpi)
        d["physical_size_mm"] = "x".join(str(v) for v in self.physical_size_mm)
        d["warnings"] = "; ".join(self.warnings)
        return d

    def pretty(self) -> str:
        m = self
        lines = [
            f"File            : {m.filename}  [{m.format}, {m.file_size_bytes/1024:.0f} KB]",
            f"Dimensions      : {m.width} x {m.height} px  ({m.megapixels} MP, "
            f"AR {m.aspect_ratio})",
            f"Channels / dtype: {m.channels} ch, {m.dtype}, "
            f"{'grey source' if m.is_grayscale_source else 'colour source'}",
            f"Bit depth       : {m.bits_per_sample} bit/sample, "
            f"{m.bits_per_pixel} bit/pixel",
            f"Resolution      : {m.resolution_dpi[0]}x{m.resolution_dpi[1]} dpi -> "
            f"{m.physical_size_mm[0]} x {m.physical_size_mm[1]} mm",
            f"Intensity       : mean {m.mean_intensity}, std {m.std_intensity}, "
            f"range {m.dynamic_range}",
            f"Contrast (C)    : {m.contrast_constant}   Entropy: {m.entropy_bits} bit",
            f"Ink coverage    : {m.ink_area_pct} % (Otsu foreground)",
            f"Colour ratio    : {round(m.color_ratio*100,1)} % of pixels saturated",
        ]
        if m.region:
            r = m.region
            lines.append(f"Region ({r['name']}) : bbox {r['bbox_px']}, "
                         f"{r['area_px']} px = {r['area_pct_of_image']} % "
                         f"({r['area_mm2']} mm2)")
        if m.warnings:
            lines.append("Warnings        : " + "; ".join(m.warnings))
        return "\n".join(lines)


# ----------------------------------------------------------------- internals
def _three(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _to8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    finite = np.isfinite(img) if img.dtype.kind == "f" else np.ones_like(img, bool)
    a = img[finite] if np.any(finite) else np.array([0])
    hi = float(max(np.max(a), 1e-6))
    return np.clip(img / hi * 255.0, 0, 255).astype(np.uint8)


def _is_effectively_gray(img: np.ndarray) -> bool:
    if img.ndim == 2:
        return True
    b, g, r = cv2.split(img[..., :3].astype(np.int16))
    maxdiff = np.maximum(np.maximum(np.abs(b - g), np.abs(g - r)), np.abs(b - r))
    return float(np.mean(maxdiff > 6)) < 0.01


def _read_dpi(path: str, fallback: int) -> Tuple[int, int]:
    if _PILImage is not None:
        try:
            with _PILImage.open(path) as im:
                info = dict(getattr(im, "info", {}))
                dpi = info.get("dpi")
                if dpi and float(dpi[0]) > 0:
                    return (int(round(float(dpi[0]))), int(round(float(dpi[1] or dpi[0]))))
                res = info.get("resolution")
                if res and float(res) > 0:
                    unit = str(info.get("resolution_unit", ""))
                    if unit in ("2", "cm", "Centimeter"):
                        return (int(round(float(res) * 2.54)),) * 2
                    return (int(round(float(res))),) * 2
        except Exception:
            pass
    return (fallback, fallback)


def _quality_warnings(m: ImageMetadata, g8: np.ndarray) -> List[str]:
    """Automatic acquisition-quality flags (used by the conservator checklist)."""
    w: List[str] = []
    if m.mean_intensity < 70:
        w.append("under-exposed scan (mean<70)")
    if m.mean_intensity > 225:
        w.append("over-exposed scan (mean>225)")
    if m.dynamic_range < 60:
        w.append("low dynamic range - faded or flat illumination")
    if m.std_intensity < 25:
        w.append("low contrast (std<25)")
    if m.megapixels < 0.35:
        w.append("resolution below 0.35 MP - text may be unreadable")
    if np.mean(g8 >= 252) > 0.30:
        w.append(">30% blown highlights (scanner glare)")
    if np.mean(g8 <= 3) > 0.25:
        w.append(">25% crushed shadows")
    if not m.is_grayscale_source and m.color_ratio > 0.6:
        w.append("highly saturated source - check white balance / paper stain")
    return w
