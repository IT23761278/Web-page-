"""
heritage_vision.core.utils
=========================

Input validation, safe image I/O, colour helpers and small numeric utilities
shared by every other module.  Everything here is pure OpenCV/NumPy so the
core can be imported on a machine without a display (needed by the tests and
by the batch/evaluation tools).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import MAX_IMAGE_PIXELS, SUPPORTED_EXTENSIONS

# ---------------------------------------------------------------------------
# 1. Validation + loading
# ---------------------------------------------------------------------------


class ImageLoadError(Exception):
    """Raised for a file that cannot be used.  Carries a user-facing reason."""


def check_extension(path: str) -> Tuple[bool, str]:
    """Rule-based check performed *before* decoding.  Returns (ok, reason)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        return False, "file has no extension - unsupported format"
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"'{ext}' is not a supported raster format"
    if ext in (".jpe", ".jpg_large"):                     # defensive
        return False, f"'{ext}' is not a supported raster format"
    return True, ""


def load_image(path: str, flags: int = cv2.IMREAD_COLOR,
               max_pixels: int = MAX_IMAGE_PIXELS,
               downscale_to: Optional[int] = None) -> np.ndarray:
    """Validate then decode an image file.

    Guarantees required by Part B ("reject unsupported or corrupted files
    safely"):
      1. path must exist and be a regular file
      2. extension must be a supported raster format
      3. file must be non-empty
      4. the first bytes must match the magic number of the claimed format
      5. OpenCV must actually return a non-empty array
      6. decoded size must be below *max_pixels*

    Every failure raises :class:`ImageLoadError` with a message that says what
    to do about it; nothing is ever returned as ``None``.
    """
    if not os.path.exists(path):
        raise ImageLoadError(f"file not found: {path}")
    if not os.path.isfile(path):
        raise ImageLoadError(f"not a regular file (folder/symlink?): {path}")

    ok, reason = check_extension(path)
    if not ok:
        raise ImageLoadError(f"{os.path.basename(path)}: {reason}")

    size = os.path.getsize(path)
    if size < 8:
        raise ImageLoadError(f"{os.path.basename(path)}: file is empty/too small "
                             f"({size} bytes) - probably truncated")
    if not _magic_ok(path):
        raise ImageLoadError(f"{os.path.basename(path)}: content does not match the "
                             "file extension (corrupted header or renamed file)")

    img = cv2.imread(path, flags)
    if img is None:
        raise ImageLoadError(f"{os.path.basename(path)}: decoder failed - the file is "
                             "corrupted, is a 0-byte placeholder, or uses an "
                             "unsupported sub-format")
    if img.size == 0 or min(img.shape[:2]) < 4:
        raise ImageLoadError(f"{os.path.basename(path)}: decoded frame is degenerate "
                             f"({img.shape})")
    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        raise ImageLoadError(f"{os.path.basename(path)}: unexpected channel count "
                             f"{img.shape[2]}")

    n_px = img.shape[0] * img.shape[1]
    if n_px > max_pixels:
        scale = (max_pixels / n_px) ** 0.5
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if downscale_to:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > downscale_to:
            s = downscale_to / float(longest)
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
    if img.ndim == 3 and img.shape[2] == 4:
        # Alpha is not usable downstream; flatten onto white paper colour.
        b, g, r = cv2.split(img[..., :3])
        a = img[..., 3:4].astype(np.float32) / 255.0
        bg = 255.0
        img = cv2.merge([
            (b.astype(np.float32) * a + bg * (1 - a)),
            (g.astype(np.float32) * a + bg * (1 - a)),
            (r.astype(np.float32) * a + bg * (1 - a)),
        ]).astype(np.uint8)
    return img


_MAGIC = {
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    ".bmp": [b"BM"],
    ".webp": [b"RIFF"],
}


def _magic_ok(path: str) -> bool:
    """Read the file signature so a renamed/corrupted file is caught early."""
    ext = os.path.splitext(path)[1].lower()
    sigs = _MAGIC.get(ext)
    if not sigs:
        return True
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return False
    return any(head.startswith(s) for s in sigs)


def list_image_files(folder: str, recursive: bool = True) -> List[str]:
    """All supported image files in *folder*, sorted by name (stable batch order)."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"not a folder: {folder}")
    out: List[str] = []
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _dirs, files in walker:
        for f in sorted(files):
            p = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS and os.path.isfile(p):
                out.append(p)
    return sorted(out)


def save_image(path: str, image: np.ndarray, quality: int = 92) -> str:
    """Write an image, creating parent folders and reporting failures."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    params: Sequence[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    ok = cv2.imwrite(path, image, list(params))
    if not ok:
        raise IOError(f"could not write {path} (unsupported extension or "
                      "non-ASCII path on Windows)")
    return path


# ---------------------------------------------------------------------------
# 2. Colour helpers
# ---------------------------------------------------------------------------


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Scale 16-bit / float data to 8-bit so all downstream code is uniform."""
    if img.dtype == np.uint8:
        return img
    finite = np.isfinite(img) if img.dtype.kind == "f" else None
    data = img[finite] if finite is not None else img
    if data.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = float(np.min(data)), float(np.max(data))
    maxv = 65535.0 if (img.dtype in (np.uint16, np.int16) and hi > 255) else 255.0
    hi = max(hi, lo + 1e-6)
    scale = maxv / hi
    out = np.clip(img * scale, 0, maxv)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Numeric helpers
# ---------------------------------------------------------------------------


def robust_percentile_stretch(gray: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Contrast stretch using percentiles instead of min/max.

    Real scans contain a few saturated dust/hot pixels; percentile clipping
    ignores them, which plain ``normalize()`` would not.
    """
    if pct <= 0:
        return gray
    lo, hi = np.percentile(gray, [pct, 100.0 - pct])
    if hi - lo < 6:            # essentially flat page -> nothing to stretch
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def odd(n: int, minimum: int = 3) -> int:
    """Force a kernel size to a positive odd number (OpenCV requirement)."""
    n = int(n)
    if n < minimum:
        n = minimum
    return n if n % 2 == 1 else n + 1


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def json_dump(data: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


class Timer:
    """Context manager collecting per-stage milliseconds (Part D evaluation)."""

    def __init__(self) -> None:
        self.records: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def stage(self, name: str):
        outer = self

        class _S:
            def __enter__(self_inner):
                outer._stack.append((name, time.perf_counter()))
                return self_inner

            def __exit__(self_inner, *exc):
                n, t0 = outer._stack.pop()
                outer.records[n] = outer.records.get(n, 0.0) + (
                    time.perf_counter() - t0) * 1000.0
                return False
        return _S()

    @property
    def total_ms(self) -> float:
        return float(sum(self.records.values()))


def mm_to_px(mm: float, dpi: int) -> float:
    return mm * dpi / 25.4


def px_to_mm(px: float, dpi: int) -> float:
    return px * 25.4 / float(dpi)


