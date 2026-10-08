"""
heritage_vision.core.colorspace
===============================

Colour-space processing (Part B, requirement 3).  The application works in
*four* spaces and every one of them has a concrete job in the heritage
domain:

``GRAY``   - the ink/paper signal itself.  Binarisation, morphology,
             texture and frequency analysis are single-channel operations.
``HSV``    - ``S`` separates ochre/brown foxing and water stains (high
             saturation) from faded paper (low saturation); ``V`` separates
             dark defects from the bright background.  This is the default
             damage space because the two axes are almost orthogonal for
             aged cellulose.
``Lab``    - perceptually uniform.  ``b*`` is exactly the yellow-brown axis
             of lignin oxidation, so it is used for de-yellowing and for the
             "stain vs ink" second opinion.
``YCrCb``  - ``Y`` is the luminance the JPEG/scan pipeline really preserved,
             ``Cr``/``Cb`` give a warm/cool split that survives uneven
             lighting better than raw RGB.  Used for illumination-aware
             thresholding and for colour preservation while enhancing.
``BINARY`` - not an RGB space, but the requirement lists it: the page mask,
             text mask and damage mask are all binary images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import to_gray

SPACES = ("BGR", "GRAY", "HSV", "Lab", "YCrCb", "BINARY")


@dataclass
class SpaceDecomposition:
    """All views of one image, computed once and reused by every stage."""
    bgr: np.ndarray
    gray: np.ndarray
    hsv: Optional[np.ndarray] = None
    lab: Optional[np.ndarray] = None
    ycrcb: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None

    def channel(self, space: str, idx: int) -> np.ndarray:
        table: Dict[str, Optional[np.ndarray]] = {
            "BGR": self.bgr, "GRAY": self.gray, "HSV": self.hsv,
            "Lab": self.lab, "YCrCb": self.ycrcb, "BINARY": self.binary,
        }
        img = table.get(space)
        if img is None:
            raise KeyError(f"space '{space}' was not requested")
        if img.ndim == 2:
            return img
        return img[..., min(idx, img.shape[2] - 1)]


def decompose(image: np.ndarray, spaces: Tuple[str, ...] = ("HSV",),
              binary: Optional[np.ndarray] = None) -> SpaceDecomposition:
    bgr = image if (image.ndim == 3 and image.shape[2] == 3) else \
        cv2.cvtColor(_as3(image), cv2.COLOR_GRAY2BGR) if image.ndim == 2 else \
        cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray = to_gray(bgr)
    dec = SpaceDecomposition(bgr=bgr, gray=gray, binary=binary)
    if "HSV" in spaces:
        dec.hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if "Lab" in spaces:
        dec.lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    if "YCrCb" in spaces:
        dec.ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return dec


def _as3(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else (
        image[..., :3] if image.shape[2] == 4 else image)


# ---------------------------------------------------------------------------
# Space-specific features used later by the analysis stage
# ---------------------------------------------------------------------------


def saturation_map(image: np.ndarray) -> np.ndarray:
    """S channel; float32 0..1.  High = chromatic stain, low = neutral paper."""
    hsv = cv2.cvtColor(_three(image), cv2.COLOR_BGR2HSV)
    return hsv[..., 1].astype(np.float32) / 255.0


def yellowness_map(image: np.ndarray) -> np.ndarray:
    """Lab b* channel normalised to 0..1 - the oxidation (browning) axis."""
    lab = cv2.cvtColor(_three(image), cv2.COLOR_BGR2Lab)
    b = lab[..., 2].astype(np.float32)
    lo, hi = np.percentile(b, [5, 95])
    if hi - lo < 1:
        return np.zeros_like(b)
    return np.clip((b - lo) / (hi - lo), 0, 1)


def luminance_map(image: np.ndarray) -> np.ndarray:
    """Y of YCrCb - the channel a scan actually preserves without white balance."""
    ycc = cv2.cvtColor(_three(image), cv2.COLOR_BGR2YCrCb)
    return ycc[..., 0]


def redness_map(image: np.ndarray) -> np.ndarray:
    cr = cv2.cvtColor(_three(image), cv2.COLOR_BGR2YCrCb)[..., 1]
    return cr.astype(np.float32)


def estimate_paper_colour(image: np.ndarray, method: str = "mode") -> Tuple[int, int, int]:
    """Dominant paper (background) colour.

    Used to decide *how far* the sheet is from neutral cream and to build the
    de-yellowing correction.  A percentile/quantile estimator is used rather
    than the mean, because ink pixels and stains would bias the mean.
    """
    bgr = _three(image).astype(np.float32)
    if method == "mode":
        small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.uint8)
        flat = small.reshape(-1, 3)
        keys = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
        vals, counts = np.unique(keys, return_counts=True)
        k = int(vals[int(np.argmax(counts))])
        # the key packed B<<16 | G<<8 | R, so unpack in the same order: the
        # returned triple is BGR like every other colour value in this project
        return ((k >> 16) & 0xFF, (k >> 8) & 0xFF, k & 0xFF)
    # brightest 15% = clean paper (excluding ink and glare)
    gray = cv2.cvtColor(_three(image), cv2.COLOR_BGR2GRAY)
    p = np.percentile(gray, 85)
    sel = gray >= p
    ch = bgr[sel]
    if ch.size == 0:
        return (245, 240, 225)
    med = np.median(ch, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def deyellow(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Neutralise the brown cast of aged paper without touching ink density.

    Method: estimate the paper colour in Lab and shift the ``a*``/``b*``
    channels so the background lands on a slightly warm neutral (b* = 12),
    leaving ``L*`` (the information channel) untouched.  ``strength`` in
    0..1 lets the conservator dial the correction - full auto-white would
    erase the historical colour evidence, which is why it is not forced.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image
    bgr = _three(image).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper_mask = gray >= np.percentile(gray, 80)
    a_ref = float(np.median(lab[..., 1][paper_mask])) - 1.5
    b_ref = float(np.median(lab[..., 2][paper_mask])) - 12.0
    lab[..., 1] -= strength * (a_ref - 128.0) * 0.0 + strength * (a_ref - 126.5)
    lab[..., 2] -= strength * (b_ref - 138.0)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def gray_from(image: np.ndarray, space: str = "GRAY") -> np.ndarray:
    """Single-channel representation chosen by the user (GUI dropdown)."""
    if space == "GRAY" or image.ndim == 2:
        return to_gray(image)
    bgr = _three(image)
    if space == "HSV":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    if space == "Lab":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[..., 0]
    if space == "YCrCb":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    if space == "HSV_S":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    if space == "Lab_B":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[..., 2]
    return to_gray(image)


def _three(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
