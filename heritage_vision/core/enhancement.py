"""
heritage_vision.core.enhancement
================================

Part B requirement 4 - *image enhancement*: contrast improvement plus one
illumination-correction method.  Both are implemented in more than one
variant so the GUI can offer a justified choice and the report can show the
quantitative comparison.

Why these particular methods for faded documents / photographs?
    * Old scans suffer from *low local contrast* (ink faded into the paper)
      **and** *global illuminating gradients* (scanner lid not closed,
      torch-lit inscription, curled page).  A single global histogram equaliser
      fixes neither: it over-amplifies the shaded corner and destroys the
      already weak text.
    * CLAHE fixes local contrast while its clip limit caps the noise
      amplification.
    * Background flattening (divide-normalisation against a morphologically
      estimated background) removes the gradient *without* moving the ink
      density, which is exactly the property a conservator needs.
    * Single-scale Retinex is offered as the alternative when the gradient is
      severe (spotlight, torch on an inscription) because it is not sensitive
      to the shape of the illumination profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import odd, robust_percentile_stretch, to_gray

# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------


def clahe_gray(gray: np.ndarray, clip_limit: float = 2.0,
               tiles: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Global HE would put the whole page's histogram through one CDF, so a
    dark corner and a bright centre fight each other.  CLAHE equalises 8x8
    tiles and clips the CDF at ``clip_limit`` x the uniform height:
      * clip 1.0 -> behaves like plain HE (noisy, blown highlights)
      * clip 2.0 -> good text gain on paper (default)
      * >4.0    -> visible blotches on foxed paper
    Tiles are interpolated (bilinar) so no seams appear.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(gray)


def clahe_color(image: np.ndarray, clip_limit: float = 2.0, tiles: int = 8,
                space: str = "Lab") -> np.ndarray:
    """Apply CLAHE to the luminance-like channel only, so colours stay true.

    Lab-L is perceptually uniform (no green cast like HSV-V can introduce) but
    YCrCb-Y is what the scan pipeline actually recorded; the user can pick.
    """
    img = to_bgr(image)
    if space.upper().startswith("Y"):
        planar = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        planar[..., 0] = clahe_gray(planar[..., 0], clip_limit, tiles)
        return cv2.cvtColor(planar, cv2.COLOR_YCrCb2BGR)
    planar = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    planar[..., 0] = clahe_gray(planar[..., 0], clip_limit, tiles)
    return cv2.cvtColor(planar, cv2.COLOR_Lab2BGR)


def stretch_contrast(image: np.ndarray, percentile: float = 1.0) -> np.ndarray:
    """Robust global contrast stretch (percentile-based, not min/max)."""
    if image.ndim == 2:
        return robust_percentile_stretch(image, percentile)
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = robust_percentile_stretch(image[..., c], percentile)
    return out


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Brightness-weight transform.  gamma<1 lifts a faded, over-dark scan.

    Uses the LUT form (``np.power`` on a 256 entry table) so it is exact and
    O(1) per pixel.  Note the convention: we divide by gamma so a slider value
    of 1.3 visibly *brightens*, which matches the label in the GUI.
    """
    if abs(gamma - 1.0) < 1e-3:
        return image
    inv = 1.0 / float(np.clip(gamma, 0.1, 10.0))
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


# ---------------------------------------------------------------------------
# Illumination correction
# ---------------------------------------------------------------------------


def background_estimate_morph(gray: np.ndarray, ksize_px: int) -> np.ndarray:
    """Background = grey closing with a structuring element much larger than
    the darkest feature (text strokes, cracks).  Closing removes the dark ink
    and keeps the bright paper, yielding a smooth illumination model.

    Chosen over a plain big median because OpenCV only supports odd kernels
    up to ~31 px for median, while the background model needs 40-200 px.
    """
    k = max(int(ksize_px), 3)
    # MORPH_RECT (not ELLIPSE): separable, therefore ~70x faster for the
    # 100-300 px kernels a background model needs; see damage._scales.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # a second smoothing pass removes the kernel's footprint
    return cv2.GaussianBlur(closed, (0, 0), max(k / 6.0, 1.0))


def background_estimate_gauss(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Background = heavily blurred image (assumes ink averages out).

    Cheaper than morphology but bleeds where ink coverage is one-sided, so it
    is offered as 'fast' mode and used for the illumination *report* anyway.
    """
    k = odd(int(max(sigma, 0.5) * 6))
    return cv2.GaussianBlur(gray, (k, k), float(sigma))


def flatten_illumination(gray: np.ndarray, background: np.ndarray,
                         floor: float = 12.0) -> np.ndarray:
    """Divide-normalisation:  out = gray / background * mean(background).

    Multiplicative shading is cancelled exactly, ink density (which is
    *additive* in log space) is preserved, and no pixel is divided by a near
    zero value thanks to ``floor``.
    """
    bg = background.astype(np.float32)
    bg = np.maximum(bg, floor)
    target = float(np.percentile(bg, 90))          # keep page bright, not muddy
    out = gray.astype(np.float32) / bg * target
    return np.clip(out, 0, 255).astype(np.uint8)


def retinex(gray: np.ndarray, sigmas: Tuple[int, ...] = (15, 80, 250),
            gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Single-scale Retinex (SSR) cascade - log difference of image and blur.

    ``log(I) - log(blur(I))`` is invariant to any smooth multiplicative
    illumination, so it handles a torch-lit inscription that divide
    normalisation would fail on.  Output is rescaled to full range.
    """
    f = gray.astype(np.float32) + 1.0
    log = np.log(f)
    acc = np.zeros_like(f)
    for s in sigmas:
        k = odd(int(max(s, 1) * 6))
        blur = cv2.GaussianBlur(f, (k, k), float(s)) + 1.0
        acc += (log - np.log(blur))
    ssr = acc / float(len(sigmas))
    lo, hi = np.percentile(ssr, [0.5, 99.5])
    if hi - lo < 1e-3:
        return gray
    out = (ssr - lo) / (hi - lo) * 255.0
    out = out * float(gain) + float(offset)
    return np.clip(out, 0, 255).astype(np.uint8)


def correct_illumination(image: np.ndarray, method: str = "morph",
                         kernel_mm: float = 12.0, dpi: int = 300,
                         sigma: int = 21) -> Dict[str, np.ndarray]:
    """Run illumination correction on a colour or grey image.

    Returns a dict with the corrected image plus the background model, so the
    GUI can *show* the estimated illumination (useful diagnostic: an
    asymmetric model means a curl or a shadow, not a stain).
    """
    color = image.ndim == 3 and image.shape[2] == 3
    gray = to_gray(image)
    kpx = max(int(round(kernel_mm * dpi / 25.4)), 9)
    if method == "retinex":
        fixed = retinex(gray)
        bg = background_estimate_gauss(gray, sigma=sigma * 3)
        if color:
            fixed = transfer_gray_correction(image, gray, fixed)
    elif method == "morph":
        bg = background_estimate_morph(gray, kpx)
        fixed = flatten_illumination(gray, bg)
        if color:
            fixed = transfer_gray_correction(image, gray, fixed)
    elif method == "gaussian":
        bg = background_estimate_gauss(gray, sigma=sigma)
        fixed = flatten_illumination(gray, bg)
        if color:
            fixed = transfer_gray_correction(image, gray, fixed)
    else:
        fixed = gray if not color else image
        bg = np.zeros_like(gray)
    return {"image": fixed, "background": bg, "method": method,
            "kernel_px": kpx}


def transfer_gray_correction(color: np.ndarray, gray_in: np.ndarray,
                              gray_out: np.ndarray) -> np.ndarray:
    """Apply a grey-domain correction to a colour image as a multiplicative gain.

    Scaling each channel by ``out/in`` keeps the hue (yellow paper stays
    yellow) while fixing the brightness distribution - important because for
    heritage material the colour *is* data (a brown stain must stay visible).
    """
    ratio = gray_out.astype(np.float32) / np.maximum(gray_in.astype(np.float32), 1.0)
    ratio = np.clip(ratio, 0.0, 4.0)
    out = color.astype(np.float32) * ratio[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sharpening (used after inpainting, restoration stage)
# ---------------------------------------------------------------------------


def unsharp_mask(image: np.ndarray, amount: float = 0.6,
                 sigma: float = 2.0, threshold: int = 2) -> np.ndarray:
    """``out = img + amount * (img - gaussian(img))`` with a small threshold.

    The threshold is what makes this safe for heritage scans: without it the
    paper grain and JPEG ringing get amplified into false 'cracks'.
    """
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), float(sigma))
    lap = image.astype(np.float32) - blurred.astype(np.float32)
    if threshold > 0:
        lap[np.abs(lap) < threshold] = 0
    out = image.astype(np.float32) + float(amount) * lap
    return np.clip(out, 0, 255).astype(image.dtype)


# ---------------------------------------------------------------------------
# Comparison engine shared by the "compare" screens / report
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    name: str
    params: Dict[str, object]
    image: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


def measure_quality(gray: np.ndarray) -> Dict[str, float]:
    """Reference-free quality descriptors used to *justify* a parameter.

    contrast   : std of intensities (higher = better separation ink/paper)
    umichelson : Michelson contrast of the central band
    sharpness  : variance of Laplacian (classic blur measure)
    entropy    : 8-bit histogram entropy (information content)
    bg_uniformity : std of the low-frequency background (lower = flatter light)
    """
    g = to_gray(gray)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    # separable box filter, not a Gaussian: same "large low-pass" job at a
    # fraction of the cost (O(1) per pixel in OpenCV for ksize > 7)
    bg = cv2.boxFilter(g.astype(np.float32), cv2.CV_32F, (101, 101), normalize=True,
                       borderType=cv2.BORDER_REFLECT)
    return {
        "contrast_std": round(float(np.std(g)), 2),
        "umichelson": round(float(np.std(g)) / max(float(np.mean(g)), 1.0), 3),
        "sharpness_lap_var": round(float(lap.var()), 2),
        "entropy": round(float(_entropy(g)), 3),
        "bg_uniformity_std": round(float(np.std(bg)), 2),
        "mean_intensity": round(float(np.mean(g)), 2),
    }


def _entropy(g: np.ndarray) -> float:
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def build_variants(gray: np.ndarray) -> List[Callable[[np.ndarray], np.ndarray]]:
    """Factory for the 'try these three contrast settings' helper in the GUI."""
    return [
        lambda g: g,
        lambda g: clahe_gray(g, 1.5, 8),
        lambda g: clahe_gray(g, 3.0, 8),
    ]


def compare_variants(gray: np.ndarray, variants: Dict[str, Callable[[np.ndarray], np.ndarray]],
                     kpi: str = "umichelson") -> List[VariantResult]:
    """Evaluate named variants with reference-free metrics and rank them.

    ``recommended`` is simply the best KPI, which the report quotes as the
    justification for the default parameter set.
    """
    out: List[VariantResult] = []
    for name, fn in variants.items():
        img = fn(gray)
        out.append(VariantResult(name=name, params={}, image=img,
                                 metrics=measure_quality(img)))
    out.sort(key=lambda r: -r.metrics.get(kpi, 0.0))
    return out
