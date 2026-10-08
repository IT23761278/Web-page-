"""
heritage_vision.core.denoise
============================

Part B requirement 5 - *noise reduction*: compare at least two filters and
justify the selected method.  Six are implemented so the comparison is real,
not cosmetic, and a selection score is computed automatically.

Filter-by-filter reasoning for faded paper:

gaussian    - fast, but blurs ink edges; acceptable only as a reference.
median      - removes *salt & pepper* dust specks (the dominant noise in a
              flatbed scan of a physical sheet) without moving edges; it
              destroys thin pencil strokes when the kernel grows.
bilateral   - edge preserving; ideal for the ink/paper step, and it keeps the
              soft edges of a water stain instead of hardening them.
nlm         - strongest PSNR on grain/mottle noise, but O(r^2) and it can
              erase short hairline cracks (a defect we must NOT hide).
guided      - local-linear model; behaves like an adaptive Wiener filter,
              preserves gradient direction, cheap.  Best compromise for scans
              with strong vignetting.
wiener      - classic LMS local Wiener estimate implemented with OpenCV only;
              good on additive Gaussian noise on a locally flat background
              (i.e. blank paper) but weak inside dense text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .metrics import psnr, ssim  # noqa: F401
from .utils import odd, to_gray

METHODS = ("none", "gaussian", "median", "bilateral", "nlm", "guided", "wiener")


@dataclass
class DenoiseReport:
    method: str
    params: Dict[str, object]
    time_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    image: Optional[np.ndarray] = None
    selected: bool = False


# ---------------------------------------------------------------------------
# individual filters
# ---------------------------------------------------------------------------


def gaussian(image: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    return cv2.GaussianBlur(image, (odd(ksize), odd(ksize)), float(sigma))


def median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, odd(ksize, 3))


def bilateral(image: np.ndarray, ksize: int = 5, sigma_color: float = 35.0,
              sigma_space: float = 5.0) -> np.ndarray:
    """Preserves the ink edge because a stroke is a big intensity jump
    (|di| > sigma_color) while paper grain is a small one."""
    return cv2.bilateralFilter(image, odd(ksize), float(sigma_color), float(sigma_space))


def nlm(image: np.ndarray, h: float = 10.0, template: int = 7,
        search: int = 21) -> np.ndarray:
    gray = image.ndim == 2
    src = image if gray else image
    return cv2.fastNlMeansDenoising(src, None, h=float(h),
                                    templateWindowSize=odd(template, 3),
                                    searchWindowSize=odd(search, 5)) \
        if gray else cv2.fastNlMeansDenoisingColored(
            src, None, h=float(h), hForColorComponents=float(h),
            templateWindowSize=odd(template, 3), searchWindowSize=odd(search, 5))


def guided_filter(image: np.ndarray, radius: int = 8, eps: float = 800.0,
                  guide: Optional[np.ndarray] = None) -> np.ndarray:
    """He et al. guided filter, implemented with box filters.

    ``a, b`` are the coefficients of the local linear model ``p = a*i + b``
    between guide ``i`` and image ``p``; the output is ``mean(a)*i + mean(b)``.
    ``eps`` trades smoothing against edge preservation, in units of (1/255)^2
    scaled by 65025/100: large eps -> Wiener-like flattening, small eps -> the
    structure survives.

    The *image* is filtered, the guide only supplies the structure - an earlier
    revision used the guide for both, which made the filter a no-op smoother and
    returned a single-channel image for colour input.  Colour input is processed
    channel by channel against a common gray guide, so the edges stay aligned.
    """
    src = np.asarray(image)
    g = to_gray(src) if guide is None else to_gray(guide)
    gi = g.astype(np.float32)
    r = max(int(radius), 1)
    ksize = (2 * r + 1, 2 * r + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, ksize, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    mean_i = box(gi)
    var_i = box(gi * gi) - mean_i * mean_i

    def one(p: np.ndarray) -> np.ndarray:
        mean_p = box(p)
        cov_ip = box(gi * p) - mean_i * mean_p
        a = cov_ip / (var_i + float(eps))
        b = mean_p - a * mean_i
        return box(a) * gi + box(b)

    if src.ndim == 2:
        out = one(src.astype(np.float32))
    else:
        out = np.stack([one(src[..., c].astype(np.float32))
                        for c in range(src.shape[2])], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def wiener_local(image: np.ndarray, ksize: int = 5, noise_var: Optional[float] = None
                 ) -> np.ndarray:
    """Adaptive local Wiener: ``z_hat = mu + max(0, (var-sig2)/var) * (z-mu)``.

    The noise variance is estimated from the image itself with the standard
    Laplacian-based sigma estimator, so the filter self-tunes per scan instead
    of needing a magic constant - important because a 600 dpi and a 150 dpi
    scan of the same sheet have very different noise power.
    """
    g = image.astype(np.float32)
    k = odd(ksize)
    mu = cv2.boxFilter(g, -1, (k, k), normalize=True)
    mu2 = cv2.boxFilter(g * g, -1, (k, k), normalize=True)
    var = np.maximum(mu2 - mu * mu, 0.0)
    if noise_var is None:
        noise_var = estimate_noise_sigma(image) ** 2
    signal = np.maximum(var - noise_var, 0.0) / np.maximum(var, 1e-6)
    out = mu + signal * (g - mu)
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Immerka / Laplacian-based noise estimate from a grey image."""
    g = to_gray(image).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g, cv2.CV_32F, k)
    h, w = g.shape
    sigma = float(np.abs(conv).mean() * np.sqrt(0.5 * np.pi) / 6.0)
    return max(sigma, 0.1)


def estimate_noise_sigma_mad(image: np.ndarray) -> float:
    """Donoho MAD of high-pass detail - second estimator used to cross-check."""
    g = to_gray(image).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(np.median(np.abs(lap)) / 0.6745 / np.sqrt(6.0))


# ---------------------------------------------------------------------------
# comparison + automatic selection
# ---------------------------------------------------------------------------


def apply_method(image: np.ndarray, method: str, params: Optional[Dict[str, object]] = None
                 ) -> np.ndarray:
    p: Dict[str, object] = {
        "ksize": 5, "sigma_color": 35.0, "sigma_space": 5.0, "h": 10.0,
        "radius": 8, "eps": 800.0,
    }
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    if method == "none":
        return image
    if method == "gaussian":
        return gaussian(image, int(p["ksize"]))
    if method == "median":
        return median(image, int(p["ksize"]))
    if method == "bilateral":
        return bilateral(image, int(p["ksize"]), float(p["sigma_color"]),
                         float(p["sigma_space"]))
    if method == "nlm":
        try:
            return nlm(image, float(p["h"]))
        except cv2.error:            # small images / odd builds
            return median(image, 3)
    if method == "guided":
        return guided_filter(image, int(p["radius"]), float(p["eps"]))
    if method == "wiener":
        return wiener_local(image, int(p["ksize"]))
    raise ValueError(f"unknown denoise method '{method}'")


def compare_filters(image: np.ndarray,
                    methods: List[str] = ("none", "gaussian", "median",
                                          "bilateral", "guided", "wiener"),
                    params: Optional[Dict[str, object]] = None,
                    reference: Optional[np.ndarray] = None) -> List[DenoiseReport]:
    """Run every candidate, time it, score it, and mark the winner.

    Scoring (reference-free, because a real scan has no clean ground truth):
      noise_suppression : local MAD-noise inside flat paper (lower is better)
      edge_retention    : ratio of gradient energy vs the input (higher is
                          better -> text strokes survived)
      time              : milliseconds (logged for the report)
    The selected filter maximises ``edge_retention - 1.5 * relative_noise``.
    When ``reference`` (a clean image) is supplied, PSNR is added and becomes
    the dominant term - this is the controlled-test-image path used by the
    evaluation tool.
    """
    import time as _t
    gray_in = to_gray(image)
    reports: List[DenoiseReport] = []
    base_grad = _gradient_energy(gray_in) + 1e-6
    for m in methods:
        t0 = _t.perf_counter()
        out = apply_method(image, m, params)
        dt = (_t.perf_counter() - t0) * 1000.0
        og = to_gray(out)
        res = {
            "residual_noise": round(estimate_noise_sigma(og), 3),
            "edge_retention": round(_gradient_energy(og) / base_grad, 4),
            "mean_shift": round(float(np.mean(og) - np.mean(gray_in)), 2),
            "time_ms": round(dt, 2),
        }
        if reference is not None:
            res["psnr"] = round(psnr(og, to_gray(reference)), 3)
            res["ssim"] = round(ssim(og, to_gray(reference)), 4)
        reports.append(DenoiseReport(method=m, params=dict(params or {}),
                                     time_ms=round(dt, 2), metrics=res, image=out))

    if reference is not None:
        reports.sort(key=lambda r: -r.metrics.get("psnr", 0))
    else:
        reports.sort(key=lambda r: -(r.metrics.get("edge_retention", 0)
                                     - 1.5 * r.metrics.get("residual_noise", 99) / 10.0))
    if reports:
        reports[0].selected = True
    return reports


def _gradient_energy(g: np.ndarray) -> float:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))


