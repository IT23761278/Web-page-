"""
heritage_vision.core.metrics
============================

Pure evaluation functions, shared by the evaluation tool, the denoiser
comparison and the before/after report.  Kept dependency-free (OpenCV + NumPy
only) so a lab machine does not need scikit-image just to compute a number.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import to_gray


# ---------------------------------------------------------------------------
# pixel-level fidelity (needs a reference image)
# ---------------------------------------------------------------------------


def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = to_gray(a).astype(np.float32)
    b = to_gray(b).astype(np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_AREA if a.size > b.size
                       else cv2.INTER_CUBIC).astype(np.float32)
    if a.max() <= 1.5 and b.max() <= 1.5:
        a, b = a * 255.0, b * 255.0
    return a, b


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _align(a, b)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def psnr(a: np.ndarray, b: np.ndarray, maxv: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB.  inf for identical images."""
    x, y = _align(a, b)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((maxv ** 2) / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004), 11x11 sigma 1.5 Gaussian window.

    Implemented with ``cv2.GaussianBlur`` on the four required local moments.
    A real implementation of the *window* matters: a uniform 11x11 box filter
    over-estimates SSIM on high-contrast text because it under-estimates the
    local variance.
    """
    x, y = _align(a, b)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = (11, 11)
    sigma = 1.5

    def blur(img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, win, sigma)

    mu_x, mu_y = blur(x), blur(y)
    mu_xx = blur(x * x)
    mu_yy = blur(y * y)
    mu_xy = blur(x * y)
    var_x = np.maximum(mu_xx - mu_x * mu_x, 0.0)
    var_y = np.maximum(mu_yy - mu_y * mu_y, 0.0)
    cov_xy = mu_xy - mu_x * mu_y
    s1 = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    s2 = (2 * cov_xy + C2) / (var_x + var_y + C2)
    return float(np.mean(s1 * s2))


def mse_masked(a, b, mask):
    """Mean squared intensity error *inside* a region (float, 0..255 scale).

    Restricting the error to the damage region is what makes a restoration
    measure meaningful: a global PSNR is dominated by the 90 % of the sheet
    that was deliberately contrast-enhanced (which by design moves pixels away
    from the original appearance), while the question "did the inpainting put
    the paper back where the damage was?" can only be answered inside the mask.
    """
    x, y = _align(a, b)
    m = np.asarray(mask) > 0
    if m.shape != x.shape:
        m = cv2.resize(m.astype(np.uint8) * 255, (x.shape[1], x.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 127
    if not m.any():
        return float("nan")
    return float(np.mean((x[m] - y[m]) ** 2))


def psnr_masked(a, b, mask, maxv=255.0):
    mse = mse_masked(a, b, mask)
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((maxv ** 2) / mse))


def u3ae(a: np.ndarray, b: np.ndarray) -> float:
    """ULCCS/'3x-scale' style absolute error - a perceptual sanity check."""
    x, y = _align(a, b)
    return float(np.mean(np.abs(x - y)))


# ---------------------------------------------------------------------------
# region-level (needs a ground-truth mask)
# ---------------------------------------------------------------------------


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 127 if np.asarray(mask).max() > 1 else np.asarray(mask) > 0


def overlap_stats(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Confusion counts + IoU (Jaccard), Dice (F1), precision, recall.

    Area is in pixels; the caller converts to mm2 when needed.  A 2-px
    boundary tolerance is *not* applied here on purpose - an un-tolerant IoU is
    the honest number for a stain whose edge is fuzzy by nature; the tolerance
    variant is reported separately as ``iou_tol2`` so both are visible.
    """
    p = _as_bool(pred)
    t = _as_bool(truth)
    if p.shape != t.shape:
        t = cv2.resize(t.astype(np.uint8) * 255, (p.shape[1], p.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 127
    tp = float(np.count_nonzero(p & t))
    fp = float(np.count_nonzero(p & ~t))
    fn = float(np.count_nonzero(~p & t))
    tn = float(np.count_nonzero(~p & ~t))
    inter = tp
    union = tp + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": inter / union if union else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / max(p.size, 1),
        "pred_area_px": float(np.count_nonzero(p)),
        "truth_area_px": float(np.count_nonzero(t)),
        "area_error_pct": 100.0 * (np.count_nonzero(p) - np.count_nonzero(t)) /
        max(np.count_nonzero(t), 1),
    }
    k = np.ones((5, 5), np.uint8)
    t_tol = cv2.dilate(t.astype(np.uint8) * 255, k) > 0
    p_tol = cv2.dilate(p.astype(np.uint8) * 255, k) > 0
    inter2 = float(np.count_nonzero(p & t_tol)) + float(np.count_nonzero(t & p_tol))
    out["iou_tol2"] = min(1.0, inter2 / max(2.0 * (tp + fp + fn), 1.0))
    return out


def count_error(pred_count: int, truth_count: int) -> Dict[str, float]:
    """Absolute + relative error of an object count (Part D 'count error')."""
    err = int(pred_count) - int(truth_count)
    rel = 100.0 * err / max(abs(truth_count), 1)
    return {"count_pred": float(pred_count), "count_truth": float(truth_count),
            "count_abs_error": float(abs(err)), "count_rel_error_pct": round(rel, 2)}


def measurement_error(measured: float, reference: float,
                      unit: str = "mm2") -> Dict[str, float]:
    """Dimensional/area measurement error against the known value."""
    d = float(measured) - float(reference)
    rel = 100.0 * d / max(abs(reference), 1e-9)
    return {f"{unit}_measured": float(measured), f"{unit}_reference": float(reference),
            f"{unit}_abs_error": d, f"{unit}_rel_error_pct": round(rel, 2)}


def contrast_improvement(before_gray: np.ndarray, after_gray: np.ndarray) -> Dict[str, float]:
    """'Contrast improvement' metric named by the domain table.

    Uses Michelson contrast (std/mean is reported too).  Improvement is the
    signed % change; a *negative* number means the enhancement destroyed
    contrast, which is exactly what the parameter sweep is looking for.
    """
    def mc(g: np.ndarray) -> float:
        g = to_gray(g).astype(np.float32)
        return float(np.std(g)) / max(float(np.mean(g)), 1e-6)

    b, a = mc(before_gray), mc(after_gray)
    def umich(g):
        g = to_gray(g).astype(np.float32)
        lo, hi = np.percentile(g, [1, 99])
        return float((hi - lo) / max(hi + lo, 1.0))
    ub, ua = umich(before_gray), umich(after_gray)
    return {
        "contrast_before": round(b, 4), "contrast_after": round(a, 4),
        "contrast_gain_pct": round(100.0 * (a - b) / max(b, 1e-6), 2),
        "umichelson_before": round(ub, 4), "umichelson_after": round(ua, 4),
        "umichelson_gain_pct": round(100.0 * (ua - ub) / max(ub, 1e-6), 2),
    }


def uniform_bg_std(gray: np.ndarray, block: int = 64) -> float:
    """Illumination uniformity: mean std of the background over the grid."""
    from .enhancement import background_estimate_gauss
    g = to_gray(gray)
    bg = background_estimate_gauss(g, sigma=25)
    H, W = bg.shape
    vals = []
    for y in range(0, H - block, block):
        for x in range(0, W - block, block):
            vals.append(float(np.std(bg[y:y + block, x:x + block])))
    return round(float(np.mean(vals)), 3) if vals else 0.0


def summarise(rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    """mean / std / median / IQR / min / max / n of the given keys.

    The median and the inter-quartile range are reported next to the mean
    because several metrics are ratios of a *tiny* ground-truth area (a lightly
    damaged page has ~200 damaged pixels), which makes the relative error
    unbounded: the mean then says +49 % while the median says +7 %.  Quoting
    only the mean would misrepresent the system, and quoting only the median
    would hide the outliers, so both are in the table.
    """
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if k in r and
                         np.isfinite(float(r[k]))], np.float64)
        if vals.size == 0:
            out[k] = float("nan")
            continue
        out[f"{k}_mean"] = round(float(vals.mean()), 4)
        out[f"{k}_median"] = round(float(np.median(vals)), 4)
        out[f"{k}_iqr"] = round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4)
        out[f"{k}_std"] = round(float(vals.std()), 4)
        out[f"{k}_min"] = round(float(vals.min()), 4)
        out[f"{k}_max"] = round(float(vals.max()), 4)
        out[f"{k}_n"] = float(vals.size)
    return out
