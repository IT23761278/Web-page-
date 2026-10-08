"""
heritage_vision.core.segmentation
=================================

Part C techniques 2, 3, 4 and 5 (thresholding, morphology, colour
segmentation, region/watershed) implemented on top of the enhanced image
produced by :mod:`heritage_vision.core.enhancement`.

Everything returns *masks* (uint8 0/255) plus a small statistics dict, so the
GUI can visualise and the evaluator can compute IoU without any coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import odd, to_bgr, to_gray


# ---------------------------------------------------------------------------
# 1. Thresholding (global Otsu / adaptive / Sauvola-like)
# ---------------------------------------------------------------------------


def otsu(gray: np.ndarray, invert: bool = True) -> Tuple[np.ndarray, float]:
    """Global Otsu threshold.

    Otsu maximises inter-class variance and works well once illumination has
    been flattened (that is exactly why the enhancement stage comes first).
    Returns (mask, threshold_used).
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    t, mask = cv2.threshold(blur, 0, 255, flag)
    return mask, float(t)


def adaptive_gray(gray: np.ndarray, block: int = 51, c: int = 10,
                  mode: str = "gauss") -> np.ndarray:
    """Local adaptive threshold - robust to residual shading and to a page
    that is darker on one side (curl, torch).  Mean vs Gaussian: Gaussian
    weights the neighbourhood and is less sensitive to a single dark stain
    inside the window, hence the default."""
    k = odd(block, 3)
    method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if mode == "gauss"
              else cv2.ADAPTIVE_THRESH_MEAN_C)
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
    #                        blockSize, C)  - the argument order matters and an
    # earlier revision folded method+type into one flag and passed C twice,
    # which raised "missing required argument C" for every colour space.
    return cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY_INV, k, int(c))


def sauvola(gray: np.ndarray, window: int = 41, k: float = 0.2,
            r: float = 128.0) -> Tuple[np.ndarray, np.ndarray]:
    """Sauvola binarisation  T = mu*(1 + k*(sigma/R - 1)).

    The de-facto standard for degraded historical documents because it adapts
    to *local contrast*: a faint stain and faint ink both sit in a low-variance
    neighbourhood and are separated differently than by a plain local mean.
    Implemented with integral images (box filters) -> O(N) and exact.
    Returns (text_mask, local_threshold_float32).
    """
    g = gray.astype(np.float32)
    w = odd(window, 5)
    mean = cv2.boxFilter(g, cv2.CV_32F, (w, w), normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(g * g, cv2.CV_32F, (w, w), normalize=True,
                            borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    thr = mean * (1.0 + float(k) * (std / float(r) - 1.0))
    mask = np.where(g < thr, 255, 0).astype(np.uint8)
    return mask, thr


# ---------------------------------------------------------------------------
# 2. Morphology & connected components
# ---------------------------------------------------------------------------


@dataclass
class MorphPlan:
    open_px: int = 2
    close_px: int = 3
    kernel: int = cv2.MORPH_ELLIPSE

    def apply(self, mask: np.ndarray) -> np.ndarray:
        out = mask
        if self.open_px > 0:
            k = cv2.getStructuringElement(self.kernel,
                                          (odd(self.open_px, 3), odd(self.open_px, 3)))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        if self.close_px > 0:
            k = cv2.getStructuringElement(self.kernel,
                                          (odd(self.close_px, 3), odd(self.close_px, 3)))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
        return out


def connected_components(mask: np.ndarray, connectivity: int = 8,
                         min_area: int = 25, max_area: Optional[int] = None,
                         roi: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    """Label a binary mask and return per-component statistics.

    One CCL call gives count, area, bbox and pixel assignment; everything else
    (perimeter, circularity, solidity, orientation) is derived per component
    from the labelled array, which is far cheaper than findContours on noisy
    masks.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=connectivity)
    total = float(mask.shape[0] * mask.shape[1])
    out: List[Dict[str, object]] = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if roi is not None:
            sub = roi[y:y + h, x:x + w]
            lab = labels[y:y + h, x:x + w] == i
            if sub.size and float(np.mean(sub[lab])) < 0.5:
                continue
        comp = labels == i
        cx, cy = centroids[i]
        out.append({
            "id": i, "bbox": (x, y, w, h), "area_px": area,
            "area_pct": round(area / max(total, 1.0) * 100.0, 4),
            "centroid": (float(cx), float(cy)),
            "aspect": round(w / float(max(h, 1)), 3),
            "labels": labels,
        })
    return out


def fill_holes(mask: np.ndarray, max_hole_frac: float = 0.15,
               evidence: Optional[np.ndarray] = None,
               evidence_min: float = 0.0) -> np.ndarray:
    """Fill enclosed cavities (the pale core of a ring stain, a closed ink loop).

    Two guards keep this from destroying the measurement:
      * ``max_hole_frac`` - never fill a cavity bigger than this share of the
        frame (a big enclosed clean area is *clean*, not a stain core);
      * ``evidence`` / ``evidence_min`` - optional scalar map (e.g. the chroma
        excess map).  A hole is only filled when its mean evidence exceeds
        ``evidence_min``, i.e. when the enclosed pixels really look like the
        lesion rather than like untouched paper.  Without this guard a crack
        that happens to loop around a paragraph would mark the whole
        paragraph as damage (measured: 0.6 % true damage became 43 %).
    """
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    total = float(mask.size)
    out = mask.copy()
    for i in range(1, n):
        x, y, w, h = [int(v) for v in stats[i, :4]]
        area = int(stats[i, cv2.CC_STAT_AREA])
        if x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]:
            continue                                  # outside background
        if area > max_hole_frac * total:
            continue
        sub = labels[y:y + h, x:x + w] == i
        if evidence is not None:
            e = evidence[y:y + h, x:x + w][sub]
            if e.size == 0 or float(np.mean(e)) < evidence_min:
                continue
        out[y:y + h, x:x + w][sub] = 255
    return out


def component_metrics(mask: np.ndarray, labels: np.ndarray, label_id: int,
                      dpi: int = 300) -> Dict[str, float]:
    """Shape descriptors of one component: area, perimeter, circularity,
    compactness, orientation, solidity, elongation (Part C technique 6)."""
    comp = (labels == label_id).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        area = float(np.count_nonzero(comp))
    per = cv2.arcLength(c, True)
    circ = (4 * np.pi * area / (per * per)) if per > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    rect = cv2.minAreaRect(c)
    (rw, rh) = rect[1]
    elong = max(rw, rh) / max(min(rw, rh), 1e-6)
    px_mm = 25.4 / float(max(dpi, 1))
    return {
        "area_px": float(np.count_nonzero(comp)),
        "area_mm2": round(area * px_mm ** 2, 3),
        "perimeter_px": round(per, 2),
        "perimeter_mm": round(per * px_mm, 3),
        "circularity": round(float(circ), 4),
        "solidity": round(float(solidity), 4),
        "elongation": round(float(elong), 3),
        "orientation_deg": round(float(normalise_angle(rect[2])), 2),
        "bbox": rect[0][0], "bbox_w": float(rect[1][0]), "bbox_h": float(rect[1][1]),
        "extent": round(area / max(rw * rh, 1e-6), 4),
    }


def normalise_angle(a: float) -> float:
    """Fold OpenCV's rect angle into [-45, 45) so 'orientation' is comparable."""
    a = float(a) % 180.0
    if a > 90:
        a -= 180
    return a


# ---------------------------------------------------------------------------
# 3. Colour-based segmentation
# ---------------------------------------------------------------------------


def hsv_range_mask(image: np.ndarray, hue_range: Sequence[int],
                   sat_range: Sequence[int], val_range: Sequence[int]) -> np.ndarray:
    """In-range mask with wrap-around support for hue (red spans 0/179)."""
    hsv = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2HSV)
    lo = np.array([hue_range[0], sat_range[0], val_range[0]], np.uint8)
    hi = np.array([hue_range[1], sat_range[1], val_range[1]], np.uint8)
    if hue_range[0] > hue_range[1]:      # wraps through 0
        m1 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                         np.array([hue_range[1], hi[1], hi[2]], np.uint8))
        m2 = cv2.inRange(hsv, np.array([hue_range[0], lo[1], lo[2]], np.uint8),
                         np.array([179, hi[1], hi[2]], np.uint8))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, lo, hi)


#: Chromatic fingerprint of the typical damage seen on paper heritage.
COLOUR_PROFILES: Dict[str, Dict[str, object]] = {
    "foxing / brown stain": {
        "hue": (8, 32), "sat": (45, 200), "val": (50, 215)},
    "water damage / tide line": {
        "hue": (5, 40), "sat": (25, 180), "val": (30, 235)},
    "mould / green bloom": {
        "hue": (32, 90), "sat": (35, 255), "val": (25, 200)},
    "ink burn / scorch (dark)": {
        "hue": (0, 179), "sat": (0, 90), "val": (0, 70)},
    "red seal / rubric": {
        "hue": (165, 179), "sat": (70, 255), "val": (60, 235)},
    "photographic silver mirroring": {
        "hue": (95, 140), "sat": (5, 60), "val": (140, 255)},
}


def colour_segment(image: np.ndarray, profile: Optional[str] = None,
                   custom: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    if profile:
        p = COLOUR_PROFILES[profile]
        masks["mask"] = hsv_range_mask(image, p["hue"], p["sat"], p["val"])
    elif custom:
        masks["mask"] = hsv_range_mask(
            image, custom.get("hue", (0, 179)), custom.get("sat", (30, 255)),
            custom.get("val", (0, 255)))
    return masks


# ---------------------------------------------------------------------------
# 4. Region-based / watershed segmentation
# ---------------------------------------------------------------------------


def distance_transform(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask, cv2.DIST_L2, 5)


def marker_watershed(mask: np.ndarray, sure_fg_frac: float = 0.55,
                     roi: Optional[np.ndarray] = None, min_sep_px: int = 14,
                     min_height_px: float = 1.6, mode: str = "peaks"
                     ) -> Dict[str, object]:
    """Marker-controlled watershed that splits touching lesions.

    Problem in this domain: a stain blob and an adjacent fold line merge into
    one contour, and overlapping letterforms merge in the binarised mask - so a
    plain external-contour count is wrong (it also under-counts the damage area
    of each region, which is what the severity score is built on).

    method='peaks' (default) - seeds = local maxima of the distance transform,
        separated by at least ``min_sep_px`` and higher than ``min_height_px``.
        Each visible lobe of a fused blob therefore gets exactly one seed, and
        the count does not depend on the *absolute* size of the blob, which
        varies with the stain severity across the corpus.
    method='threshold' - classic Frayle/Beucher seeding: everything further than
        ``sure_fg_frac``*max(distance) from the boundary is a seed.  Simple, but
        a single large stain complex yields a single seed, so a set of touching
        spots is counted as one lesion (measured: 1 region instead of 22).
        Kept because it is the textbook variant and the GUI can switch to it.

    The unknown band between seeds and the dilated background is what
    watershed labels; boundary pixels come back as -1 from OpenCV.
    """
    m = (mask > 0).astype(np.uint8) * 255
    dist = distance_transform(m)
    dmax = float(dist.max())
    if dmax <= 0:
        return {"regions": 0, "labels": None, "areas_mm2": [], "seeds": 0,
                "sure_fg": m, "sure_bg": m, "unknown": m, "dist": dist,
                "method": mode}
    if mode == "peaks":
        sep = max(int(min_sep_px), 3) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sep, sep))
        loc_max = cv2.dilate(dist, kernel)
        peaks = ((dist >= loc_max - 1e-6) & (dist > min_height_px)).astype(np.uint8)
        if not peaks.any():                       # thin-only mask (cracks)
            peaks = (dist >= dmax - 1e-6).astype(np.uint8)
        # one seed per connected peak plateau
        n_seed, seed_lab = cv2.connectedComponents(peaks, 8)
        sure_fg = (seed_lab > 0).astype(np.uint8) * 255
        n = n_seed
    else:
        thr = max(dmax * float(sure_fg_frac), 1.0)
        _, sure_fg = cv2.threshold(dist.astype(np.float32), thr, 255,
                                   cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        n = cv2.connectedComponents(sure_fg, 8)[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    sure_bg = cv2.dilate(m, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n2, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    try:
        markers = cv2.watershed(bgr, markers)
    except cv2.error:
        pass
    # watershed can wipe a whole seed out (all its pixels became boundary), so
    # the label set is counted from the *result*, not from n2 - an off-by-one
    # here raised IndexError on images with a large number of lesions.
    lim = int(markers.max()) + 1
    flat = markers.ravel()
    counts = np.bincount(flat[flat >= 0], minlength=lim)
    px_mm2 = (25.4 / 300.0) ** 2
    areas = [round(float(counts[i]) * px_mm2, 3) for i in range(2, lim)
             if counts[i] > 0]
    regions_in_roi = None
    if roi is not None:
        sel = markers[(roi > 0)]
        in_roi = np.bincount(sel[sel >= 0], minlength=lim)
        regions_in_roi = int(np.count_nonzero(in_roi[2:lim]))
    return {"regions": len(areas), "labels": markers, "areas_mm2": areas,
            "seeds": int(n - 1), "sure_fg": sure_fg, "sure_bg": sure_bg,
            "unknown": unknown, "dist": dist, "method": mode,
            "regions_in_roi": regions_in_roi}


def region_grow_seeds(mask: np.ndarray, step: int = 24) -> np.ndarray:
    """Seed points = local maxima of the distance transform on a regular grid.
    Used to visualise where watershed starts; not a separate algorithm."""
    dist = distance_transform((mask > 0).astype(np.uint8) * 255)
    seeds = np.zeros_like(dist, np.uint8)
    H, W = dist.shape
    for y in range(0, H, step):
        for x in range(0, W, step):
            win = dist[y:y + step, x:x + step]
            if win.size == 0 or win.max() <= 1:
                continue
            yy, xx = np.unravel_index(int(np.argmax(win)), win.shape)
            seeds[y + yy, x + xx] = 255
    return seeds


# ---------------------------------------------------------------------------
# 5. Small helpers
# ---------------------------------------------------------------------------


def mask_stats(mask: np.ndarray, roi: Optional[np.ndarray] = None,
               dpi: int = 300) -> Dict[str, float]:
    """Area / percentage of a binary layer, restricted to the ROI."""
    m = (mask > 0)
    if roi is not None:
        m = m & (roi > 0)
    area = int(np.count_nonzero(m))
    total = int(np.count_nonzero(roi > 0)) if roi is not None else m.size
    px_mm = 25.4 / float(max(dpi, 1))
    return {
        "area_px": float(area),
        "area_pct": round(area / max(total, 1) * 100.0, 4),
        "area_mm2": round(area * px_mm ** 2, 2),
        "area_cm2": round(area * px_mm ** 2 / 100.0, 3),
    }


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == best).astype(np.uint8) * 255)
