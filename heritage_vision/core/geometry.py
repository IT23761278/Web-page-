"""
heritage_vision.core.geometry
=============================

Part B requirement 6 - *geometric processing*: a region-of-interest
selection plus at least one transformation required by the domain.

For Heritage & Culture the two transformations that actually matter are
    1. deskew / rotation correction - a hand-placed sheet or a photograph of a
       wall inscription always arrives with a small rotation, which breaks the
       row-projection analysis used for line counting;
    2. perspective rectification - a phone photo of a page needs a 4-point
       homography, otherwise measured areas (mm2) are wrong in the far corner.

Both are geometric, both are needed before any measurement is quoted, and the
module reports the applied transforms so the numbers in the report are
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import to_bgr, to_gray


@dataclass
class GeometryLog:
    applied: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, **kw: float) -> None:
        self.applied.append(name)
        for k, v in kw.items():
            self.details[f"{name}_{k}"] = float(v)

    def as_dict(self) -> Dict[str, object]:
        return {"applied": "+".join(self.applied) or "none", **self.details}


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


def crop(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    """Safe crop: clamps to the frame and refuses degenerate boxes."""
    x, y, w, h = [int(v) for v in box]
    H, W = image.shape[:2]
    x0 = max(0, min(x, W - 1))
    y0 = max(0, min(y, H - 1))
    x1 = max(x0 + 1, min(x + w, W))
    y1 = max(y0 + 1, min(y + h, H))
    return image[y0:y1, x0:x1]


def crop_mask(mask: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    out = np.zeros_like(mask)
    H, W = mask.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def make_roi(shape: Tuple[int, int], mode: str = "full",
             box: Optional[Sequence[int]] = None,
             margin_px: int = 0) -> np.ndarray:
    """Binary ROI mask.  mode: full | box | auto (largest contour) | centre."""
    H, W = shape[:2]
    roi = np.ones((H, W), np.uint8) * 255
    if mode == "box" and box is not None:
        roi = crop_mask(roi, box)
    elif mode == "centre":
        mx, my = int(W * 0.12), int(H * 0.12)
        roi = roi[my:H - my, mx:W - mx]
        roi = _embed(roi, (H, W), (mx, my))
    elif mode == "auto":
        raise ValueError("auto ROI needs the image, use auto_page()")
    if margin_px > 0 and mode != "box":
        k = max(int(margin_px), 1) * 2
        kernel = np.ones((3, 3), np.uint8)
        shrunk = cv2.erode(roi, kernel, iterations=k)
        if shrunk.sum() > 0.10 * roi.sum():
            roi = shrunk
    return roi


def _embed(small: np.ndarray, size: Tuple[int, int], at: Tuple[int, int]) -> np.ndarray:
    H, W = size
    out = np.zeros((H, W), np.uint8)
    h, w = small.shape[:2]
    out[at[1]:at[1] + h, at[0]:at[0] + w] = small
    return out


# ---------------------------------------------------------------------------
# Rotation / deskew
# ---------------------------------------------------------------------------


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 6.0,
                         method: str = "projection", min_gain: float = 1.15
                         ) -> Tuple[float, Dict[str, float]]:
    """Return the rotation (degrees) that makes the text rows horizontal.

    method='projection' - sum of squared row projections; the true baseline
        orientation maximises the peak-to-mean ratio because rows line up.
        Robust on scanned paper, no assumption about stroke length.
    method='hough'     - median angle of long line segments; better for
        inscriptions/tables, worse for cursive handwriting.
    Both are reported so the choice can be justified in the VIVA.

    Sign convention (important): the returned value is the angle to *apply*
    with :func:`rotate` in order to make the rows horizontal, i.e. it is the
    correction, not the distortion.  For a page that was rotated +2 deg the
    function returns -2.0 deg (verified in tests/test_geometry.py).
    """
    g = to_gray(gray)
    edges = cv2.Canny(g, 60, 160)
    info: Dict[str, float] = {}
    if method == "hough":
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                                minLineLength=max(40, g.shape[1] // 12),
                                maxLineGap=8)
        if lines is None or len(lines) < 3:
            return 0.0, {"confidence": 0.0}
        flat = lines.reshape(-1, 4)
        ang = np.degrees(np.arctan2((flat[:, 3] - flat[:, 1]).astype(np.float64),
                                    (flat[:, 2] - flat[:, 0]).astype(np.float64)))
        ang = np.abs(((ang + 90) % 180) - 90)
        keep = np.abs(ang) <= max_angle
        if keep.sum() < 3:
            return 0.0, {"confidence": 0.0}
        med = float(np.median(ang[keep]))
        info = {"confidence": float(keep.mean()), "lines": float(keep.sum())}
        return med, info

    best, best_score = 0.0, -np.inf
    profile_scores: List[Tuple[float, float]] = []
    step = max(max_angle / 60.0, 0.1)
    zero = cv2.warpAffine(edges, cv2.getRotationMatrix2D(
        (g.shape[1] / 2, g.shape[0] / 2), 0.0, 1.0), (g.shape[1], g.shape[0]))
    p0 = zero.sum(axis=1).astype(np.float64)
    score0 = float((p0 / max(p0.sum(), 1.0)).var())
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), a, 1.0)
        rot = cv2.warpAffine(edges, M, (g.shape[1], g.shape[0]),
                             flags=cv2.INTER_NEAREST, borderValue=0)
        prof = rot.sum(axis=1).astype(np.float64)
        prof /= max(prof.sum(), 1.0)
        # variance of the projection = rows are aligned
        score = float(prof.var())
        profile_scores.append((float(a), score))
        if score > best_score:
            best_score, best = score, float(a)
    # parabolic refinement around the discrete maximum
    arr = np.array(profile_scores)
    i = int(np.argmax(arr[:, 1]))
    if 0 < i < len(arr) - 1:
        y0, y1, y2 = arr[i - 1, 1], arr[i, 1], arr[i + 1, 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            off = 0.5 * (y0 - y2) / denom
            best = float(arr[i, 0] + off * step)
    # Row periodicity at the chosen angle: max/mean of the normalised row
    # profile.  A page of text has clear rows (ratio > 3), a photograph of a
    # building does not, and for the latter the argmax of the variance is
    # meaningless - it invented 13 deg on a photo plate in an earlier revision.
    M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), best, 1.0)
    rot = cv2.warpAffine(edges, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_NEAREST,
                         borderValue=0)
    prof = rot.sum(axis=1).astype(np.float64)
    prof /= max(prof.sum(), 1.0)
    periodicity = float(prof.max() / max(prof.mean(), 1e-12))
    edge_of_window = abs(abs(best) - max_angle) < step
    info = {"score": best_score, "angle_step": step, "score_zero": score0,
            "periodicity": periodicity}
    if (best_score < min_gain * max(score0, 1e-12) or abs(best) < 0.2
            or periodicity < 2.2 or edge_of_window):
        info["rejected"] = 1.0
        return 0.0, info
    info["rejected"] = 0.0
    return best, info


def rotate(image: np.ndarray, angle_deg: float, expand: bool = True,
           border_value=(0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate by ``angle_deg`` returning (image, valid-mask).

    ``expand`` keeps all pixels (used for whole-page deskew) and the returned
    mask tells later stages which area is real content and which is padding -
    otherwise the black corner would be detected as a 'burn mark'.
    """
    if abs(angle_deg) < 1e-3:
        H, W = image.shape[:2]
        return image, np.full((H, W), 255, np.uint8)
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        M[0, 2] += nw / 2.0 - center[0]
        M[1, 2] += nh / 2.0 - center[1]
        size = (nw, nh)
    else:
        size = (w, h)
    out = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, size,
                          flags=cv2.INTER_NEAREST, borderValue=0)
    return out, mask


# ---------------------------------------------------------------------------
# Perspective rectification
# ---------------------------------------------------------------------------


def find_page_corners(image: np.ndarray, min_area_frac: float = 0.25,
                      polarity: str = "auto") -> Tuple[Optional[np.ndarray], str]:
    """Locate the 4 outer corners of the physical sheet in a photograph.

    A sheet is the largest *rectangular* closed region of the frame, but whether
    it is brighter or darker than its surroundings depends entirely on the
    mount (white paper on a black scanner bed, a dark glass-plate negative on a
    light table).  So both polarities are segmented, both candidate quads are
    measured, and the one with the better rectangularity wins - which is also
    the criterion a human uses ("that shape has four right angles, so it is the
    sheet").  This replaced a chroma-based rule that worked on paper but failed
    on photographic plates (mean corner error 137 px instead of 1 px).
    """
    bgr = to_bgr(image)
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(to_gray(bgr), (5, 5), 0)
    best: Tuple[float, Optional[np.ndarray], str] = (-1.0, None, "no-page-contour")
    for pol in (("bright", "dark") if polarity == "auto" else (polarity,)):
        thr = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if pol == "dark":
            thr = cv2.bitwise_not(thr)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = [c for c in cnts if cv2.contourArea(c) >= min_area_frac * h * w]
        if not cand:
            continue
        cnt = max(cand, key=cv2.contourArea)
        for c in (cnt, cv2.convexHull(cnt)):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            pts = (approx.reshape(-1, 2).astype(np.float32) if len(approx) == 4
                   else cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32))
            rect_area = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(c)))
            rectangularity = cv2.contourArea(c) / max(rect_area, 1.0)
            area_frac = cv2.contourArea(c) / float(h * w)
            # A region that touches all four image edges is the frame, not the
            # sheet: without this penalty the "dark sheet" polarity wins with a
            # perfect quad covering the whole image (measured: 40 px corner
            # error on every synthetic scan, where the bright polarity is exact).
            rect = cv2.boundingRect(c)
            touches = ((rect[1] <= 2) + (rect[0] <= 2)
                       + (rect[1] + rect[3] >= h - 2) + (rect[0] + rect[2] >= w - 2))
            if touches >= 4 or area_frac > 0.97:
                continue
            score = rectangularity * (1.05 if len(approx) == 4 else 1.0) * min(area_frac * 2, 1.0)
            if score > best[0]:
                best = (score, pts, f"quad-{'bright' if pol=='bright' else 'dark'}"
                        + ("" if len(approx) == 4 else "-min-area-rect"))
    if best[1] is None:
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = [c for c in cnts if cv2.contourArea(cv2.convexHull(c)) >= min_area_frac * h * w]
        if not cand:
            return None, "no-page-contour"
        big = max(cand, key=lambda c: cv2.contourArea(cv2.convexHull(c)))
        return cv2.boxPoints(cv2.minAreaRect(big)).astype(np.float32), "edge-min-area-rect"
    return best[1], best[2]


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return corners as [tl, tr, br, bl] - required for a stable warp."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.zeros((4, 2), np.float32)
    out[0] = pts[np.argmin(s)]          # top-left has smallest x+y
    out[2] = pts[np.argmax(s)]          # bottom-right has largest x+y
    out[1] = pts[np.argmin(d)]          # top-right has smallest y-x
    out[3] = pts[np.argmax(d)]          # bottom-left has largest y-x
    return out


def four_point_transform(image: np.ndarray, pts: np.ndarray
                         ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Warp a quadrilateral to an axis-aligned rectangle of natural size."""
    tl, tr, br, bl = order_points(pts)
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxW = int(max(round(wA), round(wB), 1))
    maxH = int(max(round(hA), round(hB), 1))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.stack([tl, tr, br, bl]).astype(np.float32), dst)
    out = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    # homography quality: aspect distortion and projective term
    info = {
        "width_px": float(maxW), "height_px": float(maxH),
        "aspect_after": round(maxW / float(maxH), 3),
        "skew_corner_dev_px": round(float(np.mean([
            np.linalg.norm(tl - bl) - np.linalg.norm(tr - br),
            np.linalg.norm(tl - tr) - np.linalg.norm(bl - br)])), 2),
    }
    return out, info


def rectify_to_a4_ratio(image: np.ndarray, ratio: float = 1.0 / np.sqrt(2.0)
                        ) -> np.ndarray:
    """Optional normalisation to the target sheet ratio (A4 = 1:sqrt(2)).

    Only pads/crops symmetrically - never rescales non-uniformly, because that
    would corrupt the mm2 measurements reported by the analysis stage.
    """
    h, w = image.shape[:2]
    cur = w / float(h)
    if abs(cur - ratio) < 0.02:
        return image
    if cur < ratio:      # too narrow -> pad width
        new_w = int(round(h * ratio))
        pad = new_w - w
        left, right = pad // 2, pad - pad // 2
        b = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_REPLICATE)
        return b
    new_h = int(round(w / ratio))
    pad = new_h - h
    top, bot = pad // 2, pad - pad // 2
    return cv2.copyMakeBorder(image, top, bot, 0, 0, cv2.BORDER_REPLICATE)


def auto_crop(image: np.ndarray, margin_px: int = 12
              ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Trim the surrounding desk/background, keeping a safety margin.

    The bounding box comes from :func:`find_page_corners`, i.e. from the sheet
    quad - which is the same decision the rectification step makes, so the two
    can never disagree about where the page is.  A previous version re-thresholded
    here with a "the bright side is the sheet" heuristic and cut the *desk*
    instead whenever the page covered less than half the frame.

    ``margin_px`` exists because damage at the physical edge (a torn corner!)
    must not be trimmed away.  If the detected quad is the whole frame nothing is
    cut, so a scan that was already cropped is left untouched.
    """
    H, W = image.shape[:2]
    pts, _how = find_page_corners(image)
    if pts is None:
        return image, (0, 0, W, H)
    x, y, w, h = cv2.boundingRect(np.asarray(pts, np.int32).reshape(-1, 2))
    x0, y0 = max(0, int(x) - int(margin_px)), max(0, int(y) - int(margin_px))
    x1, y1 = min(W, int(x + w) + int(margin_px)), min(H, int(y + h) + int(margin_px))
    if (x1 - x0) >= W * 0.995 and (y1 - y0) >= H * 0.995:
        return image, (0, 0, W, H)
    return crop(image, (x0, y0, x1 - x0, y1 - y0)), (x0, y0, x1 - x0, y1 - y0)


def scale_to_dpi(image: np.ndarray, src_dpi: int, dst_dpi: int) -> np.ndarray:
    """Resample so that 1 px always means the same physical length.

    All mm-based thresholds (minimum damage area, stroke width) assume one
    scale; this is the single place where the scale is normalised.
    """
    if not src_dpi or src_dpi == dst_dpi:
        return image
    s = dst_dpi / float(src_dpi)
    if abs(s - 1.0) < 0.01:
        return image
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (max(int(round(image.shape[1] * s)), 1),
                              max(int(round(image.shape[0] * s)), 1)),
                      interpolation=interp)
