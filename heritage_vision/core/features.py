"""
heritage_vision.core.features
=============================

Part C technique 6 (shape / area / perimeter / circularity / orientation /
texture) plus the texture measures a conservator actually uses.

Texture choices and why:
    LBP       - monotone grey-level invariant, captures the fibrous "grain"
                of paper vs the flat look of a scanner cover-up; cheap and
                deterministic (no learned features, so it is explainable).
    GLCM      - contrast, homogeneity, energy, correlation on 4 directions;
                the classical description of stains/mottling.
    Gabor     - oriented periodic texture (woven canvas support of a painting,
                engraving hatching); also our proxy for "directional wear".
    DoG energy- variance in a narrow mid-frequency band - a single-number
                proxy for micro-detail, used for the sharpness term.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .utils import to_gray


# ---------------------------------------------------------------------------
# 1. Shape descriptors
# ---------------------------------------------------------------------------


def describe_contour(cnt: np.ndarray, dpi: int = 300,
                     mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Complete geometric signature of one region.

    circularity = 4*pi*A/P^2  : 1 = disc (a foxing spot), <0.3 = elongated
                                 (a crack, a fold, a tear).  This single number
                                 is what separates 'stain' from 'crack' in the
                                 classification step.
    solidity    = A/A_hull    : low = ragged / feathered edges (active damage).
    extent      = A/A_rect    : how much of its bounding box the defect fills.
    """
    cnt = np.asarray(cnt, np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(cnt)
    if area <= 0 and mask is not None:
        area = float(np.count_nonzero(mask))
    per = cv2.arcLength(cnt, True)
    rect = cv2.minAreaRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    px_mm = 25.4 / float(max(dpi, 1))
    circularity = float((4 * np.pi * area / (per * per)) if per > 0 else 0.0)
    w0, h0 = rect[1]
    elong = max(w0, h0) / max(min(w0, h0), 1e-6)
    angle = float(rect[2])
    # OpenCV >= 4.5 returns the angle in [0, 90).  Rotate it by -90 when the
    # width side is the longer one so that "orientation" always means "the
    # direction of the long axis" (i.e. the direction a crack runs in).
    if w0 <= h0:
        angle = angle - 90.0
    else:
        angle = angle
    return {
        "area_px": round(area, 1),
        "area_mm2": round(area * px_mm ** 2, 3),
        "perimeter_px": round(per, 2),
        "perimeter_mm": round(per * px_mm, 3),
        "circularity": round(circularity, 4),
        "compactness": round((per * per) / max(4 * np.pi * area, 1e-6), 3),
        "solidity": round(area / hull_area if hull_area > 0 else 1.0, 4),
        "extent": round(area / max(bw * bh, 1), 4),
        "elongation": round(elong, 3),
        "orientation_deg": round(((angle + 45.0) % 180.0) - 45.0, 2),
        "bbox_px": [int(bx), int(by), int(bw), int(bh)],
        "bbox_diag_mm": round(float(np.hypot(bw, bh)) * px_mm, 2),
        "centroid_px": [round(float(rect[0][0]), 1), round(float(rect[0][1]), 1)],
        "hull_area_px": round(hull_area, 1),
        "vertices": int(cnt.reshape(-1, 2).shape[0]),
    }


def classify_defect(f: Dict[str, float], stroke_width_mm: float = 0.25) -> str:
    """Shape -> damage class.  Rule-based, therefore fully explainable.

    crack/tear    : very elongated and low circularity, thin.
    fold/crease   : elongated but fills its bounding box and is not ragged.
    foxing/stain  : compact blob, moderate saturation-driven solidity.
    ink-blodge    : compact but large and thick (bleeding ink, not paper wear).
    stain-complex : merged cluster of blobs.
    abrasion      : low solidity -> feathered, torn fibre edge.
    """
    el = float(f.get("elongation", 1.0))
    ci = float(f.get("circularity", 0.0))
    so = float(f.get("solidity", 1.0))
    ex = float(f.get("extent", 0.0))
    area_mm2 = float(f.get("area_mm2", 0.0))
    per_mm = float(f.get("perimeter_mm", 0.0))
    mean_width_mm = (2.0 * area_mm2 / per_mm) if per_mm > 0 else 0.0
    if el >= 3.5 and ci < 0.30 and mean_width_mm < 4.0 * stroke_width_mm:
        return "crack/tear"
    if el >= 3.5 and ex >= 0.55:
        return "fold/crease"
    if ci >= 0.60 and el < 2.0:
        return "ink-blodge" if mean_width_mm > 3.0 * stroke_width_mm and area_mm2 > 4.0 \
            else "foxing/stain"
    if ci >= 0.35 and el < 2.6:
        return "stain-complex"
    if so < 0.55:
        return "abrasion"
    return "damage-other"


def shape_histogram(features: List[Dict[str, float]], key: str = "circularity",
                     bins: int = 10) -> Dict[str, object]:
    """Distribution of one descriptor over all regions (reported + plotted)."""
    vals = np.array([float(f.get(key, 0.0)) for f in features], np.float32)
    if vals.size == 0:
        return {"counts": [0] * bins, "edges": [0.0] * (bins + 1),
                "mean": 0.0, "median": 0.0, "p90": 0.0}
    counts, edges = np.histogram(vals, bins=bins,
                                 range=(0.0, float(max(vals.max(), 1.0))))
    return {"counts": [int(c) for c in counts],
            "edges": [round(float(e), 3) for e in edges],
            "mean": round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4)}


# ---------------------------------------------------------------------------
# 2. Texture
# ---------------------------------------------------------------------------

_UNIFORM_TABLE: Optional[np.ndarray] = None


def _uniform_table(points: int = 8) -> np.ndarray:
    """Map every LBP code to a uniform pattern id (<=2 bit transitions)."""
    global _UNIFORM_TABLE
    if _UNIFORM_TABLE is not None and points == 8:
        return _UNIFORM_TABLE
    table = np.zeros(2 ** points, np.int32)
    nxt = 0
    seen: Dict[int, int] = {}
    for v in range(2 ** points):
        bits = [(v >> i) & 1 for i in range(points)]
        trans = sum(bits[i] != bits[(i + 1) % points] for i in range(points))
        if trans <= 2:
            if v not in seen:
                seen[v] = nxt
                nxt += 1
            table[v] = seen[v]
        else:
            table[v] = nxt                     # single "non-uniform" bin
    if points == 8:
        _UNIFORM_TABLE = table
    return table


def lbp_codes(gray: np.ndarray, radius: int = 1, points: int = 8) -> np.ndarray:
    """Dilated uniform LBP code image (P=8, R=1) built with integer maths."""
    g = to_gray(gray).astype(np.int16)
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    code = np.zeros((H, W), np.uint8)
    for i in range(points):
        ang = 2.0 * np.pi * i / points
        oy = int(round(-radius * np.sin(ang)))
        ox = int(round(radius * np.cos(ang)))
        sy = np.clip(yy + oy, 0, H - 1)
        sx = np.clip(xx + ox, 0, W - 1)
        code |= ((g[sy, sx] > g).astype(np.uint8) << i)
    return code


def lbp_histogram(gray: np.ndarray, radius: int = 1, points: int = 8,
                  bins: Optional[int] = None, normalize: bool = True) -> np.ndarray:
    code = lbp_codes(gray, radius, points)
    table = _uniform_table(points)
    codes = table[code]
    nb = int(bins or (points + 2))
    hist = np.bincount(codes.ravel(), minlength=nb).astype(np.float64)
    return hist / max(hist.sum(), 1.0) if normalize else hist


def histogram_entropy(hist: np.ndarray) -> float:
    h = np.asarray(hist, np.float64).ravel()
    h = h / max(h.sum(), 1e-12)
    nz = h[h > 0]
    return float(-(nz * np.log2(nz)).sum())


def glcm(gray: np.ndarray, levels: int = 16, distance: int = 1) -> Dict[str, float]:
    """Grey-level co-occurrence statistics averaged over 4 directions."""
    g = to_gray(gray)
    q = np.clip((g.astype(np.float32) / 256.0 * levels).astype(np.int32), 0, levels - 1)
    H, W = q.shape
    dirs = [(0, distance), (distance, 0), (distance, distance), (-distance, distance)]
    acc = np.zeros((levels, levels), np.float64)
    for dy, dx in dirs:
        y0, y1 = max(0, -dy), min(H, H - dy)
        x0, x1 = max(0, -dx), min(W, W - dx)
        a = q[y0:y1, x0:x1]
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        acc += np.bincount((a.ravel() * levels + b.ravel()),
                           minlength=levels * levels).reshape(levels, levels)
    acc = acc + acc.T          # symmetrise: P(i,j) + P(j,i) for d and -d
    p = acc / max(acc.sum(), 1.0)
    i = np.arange(levels, dtype=np.float64)
    I, J = np.meshgrid(i, i, indexing="ij")
    mu_i = float((p * I).sum())
    mu_j = float((p * J).sum())
    var_i = max(float((p * (I - mu_i) ** 2).sum()), 1e-12)
    var_j = max(float((p * (J - mu_j) ** 2).sum()), 1e-12)
    return {
        "glcm_contrast": round(float((p * (I - J) ** 2).sum()), 4),
        "glcm_homogeneity": round(float((p / (1.0 + (I - J) ** 2)).sum()), 4),
        "glcm_energy": round(float((p ** 2).sum()), 6),
        "glcm_entropy": round(float(-(p[p > 0] * np.log2(p[p > 0])).sum()), 4),
        "glcm_correlation": round(float((p * (I - mu_i) * (J - mu_j)).sum() /
                                        np.sqrt(var_i * var_j)), 4),
    }


def gabor_energy(gray: np.ndarray, frequencies: Sequence[float] = (0.08, 0.16),
                 thetas: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
                 sigma: float = 3.0) -> Dict[str, float]:
    """Mean squared response of a Gabor bank -> oriented texture strength.

    A canvas support or an engraved hatch pattern produces a strong response
    at one orientation; a plain paper stain does not.  Used as the
    'directional wear' indicator.
    """
    g = to_gray(gray).astype(np.float32)
    g = g - float(np.mean(g))
    out: Dict[str, float] = {}
    for f in frequencies:
        best = 0.0
        for t in thetas:
            ksize = int(max(6 * sigma, 7))
            if ksize % 2 == 0:
                ksize += 1
            try:
                ker = cv2.getGaborKernel((ksize, ksize), sigma, np.deg2rad(float(t)),
                                         1.0 / max(float(f), 1e-3), 0.56, 0,
                                         cv2.CV_32F)
            except cv2.error:
                continue
            resp = cv2.filter2D(g, cv2.CV_32F, ker)
            best = max(best, float(np.mean(resp * resp)))
        out[f"gabor_energy_f{f:.2f}"] = round(best, 4)
    return out


def dog_detail(gray: np.ndarray, sigma1: float = 0.8, sigma2: float = 2.0) -> Dict[str, float]:
    """Difference-of-Gaussians mid-band energy - micro-detail / sharpness."""
    g = to_gray(gray).astype(np.float32)
    band = cv2.GaussianBlur(g, (0, 0), sigma1) - cv2.GaussianBlur(g, (0, 0), sigma2)
    return {"dog_energy": round(float(np.mean(band * band)), 3),
            "dog_rms": round(float(np.sqrt(np.mean(band * band))), 4)}


def local_contrast_rms(gray: np.ndarray, ksize: int = 5) -> float:
    g = to_gray(gray).astype(np.float32)
    mean = cv2.boxFilter(g, -1, (ksize, ksize))
    return round(float(np.sqrt(np.mean((g - mean) ** 2))), 3)


def texture_bundle(gray: np.ndarray) -> Dict[str, float]:
    """One call -> the whole texture fingerprint used by the report table."""
    d: Dict[str, float] = {}
    d.update(glcm(gray))
    d.update(dog_detail(gray))
    d["lbp_entropy"] = round(histogram_entropy(lbp_histogram(gray)), 4)
    d["local_contrast_rms"] = local_contrast_rms(gray)
    d.update(gabor_energy(gray))
    return d


def ink_density_profile(gray: np.ndarray, mask: np.ndarray,
                        axis: int = 0, bins: int = 24) -> Dict[str, object]:
    """Row/column ink-density profile - evidence for baseline alignment and for
    'text present in this band' (used by the readability measure)."""
    m = (mask > 0).astype(np.float32)
    proj = m.sum(axis=axis)
    if bins > 1 and proj.size > bins:
        edges = np.linspace(0, proj.size, bins + 1).astype(int)
        proj = np.array([proj[edges[i]:edges[i + 1]].sum() for i in range(bins)])
    peak = float(proj.max()) if proj.size else 0.0
    # simple local-maximum counter above 25% of the peak -> estimated line count
    padded = np.concatenate([[0.0], proj, [0.0]])
    is_peak = (padded[1:-1] > padded[:-2]) & (padded[1:-1] >= padded[2:]) & \
              (padded[1:-1] > 0.25 * max(peak, 1e-6))
    return {"profile": [round(float(v), 1) for v in proj],
            "peaks": int(is_peak.sum()),
            "peak": round(peak, 1), "mean": round(float(proj.mean()), 2)}
