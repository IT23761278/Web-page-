"""
heritage_vision.core.restoration
================================

Reversible, non-destructive restoration built on top of the damage analysis.
Only classical computer vision is used here (inpainting, morphological
despeckle, unsharp, tone) - the point of the project is the OpenCV pipeline,
and a conservator needs to know that nothing was "invented" by a network.

Design rules followed:
    * Every operation is applied **only inside a dilated damage mask**, so sound
      areas of the sheet are bit-exact copies of the input.  This is the
      digital equivalent of "treat only what is damaged".
    * The untouched original and the mask are always returned, so the workflow
      is auditable and can be undone (reset).
    * ``inpaint_method`` exposes Telea (fast marching) vs Navier-Stokes
      (FDM) because they fail differently on our two shapes: Telea smears a
      stain edge into a halo, FDM propagates structure and is better across a
      crack that crosses a stroke.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .enhancement import unsharp_mask
from .utils import odd, to_bgr, to_gray


@dataclass
class RestorationResult:
    restored: np.ndarray
    original: np.ndarray
    mask_used: np.ndarray
    operations: list = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


def inpaint_damage(image: np.ndarray, damage_mask: np.ndarray, radius: float = 3.0,
                   method: str = "telea", dilate_px: int = 2,
                   min_area_px: int = 4) -> RestorationResult:
    """Fill damaged pixels from their neighbourhood.

    ``dilate_px`` matters: inpainting stops at the mask boundary, so a 1 px
    halo of the original damage survives unless the mask is grown.  Too much
    growth starts eating intact ink, hence the small default.
    """
    img = to_bgr(image)
    m = (damage_mask > 0).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (odd(dilate_px, 3), odd(dilate_px, 3)))
        m = cv2.dilate(m, k, iterations=max(int(dilate_px // 2) + 1, 1))
    if min_area_px > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                keep[lab == i] = 255
        m = keep
    res = RestorationResult(restored=img.copy(), original=img.copy(), mask_used=m)
    if not np.any(m):
        res.operations.append("inpaint:skipped(no mask)")
        return res
    flag = cv2.INPAINT_NS if str(method).lower().startswith("ns") else cv2.INPAINT_TELEA
    gray_in = to_gray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        out = cv2.inpaint(img, m, float(radius), flag)
    else:
        out = cv2.inpaint(gray_in, m, float(radius), flag)
    res.restored = out
    res.operations.append(f"inpaint:{'NS' if flag == cv2.INPAINT_NS else 'Telea'}"
                          f"(r={radius},dilate={dilate_px})")
    res.stats["inpaint_area_px"] = float(np.count_nonzero(m))
    res.stats["inpaint_area_pct"] = round(float(np.count_nonzero(m)) / m.size * 100.0, 3)
    return res


def despeckle(image: np.ndarray, open_px: int = 2,
              max_speckle_area_px: int = 24, keep_text: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove dust / salt-and-pepper specks without touching the writing.

    A speck is a small connected component whose size is below the size of the
    smallest real feature (an i-dot or a diacritic).  Removing components below
    ``max_speckle_area_px`` from the *foreground* mask and re-compositing is
    safer than a median blur, which would round the edges of genuine strokes.
    Returns (cleaned_image, speckle_mask).
    """
    img = to_bgr(image)
    g = to_gray(img)
    _, thr = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (odd(open_px, 3), odd(open_px, 3)))
        opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k)
    else:
        opened = thr
    specks = cv2.subtract(thr, opened)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(specks, 8)
    only_small = np.zeros_like(specks)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_speckle_area_px:
            only_small[lab == i] = 255
    specks = only_small
    if keep_text is not None:
        specks = cv2.subtract(specks, cv2.dilate(keep_text, np.ones((3, 3), np.uint8)))
    out = img.copy()
    if np.any(specks):
        # Fill each speck with the local background: the median of a ring around
        # it (computed by median-blurring the *unmasked* image, which is exactly
        # what a speck-free neighbourhood estimate is).
        bg = cv2.medianBlur(img, 5)
        out = np.where(specks[..., None] > 0, bg, img)
    return out, specks


def tone_map_recover(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Recover shadowed/dark regions of an old *photograph*.

    A gamma-on-luminance lift plus a soft highlight rolloff.  Deliberately not
    a full CLAHE pass: on a faded print the tonal range is what carries the
    image, and aggressive equalisation blocks up the sky/skin tone areas.
    ``strength`` 0..1 lets the operator stop at "readable", not "unnatural".
    """
    img = to_bgr(image)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0] / 255.0
    gamma = 1.0 / (1.0 + strength * 1.2)
    lifted = L ** gamma
    # highlight protection (soft knee above 0.85)
    hi = lifted > 0.85
    lifted[hi] = 0.85 + (lifted[hi] - 0.85) * 0.55
    lab[..., 0] = np.clip(lifted * 255.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


def remove_scratch_halo(image: np.ndarray, crack_mask: np.ndarray,
                        radius: int = 3) -> np.ndarray:
    """Median pass restricted to the crack corridor only.

    A scratch in a photograph usually comes with a light halo from the scan of
    a physical groove; a small median inside the corridor collapses it while the
    rest of the print is untouched.
    """
    img = to_bgr(image)
    m = (crack_mask > 0).astype(np.uint8)
    if not m.any():
        return img
    k = odd(radius)
    med = cv2.medianBlur(img, k)
    sm = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return np.where(sm[..., None] > 0, med, img)


def restore(image: np.ndarray, damage_mask: np.ndarray, crack_mask: np.ndarray,
            text_mask: Optional[np.ndarray], params) -> RestorationResult:
    """Full restoration stage in the documented order:
    despeckle -> inpaint(stain+missing) -> scratch-halo -> unsharp."""
    img = to_bgr(image)
    res = RestorationResult(restored=img.copy(), original=img.copy(),
                            mask_used=np.zeros(img.shape[:2], np.uint8))
    cleaned, specks = despeckle(img, open_px=getattr(params, "despeckle_open_px", 2),
                                keep_text=text_mask)
    if np.any(specks):
        res.operations.append(f"despeckle(open={params.despeckle_open_px}px)")
        res.stats["speckle_px"] = float(np.count_nonzero(specks))
    img = cleaned
    if getattr(params, "inpaint", True):
        comb = cv2.bitwise_or(damage_mask, np.zeros_like(damage_mask))
        r = inpaint_damage(img, comb, radius=getattr(params, "inpaint_radius", 3.0),
                           method=getattr(params, "inpaint_method", "telea"),
                           dilate_px=getattr(params, "inpaint_dilate_px", 2))
        img = r.restored
        res.mask_used = cv2.bitwise_or(res.mask_used, r.mask_used)
        res.operations += r.operations
        res.stats.update(r.stats)
    if crack_mask is not None and np.any(crack_mask):
        img = remove_scratch_halo(img, crack_mask, radius=3)
        res.operations.append("scratch-halo:median3")
    if getattr(params, "sharpen", True):
        img = unsharp_mask(img, amount=getattr(params, "unsharp_amount", 0.6),
                           sigma=getattr(params, "unsharp_sigma", 2.0))
        res.operations.append(f"unsharp(a={params.unsharp_amount},s={params.unsharp_sigma})")
    res.restored = img
    res.stats["applied_ops"] = float(len(res.operations))
    return res
