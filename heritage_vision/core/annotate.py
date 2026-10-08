"""
heritage_vision.core.annotate
=============================

Visual decision output: the annotated image is what a user (and the examiner)
looks at first, so it must carry the *decision*, not just pretty colours.

Layout of the overlay
    top bar      : filename + PASS/WARN/FAIL + condition + quality
    left column  : numeric summary (damage %, mm2, region count, classes)
    regions      : contour outline + bbox + severity badge for the top N
                   highest-severity findings (drawing all 400 foxing spots
                   would be unreadable)
    colours      : class-coded (crack = red, stain = orange, missing = magenta,
                   fold = yellow), ROI = cyan dashed, text = green
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

CLASS_COLORS = {
    "crack/tear": (40, 40, 235),
    "fold/crease": (0, 210, 230),
    "foxing/stain": (0, 140, 255),
    "stain-complex": (0, 100, 255),
    "ink-blodge": (200, 60, 220),
    "abrasion": (200, 200, 60),
    "damage-other": (150, 150, 150),
}
ROI_COLOR = (255, 200, 60)
TEXT_COLOR = (60, 200, 60)

_VERDICT_COLOR = {"PASS": (60, 190, 60), "WARN": (0, 170, 255),
                  "FAIL": (40, 40, 235)}


def _font(scale: float, thickness: int):
    return cv2.FONT_HERSHEY_SIMPLEX, scale, thickness


def draw_report(image: np.ndarray, damage, result, max_regions: int = 24,
                show_text_boxes: bool = False, show_legend: bool = True
                ) -> np.ndarray:
    """Compose the annotated view from the pipeline outputs."""
    img = image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.uint8).copy()
    h, w = img.shape[:2]
    s = max(1.0, min(w, h) / 900.0)          # scale everything with the image
    line = max(1, int(round(2 * s)))
    fs = 0.42 * s
    th = max(1, int(round(1.2 * s)))

    # ---- damage overlay (semi-transparent fill + outline) ---------------
    masks = getattr(damage, "masks", {}) or {}
    overlay = img.copy()
    for key, colour in (("stain", CLASS_COLORS["foxing/stain"]),
                        ("crack", CLASS_COLORS["crack/tear"]),
                        ("missing", (200, 60, 220))):
        m = masks.get(key)
        if m is not None and np.any(m):
            overlay[m > 0] = colour
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, dst=img)

    # ---- ROI ------------------------------------------------------------
    roi = getattr(result, "roi", None)
    if roi is not None and np.any(roi):
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            _dash_rect(img, (x, y, bw, bh), ROI_COLOR, line, int(10 * s))

    # ---- per-region boxes ------------------------------------------------
    cond = getattr(result, "condition", None)
    for r in (damage.regions[:max_regions] if damage else []):
        x, y, bw, bh = r.bbox
        col = CLASS_COLORS.get(r.class_name, (150, 150, 150))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, line)
        tag = f"{r.class_name} {r.area_mm2:.1f}mm2 s={r.severity:.2f}"
        ty = max(int(12 * s), y - int(5 * s))
        _text(img, tag, (x, ty), fs, col, th)

    # ---- text layer -------------------------------------------------------
    if show_text_boxes and getattr(result, "text_mask", None) is not None:
        tm = result.text_mask
        cnts, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts[:400]:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), TEXT_COLOR, 1)

    # ---- header ----------------------------------------------------------
    if show_legend:
        _header(img, result, s, line, fs, th)
    return img


def _header(img: np.ndarray, result, s: float, line: int, fs: float, th: int) -> None:
    h, w = img.shape[:2]
    bar_h = int(78 * s)
    bar = np.zeros((bar_h, w, 3), np.uint8)
    cv2.rectangle(bar, (0, 0), (w, bar_h), (25, 25, 25), -1)
    cond = getattr(result, "condition", None)
    qual = getattr(result, "quality", None)
    dmg = getattr(result, "damage", None)
    verdict = cond.pass_fail if cond else "N/A"
    col = _VERDICT_COLOR.get(verdict, (200, 200, 200))
    y1, y2 = int(26 * s), int(56 * s)
    _text(bar, f"{result.name}", (int(10 * s), y1), fs * 1.05, (235, 235, 235), th)
    if dmg:
        _text(bar, f"damage {dmg.damage_pct:.2f}% / {dmg.damage_mm2:.1f}mm2 "
                   f"({dmg.n_damage_total} regions)", (int(10 * s), y2),
              fs, (200, 220, 200), th)
    if cond:
        txt = f"CONDITION {cond.score:.1f} {cond.severity_class}"
        _text(bar, txt, (int(w * 0.60), y1), fs, (255, 255, 255), th)
    if qual:
        _text(bar, f"QUALITY {qual.score:.1f} {qual.grade}", (int(w * 0.60), y2),
              fs, (255, 255, 255), th)
    cv2.rectangle(bar, (w - int(112 * s), int(8 * s)), (w - int(8 * s), bar_h - int(8 * s)),
                  col, -1)
    _text(bar, verdict, (w - int(104 * s), int(y1 - int(2 * s)) + int(15 * s)),
          fs * 1.15, (10, 10, 10), max(1, int(th * 1.4)))
    img[0:bar_h] = bar


def _text(img: np.ndarray, s: str, org, scale: float, color, thickness: int) -> None:
    (tw, tH), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    if x + tw > img.shape[1] - 2:
        x = max(2, img.shape[1] - tw - 2)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _dash_rect(img: np.ndarray, box, color, thickness: int, dash: int) -> None:
    x, y, w, h = [int(v) for v in box]
    pts = [(x, y, x + w, y), (x + w, y, x + w, y + h),
           (x + w, y + h, x, y + h), (x, y + h, x, y)]
    for x0, y0, x1, y1 in pts:
        n = max(int(np.hypot(x1 - x0, y1 - y0) / max(dash, 1)), 1)
        for i in range(0, n, 2):
            t0 = i / n
            t1 = min((i + 1) / n, 1.0)
            cv2.line(img, (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0)),
                     (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1)),
                     color, thickness, cv2.LINE_AA)


def side_by_side(a: np.ndarray, b: np.ndarray, label_a: str = "original",
                 label_b: str = "processed", gap: int = 12,
                 height: Optional[int] = None) -> np.ndarray:
    """Before/after panel used in the report and in the GUI comparison tab."""
    a = a if a.ndim == 3 else cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    b = b if b.ndim == 3 else cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    if height:
        a = cv2.resize(a, (int(a.shape[1] * height / a.shape[0]), height))
        b = cv2.resize(b, (int(b.shape[1] * height / b.shape[0]), height))
    h = max(a.shape[0], b.shape[0])
    wa, wb = a.shape[1], b.shape[1]
    canvas = np.full((h + 26, wa + wb + gap, 3), 30, np.uint8)
    canvas[26:26 + a.shape[0], 0:wa] = a[:h]
    canvas[26:26 + b.shape[0], wa + gap:wa + gap + wb] = b[:h]
    cv2.putText(canvas, label_a, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, label_b, (wa + gap + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def mask_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 0, 235),
                 alpha: float = 0.4) -> np.ndarray:
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = img.copy()
    sel = mask > 0
    if np.any(sel):
        out[sel] = (np.array(color, np.float32) * alpha +
                    out[sel].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return out
