"""
tools/make_dataset.py
=====================

Dataset builder for Part A ("collect or create a dataset of >= 30 images, at
least 10 produced by the student") and for the controlled test images required
by Part D.

It has two jobs:

1. ``--render``  create *clean* synthetic source pages (a manuscript page, a
   typewritten report, a printed leaf, a plan/drawing sheet, a photograph plate)
   so that a pixel-exact "no damage" reference and an exact ink mask exist.
   Without such a reference it is impossible to measure PSNR/SSIM or an area
   error honestly on heritage material.

2. ``--degrade`` apply a documented degradation model to a clean image and
   record, per case, the ground-truth damage mask, ink mask, parameters and
   sidecar ``*.gt.json``.  The degradation model reproduces what a real sheet
   of 19th-century paper or a silver-gelatin print actually shows:

      paper yellowing (lignin oxidation)      Lab b* shift + low-freq mottling
      uneven lighting / scanner vignetting    radial + linear gradient
      foxing spots (iron-gall + humidity)     brown blobs, soft edge, 1-4 mm
      water tide lines                        large region + dark border ring
      cracks / tears                          thin curvilinear dark ridges
      loss of material                        bright holes inside the sheet
      ink corrosion / bleed-through           dark halo around text
      dust and specks                         salt & pepper
      scanner blur, film grain, JPEG blocking Gaussian blur, additive noise,
                                              quality-60 JPEG
      acquisition rotation / perspective      deskew + rectification test

Usage
-----
    python -m heritage_vision.tools.make_dataset --render   --out dataset/clean
    python -m heritage_vision.tools.make_dataset --degrade --in dataset/clean \
        --out dataset/student_created --cases 3 --seed 7

Every written image is registered in ``dataset_manifest.csv`` (source,
resolution, format, purpose) as required by Part A.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:                                                   # pragma: no cover
    _HAVE_PIL = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heritage_vision.core.utils import save_image                    # noqa: E402

GT_SUFFIX = ".gt.json"

# ---------------------------------------------------------------------------
# small procedural helpers
# ---------------------------------------------------------------------------

WORDS = ("the quick brown fox jumps over lazy dogs while ancient scribes record "
         "chronicles of harvest tide and merchant guilds in marginal notes upon "
         "this vellum leaf of a forgotten archive volume numbered nine").split()


def _font(size: int):
    if not _HAVE_PIL:
        return None
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/georgia.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_width(text: str, f) -> float:
    if f is None or not _HAVE_PIL:
        return len(text) * 8.0
    try:
        return f.getlength(text)
    except Exception:                                  # older Pillow
        return f.getsize(text)[0]


def _render_line(text: str, f, x0: int, width: int, height: int) -> np.ndarray:
    """Rasterise one text line into a float32 coverage map in [0,1]."""
    if not _HAVE_PIL:
        out = np.zeros((height, width), np.float32)
        cv2.putText(out, text, (x0, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    1.0, 1, cv2.LINE_AA)
        return out
    pad = 6
    pil = Image.new("L", (width + pad, height + pad), 0)
    d = ImageDraw.Draw(pil)
    d.text((x0, pad // 2), text, font=f, fill=255)
    g = np.asarray(pil, np.float32) / 255.0
    return g


def fbm(shape: Tuple[int, int], octaves: int = 4, base: int = 64,
        rng: Optional[random.Random] = None, persistence: float = 0.55
        ) -> np.ndarray:
    """Fractal Brownian motion in [0,1] from repeated upsampled noise.

    Used for paper mottling / fibre texture: a few octaves of blurred random
    noise is a surprisingly good model of handmade paper and it is cheap.
    """
    rng = rng or random.Random(0)
    H, W = shape
    acc = np.zeros((H, W), np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        size = max(2, int(base / (2 ** o)))
        sh = (max(2, int(round(H / size))), max(2, int(round(W / size))))
        n = np.array([[rng.random() for _ in range(sh[1])] for _ in range(sh[0])],
                     np.float32)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        acc += amp * (n - n.mean())
        tot += amp
        amp *= persistence
    acc = acc / max(tot, 1e-6)
    acc = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-6)
    return acc.astype(np.float32)


def fibre_texture(shape: Tuple[int, int], rng: random.Random, n: int = 900,
                  length: int = 60) -> np.ndarray:
    """Short horizontal strokes -> laid-paper / canvas weave look."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    for _ in range(n):
        y = rng.randrange(H)
        x = rng.randrange(W)
        L = rng.randint(length // 3, length)
        a = rng.uniform(-0.12, 0.12)
        dx, dy = int(round(L * math.cos(a))), int(round(L * math.sin(a)))
        cv2.line(out, (x, y), (x + dx, y + dy), rng.uniform(0.25, 0.8), 1,
                 cv2.LINE_AA)
    return cv2.GaussianBlur(out, (3, 3), 0)


# ---------------------------------------------------------------------------
# 1. CLEAN SOURCE RENDERING
# ---------------------------------------------------------------------------


def render_clean_page(kind: str, size: Tuple[int, int], seed: int) -> Dict[str, np.ndarray]:
    """Create one synthetic 'perfect' sheet.

    Returns dict with ``image`` (uint8 BGR, ink on paper, already toned),
    ``ink`` (exact ink mask) and ``sheet`` (page mask).  Everything is derived
    from a fixed seed so the dataset is reproducible - a requirement the report
    states explicitly.
    """
    W, H = size
    rng = random.Random(seed)
    paper_col = np.array([236, 240, 246], np.float32)          # BGR cream
    img = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        img[..., c] = paper_col[c] + (fbm((H, W), 4, 72, rng) - 0.5) * 14.0
    img += (fibre_texture((H, W), rng, int(W * H / 2600), max(W // 14, 30))[..., None]) \
        * np.array([[-3.0, -2.5, -2.0]], np.float32)

    ink = np.zeros((H, W), np.uint8)
    draw_layer = np.zeros((H, W), np.float32)
    if kind == "photo_plate":
        # A "photographic plate": a simple architectural study, so the photo
        # branch of the pipeline has a clean reference as well.
        base = 138 + rng.random() * 26          # mid-grey print density
        tone = np.zeros((H, W), np.float32)
        # sky gradient + building silhouette + horizon
        yy, xx = np.mgrid[0:H, 0:W]
        tone = (1 - yy / H) * 52 + base                       # sky, brighter at top
        bx = int(W * rng.uniform(0.18, 0.28))
        bw = int(W * rng.uniform(0.35, 0.6))
        bh = int(H * rng.uniform(0.35, 0.6))
        tone[H - bh:H, bx:bx + bw] = base - 46                # facade in shade
        for i in range(5):                                     # windows catch light
            wx = bx + int(bw * (i + 0.5) / 5.5)
            tone[H - int(bh * 0.82):H - int(bh * 0.35), wx - 9:wx + 9] = base + 34
        tone[:int(H * 0.90)] = tone[:int(H * 0.90)]           # no-op, keeps shape
        tone += (fbm((H, W), 5, 40, rng) - 0.5) * 22
        tone += np.random.default_rng(seed).normal(0, 6, (H, W))
        img = np.stack([tone] * 3, -1).astype(np.float32)
        img[..., 2] *= 1.04                    # a warm print, not neutral
        img[..., 0] *= 0.96
        ink = (tone < base - 20).astype(np.uint8) * 255
        sheet = np.full((H, W), 255, np.uint8)
        return {"image": np.clip(img, 0, 255).astype(np.uint8),
                "ink": ink, "sheet": sheet}

    margin = int(W * 0.09)
    line_h = int(H / rng.uniform(18, 26))
    content_w = W - 2 * margin
    y = margin + rng.randint(0, line_h)
    block_top = y
    while y < H - margin:
        if rng.random() < 0.07:                       # paragraph break
            y += int(line_h * 0.85)
            continue
        fs = int(line_h * rng.uniform(0.62, 0.80))
        f = _font(fs)
        # word-wrap with the real font metrics so nothing is clipped
        words = [rng.choice(WORDS) for _ in range(200)]
        lines: List[str] = []
        cur = ""
        for wd in words:
            trial = wd if not cur else cur + " " + wd
            tw = _text_width(trial, f)
            if tw > content_w and cur:
                lines.append(cur)
                cur = wd
            else:
                cur = trial
            if len(lines) >= 1:
                break
        if cur:
            lines.append(cur)
        for ln in lines[:1]:
            g = _render_line(ln, f, margin, content_w, int(line_h * 1.5))
            shift = rng.randint(-1, 1)                 # baseline jitter
            hgt = min(g.shape[0], H - y - max(shift, 0))
            if hgt > 2:
                draw_layer[y + max(shift, 0):y + max(shift, 0) + hgt, :g.shape[1]] += \
                    g[:hgt]
        y += line_h
    if rng.random() < 0.6:                                    # ruled frame
        cv2.rectangle(draw_layer, (margin - 30, block_top - 20),
                      (W - margin + 30, H - margin + 20), 0.6, 2)
        ink = cv2.rectangle(ink, (margin - 30, block_top - 20),
                            (W - margin + 30, H - margin + 20), 255, 2)
    # iron-gall ink: brown-black, density varies with the paper texture
    density = np.clip(draw_layer, 0, 1)
    # real iron-gall / carbon ink is dense in the middle of a stroke and lighter
    # at the edges, so the coverage map is gamma-boosted before compositing
    density = np.clip(density ** 0.65 * (0.80 + 0.20 * fbm((H, W), 3, 120, rng)), 0, 1)
    ink_col = np.array([22, 24, 38], np.float32)               # BGR dark brown-black
    img = img * (1 - density[..., None]) + ink_col[None, None, :] * density[..., None]
    sheet = np.full((H, W), 255, np.uint8)
    return {"image": np.clip(img, 0, 255).astype(np.uint8),
            "ink": (density > 0.20).astype(np.uint8) * 255, "sheet": sheet}


# ---------------------------------------------------------------------------
# 2. DEGRADATION MODEL
# ---------------------------------------------------------------------------


def _blob_mask(shape, cx, cy, rx, ry, angle, roughness=0.35, rng=None):
    """Irregular soft-edged blob (foxing spot / water stain)."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    rng = rng or random.Random(0)
    n = 46
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = 1.0 + roughness * (math.sin(3 * t + rng.random() * 2) * 0.5 +
                               (rng.random() - 0.5) * 0.8)
        pts.append((int(cx + rx * r * math.cos(t)), int(cy + ry * r * math.sin(t))))
    pts = np.array(pts, np.int32)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    pts = cv2.transform(pts.reshape(-1, 1, 2).astype(np.float32), M).astype(np.int32)
    cv2.fillPoly(m, [pts.reshape(-1, 2)], 255)
    return m


def _polyline_curve(shape, rng: random.Random, n_pts: int = 14,
                    min_len: int = 80) -> Tuple[np.ndarray, int]:
    """Wiggly crack path with a locally consistent direction."""
    H, W = shape
    x, y = rng.uniform(0.1 * W, 0.9 * W), rng.uniform(0.1 * H, 0.9 * H)
    ang = rng.uniform(0, 2 * math.pi)
    pts = [(int(x), int(y))]
    total = 0.0
    for _ in range(n_pts):
        step = rng.uniform(min_len * 0.35, min_len * 0.8)
        ang += rng.uniform(-0.5, 0.5)
        x += step * math.cos(ang)
        y += step * math.sin(ang)
        total += step
        if not (0 < x < W and 0 < y < H):
            break
        pts.append((int(x), int(y)))
    return np.array(pts, np.int32), int(total)


def _desk_background(shape: Tuple[int, int], rng: random.Random) -> np.ndarray:
    """The surface the sheet lies on (scanner bed / archival mount board).

    Modelling it matters for two reasons: it makes the page mask meaningful (the
    sheet no longer fills the frame, so "loss of material" can be detected as
    "the sheet outline is not convex" / "background colour inside the sheet"),
    and it is what a photograph of a document on a table actually looks like.
    """
    H, W = shape
    base = np.array([rng.randint(26, 52)] * 3, np.float32)
    d = np.zeros((H, W, 3), np.float32)
    d[:] = base
    tex = fbm((H, W), 3, 90, rng)
    d += (tex - 0.5)[..., None] * 12.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1.0 - 0.35 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 3.0
    d *= vig[..., None]
    return np.clip(d, 0, 255).astype(np.float32)


def degrade(clean: np.ndarray, ink: np.ndarray, sheet: np.ndarray, seed: int,
            severity: str = "moderate", photo: bool = False,
            skew: Optional[float] = None, perspective: bool = False,
            desk: bool = True, pad_frac: float = 0.035,
            return_canvas: bool = False) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return (degraded_image, gt_dict).  All GT masks are 0/255 uint8.

    Order of operations is the physical order, and it is why the result looks
    convincing:

        1. chemical damage to the paper  (foxing, tide lines, fading of the
           ink, yellowing, uneven lighting while the sheet was photographed)
        2. mechanical damage  (cracks, tears, loss of material) - material loss
           is punched *after* the discoloration because a hole is the absence of
           paper, so it shows the mount colour, undimmed by 150 years of aging
        3. acquisition  (desk blur, sensor grain, JPEG blocking, rotation or
           perspective of the camera)

    Every parameter that was used is written into the returned dict, so a test
    case can be regenerated exactly and the measured damage area is known.
    """
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 31))
    H, W = clean.shape[:2]
    n_stain = {"light": 3, "moderate": 9, "severe": 22}[severity]
    n_crack = {"light": 1, "moderate": 3, "severe": 7}[severity]
    n_hole = {"light": 0, "moderate": 1, "severe": 3}[severity]
    fade = {"light": 0.25, "moderate": 0.5, "severe": 0.78}[severity]
    noise = {"light": 4.0, "moderate": 9.0, "severe": 17.0}[severity]
    blur = {"light": 0.0, "moderate": 0.7, "severe": 1.35}[severity] + (0.4 if photo else 0.0)

    img = clean.astype(np.float32).copy()
    gt_stain = np.zeros((H, W), np.uint8)
    gt_crack = np.zeros((H, W), np.uint8)

    # -- 1. chemical damage -------------------------------------------------
    stain_col = np.array([60, 105, 150], np.float32)              # BGR warm brown
    for _ in range(n_stain):
        big = rng.random() < 0.2
        # size relative to the *short* side, so a landscape scan gets stains of
        # the same physical size as a portrait one (and the centre search below
        # cannot degenerate into an empty range, which crashed a 289 px page)
        rx = int(min(H, W) * rng.uniform(0.010, 0.030) * (4 if big else 1))
        ry = int(rx * rng.uniform(0.7, 1.3))
        rx = min(max(rx, 2), max((W - 3) // 2, 2))
        ry = min(max(ry, 2), max((H - 3) // 2, 2))
        cx = rng.randint(rx, W - rx)
        cy = rng.randint(ry, H - ry)
        m = _blob_mask((H, W), cx, cy, rx, ry, rng.uniform(0, 180),
                       roughness=rng.uniform(0.15, 0.5), rng=rng)
        m = cv2.GaussianBlur(m, (0, 0), max(rx * 0.14, 1.0))
        a = (m.astype(np.float32)[..., None] / 255.0) * rng.uniform(0.35, 0.8)
        img = img * (1 - a) + stain_col[None, None, :] * a * 0.75
        gt_stain = cv2.bitwise_or(gt_stain, (m > 128).astype(np.uint8) * 255)
    if rng.random() < (0.35 if severity != "light" else 0.1):      # water tide line
        cx, cy = rng.randint(W // 4, 3 * W // 4), rng.randint(H // 4, 3 * H // 4)
        r = int(min(H, W) * rng.uniform(0.18, 0.30))
        m = _blob_mask((H, W), cx, cy, r, int(r * rng.uniform(0.7, 1.1)),
                       rng.uniform(0, 180), 0.3, rng)
        ring = cv2.subtract(m, cv2.erode(m, np.ones((9, 9), np.uint8)))
        mm = cv2.GaussianBlur(m, (0, 0), r * 0.06).astype(np.float32) / 255.0
        img = img * (1 - 0.20 * mm[..., None]) + stain_col[None, None, :] * 0.20 * mm[..., None]
        rr = cv2.GaussianBlur(ring, (0, 0), 4).astype(np.float32) / 255.0
        img = img * (1 - 0.55 * rr[..., None]) + (stain_col * 0.55)[None, None, :] * rr[..., None]
        gt_stain = cv2.bitwise_or(gt_stain, (m > 128).astype(np.uint8) * 255)
    inkf = (ink > 0).astype(np.float32)
    inkf = cv2.GaussianBlur(inkf, (0, 0), 0.8)[..., None]
    img = img + (255.0 - img) * (fade * 0.62) * inkf                # ink fading
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2Lab).astype(np.float32)
    mottle = fbm((H, W), 3, 140, rng)
    lab[..., 2] = np.clip(lab[..., 2] + 10 + 16 * mottle, 0, 255)   # yellowing
    lab[..., 1] = np.clip(lab[..., 1] + 3, 0, 255)
    lab[..., 0] = np.clip(lab[..., 0] * (0.86 + 0.2 * mottle), 0, 255)
    img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 1.0 - 0.22 * (((xx - rng.uniform(0.2, 0.8) * W) / W) ** 2 +
                         ((yy - rng.uniform(0.2, 0.8) * H) / H) ** 2)
    lin = 1.0 - 0.10 * (xx / W)
    img *= (grad * lin)[..., None]                                    # uneven light

    # -- 2. mechanical damage ----------------------------------------------
    for _ in range(n_crack):
        pts, length = _polyline_curve((H, W), rng, n_pts=rng.randint(6, 20),
                                       min_len=int(min(H, W) * 0.10))
        if len(pts) < 2:
            continue
        wdt = rng.choice([1, 2, 2, 3])
        cm = np.zeros((H, W), np.uint8)
        cv2.polylines(cm, [pts], False, 255, wdt, cv2.LINE_AA)
        img = img * (1 - 0.75 * (cm[..., None] / 255.0)) + \
            np.array([20.0, 22.0, 26.0])[None, None, :] * 0.75 * (cm[..., None] / 255.0)
        bright = cv2.subtract(cv2.dilate(cm, np.ones((3, 3), np.uint8)), cm)
        img += (bright[..., None].astype(np.float32) / 255.0) * 18.0
        gt_crack = cv2.bitwise_or(gt_crack, cm)

    # ---- the sheet is placed on the desk; GT masks move to canvas size ----
    pad = max(int(round(min(H, W) * pad_frac)), 4) if desk else 0
    ch, cw = H + 2 * pad, W + 2 * pad
    canvas = _desk_background((ch, cw), rng) if desk else np.clip(img, 0, 255).astype(np.float32)
    # the surface *before* the page is pasted on it - a hole shows exactly this
    desk_under = canvas.copy()
    canvas[pad:pad + H, pad:pad + W] = np.clip(img, 0, 255).astype(np.float32)

    def to_canvas(m: np.ndarray) -> np.ndarray:
        out = np.zeros((ch, cw), m.dtype)
        out[pad:pad + m.shape[0], pad:pad + m.shape[1]] = m
        return out

    sheet_mask = np.zeros((ch, cw), np.uint8)
    sheet_mask[pad:pad + H, pad:pad + W] = 255
    gt_stain = to_canvas(gt_stain)
    gt_crack = to_canvas(gt_crack)
    gt_ink = to_canvas(ink)

    # material loss: punched last, so it shows the mount board undimmed
    gt_hole = np.zeros((ch, cw), np.uint8)
    for _ in range(n_hole):
        cx = rng.randint(max(W // 8, 4), 7 * W // 8) + pad
        cy = rng.randint(max(H // 8, 4), 7 * H // 8) + pad
        r = int(min(H, W) * rng.uniform(0.012, 0.04))
        m = _blob_mask((ch, cw), cx, cy, r, int(r * rng.uniform(0.6, 1.4)),
                       rng.uniform(0, 180), 0.5, rng)
        edge = cv2.dilate(m, np.ones((5, 5), np.uint8))
        # through the hole we see the mount board itself, with its own vignette
        # and texture - not a flat grey fill (an earlier version sampled the
        # *local* neighbourhood, which inside a page is paper, so the "hole"
        # came out brighter than the sheet and was undetectable by design)
        canvas = np.where(m[..., None] > 0,
                          np.clip(desk_under * 1.06 + 6.0, 0, 255), canvas)
        canvas = np.where(cv2.subtract(edge, m)[..., None] > 0, canvas * 0.72 + 46.0, canvas)
        gt_hole = cv2.bitwise_or(gt_hole, m)

    # -- 3. acquisition -----------------------------------------------------
    if blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas = canvas + np.random.default_rng(seed + 1).normal(
        0, noise, canvas.shape).astype(np.float32)
    if photo:                                    # silver-gelatin grain + scratches
        g = np.random.default_rng(seed + 2).normal(
            0, noise * 1.4, canvas.shape[:2]).astype(np.float32)
        canvas += g[..., None] * np.array([[[0.90, 0.95, 1.00]]], np.float32)
        for _ in range(rng.randint(2, 8)):
            y0 = rng.randint(0, canvas.shape[0] - 1)
            x0 = rng.randint(0, canvas.shape[1] - 1)
            L = rng.randint(20, min(canvas.shape[:2]) // 2)
            bright_line = rng.random() < 0.5      # a scratch reflects the light
            col = (250.0, 248.0, 246.0) if bright_line else (12.0, 12.0, 14.0)
            p1 = (x0, y0)
            p2 = ((x0 + rng.randint(-8, 8), y0 + L) if bright_line
                  else (x0 + L, y0 + rng.randint(-6, 6)))
            canvas = cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
    dust = rng.uniform(0.00012, 0.0006)           # airborne dust on the glass
    nd = int(ch * cw * dust)
    rg = np.random.default_rng(seed + 3)
    dy = rg.integers(0, ch, nd)
    dx = rg.integers(0, cw, nd)
    light = rg.random(nd) > 0.55
    flat = canvas.reshape(-1, 3).copy()
    flat[dy * cw + dx] = np.where(light[:, None] > 0, 250.0, 8.0)
    canvas = flat.reshape(canvas.shape).astype(np.float32)

    # -- 4. geometry: rotation and/or perspective of the whole frame --------
    angle = skew if skew is not None else rng.uniform(-3.2, 3.2)
    Mrot = np.float32([[1, 0, 0], [0, 1, 0]])
    if abs(angle) > 0.2:
        Mrot = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), angle, 1.0)
    Mp = None
    if perspective:
        src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
        jw = lambda: rng.uniform(-0.05, 0.05)
        dst = np.array([[jw() * cw, jw() * ch], [cw - 1 + jw() * cw, jw() * ch],
                        [cw - 1 + jw() * cw, ch - 1 + jw() * ch],
                        [jw() * cw, ch - 1 + jw() * ch]], np.float32)
        Mp = cv2.getPerspectiveTransform(src, dst)
    warp = Mp if Mp is not None else Mrot

    def wp(x: np.ndarray, interp: int, border=cv2.BORDER_CONSTANT, bg=0) -> np.ndarray:
        fn = cv2.warpPerspective if Mp is not None else cv2.warpAffine
        return fn(x, warp, (cw, ch), flags=interp, borderMode=border, borderValue=bg)

    img = wp(np.clip(canvas, 0, 255).astype(np.uint8), cv2.INTER_LINEAR)
    ok_, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                         62 if severity == "severe" else 78])
    if ok_:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    sheet_mask = wp(sheet_mask, cv2.INTER_NEAREST)
    gt_stain = wp(gt_stain, cv2.INTER_NEAREST)
    gt_crack = wp(gt_crack, cv2.INTER_NEAREST)
    gt_hole = wp(gt_hole, cv2.INTER_NEAREST)
    gt_ink = wp(gt_ink, cv2.INTER_NEAREST)

    # exact sheet quad in the final frame -> ground truth for rectification
    cnts, _ = cv2.findContours(sheet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        quad = approx.reshape(-1, 2).astype(np.float32) if len(approx) == 4 \
            else cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        corners = quad.tolist()

    gt_damage = cv2.bitwise_or(cv2.bitwise_or(gt_stain, gt_crack), gt_hole)
    gt_damage = cv2.morphologyEx(gt_damage, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    gt = {
        "gt_damage_mask": gt_damage, "gt_stain_mask": gt_stain,
        "clean_image_canvas": np.clip(canvas, 0, 255).astype(np.uint8),
        "gt_crack_mask": gt_crack, "gt_hole_mask": gt_hole,
        "gt_text_mask": gt_ink, "gt_sheet_mask": sheet_mask,
        "params": {"seed": seed, "severity": severity, "n_stain": n_stain,
                   "n_crack": n_crack, "n_hole": n_hole, "fade": fade,
                   "noise_sigma": noise, "blur_sigma": round(blur, 2),
                   "jpeg_quality": 62 if severity == "severe" else 78,
                   # for a perspective case the injected rotation is folded into
                   # the homography, so "the angle a deskew step should find" is
                   # not the number we asked for - record None and let the
                   # evaluator skip the deskew score for that case instead of
                   # scoring it against a wrong reference.
                   "skew_deg": (None if perspective else round(float(angle), 3)),
                   "perspective": perspective,
                   "desk": desk, "pad_px": pad, "photo": photo},
        "corners": np.asarray(corners, np.float32).reshape(-1, 2).tolist() if corners is not None else None,
    }
    if not return_canvas:
        # the reference canvas is 3 MB per case; callers that only need the
        # degraded image and the masks should not pay for it
        gt.pop("clean_image_canvas", None)
    return img, gt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_render(args) -> List[str]:
    out = args.out
    os.makedirs(out, exist_ok=True)
    written = []
    kinds = args.kinds.split(",") if args.kinds else \
        ["manuscript", "typed", "printed", "photo_plate"]
    for i, kind in enumerate(kinds):
        for size, tag in (((args.width, args.height), "a"),
                          ((int(args.width * 0.62), int(args.height * 0.62)), "b")):
            seed = args.seed + i * 17 + (3 if tag == "b" else 0)
            d = render_clean_page(kind.strip(), (size[0], size[1]), seed)
            name = f"clean_{kind.strip()}_{tag}"
            for key, suffix in (("image", ""), ("ink", "_ink"), ("sheet", "_sheet")):
                p = os.path.join(out, name + suffix + ".png")
                cv2.imwrite(p, d[key])
                written.append(p)
            with open(os.path.join(out, name + ".clean.json"), "w", encoding="utf-8") as fh:
                json.dump({"kind": kind, "seed": seed, "size": [size[0], size[1]],
                           "note": "procedurally rendered clean sheet "
                                   "(no damage, no noise)"}, fh, indent=2)
            print(f"[render] {name}  {size[0]}x{size[1]}")
    print(f"[render] {len(written)} files -> {out}")
    return written


def load_clean_source(path: str, gt_dir: Optional[str] = None,
                      max_side: int = 1600
                      ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """One *clean* image -> (name, clean, ink, sheet), the input of degrade().

    Three kinds of source are supported, in this order of preference for the ink
    layer, and the choice is recorded so the evaluator can rebuild the exact
    same tuple later:

      * a rendered page from dataset/clean has ``<name>_ink.png`` /
        ``<name>_sheet.png`` beside it (255 = ink);
      * a real scan may have a bitonal ground truth in ``<gt_dir>/<name>.png``
        (DIBCO publishes one).  Its convention is 0 = writing, and some mirrors
        invert that, so the polarity is resolved by "ink is the minority of a
        page".  Using the published mask is much better than thresholding an
        aged page, because the aged page *is* what we are trying to analyse;
      * neither: Otsu on the grayscale image, i.e. the same assumption the
        pipeline itself makes, clearly labelled as an approximation.
    """
    clean = cv2.imread(path, cv2.IMREAD_COLOR)
    if clean is None:
        return None
    h, w = clean.shape[:2]
    if max_side and max(h, w) > max_side:
        sc = float(max_side) / max(h, w)
        clean = cv2.resize(clean, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    base = os.path.splitext(os.path.basename(path))[0]
    stem = os.path.splitext(path)[0]
    ink = (cv2.imread(stem + "_ink.png", cv2.IMREAD_GRAYSCALE)
           if os.path.exists(stem + "_ink.png") else None)
    if ink is None and gt_dir:
        for ext in (".png", ".tif", ".bmp", ".jpg"):
            cand = os.path.join(gt_dir, base + ext)
            g = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if g.shape[:2] != clean.shape[:2]:
                    g = cv2.resize(g, (clean.shape[1], clean.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                ink = np.where(g < 128, 255, 0).astype(np.uint8)
                if float(np.count_nonzero(ink)) > 0.5 * ink.size:
                    ink = cv2.bitwise_not(ink)
                break
    if ink is None:
        g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sheet = (cv2.imread(stem + "_sheet.png", cv2.IMREAD_GRAYSCALE)
             if os.path.exists(stem + "_sheet.png") else None)
    if sheet is None:
        sheet = np.full(clean.shape[:2], 255, np.uint8)
    return base, clean, ink, sheet


def build_degrade(args) -> List[Dict[str, object]]:
    out = args.out
    os.makedirs(out, exist_ok=True)
    sources: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    origin: Dict[str, Dict[str, object]] = {}
    if args.clean_dir and os.path.isdir(args.clean_dir):
        for p in sorted(glob_json(args.clean_dir)):
            # *_ink.png / *_sheet.png are layers of a page, not pages: treating
            # a mask as a source produced 3x the intended number of cases once
            # the loader stopped requiring the suffix to exist
            if p.endswith(("_ink.png", "_sheet.png")):
                continue
            try:
                got = load_clean_source(p, max_side=0)
                if got is None:
                    continue
                sources.append(got)
                origin[got[0]] = {"source": os.path.abspath(p), "gt_dir": None,
                                  "max_side": 0}
            except Exception as exc:
                print(f"[degrade] skipping {p}: {exc}")
    if args.from_dir and os.path.isdir(args.from_dir):
        from heritage_vision.core.utils import list_image_files
        # non-recursive: a gt/ subfolder holds reference masks, not sources
        for p in list_image_files(args.from_dir, recursive=False)[:args.max_sources]:
            got = load_clean_source(p, gt_dir=os.path.join(args.from_dir, "gt"),
                                    max_side=args.max_side)
            if got is None:
                continue
            sources.append(got)
            origin[got[0]] = {"source": os.path.abspath(p),
                              "gt_dir": os.path.join(args.from_dir, "gt"),
                              "max_side": args.max_side}
    if not sources:
        print("[degrade] no clean sources found - run with --render first, or pass "
              "--from-dir <folder of clean scans>")
        return []

    severities = args.severities.split(",")
    manifest: List[Dict[str, object]] = []
    idx = 0
    for src_name, clean, ink, sheet in sources:
        for ci in range(args.cases):
            sev = severities[ci % len(severities)].strip()
            idx += 1
            seed = args.seed + idx * 101
            photo = "photo" in src_name.lower()
            name = f"{idx:03d}_{src_name}_{sev}"
            t0 = __import__("time").perf_counter()
            img, gt = degrade(clean, ink, sheet, seed, severity=sev, photo=photo,
                             perspective=args.perspective and ci % 2 == 1,
                             return_canvas=bool(getattr(args, "keep_canvas", False)))
            deg_path = os.path.join(out, name + ".jpg")
            save_image(deg_path, img, quality=95)
            gt_dir = os.path.join(out, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            files = {}
            for key in ("gt_damage_mask", "gt_stain_mask", "gt_crack_mask",
                        "gt_hole_mask", "gt_text_mask", "gt_sheet_mask"):
                fp = os.path.join(gt_dir, f"{name}.{key.replace('gt_', '')}.png")
                cv2.imwrite(fp, gt[key])
                files[key] = os.path.relpath(fp, out)
            # The pre-damage reference canvas is 3 MB as a PNG and there are 24
            # of them, so it is *not* stored by default: degrade() is a
            # deterministic function of (source, seed, severity, ...), so the
            # evaluator re-creates it bit-exactly from these parameters.  That
            # is better than storing it, because the sidecar then also proves
            # that every test image can be regenerated - which is what a
            # controlled test set is for.
            if getattr(args, "keep_canvas", False):
                fp = os.path.join(gt_dir, f"{name}.clean_image_canvas.png")
                cv2.imwrite(fp, gt["clean_image_canvas"])
                files["clean_image_canvas"] = os.path.relpath(fp, out)
                canvas_regen = {"source": origin.get(src_name, {}).get("source", ""),
                                "stored": files["clean_image_canvas"]}
            else:
                files["clean_image_canvas"] = None
                org = origin.get(src_name, {})
                canvas_regen = {"function": "heritage_vision.tools.make_dataset.degrade",
                                "loader": "heritage_vision.tools.make_dataset.load_clean_source",
                                "source": org.get("source", ""),
                                "gt_dir": org.get("gt_dir"), "max_side": org.get("max_side", 0),
                                "seed": gt["params"]["seed"], "severity": sev,
                                "photo": photo,
                                "perspective": bool(gt["params"]["perspective"]),
                                "desk": bool(gt["params"]["desk"]),
                                "pad_frac": 0.035}
            clean_path = ""
            sidecar = {
                "case_id": name,
                "degraded_image": os.path.relpath(deg_path, out),
                "clean_image": clean_path,
                "source": src_name, "severity": sev,
                "source_image": os.path.relpath(os.path.join(args.clean_dir, src_name + ".png"),
                                                _ROOT).replace(os.sep, "/"),
                "canvas_regen": canvas_regen,
                **{k: v for k, v in gt.items() if isinstance(v, (str, int, float))},
                **{k: v for k, v in files.items()},
                "gt_damage_mask": files["gt_damage_mask"],
                "gt_text_mask": files["gt_text_mask"],
                "gt_sheet_mask": files["gt_sheet_mask"],
                "clean_canvas": files["clean_image_canvas"],
                "gt": {k: (v if isinstance(v, (int, float, str, bool, list)) else
                            "mask") for k, v in gt.items()},
                "params": gt["params"],
                "corners": gt["corners"],
                "created": datetime.now().isoformat(timespec="seconds"),
                "generator": "heritage_vision.tools.make_dataset",
            }
            with open(os.path.join(out, name + GT_SUFFIX), "w", encoding="utf-8") as fh:
                json.dump(sidecar, fh, indent=2, default=str)
            manifest.append({
                "file": os.path.basename(deg_path),
                "folder": os.path.relpath(out, _ROOT),
                "source": f"student-created (degradation of {src_name})",
                "width": img.shape[1], "height": img.shape[0],
                "format": "JPEG", "purpose": f"controlled damage test ({sev})",
                "gt_damage_pct": round(float(np.count_nonzero(gt["gt_damage_mask"])) /
                                        gt["gt_damage_mask"].size * 100.0, 3),
                "build_ms": round((__import__("time").perf_counter() - t0) * 1000, 1),
            })
            print(f"[degrade] {name}  damage "
                  f"{manifest[-1]['gt_damage_pct']:.2f}%  ({sidecar['params']['n_stain']} "
                  f"stains, {sidecar['params']['n_crack']} cracks)")
    if args.manifest:
        # rebuilt from the tree, so the file describes the dataset as it is now
        build_manifest(os.path.dirname(os.path.abspath(args.manifest)) or "dataset",
                       args.manifest)
    print(f"[degrade] {len(manifest)} cases -> {out}")
    return manifest


MANIFEST_COLUMNS = ["file", "folder", "source", "origin", "format", "width", "height",
                    "bit_depth", "channels", "dpi", "purpose", "gt_layers",
                    "gt_damage_pct", "skew_deg", "perspective", "notes"]


def build_manifest(root: str, path: str,
                   folders: Sequence[str] = ("clean", "student_created", "real_scans",
                                             "unseen_test")) -> str:
    """Rewrite ``dataset_manifest.csv`` from what is actually on disk.

    Part A of the brief asks for a manifest that lists, for every image, where
    it came from, its resolution, its format and its role in the project.  It is
    generated from the folder tree (and from the ``.gt.json`` sidecars and the
    ``CREDITS.csv`` files that sit next to the external images) rather than
    written during generation, so it can never drift out of sync with the data.
    """
    import csv
    rows: List[Dict[str, object]] = []
    for folder in folders:
        d = os.path.join(root, folder)
        if not os.path.isdir(d):
            continue
        credits: Dict[str, Dict[str, str]] = {}
        cpath = os.path.join(d, "CREDITS.csv")
        if os.path.exists(cpath):
            with open(cpath, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    credits[r.get("file", "")] = r
        from heritage_vision.core.utils import list_image_files
        for fp in list_image_files(d, recursive=False):
            f = os.path.basename(fp)
            im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if im is None:
                rows.append({"file": f, "folder": folder, "source": "unreadable",
                             "origin": "?", "format": os.path.splitext(f)[1][1:],
                             "width": 0, "height": 0, "bit_depth": 0, "channels": 0,
                             "dpi": "", "purpose": "UNREADABLE - excluded from tests",
                             "gt_layers": 0, "gt_damage_pct": "", "skew_deg": "",
                             "perspective": "", "notes": "file could not be decoded"})
                continue
            h, w = im.shape[:2]
            ch = 1 if im.ndim == 2 else im.shape[2]
            depth = {"uint8": 8, "uint16": 16}.get(str(im.dtype), 8)
            side = os.path.splitext(fp)[0] + GT_SUFFIX
            note, gt_layers, dmg, skew, persp, src, purpose = "", 0, "", "", "", "", ""
            origin = ("rendered (student-created, no copyright)" if folder == "clean" else
                      "student-created (degraded synthetic scan)" if folder == "student_created"
                      else "external public dataset" if folder == "real_scans"
                      else "held-out test set")
            if os.path.exists(side):
                with open(side, encoding="utf-8") as fh:
                    sd = json.load(fh)
                src = sd.get("source", "")
                purpose = f"controlled damage test ({sd.get('severity', '?')})"
                prm = sd.get("params", {})
                gt_layers = sum(1 for k in ("gt_damage_mask", "gt_stain_mask", "gt_crack_mask",
                                            "gt_hole_mask", "gt_text_mask", "gt_sheet_mask")
                                if sd.get(k))
                dm = sd.get("gt_damage_mask")
                if dm:
                    m = cv2.imread(os.path.join(d, dm), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        # "% of the sheet", not "% of the frame": the desk pad
                        # around the page is not part of the object being graded
                        sm = cv2.imread(os.path.join(
                            os.path.dirname(os.path.join(d, dm)),
                            os.path.basename(dm).replace("damage_mask",
                                                         "sheet_mask")),
                            cv2.IMREAD_GRAYSCALE)
                        denom = float(np.count_nonzero(sm > 0)) if sm is not None \
                            else float(m.size)
                        dmg = round(100.0 * float(np.count_nonzero(m > 0))
                                    / max(denom, 1.0), 3)
                skew = prm.get("skew_deg", "")
                persp = bool(prm.get("perspective", False))
                note = (f"seed={prm.get('seed')}, stains={prm.get('n_stain')}, "
                        f"cracks={prm.get('n_crack')}, holes={prm.get('n_hole')}, "
                        f"noise={prm.get('noise_sigma')}, jpeg={prm.get('jpeg_quality')}")
            cr = credits.get(f, {})
            if cr:
                src = cr.get("source", src)
                purpose = cr.get("purpose", purpose)
                note = (note + "; " if note else "") + cr.get("notes", "")
                origin = cr.get("origin", origin)
            if not purpose:            # no sidecar, no CREDITS entry (the clean
                base = os.path.splitext(f)[0]        # renders and their masks)
                purpose = (("ground-truth ink layer (text-preservation reference)"
                            if base.endswith("_ink") else
                            "ground-truth sheet mask (page extent reference)"
                            if base.endswith("_sheet") else
                            "reference clean sheet (source for the degraded sets)"
                            if folder == "clean" else
                            "real historical page (read-only benchmark)"
                            if folder == "real_scans" else
                            "held-out test image (never used for tuning)"
                            if folder == "unseen_test" else
                            "controlled damage test"))
            rows.append({"file": f, "folder": folder, "source": src or os.path.splitext(f)[0],
                         "origin": origin, "format": os.path.splitext(f)[1][1:].upper(),
                         "width": w, "height": h, "bit_depth": depth, "channels": ch,
                         "dpi": "", "purpose": purpose or cr.get("purpose", ""),
                         "gt_layers": gt_layers, "gt_damage_pct": dmg, "skew_deg": skew,
                         "perspective": persp, "notes": note})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wt = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        wt.writeheader()
        for r in rows:
            wt.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    print(f"[manifest] {len(rows)} images -> {path}")
    return path


def write_manifest_rows(rows: Sequence[Dict[str, object]], path: str) -> str:
    """Append to the dataset manifest (Part A documentation requirement)."""
    import csv
    exists = os.path.exists(path)
    keys = ["file", "folder", "source", "width", "height", "format", "purpose",
            "gt_damage_pct", "build_ms"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return path


def glob_json(folder: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(folder, "clean_*.png")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HeritageVision dataset builder")
    ap.add_argument("--render", action="store_true", help="create clean source pages")
    ap.add_argument("--degrade", action="store_true", help="apply the degradation model")
    ap.add_argument("--out", default=os.path.join(_ROOT, "dataset", "student_created"))
    ap.add_argument("--clean-dir", default=os.path.join(_ROOT, "dataset", "clean"))
    ap.add_argument("--from-dir", default="", help="folder of real clean scans to age")
    ap.add_argument("--max-sources", type=int, default=40)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--cases", type=int, default=3, help="degradations per source")
    ap.add_argument("--severities", default="light,moderate,severe")
    ap.add_argument("--kinds", default="", help="comma list for --render")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--perspective", action="store_true",
                    help="every 2nd case gets a perspective distortion (rectification test)")
    ap.add_argument("--manifest", default=os.path.join(_ROOT, "dataset",
                                                        "dataset_manifest.csv"))
    ap.add_argument("--keep-canvas", action="store_true",
                    help="write the 3 MB pre-damage reference PNG per case (the "
                         "evaluator can re-create it from the sidecar instead)")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild dataset_manifest.csv by scanning the dataset "
                         "tree, without generating anything")
    ap.add_argument("--dataset-root", default=os.path.join(_ROOT, "dataset"))
    a = ap.parse_args(argv)
    if a.manifest_only:
        build_manifest(a.dataset_root, a.manifest)
        return 0
    if not (a.render or a.degrade):
        ap.print_help()
        return 2
    if a.render:
        a.out_clean = a.out
        a_out = a.out
        a.out = a.clean_dir
        build_render(a)
        a.out = a_out
    if a.degrade:
        build_degrade(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
