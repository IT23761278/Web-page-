#!/usr/bin/env python3
"""HeritageVision - desktop GUI for condition assessment and enhancement of
historical paper items (Part D of SE3071).

Everything the window can do is also reachable without one: the behaviour lives
in :class:`AppController`, and ``python app.py --selftest`` exercises exactly
that controller (load -> process -> ROI -> export -> batch).  Keeping the view
thin is what makes the tool testable, and it means a minimal Linux install that
has no ``tkinter`` gets a clear message instead of a broken window.

Run:
    python app.py                        # open the window
    python app.py dataset/student_created/003_clean_manuscript_a_severe.jpg
    python app.py --selftest [image]     # no display needed, checks every path
    python app.py --folder <dir> --out <dir>     # batch only, no window
    python -m heritage_vision.tools.evaluate --dataset dataset/unseen_test

Layout
    left    file/folder, the parameter panel, Run / Reset / Export / ROI-clear
    centre  tabs: original | processed | annotated | before/after | masks.
            Drag a rectangle on the image to restrict the analysis to a region.
    right   the decision numbers: damage %, region counts and classes, area in
            mm2, quality index, condition index, PASS/WARNING/FAIL, warnings,
            measures, timings, the top regions by severity.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heritage_vision import __author__, __version__                 # noqa: E402
from heritage_vision.core import annotate as ann                    # noqa: E402
from heritage_vision.core.config import PipelineParams              # noqa: E402
from heritage_vision.core.export import (write_csv, write_json,       # noqa: E402
                                         write_text_report)
from heritage_vision.core.pipeline import HeritagePipeline          # noqa: E402
from heritage_vision.core.utils import list_image_files             # noqa: E402

try:                                                    # the GUI is optional
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    HAS_TK = True
except Exception as _tk_err:                            # pragma: no cover
    HAS_TK = False
    TK_ERROR = str(_tk_err)
PIL_ONLY = False
try:                # Pillow alone is not enough: the Tk bridge ImageTk is what
    from PIL import Image, ImageTk    # we need to paint into a tkinter canvas
    HAS_PIL = True
except Exception:                       # pragma: no cover
    HAS_PIL = False
    try:                                # distinguish "no Pillow" from "no Tk"
        from PIL import Image as _I     # noqa: F401
        PIL_ONLY = True
    except Exception:
        PIL_ONLY = False

PREVIEW_MAX = 1000          #: working resolution while a window is open
VIEW_FIT = (860, 700)       #: px, the canvas the image is fitted into

# ---------------------------------------------------------------------------
# the parameter panel, declared once, so the widgets, the validation ranges and
# PipelineParams cannot drift apart.  (attribute, label, kind, lo, hi, step,
# help)  - "kind" is bool | int | float | combo.
# ---------------------------------------------------------------------------
CONTROLS: List[Tuple[str, str, str, Any, Any, Any, str]] = [
    ("clahe_enabled", "CLAHE contrast", "bool", None, None, None,
     "adaptive histogram equalisation - the main legibility control"),
    ("clahe_clip_limit", "CLAHE clip limit", "float", 0.5, 8.0, 0.1,
     "amplification cap; above ~3 the paper grain starts to shout"),
    ("clahe_tile_grid", "CLAHE tiles", "int", 2, 32, 1,
     "neighbourhood size; larger is more global"),
    ("gamma", "Gamma", "float", 0.4, 2.5, 0.05,
     "<1 brightens faded paper, >1 darkens a washed-out scan"),
    ("percentile_stretch", "Contrast stretch %", "float", 0.0, 10.0, 0.1,
     "robust min/max stretch, this % cut from each tail (0 = off)"),
    ("bg_normalisation", "Illumination", "combo", ("none", "morph", "gaussian"),
     None, None, "removes the uneven light of a flatbed or a phone photo"),
    ("bg_kernel_mm", "Background scale (mm)", "float", 2.0, 80.0, 1.0,
     "features bigger than this count as lighting, not as ink"),
    ("denoise_method", "Denoise filter", "combo",
     ("bilateral", "median", "gaussian", "nlm", "guided", "wiener", "auto",
      "none"), None, None,
     "bilateral keeps ink edges; auto takes the winner of the comparison"),
    ("denoise_ksize", "Filter size (px)", "int", 3, 9, 2,
     "odd neighbourhood of the filter"),
    ("binarisation", "Binarisation", "combo", ("otsu", "adaptive", "sauvola"),
     None, None, "global, local, or Sauvola for stained paper"),
    ("deskew", "Deskew", "bool", None, None, None,
     "rotation search on the projection profile; refuses when unsure"),
    ("rectify", "Rectify page", "bool", None, None, None,
     "perspective warp to the detected sheet quad"),
    ("auto_crop", "Auto crop to page", "bool", None, None, None,
     "drop everything outside the detected sheet"),
    ("analysis_source", "Detect on", "combo", ("raw", "enhanced"), None, None,
     "raw scan (colour evidence intact) or the enhanced view"),
    ("damage_sensitivity", "Damage sensitivity", "float", 0.4, 2.5, 0.05,
     "1/sigma of the stain and crack thresholds; higher finds more, and more noise"),
    ("damage_min_area_mm2", "Min damage area (mm2)", "float", 0.05, 50.0, 0.25,
     "smaller blobs are ignored - the smallest spot a conservator would note"),
    ("inpaint", "Inpaint damage", "bool", None, None, None,
     "fills the detected damage to produce a legible surrogate copy"),
    ("inpaint_radius", "Inpaint radius", "float", 1.0, 15.0, 0.5,
     "Telea / Navier-Stokes search radius in px"),
    ("inpaint_method", "Inpaint method", "combo", ("telea", "ns"), None, None,
     "Telea is faster; Navier-Stokes continues strokes better"),
    ("sharpen", "Unsharp mask", "bool", None, None, None,
     "edge boost after the restoration, never inside it"),
    ("target_dpi", "Target dpi", "int", 72, 1200, 4,
     "only used to convert pixels to mm for the area measures"),
]

#: not a PipelineParams field - it asks the pipeline to measure every filter
EXTRA_CONTROLS = [("compare_filters", "Compare all filters",
                   "runs median/gaussian/bilateral/guided/Wiener and reports the "
                   "PSNR + edge retention of each (Part B comparison)")]

VIEWS = ("original", "processed", "annotated", "before/after", "damage mask",
         "text mask")

METRIC_KEYS = ("hq_residue_reduction_pct", "hq_over_reach_pct", "text_ink_pct",
               "roi_area_pct", "spectrum_hf_ratio", "fft_blur_sigma_px",
               "texture_glcm_contrast")


# ---------------------------------------------------------------------------
# controller: all behaviour, no widgets (this is what the self-test drives)
# ---------------------------------------------------------------------------
class AppController:
    """State and actions of the application.

    ``thread=False`` runs the work synchronously, which is what the self-test
    and any scripted use want; the Tk view uses a worker thread so the window
    keeps painting while a 2-4 s analysis runs.
    """

    def __init__(self, thread: bool = False) -> None:
        self.params = PipelineParams()
        self.pipeline = HeritagePipeline(self.params)
        self.compare_filters = True
        self.path: Optional[str] = None
        self.image: Optional[np.ndarray] = None            #: full-res original
        self.result: Any = None                            #: last ProcessResult
        self.roi_box: Optional[Tuple[int, int, int, int]] = None   #: x, y, w, h
        self.view = "annotated"
        self.log: List[str] = []
        self._thread = thread
        self._busy = False
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.on_update: Optional[Any] = None               #: view callback

    # ------------------------------------------------------------- settings
    def set(self, attr: str, value: Any) -> Tuple[bool, str]:
        """One validated parameter change: (accepted, message).

        The view never clamps anything itself - the value goes in raw, and
        ``PipelineParams.validated()`` returns the clamped copy, so the GUI and
        the CLI cannot disagree about what a legal value is.
        """
        if attr == "compare_filters":
            self.compare_filters = bool(value)
            return True, ""
        if not hasattr(self.params, attr):
            return False, f"unknown parameter: {attr}"
        try:
            setattr(self.params, attr, value)
        except Exception as exc:
            return False, f"{attr}: {exc}"
        self.params = self.params.validated()
        self.pipeline.params = self.params
        self.log.append(f"set {attr} = {getattr(self.params, attr)}")
        return True, ""

    def reset(self) -> None:
        self.params = PipelineParams()
        self.pipeline.params = self.params
        self.compare_filters = True
        self.roi_box = None
        self.log.append("parameters reset to the defaults")

    def values(self) -> Dict[str, Any]:
        out = {a: getattr(self.params, a) for a, *_ in CONTROLS}
        out["compare_filters"] = self.compare_filters
        return out

    # ------------------------------------------------------------- loading
    def load(self, path: str) -> Tuple[bool, str]:
        """Read and validate an image file.

        Corrupt files, unsupported formats and unreadable headers are rejected
        here with a message for the user, never as an exception escaping the
        window (Part D error-handling requirement).
        """
        if not os.path.exists(path):
            return False, f"no such file: {path}"
        if os.path.getsize(path) == 0:
            return False, f"{os.path.basename(path)} is empty (0 bytes)"
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                       ".webp", ".jp2"):
            return False, (f"'{ext}' is not a supported format "
                           "(png, jpg, tif, bmp, webp, jp2)")
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("the decoder returned nothing - the file is "
                                 "corrupt or truncated")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.log.append("grayscale input expanded to 3 channels")
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.log.append("alpha channel dropped (BGR used)")
            elif img.shape[2] not in (1, 3):
                raise ValueError(f"unsupported channel count {img.shape[2]}")
            if img.dtype != np.uint8:                      # 16-bit / float scan
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = np.clip(img, 0, 255).astype(np.uint8)
                self.log.append(f"{img.dtype} input tone-mapped to 8 bit")
            self.path, self.image = path, img
            self.roi_box = None
            self.result = None
            self.log.append(f"loaded {os.path.basename(path)} "
                            f"{img.shape[1]}x{img.shape[0]}")
            self._run(os.path.basename(path))
            return True, ""
        except Exception as exc:
            msg = f"could not open {os.path.basename(path)}: {exc}"
            self.log.append(msg)
            return False, msg

    # ------------------------------------------------------------- analysis
    def process(self) -> Tuple[bool, str]:
        if self.image is None:
            return False, "load an image first"
        self._run(os.path.basename(self.path or "image"))
        return True, ""

    def set_roi(self, box: Optional[Tuple[int, int, int, int]]) -> Tuple[bool, str]:
        """Restrict the analysis to a rectangle of the *original* image."""
        if box is not None:
            x, y, w, h = [int(v) for v in box]
            if self.image is not None:
                H, W = self.image.shape[:2]
                x, y = max(x, 0), max(y, 0)
                w, h = min(max(w, 8), W - x), min(max(h, 8), H - y)
            box = (x, y, w, h)
        self.roi_box = box
        self.log.append("ROI cleared" if box is None else f"ROI = {box}")
        if self.image is not None:
            self._run(os.path.basename(self.path or "image"))
        return True, ""

    def _run(self, name: str) -> None:
        if self._thread:
            self._busy = True
            threading.Thread(target=self._work, args=(name,), daemon=True).start()
        else:
            self._deliver(self._compute(name))

    def _work(self, name: str) -> None:                      # pragma: no cover
        try:
            out = self._compute(name)
        except Exception:                                    # never kill the UI
            out = {"ok": False, "error": traceback.format_exc(limit=4)}
        self._q.put(out)
        if self.on_update is not None:
            self.on_update()

    def _compute(self, name: str) -> Dict[str, Any]:
        """One analysis.  In window mode it runs on a reduced copy, so the
        preview stays responsive; the ROI box is scaled by the same factor so
        the region the user drew still points at the same paper."""
        if self.image is None:
            return {"ok": False, "error": "no image loaded"}
        img, box = self.image, self.roi_box
        if self._thread:
            h, w = img.shape[:2]
            s = min(PREVIEW_MAX / float(max(h, w)), 1.0)
            if s < 1.0:
                img = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
                if box is not None:
                    box = tuple(int(round(v * s)) for v in
                                (box[0], box[1], box[2], box[3]))
        try:
            res = self.pipeline.process(img, name=name, roi_box=box,
                                        run_denoise_comparison=bool(self.compare_filters))
        except Exception:                                    # pragma: no cover
            return {"ok": False, "error": traceback.format_exc(limit=4)}
        res.input_image = self.image
        return {"ok": res.ok, "result": res, "error": res.error}

    def poll(self) -> Optional[Dict[str, Any]]:
        """Called from the Tk event loop: take a finished worker result."""
        try:
            out = self._q.get_nowait()
            self._busy = False
            self._deliver(out)
            return out
        except queue.Empty:
            return None

    def _deliver(self, out: Dict[str, Any]) -> None:
        if out.get("ok"):
            self.result = out["result"]
            self.log.append(self.result.summary())
        else:
            self.log.append("ERROR " + str(out.get("error", ""))[:300])

    # ------------------------------------------------------------- views
    def view_image(self, which: Optional[str] = None,
                   fit: Tuple[int, int] = VIEW_FIT) -> Optional[np.ndarray]:
        """The BGR image of one tab, fitted to the canvas (never upscaled)."""
        which = which or self.view
        r, src = self.result, None
        if which == "original":
            src = self.image
        elif r is None:
            src = self.image
        elif which == "processed":
            src = r.processed
        elif which == "annotated":
            src = r.annotated
        elif which == "before/after":
            if self.image is not None and r.processed is not None:
                src = ann.side_by_side(self.image, r.processed,
                                       "before (input)", "after (processed)")
        elif which == "damage mask":
            src = r.damage_mask
        elif which == "text mask":
            src = r.text_mask
        if src is None:
            src = self.image
        if src is None:
            return None
        return _fit(src, fit)

    def report_lines(self) -> List[str]:
        """The text panel: every number the decision was built from."""
        r = self.result
        if r is None:
            return ["no result yet - open an image"]
        if not r.ok:
            return ["FAILED\n" + str(r.error)]
        d, q, c, rd = r.damage, r.quality, r.condition, r.readability
        L: List[str] = [r.name, ""]
        if r.metadata is not None:
            L += r.metadata.pretty().splitlines() + [""]
        L += [f"damage            {d.damage_pct:6.2f} % of ROI "
              f"({d.damage_mm2:.1f} mm2)",
              f"regions           {d.n_damage_total} (stains {d.n_stain}, "
              f"cracks {d.n_crack}, losses {d.n_missing})",
              f"text affected     {d.n_text_loss} region(s)",
              f"largest region    {d.largest_damage_mm2:.2f} mm2",
              f"density           {d.damage_density_per_cm2:.2f} regions/cm2",
              f"mean intensity    {d.mean_intensity_before:.1f} -> "
              f"{d.mean_intensity_after:.1f}", ""]
        if d.class_counts:
            L.append("detected classes")
            L += [f"   {k:16s} {v}" for k, v in
                  sorted(d.class_counts.items(), key=lambda x: -x[1])]
            L.append("")
        if d.notes:
            L.append("notes")
            L += [f"   - {n}" for n in d.notes]
            L.append("")
        L += [f"quality index     {q.score:5.1f}  {q.grade}",
              f"condition index   {c.score:5.1f}  {c.severity_class}  ->  "
              f"{c.pass_fail}", ""]
        if c.warning:
            L += ["decision", f"   ! {c.warning}", ""]
        if c.actions:
            L.append("suggested actions")
            L += [f"   - {a}" for a in c.actions]
            L.append("")
        if q.flags:
            L.append("quality flags")
            L += [f"   - {a}" for a in q.flags]
            L.append("")
        if q.recommendations:
            L.append("recommendations")
            L += [f"   - {a}" for a in q.recommendations]
            L.append("")
        L += [f"readability       {rd.before.get('score', 0):.0f} -> "
              f"{rd.after.get('score', 0):.0f} ({rd.improvement_pct:+.1f} %)  "
              f"[{rd.verdict}]",
              f"                 lines {rd.lines}  words {rd.words}  "
              f"components {rd.characters}", ""]
        if r.metrics:
            L.append("measures")
            for k in METRIC_KEYS:
                if k in r.metrics:
                    v = r.metrics[k]
                    L.append(f"   {k:26s} {v:.3f}" if isinstance(v, float)
                             else f"   {k:26s} {v}")
            L.append("")
        if r.denoise_comparison:
            L.append("filter comparison (residual noise / time)")
            for rep in r.denoise_comparison:
                m = rep.metrics or {}
                L.append(f"   {rep.method:10s}"
                         + (f"{'  <-- chosen' if rep.selected else ''}  "
                            + "  ".join(f"{k}={v:.3f}" for k, v in m.items()
                                        if isinstance(v, float))))
            L.append("")
        if r.geometry:
            L.append("geometry")
            L += [f"   {k:16s} {v}" for k, v in r.geometry.items()]
            L.append("")
        if r.extra.get("restoration_stats"):
            L.append("restoration")
            L += [f"   {k:16s} {v}" for k, v in r.extra["restoration_stats"].items()]
            L.append("")
        L.append("timings (ms)")
        L.append("   " + "  ".join(f"{k}={v:.0f}" for k, v in r.timings.items()))
        if r.damage.regions:
            L += ["", "regions by severity"]
            for reg in sorted(r.damage.regions, key=lambda x: -x.severity)[:10]:
                L.append(f"   #{reg.id:02d} {reg.class_name:14s} {reg.layer:7s}"
                         f" {reg.area_mm2:7.2f} mm2  sev {reg.severity:.2f}"
                         f"{'  [text]' if reg.overlaps_text else ''}")
        return L

    def current_row(self) -> Dict[str, Any]:
        r = self.result
        if r is None or not r.ok:
            return {}
        row = r.to_dict()
        row["parameters"] = self.params.as_dict()
        row["roi"] = list(self.roi_box) if self.roi_box else []
        row["source_file"] = self.path or ""
        return row

    # ------------------------------------------------------------- exports
    def export(self, folder: str, ext: str = ".png") -> Tuple[bool, str, List[str]]:
        """Images + JSON + CSV (single row) + region CSV + text report."""
        r = self.result
        if r is None or not r.ok:
            return False, "nothing to export - run the analysis first", []
        try:
            os.makedirs(folder, exist_ok=True)
            written = r.save(folder, ext=ext)          # processed/annotated/stages
            stem = os.path.splitext(r.name)[0] or "image"
            row = self.current_row()
            write_csv([row], os.path.join(folder, stem + "_summary.csv"))
            write_text_report([r], os.path.join(folder, stem + "_report.txt"))
            if r.damage is not None and r.damage.regions:
                rp = os.path.join(folder, stem + "_regions.csv")
                write_csv([asdict(g) for g in r.damage.regions], rp)
                written.append(rp)
            written += [os.path.join(folder, stem + s) for s in
                        ("_summary.csv", "_report.txt")]
            self.log.append(f"exported {len(written)} files -> {folder}")
            return True, f"{len(written)} files written to\n{folder}", written
        except Exception as exc:
            return False, f"export failed: {exc}", []

    def batch(self, folder: str, out_dir: str, ext: str = ".png",
              max_images: Optional[int] = None,
              progress: Optional[Any] = None) -> Tuple[bool, str, List[Any]]:
        """Run the pipeline over a whole folder (Part D batch requirement)."""
        try:
            files = list_image_files(folder)
        except NotADirectoryError:
            return False, f"not a folder: {folder}", []
        if not files:
            return False, f"no readable images in {folder}", []
        os.makedirs(out_dir, exist_ok=True)
        results, rejected = self.pipeline.process_folder(
            folder, out_dir=out_dir, max_images=max_images, progress=progress)
        rows = [r.to_dict() for r in results]
        ok_n = sum(1 for r in results if r.ok)
        try:
            write_csv(rows, os.path.join(out_dir, "results.csv"))
            agg = {"n": len(rows), "ok": ok_n, "rejected": len(rejected),
                   "mean_time_ms": round(float(np.mean([r.timings.get("TOTAL", 0.0)
                                                        for r in results])), 1),
                   "mean_damage_pct": round(float(np.mean([r.damage.damage_pct
                                                           for r in results if r.ok])), 3),
                   "mean_condition": round(float(np.mean([r.condition.score
                                                          for r in results if r.ok])), 2),
                   "flags": {k: sum(1 for r in results if r.ok
                                    and r.condition.pass_fail == k)
                             for k in ("PASS", "WARN", "FAIL")},
                   "params": self.params.as_dict()}
            write_json(agg, os.path.join(out_dir, "summary.json"))
            write_text_report(results, os.path.join(out_dir, "report.txt"),
                              title=f"HeritageVision batch report - {folder}")
        except Exception as exc:                            # pragma: no cover
            self.log.append(f"batch export warning: {exc}")
        msg = (f"{ok_n}/{len(rows)} images ok, {len(rejected)} rejected, "
               f"mean {agg['mean_time_ms']:.0f} ms/image")
        self.log.append("batch: " + msg)
        return True, msg, results


def _fit(img: np.ndarray, fit: Tuple[int, int]) -> np.ndarray:
    """Downscale to the canvas size; small images are shown 1:1."""
    h, w = img.shape[:2]
    s = min(fit[0] / float(w), fit[1] / float(h), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                         interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# the Tk view: widgets, drawing, and nothing else
# ---------------------------------------------------------------------------
class HeritageApp:                                          # pragma: no cover
    def __init__(self, root: "tk.Tk", ctl: AppController) -> None:
        self.root, self.ctl = root, ctl
        self.vars: Dict[str, Any] = {}
        self.photo = None
        self.drag0: Optional[Tuple[int, int]] = None
        self.rect_id: Optional[int] = None
        self.scale = 1.0
        ctl.on_update = self._poll_later
        root.title(f"HeritageVision {__version__} - historical paper condition "
                   f"assessment  ({__author__})")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self._build_left()
        self._build_centre()
        self._build_right()
        self.set_status("Open an image, or a folder for batch mode.")
        root.after(150, self._poll)

    # ---------------------------------------------------------- widgets
    def _build_left(self) -> None:
        f = ttk.Frame(self.root, padding=6)
        f.grid(row=0, column=0, sticky="nsw")
        f.rowconfigure(5, weight=1)
        ttk.Label(f, text="Input").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(f)
        row.grid(row=1, column=0, sticky="ew")
        ttk.Button(row, text="Image...", command=self.pick_image).pack(side="left")
        ttk.Button(row, text="Folder...", command=self.pick_folder).pack(side="left")
        ttk.Label(f, text="View").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.view_var = tk.StringVar(value=self.ctl.view)
        box = ttk.Frame(f)
        box.grid(row=3, column=0, sticky="ew")
        for i, name in enumerate(VIEWS):
            ttk.Radiobutton(box, text=name, value=name, variable=self.view_var,
                            command=self.redraw).grid(row=i // 2, column=i % 2,
                                                      sticky="w")
        ttk.Label(f, text="Parameters").grid(row=4, column=0, sticky="w",
                                            pady=(8, 0))
        wrap = ttk.Frame(f)
        wrap.grid(row=5, column=0, sticky="nsew")
        inner = ttk.Frame(wrap)
        inner.pack(fill="both", expand=True)
        self._build_controls(inner)
        bar = ttk.Frame(f)
        bar.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="Re-run", command=self.process).pack(side="left")
        ttk.Button(bar, text="Reset", command=self.reset).pack(side="left")
        ttk.Button(bar, text="Export...", command=self.export).pack(side="left")
        ttk.Button(bar, text="ROI clear", command=self.clear_roi).pack(side="left")

    def _build_controls(self, parent: "ttk.Frame") -> None:
        vals = self.ctl.values()
        r = 0
        for attr, label, kind, lo, hi, step, help_ in CONTROLS:
            v = vals[attr]
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
            if kind == "bool":
                var: Any = tk.BooleanVar(value=bool(v))
                ttk.Checkbutton(parent, variable=var,
                                command=lambda a=attr, x=var: self.change(a, x.get())
                                ).grid(row=r, column=1, sticky="w")
            elif kind == "combo":
                var = tk.StringVar(value=str(v))
                cb = ttk.Combobox(parent, values=list(lo), textvariable=var,
                                  width=9, state="readonly")
                cb.grid(row=r, column=1, sticky="w")
                cb.bind("<<ComboboxSelected>>",
                        lambda e, a=attr, x=var: self.change(a, x.get()))
            elif kind == "int":
                var = tk.IntVar(value=int(v))
                ttk.Scale(parent, from_=lo, to=hi, value=int(v), orient="horizontal",
                          command=lambda s, a=attr, x=var: self.change(
                              a, int(round(float(s))))).grid(row=r, column=1,
                                                              sticky="ew")
            else:
                var = tk.DoubleVar(value=float(v))
                ttk.Scale(parent, from_=lo, to=hi, value=float(v), orient="horizontal",
                          command=lambda s, a=attr, x=var: self.change(
                              a, round(float(s), 3))).grid(row=r, column=1,
                                                            sticky="ew")
            self.vars[attr] = var
            ttk.Label(parent, text=help_.split(";")[0], foreground="#666").grid(
                row=r, column=2, sticky="w")
            r += 1
        for attr, label, help_ in EXTRA_CONTROLS:
            var = tk.BooleanVar(value=bool(getattr(self.ctl, attr, True)))
            ttk.Checkbutton(parent, text=label, variable=var,
                            command=lambda a=attr, x=var: self.change(a, x.get())
                            ).grid(row=r, column=0, columnspan=3, sticky="w")
            self.vars[attr] = var
            r += 1
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="drag on the image = region of interest",
                  foreground="#666").grid(row=r, column=0, columnspan=3, sticky="w")

    def _build_centre(self) -> None:
        fr = ttk.Frame(self.root, padding=(0, 6))
        fr.grid(row=0, column=1, sticky="nsew")
        fr.rowconfigure(1, weight=1)
        fr.columnconfigure(0, weight=1)
        self.title_lbl = ttk.Label(fr, text="no image", font=("TkDefaultFont", 10,
                                                               "bold"))
        self.title_lbl.grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(fr, width=VIEW_FIT[0], height=VIEW_FIT[1],
                                background="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.drag_start)
        self.canvas.bind("<B1-Motion>", self.drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.drag_end)
        self.prog = ttk.Progressbar(fr, mode="indeterminate")
        self.prog.grid(row=2, column=0, sticky="ew")
        self.status = ttk.Label(fr, text="", wraplength=VIEW_FIT[0], foreground="#444")
        self.status.grid(row=3, column=0, sticky="w")

    def _build_right(self) -> None:
        f = ttk.Frame(self.root, padding=6)
        f.grid(row=0, column=2, sticky="nsew")
        self.report = tk.Text(f, width=54, height=42, wrap="word",
                              font=("TkFixedFont", 9))
        self.report.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(f, command=self.report.yview).pack(side="right", fill="y")
        self.report.configure(state="disabled")

    # ---------------------------------------------------------- callbacks
    def pick_image(self) -> None:
        p = filedialog.askopenfilename(
            title="Choose a scan or photograph",
            filetypes=[("images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                       ("all files", "*.*")])
        if not p:
            return
        ok, err = self.ctl.load(p)
        if not ok:
            messagebox.showerror("Cannot open image", err)
        self.after_change()

    def pick_folder(self) -> None:
        d = filedialog.askdirectory(title="Choose a folder to batch process")
        if not d:
            return
        out = filedialog.askdirectory(title="Where should the results go?",
                                      initialdir=d) or os.path.join(d, "heritage_out")
        self.set_status(f"batch over {os.path.basename(d)} ...")
        ok, msg, _ = self.ctl.batch(d, out,
                                    progress=lambda i, n, r: self.set_status(
                                        f"{i}/{n} {r.name}"))
        (messagebox.showwarning if not ok else messagebox.showinfo)("Batch", msg)
        self.set_status(msg + f"   ->   {out}")

    def change(self, attr: str, value: Any) -> None:
        ok, msg = self.ctl.set(attr, value)
        if not ok:
            self.set_status(msg)

    def process(self) -> None:
        ok, err = self.ctl.process()
        if not ok:
            messagebox.showinfo("Nothing to do", err)
        self.after_change()

    def reset(self) -> None:
        self.ctl.reset()
        vals = self.ctl.values()
        for attr, _l, kind, *_ in CONTROLS:
            var = self.vars.get(attr)
            if var is None:
                continue
            v = vals[attr]
            var.set(bool(v) if kind == "bool" else
                    (str(v) if kind == "combo" else v))
        self.after_change()

    def clear_roi(self) -> None:
        self.ctl.set_roi(None)
        self.after_change()

    def export(self) -> None:
        d = filedialog.askdirectory(title="Export folder")
        if not d:
            return
        ok, msg, _ = self.ctl.export(d)
        (messagebox.showerror if not ok else messagebox.showinfo)("Export", msg)
        self.refresh_report()

    # ---------------------------------------------------------- rendering
    def after_change(self) -> None:
        self.prog.start(12)
        self.set_status("analysing ...")
        self.root.after(80, self._wait_done)

    def _wait_done(self) -> None:                           # pragma: no cover
        if self.ctl._busy:
            self.root.after(120, self._wait_done)
            return
        self.prog.stop()
        self.redraw()
        self.refresh_report()
        r = self.ctl.result
        if r is not None and r.ok:
            self.set_status(r.summary()[:200])
            self.title_lbl.configure(
                text=f"{r.name}   -   drag on the image to restrict the analysis "
                     f"to a region")
        elif self.ctl.image is not None:
            self.set_status("analysis failed - the report panel has the reason")

    def _poll_later(self) -> None:
        try:
            self.root.after(30, self._poll)
        except Exception:
            pass

    def _poll(self) -> None:
        if not self.ctl._busy:
            out = self.ctl.poll()
            if out is not None:
                self._wait_done()
        self.root.after(200, self._poll)

    def redraw(self) -> None:
        self.ctl.view = self.view_var.get()
        img = self.ctl.view_image(self.ctl.view)
        self.canvas.delete("all")
        if img is None:
            return
        if not HAS_PIL:
            self.set_status(("Pillow is installed but its Tk bridge (ImageTk) is "
                             "unusable - " if PIL_ONLY else "Pillow is missing - ")
                            + "showing the preview in an OpenCV window instead")
            cv2.imshow("HeritageVision preview", img)
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.scale = (self.image.shape[1] / float(img.shape[1])
                      if self.image is not None else 1.0)

    def refresh_report(self) -> None:
        txt = "\n".join(self.ctl.report_lines())
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", txt)
        self.report.configure(state="disabled")

    # ---------------------------------------------------------- ROI drag
    def drag_start(self, ev) -> None:
        self.drag0 = (int(ev.x), int(ev.y))
        self.rect_id = self.canvas.create_rectangle(ev.x, ev.y, ev.x, ev.y,
                                                    outline="#ff5050", width=2)

    def drag_move(self, ev) -> None:
        if self.drag0 and self.rect_id:
            self.canvas.coords(self.rect_id, self.drag0[0], self.drag0[1],
                               ev.x, ev.y)

    def drag_end(self, ev) -> None:
        if not self.drag0 or self.ctl.image is None:
            self.drag0 = None
            return
        x0, y0 = self.drag0
        x1, y1 = int(ev.x), int(ev.y)
        self.drag0 = None
        s = max(float(self.scale), 1e-6)
        xa, xb = sorted((max(x0, 0), max(x1, 0)))
        ya, yb = sorted((max(y0, 0), max(y1, 0)))
        if xb - xa < 6 or yb - ya < 6:          # a click, not a drag -> clear
            self.ctl.set_roi(None)
        else:
            self.ctl.set_roi((int(xa * s), int(ya * s),
                              max(int((xb - xa) * s), 8), max(int((yb - ya) * s), 8)))
        self.after_change()

    def set_status(self, msg: str) -> None:
        self.status.configure(text=msg)
        self.root.update_idletasks()


# ---------------------------------------------------------------------------
# self test + command line
# ---------------------------------------------------------------------------
def selftest(image: Optional[str] = None, verbose: bool = True) -> int:
    """Drive the controller exactly as the buttons do, with no display."""
    import tempfile
    img = image or os.path.join(_ROOT, "dataset", "student_created",
                                "003_clean_manuscript_a_severe.jpg")
    print(f"[selftest] tkinter: {'yes' if HAS_TK else 'NO'}   "
          f"Pillow: {'yes' if HAS_PIL else 'NO'}")
    print(f"[selftest] image: {img}")
    if not os.path.exists(img):
        print("[selftest] FAIL - image not found")
        return 1
    ctl = AppController(thread=False)
    ok, err = ctl.load(img)
    if not ok:
        print("[selftest] FAIL load:", err)
        return 1
    r = ctl.result
    if r is None or not r.ok:
        print("[selftest] FAIL analysis:", r.error if r else "no result")
        return 1
    checks = [("processed image", r.processed is not None),
              ("annotated image", r.annotated is not None),
              ("damage mask", r.damage_mask is not None and r.damage_mask.any()),
              ("text mask", r.text_mask is not None),
              ("regions listed", bool(r.damage.regions)),
              ("quality index in 0..100", 0.0 <= r.quality.score <= 100.0),
              ("condition flag", r.condition.pass_fail in ("PASS", "WARN", "FAIL")),
              ("readability before/after", bool(r.readability.before) and
               bool(r.readability.after)),
              ("stage images", len(r.stages) >= 5),
              ("filter comparison", len(r.denoise_comparison) >= 2),
              ("before/after view", ctl.view_image("before/after") is not None),
              ("mask view", ctl.view_image("damage mask") is not None),
              ("report panel lines", len(ctl.report_lines()) > 20)]
    print("[selftest] " + r.summary())
    bad = 0
    for name, passed in checks:
        print(f"           {'ok  ' if passed else 'FAIL'} {name}")
        bad += 0 if passed else 1
    # clamping: out-of-range values must be pulled back into the range the
    # widget offers, never crash and never reach the pipeline unbounded
    orig = ctl.values()
    for attr, _l, kind, lo, hi, _s, _h in CONTROLS:
        if kind in ("int", "float"):
            for probe, want in ((hi + 1000.0, hi), (lo - 1000.0, lo)):
                ctl.set(attr, probe)
                got = float(getattr(ctl.params, attr))
                if abs(got - want) > (0.75 if kind == "int" else 1e-6):
                    print(f"           FAIL clamp {attr}: {probe} -> {got}, "
                          f"expected {want}")
                    bad += 1
            mid = (lo + hi) / 2.0
            ctl.set(attr, mid)
            got = float(getattr(ctl.params, attr))
            want = int(round(mid)) if kind == "int" else mid
            if attr in ("denoise_ksize", "adaptive_block"):
                # these two are forced to odd values by validated(), because
                # every OpenCV kernel that uses them must be odd
                want = max(want | 1, 3) if want % 2 == 0 else want
            if abs(got - want) > 1e-6:
                print(f"           FAIL round-trip {attr}: {mid} -> {got}")
                bad += 1
    for attr, v in orig.items():
        ctl.set(attr, v)
    print("           ok   parameter clamping + round-trip")
    # an unknown parameter must be refused, not invented
    ok2, msg2 = ctl.set("not_a_parameter", 3)
    assert not ok2 and "unknown" in msg2
    print("           ok   unknown parameter refused")
    # ROI
    H, W = ctl.image.shape[:2]
    ctl.set_roi((int(W * 0.1), int(H * 0.1), int(W * 0.4), int(H * 0.3)))
    assert ctl.result is not None and ctl.result.ok, ctl.result.error
    print(f"[selftest] ROI run ok - damage {ctl.result.damage.damage_pct:.2f} % "
          f"of the ROI")
    ctl.set_roi(None)
    # corrupt file rejection
    with tempfile.TemporaryDirectory() as td:
        bogus = os.path.join(td, "broken.jpg")
        with open(bogus, "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0 garbage, not a jpeg")
        ok3, err3 = ctl.load(bogus)
        print(f"[selftest] corrupt file rejected: {not ok3}  "
              f"({err3[:60]}...)")
        bad += 1 if ok3 else 0
        ok4, msg4, files = ctl.export(td)
        names = sorted(os.path.basename(f) for f in files)
        need = ("_processed.png", "_annotated.png", "_report.json",
                "_summary.csv", "_regions.csv", "_report.txt")
        miss = [n for n in need if not any(x.endswith(n) for x in names)]
        print(f"[selftest] export: {len(names)} files, missing={miss or 'none'}")
        bad += len(miss)
        ok5, msg5, res5 = ctl.batch(os.path.join(_ROOT, "dataset", "clean"),
                                    os.path.join(td, "batch"), ext=".jpg",
                                    max_images=3)
        print(f"[selftest] batch: {'ok' if ok5 else 'FAIL'} - {msg5}")
        has_csv = os.path.exists(os.path.join(td, "batch", "results.csv"))
        print(f"[selftest] batch results.csv: {'ok' if has_csv else 'FAIL'}")
        bad += 0 if (ok5 and has_csv) else 1
    print("[selftest]", "PASS" if bad == 0 else f"FAIL ({bad} problem(s))")
    if verbose:
        print("\n".join("  log| " + x for x in ctl.log[-6:]))
    return 0 if bad == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HeritageVision GUI (Part D)")
    ap.add_argument("image", nargs="?", help="image to open")
    ap.add_argument("--selftest", action="store_true",
                    help="run the GUI logic without a display and exit")
    ap.add_argument("--folder", help="batch mode: process this folder and exit")
    ap.add_argument("--out", help="batch output folder",
                    default=os.path.join(_ROOT, "outputs", "gui_batch"))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.image)
    ctl = AppController(thread=True)
    if a.folder:
        ok, msg, _ = ctl.batch(a.folder, a.out or os.path.join(a.folder,
                                                                "heritage_out"))
        print(("[batch] ok: " if ok else "[batch] failed: ") + msg)
        return 0 if ok else 1
    if not HAS_TK:
        print("This Python has no tkinter, so no window can be opened here.\n"
              f"  import error: {TK_ERROR}\n"
              "Install it (Debian/Ubuntu: sudo apt-get install python3-tk), or use\n"
              "the command line instead:\n"
              "  python app.py --folder <dir> --out <dir>\n"
              "  python -m heritage_vision.tools.run_batch -i <image|folder> "
              "-o <out>\n"
              "The logic behind every button is verified by:  python app.py "
              "--selftest")
        return 2
    root = tk.Tk()                                        # pragma: no cover
    app = HeritageApp(root, ctl)
    if a.image:
        ok, err = ctl.load(a.image)
        if not ok:
            messagebox.showerror("Cannot open image", err)
        root.after(200, app._wait_done)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
