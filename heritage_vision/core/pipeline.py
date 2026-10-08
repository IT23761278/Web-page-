"""
heritage_vision.core.pipeline
=============================

The integrated pipeline (Part B/C).  All stages are implemented in separate
modules and *wired together here*, in the order required by section 4 of the
assignment:

    1 input image or folder
    2 validation + metadata                      (utils.load_image, metadata)
    3 pre-processing + enhancement               (geometry, enhancement, denoise)
    4 ROI selection or segmentation                (segmentation, damage.page_mask)
    5 feature extraction + domain analysis         (damage, features, matching)
    6 decision / measurement / score / visualisation (scoring, annotate)
    7 export + evaluation log                      (export)

Every stage returns an artefact that the next stage consumes, and every
intermediate image is kept in ``result.stages`` so the GUI can show it and the
report can quote it.  Timing per stage is recorded (Part D "processing time").
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import damage as dmg
from . import enhancement as enh
from . import denoise as den
from . import features as ft
from . import frequency as fq
from . import geometry as geo
from . import restoration as res
from . import scoring as sc
from .config import PipelineParams
from .colorspace import deyellow, decompose
from .metadata import ImageMetadata
from .segmentation import MorphPlan, adaptive_gray, mask_stats, otsu, sauvola
from .utils import Timer, json_dump, load_image, list_image_files, save_image, to_bgr, to_gray
from .utils import odd


@dataclass
class ProcessResult:
    """Everything one processed image produces (also the CSV row source)."""
    name: str = ""
    path: str = ""
    ok: bool = True
    error: str = ""
    params: Optional[PipelineParams] = None
    metadata: Optional[ImageMetadata] = None
    input_image: Optional[np.ndarray] = None
    processed: Optional[np.ndarray] = None      #: enhanced + restored output
    annotated: Optional[np.ndarray] = None      #: visual decision overlay
    roi: Optional[np.ndarray] = None
    text_mask: Optional[np.ndarray] = None
    damage_mask: Optional[np.ndarray] = None
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    damage: Optional[dmg.DamageReport] = None
    quality: Optional[sc.QualityAssessment] = None
    condition: Optional[sc.ConditionAssessment] = None
    readability: Optional[sc.ReadabilityAssessment] = None
    geometry: Dict[str, Any] = field(default_factory=dict)
    denoise: Optional[den.DenoiseReport] = None
    denoise_comparison: List[den.DenoiseReport] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ api
    def summary(self) -> str:
        if not self.ok:
            return f"FAILED {self.name}: {self.error}"
        d, q, c = self.damage, self.quality, self.condition
        return (f"{self.name}: damage {d.damage_pct:.2f}% ({d.damage_mm2:.1f} mm2, "
                f"{d.n_damage_total} regions) | condition {c.score:.1f} "
                f"{c.severity_class}/{c.pass_fail} | quality {q.score:.1f} {q.grade} | "
                f"readability {self.readability.before.get('score',0):.0f}->"
                f"{self.readability.after.get('score',0):.0f} | "
                f"{self.timings.get('TOTAL',0):.0f} ms")

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": self.name, "file": os.path.basename(self.path) if self.path else "",
            "status": "OK" if self.ok else "ERROR", "error": self.error,
            "width": self.metadata.width if self.metadata else 0,
            "height": self.metadata.height if self.metadata else 0,
            "channels": self.metadata.channels if self.metadata else 0,
            "bit_depth": self.metadata.bits_per_sample if self.metadata else 8,
            "file_size_kb": round(self.metadata.file_size_bytes / 1024.0, 1)
            if self.metadata else 0,
            "format": self.metadata.format if self.metadata else "",
            "total_time_ms": round(self.timings.get("TOTAL", 0.0), 2),
        }
        row.update({f"t_{k}": round(float(v), 2) for k, v in self.timings.items()
                    if k != "TOTAL"})
        if self.metadata:
            reg = self.metadata.region or {}
            row["roi_area_pct"] = reg.get("area_pct_of_image", 0)
        if self.damage:
            row.update(self.damage.as_row())
        if self.quality:
            row.update(self.quality.as_dict())
        if self.condition:
            row.update(self.condition.as_dict())
        if self.readability:
            row.update(self.readability.as_dict())
        if self.denoise:
            row["denoise_method"] = self.denoise.method
            row.update({f"dn_{k}": v for k, v in self.denoise.metrics.items()})
        row.update({f"g_{k}": v for k, v in self.geometry.items()})
        row.update(self.metrics)
        return row

    def save(self, folder: str, ext: str = ".png", include_stages: bool = True,
             quality: int = 92) -> List[str]:
        """Write processed/annotated images + per-stage debug images."""
        os.makedirs(folder, exist_ok=True)
        base = os.path.splitext(self.name)[0] or "image"
        written: List[str] = []
        if self.processed is not None:
            written.append(save_image(os.path.join(folder, base + "_processed" + ext),
                                      self.processed, quality))
        if self.annotated is not None:
            written.append(save_image(os.path.join(folder, base + "_annotated" + ext),
                                      self.annotated, quality))
        if include_stages:
            for k, img in self.stages.items():
                if img is None:
                    continue
                out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                written.append(save_image(
                    os.path.join(folder, f"{base}_stage-{k}{ext}"), out, quality))
        json_dump({"summary": self.summary(), **self.to_dict(),
                    "log": self.log,
                    "params": self.params.as_dict() if self.params else {}},
                   os.path.join(folder, base + "_report.json"))
        written.append(os.path.join(folder, base + "_report.json"))
        return written


class HeritagePipeline:
    """Stateless per-call processing object (safe to reuse in a worker thread)."""

    STAGES = ("input", "geometry", "background", "enhanced", "denoised",
              "binary", "roi", "text", "damage", "restored")

    def __init__(self, params: Optional[PipelineParams] = None) -> None:
        self.params = (params or PipelineParams()).validated()

    # ------------------------------------------------------------------
    def set_params(self, **kw: Any) -> PipelineParams:
        for k, v in kw.items():
            if hasattr(self.params, k) and v is not None:
                setattr(self.params, k, v)
        self.params = self.params.validated()
        return self.params

    # ------------------------------------------------------------------
    def process_file(self, path: str, roi_box: Optional[Sequence[int]] = None,
                      downscale_to: Optional[int] = None) -> ProcessResult:
        try:
            img = load_image(path, downscale_to=downscale_to)
        except Exception as exc:                     # ImageLoadError / OSError
            r = ProcessResult(name=os.path.basename(path), path=path, ok=False,
                              error=str(exc))
            return r
        r = self.process(img, name=os.path.basename(path), roi_box=roi_box)
        r.path = path
        try:
            r.metadata = ImageMetadata.from_file(path, img, self.params.target_dpi)
            if r.roi is not None:
                r.metadata.set_region(r.roi, "roi", self.params.target_dpi)
        except Exception as exc:
            r.log.append(f"metadata warning: {exc}")
        return r

    # ------------------------------------------------------------------
    def process(self, image: np.ndarray, name: str = "image",
                roi_box: Optional[Sequence[int]] = None,
                run_denoise_comparison: bool = False) -> ProcessResult:
        """The whole Part-B + Part-C pipeline for a single image."""
        p = self.params
        t = Timer()
        r = ProcessResult(name=name, params=p, input_image=image.copy())
        try:
            with t.stage("preprocess"):
                cur = to_bgr(image)
                r.stages["input"] = cur.copy()
                raw_after_geo = cur.copy()
                # ---- 2b. geometry -----------------------------------------
                cur, valid, glog = self._geometry(cur, roi_box)
                raw_after_geo = cur.copy()      #: same frame as all later masks
                r.geometry = glog
                r.log += glog["log_lines"]
                # ---- 2c. scale normalisation -------------------------------
                scale = p.target_dpi / float(glog.get("src_dpi", p.target_dpi) or
                                             p.target_dpi)
                if abs(scale - 1.0) > 0.02:
                    cur = geo.scale_to_dpi(cur, int(glog.get("src_dpi", p.target_dpi)),
                                           p.target_dpi)
                    valid = geo.scale_to_dpi(valid, int(glog.get("src_dpi", p.target_dpi)),
                                             p.target_dpi)
                    r.geometry["dpi_scale"] = round(scale, 4)
            with t.stage("enhancement"):
                # ---- 3a. colour spaces -------------------------------------
                dec = decompose(cur, spaces=("HSV", "Lab", "YCrCb"))
                # ---- 3b. illumination correction ---------------------------
                bgres = enh.correct_illumination(
                    dec.gray, method=p.bg_normalisation, kernel_mm=p.bg_kernel_mm,
                    dpi=p.target_dpi, sigma=p.bg_smooth)
                r.stages["background"] = bgres["background"]
                gray_corr = bgres["image"]
                img_corr = gray_corr if cur.ndim == 2 else bgres["image"]
                if p.bg_normalisation != "none" and cur.ndim == 3:
                    img_corr = enh.transfer_gray_correction(cur, dec.gray, gray_corr)
                # ---- 3c. de-yellowing (optional, preserves colour evidence)
                if p.damage_space == "Lab":
                    img_corr = deyellow(img_corr, 0.5)
                # ---- 3d. contrast -----------------------------------------
                if p.clahe_enabled:
                    img_corr = enh.clahe_color(img_corr, p.clahe_clip_limit,
                                               p.clahe_tile_grid, space="Lab")
                if p.percentile_stretch > 0:
                    img_corr = enh.stretch_contrast(img_corr, p.percentile_stretch)
                if abs(p.gamma - 1.0) > 1e-3:
                    img_corr = enh.gamma_correction(img_corr, p.gamma)
                r.stages["enhanced"] = img_corr
            with t.stage("denoise"):
                # ---- 3e. noise reduction ----------------------------------
                comparison = den.compare_filters(
                    img_corr, methods=("none", "gaussian", "median", "bilateral",
                                       "guided", "wiener"),
                    params={"ksize": p.denoise_ksize,
                            "sigma_color": p.bilateral_sigma_color,
                            "sigma_space": p.bilateral_sigma_space, "h": p.nlm_h}) \
                    if run_denoise_comparison else []
                r.denoise_comparison = comparison
                chosen = p.denoise_method
                if comparison and chosen == "auto":
                    chosen = comparison[0].method
                out = den.apply_method(img_corr, chosen,
                                       {"ksize": p.denoise_ksize,
                                        "sigma_color": p.bilateral_sigma_color,
                                        "sigma_space": p.bilateral_sigma_space,
                                        "h": p.nlm_h})
                rep = next((c for c in comparison if c.method == chosen), None) \
                    or den.DenoiseReport(method=chosen, params={}, time_ms=0.0,
                                         image=out)
                rep.image = out
                rep.selected = True
                if not rep.metrics:
                    rep.metrics = {"residual_noise": den.estimate_noise_sigma(out),
                                   "edge_retention": 1.0, "time_ms": 0.0}
                r.denoise = rep
                r.stages["denoised"] = out
                work = out
            with t.stage("segmentation"):
                # ---- 4. ROI + binary layers --------------------------------
                gwork = to_gray(work)
                page = dmg.page_mask(work)
                roi = cv2.bitwise_and(page, valid)
                if roi_box is not None:
                    roi = geo.crop_mask(roi, roi_box)
                if roi.sum() < 0.02 * roi.size * 255:
                    roi = valid
                r.roi = roi
                r.stages["roi"] = roi
                binary = self._binarise(gwork, roi)
                r.stages["binary"] = binary
                plan = MorphPlan(open_px=p.morph_open_px, close_px=p.morph_close_px)
                text = plan.apply(binary)
                text = self._filter_components(text, roi, p)
                r.text_mask = text
                r.stages["text"] = text
            with t.stage("analysis"):
                # ---- 5. domain analysis -----------------------------------
                # which frame the detectors see (see PipelineParams.analysis_source)
                src_for_analysis = (raw_after_geo
                                     if getattr(p, "analysis_source", "raw") == "raw"
                                     else work)
                if src_for_analysis.shape[:2] != work.shape[:2]:
                    src_for_analysis = cv2.resize(src_for_analysis,
                                                  (work.shape[1], work.shape[0]))
                rep = dmg.analyse_damage(src_for_analysis, p, text, roi,
                                         enhanced=work, name=name)
                r.damage = rep
                r.damage_mask = rep.masks["damage"]
                r.stages["damage"] = rep.masks["damage"]
                r.stages["crack"] = rep.masks["vesselness"]
                r.metrics["texture_glcm_contrast"] = ft.texture_bundle(gwork)["glcm_contrast"]
                r.metrics["spectrum_hf_ratio"] = fq.spectrum_metrics(gwork)["hf_power_ratio"]
                r.metrics["fft_blur_sigma_px"] = fq.estimate_blur_std(gwork)
                r.extra["shape_histogram"] = ft.shape_histogram(
                    [self._region_to_feat(x) for x in rep.regions], "circularity")
                # ---- 6. restoration ---------------------------------------
                rr = res.restore(work, rep.masks["damage"], rep.masks["crack"],
                                 text, p)
                r.processed = rr.restored
                r.extra["restoration_ops"] = rr.operations
                r.extra["restoration_stats"] = rr.stats
                r.stages["restored"] = rr.restored
                r.extra["restored_damage_mask"] = rr.mask_used
                # "inpainting only" on the untouched input frame.  Charging the
                # full processed output against the pre-damage reference would
                # also penalise the deliberate contrast enhancement, so the
                # fidelity number is measured on the inpainting by itself.
                raw = raw_after_geo
                if raw.shape[:2] != work.shape[:2]:
                    raw = cv2.resize(raw, (work.shape[1], work.shape[0]))
                only = res.inpaint_damage(raw, rep.masks["damage"],
                                          radius=p.inpaint_radius,
                                          method=p.inpaint_method,
                                          dilate_px=p.inpaint_dilate_px)
                r.stages["inpaint_only"] = only.restored
                rq = sc.restoration_quality(to_gray(raw), to_gray(only.restored),
                                            rep.masks["damage"])
                r.metrics.update({("hq_" + k): v for k, v in rq.items()})
                r.extra["restoration_quality"] = rq
            with t.stage("scoring"):
                # ---- 6b. decisions ----------------------------------------
                roi_stats = mask_stats(roi, dpi=p.target_dpi)
                r.quality = sc.assess_quality(to_gray(raw_after_geo), roi,
                                              p.target_dpi, p)
                r.condition = sc.assess_condition(rep, p, float(roi_stats["area_px"]))
                raw_gray = to_gray(raw_after_geo)
                if raw_gray.shape != roi.shape:
                    raw_gray = cv2.resize(raw_gray, (roi.shape[1], roi.shape[0]))
                before = sc.measure_readability(raw_gray, text, roi, p.target_dpi)
                r.extra["raw_after_geometry"] = None
                gafter = to_gray(r.processed)
                binary_after = self._binarise(gafter, roi)
                text_after = self._filter_components(
                    MorphPlan(p.morph_open_px, p.morph_close_px).apply(binary_after),
                    roi, p)
                after = sc.measure_readability(gafter, text_after, roi, p.target_dpi)
                r.readability = sc.ReadabilityAssessment(
                    before=before, after=after,
                    improvement_pct=round(
                        (after["score"] - before["score"]) /
                        max(before["score"], 1.0) * 100.0, 2))
                ts = sc.text_statistics(text, roi, p.target_dpi, p.min_text_area_px)
                r.readability.lines = ts["lines"]
                r.readability.words = ts["words"]
                r.readability.characters = ts["components"]
                areas = [x.area_mm2 for x in rep.regions] or [0.0]
                r.readability.mean_char_area_mm2 = round(float(np.mean(areas)), 4)
                r.readability.verdict = ("IMPROVED" if after["score"] > before["score"] + 2
                                         else "NEUTRAL" if after["score"] >= before["score"] - 2
                                         else "DEGRADED")
                r.metrics.update({"roi_area_pct": roi_stats["area_pct"],
                                  "roi_area_mm2": roi_stats["area_mm2"],
                                  "text_ink_pct": float(np.count_nonzero(
                                      cv2.bitwise_and(text, roi)) /
                                      max(roi_stats["area_px"], 1.0)) * 100.0})
                r.extra["authenticity"] = sc.authenticity_indicators(gwork, p.target_dpi)
            with t.stage("annot"):
                r.annotated = self.annotate(cur, rep, r)
        except Exception as exc:                     # never crash the UI thread
            r.ok = False
            r.error = f"{type(exc).__name__}: {exc}"
            r.log.append("PIPELINE ERROR " + r.error)
            import traceback
            r.extra["traceback"] = traceback.format_exc(limit=6)
        finally:
            r.timings = dict(t.records)
            r.timings["TOTAL"] = t.total_ms
            if r.ok:
                r.log.append(f"done in {t.total_ms:.0f} ms")
        return r

    # ------------------------------------------------------------------
    def _geometry(self, cur: np.ndarray, roi_box: Optional[Sequence[int]]):
        p = self.params
        log: List[str] = []
        H, W = cur.shape[:2]
        glog = {"src_dpi": p.target_dpi, "log_lines": log}
        valid = np.full((H, W), 255, np.uint8)
        if p.rectify:
            pts, how = geo.find_page_corners(cur)
            if pts is not None:
                cur2, info = geo.four_point_transform(cur, pts)
                valid = np.full(cur2.shape[:2], 255, np.uint8)
                cur = cur2
                glog.update({"rectify_method": how, **{k: float(v) for k, v in info.items()
                                                       if isinstance(v, (int, float))}})
                glog["rectified"] = 1.0
                log.append(f"rectified via {how} "
                           f"({info['width_px']}x{info['height_px']} px)")
            else:
                glog["rectified"] = 0.0
                log.append("rectification skipped: page quad not found")
        if p.deskew:
            ang, info = geo.estimate_skew_angle(to_gray(cur), method="projection",
                                                max_angle=p.deskew_max_angle)
            conf = float(info.get("confidence", info.get("score", 0.0)))
            if abs(ang) >= 0.25:
                # estimate_skew_angle returns the CORRECTION angle
                cur, m2 = geo.rotate(cur, ang)
                valid = cv2.bitwise_and(cv2.resize(valid, (cur.shape[1], cur.shape[0]))
                                        if valid.shape[:2] != cur.shape[:2] else valid, m2)
                glog["deskew_angle_deg"] = round(ang, 3)
                glog["deskew_confidence"] = round(conf, 4)
                glog["deskew_rejected"] = float(info.get("rejected", 0.0))
                log.append(f"deskew {ang:+.2f} deg (projection score "
                           f"{info.get('score', 0):.3e})")
            else:
                glog["deskew_angle_deg"] = 0.0
                log.append(f"deskew skipped (angle {ang:+.2f} deg below 0.25)")
        if p.auto_crop:
            cur, box = geo.auto_crop(cur, p.crop_margin_px)
            valid = geo.crop_mask(valid, box) if valid.shape[:2] == cur.shape[:2] \
                else np.full(cur.shape[:2], 255, np.uint8)
            glog["auto_crop_box"] = ",".join(str(v) for v in box)
            log.append(f"auto-crop to {box}")
        glog["applied"] = "+".join(
            [n for n, on in (("rectify", p.rectify), ("deskew", bool(glog.get("deskew_angle_deg"))),
                              ("autocrop", p.auto_crop)) if on]) or "none"
        glog["log_lines"] = log
        return cur, valid, glog

    # ------------------------------------------------------------------
    def _binarise(self, gray: np.ndarray, roi: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.params
        g = gray if gray.ndim == 2 else to_gray(gray)
        if p.binarisation == "adaptive":
            mask = adaptive_gray(g, p.adaptive_block, p.adaptive_c)
        elif p.binarisation == "sauvola":
            mask, _ = sauvola(g, window=odd(p.adaptive_block), k=0.2)
        else:
            # Otsu on the ROI only, so a dark border cannot shift the threshold
            if roi is not None and np.any(roi > 0):
                vals = g[roi > 0]
                t = float(cv2.threshold(vals.reshape(-1, 1), 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
                mask = (g < t).astype(np.uint8) * 255
            else:
                mask, _ = otsu(g)
        if roi is not None:
            mask = cv2.bitwise_and(mask, mask, mask=(roi > 0).astype(np.uint8))
        return mask

    # ------------------------------------------------------------------
    def _filter_components(self, mask: np.ndarray, roi: np.ndarray,
                           p: PipelineParams) -> np.ndarray:
        """Drop components that are too small to be writing or too big to be a
        glyph (page shadow, scanner artefact) and that lie outside the ROI."""
        roi_area = max(float(np.count_nonzero(roi > 0)), 1.0)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        out = np.zeros_like(mask)
        for i in range(1, n):
            x, y, w, h, a = [int(v) for v in stats[i]]
            if a < p.min_text_area_px:
                continue
            if a > p.max_text_area_frac * roi_area:
                continue
            out[lab == i] = 255
        return cv2.bitwise_and(out, out, mask=(roi > 0).astype(np.uint8))

    # ------------------------------------------------------------------
    @staticmethod
    def _region_to_feat(r) -> Dict[str, float]:
        return {"circularity": r.circularity, "elongation": r.elongation,
                "solidity": r.solidity, "area_mm2": r.area_mm2}

    # ------------------------------------------------------------------
    def annotate(self, image: np.ndarray, rep: dmg.DamageReport,
                 r: ProcessResult) -> np.ndarray:
        """Draw the decision layer (class-coloured outlines, ROI box, header)."""
        from .annotate import draw_report
        return draw_report(image, rep, r)

    # ------------------------------------------------------------------
    def process_folder(self, folder: str, out_dir: Optional[str] = None,
                       recursive: bool = True, max_images: Optional[int] = None,
                       progress=None) -> Tuple[List[ProcessResult], List[str]]:
        """Batch processing (Part D).  Returns (results, rejected_files)."""
        files = list_image_files(folder, recursive=recursive)
        if max_images:
            files = files[:max_images]
        results: List[ProcessResult] = []
        rejected: List[str] = []
        for i, f in enumerate(files):
            res_obj = self.process_file(f, downscale_to=2400)
            results.append(res_obj)
            if not res_obj.ok:
                rejected.append(f"{os.path.basename(f)}: {res_obj.error}")
            if progress:
                progress(i + 1, len(files), res_obj)
            if out_dir:
                sub = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0])
                try:
                    res_obj.save(sub)
                except Exception as exc:
                    rejected.append(f"{os.path.basename(f)}: export failed {exc}")
        return results, rejected
