"""
tools/evaluate.py
=================

Part D evaluation harness.  Runs the application over the controlled test set
(``*.gt.json`` cases produced by tools/make_dataset.py) and prints/writes the
measures the marking sheet asks for, in four groups:

1. segmentation quality (damage mask vs recorded damage mask)
       IoU, Dice, precision, recall, F1, area error in mm2
2. detection quality (region count vs connected components of the GT mask)
       count error (absolute and relative), per-class counts
3. enhancement / restoration fidelity (degraded, enhanced and restored against
   the pre-damage clean render)
       PSNR, SSIM, RMSE, contrast improvement
4. geometry accuracy (recorded acquisition parameters)
       deskew angle error in degrees, page-corner error in px
       (rectification), text preservation IoU

Everything is per-image CSV plus a Markdown summary that is pasted directly
into the technical report, so no number in the report is hand-typed.

Usage
-----
    python -m heritage_vision.tools.evaluate --dataset dataset/unseen_test --out outputs/eval_unseen
    python -m heritage_vision.tools.evaluate --dataset dataset/student_created --out outputs/eval_tuning
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heritage_vision.core import metrics as M                  # noqa: E402
from heritage_vision.core.config import PipelineParams          # noqa: E402
from heritage_vision.core.evaluate import (PRIMARY_METRICS,    # noqa: E402
                                           find_test_cases, load_test_case)
from heritage_vision.core.export import write_csv, write_json   # noqa: E402
from heritage_vision.core.geometry import order_points          # noqa: E402
from heritage_vision.core.pipeline import HeritagePipeline      # noqa: E402
from heritage_vision.core.utils import load_image               # noqa: E402


def evaluate_geometry(pipe: HeritagePipeline, case) -> Dict[str, float]:
    """Deskew and rectification accuracy against the recorded parameters."""
    out: Dict[str, float] = {}
    try:
        img = load_image(case.degraded_path)
    except Exception:
        return out
    rec = case.meta.get("params", {})
    # --- deskew: measured angle vs applied angle -------------------------
    from heritage_vision.core.geometry import estimate_skew_angle
    from heritage_vision.core.utils import to_gray
    ang, info = estimate_skew_angle(to_gray(img), method="projection")
    truth = rec.get("skew_deg")
    out["skew_measured_deg"] = round(float(ang), 3)
    out["deskew_rejected"] = float(info.get("rejected", 0.0))
    if truth is None:
        out["skew_abs_error_deg"] = float("nan")   # perspective case
    else:
        # estimate_skew_angle returns the CORRECTION, the generator recorded the
        # DISTORTION, so the residual error is |correction + distortion| and a
        # correct estimator scores 0 (the sign convention was a bug once).
        truth = float(truth)
        out["skew_truth_deg"] = round(truth, 3)
        out["skew_abs_error_deg"] = round(abs(float(ang) + truth), 3)
    # --- rectification: detected quad vs recorded quad -------------------
    from heritage_vision.core.geometry import find_page_corners
    pts, how = find_page_corners(img)
    gt_pts = case.meta.get("corners")
    if pts is not None and gt_pts:
        p = order_points(np.asarray(pts, np.float32))
        g = order_points(np.asarray(gt_pts, np.float32))
        if p.shape == g.shape:
            d = np.linalg.norm(p - g, axis=1)
            out["corner_method"] = how
            out["corner_mean_error_px"] = round(float(d.mean()), 2)
            out["corner_max_error_px"] = round(float(d.max()), 2)
    return out


_CANVAS_CACHE: Dict[str, Any] = {}


def _reference_canvas(case, args=None):
    """The pre-damage image the degraded file was made from.

    Stored on disk only with ``--keep-canvas``; otherwise it is re-created by
    calling the generator with the parameters recorded in the sidecar.  The
    degradation is a deterministic function of (source, seed, severity, ...), so
    the reconstruction is exact, and the same code path is what produced the
    test image in the first place.
    """
    import cv2
    cc = case.meta.get("clean_canvas") or case.meta.get("clean_image")
    if cc:
        cp = cc if os.path.isabs(cc) else os.path.join(os.path.dirname(case.degraded_path), cc)
        if os.path.exists(cp):
            return cv2.imread(cp, cv2.IMREAD_COLOR)
    if args is not None and getattr(args, "canvas", "auto") == "off":
        return None
    regen = case.meta.get("canvas_regen")
    if not regen:
        return None
    key = json.dumps(regen, sort_keys=True, default=str)
    if key in _CANVAS_CACHE:
        return _CANVAS_CACHE[key]
    if args is not None and getattr(args, "canvas", "auto") == "stored-only":
        return None
    from heritage_vision.tools import make_dataset as md
    src = regen.get("source") or ""
    if not os.path.exists(src):
        print(f"[evaluate] cannot rebuild reference for {case.name}: "
              f"source {src} missing (was the dataset moved?)")
        return None
    got = md.load_clean_source(src, gt_dir=regen.get("gt_dir"),
                               max_side=int(regen.get("max_side", 0) or 0))
    if got is None:
        return None
    _name, clean, ink, sheet = got
    img2, gt = md.degrade(clean, ink, sheet, int(regen["seed"]),
                         severity=regen["severity"], photo=bool(regen["photo"]),
                         perspective=bool(regen["perspective"]), desk=bool(regen["desk"]),
                         pad_frac=float(regen.get("pad_frac", 0.035)),
                         return_canvas=True)
    canvas = gt.get("clean_image_canvas")
    if canvas is None:
        canvas = img2
    _CANVAS_CACHE[key] = canvas
    return canvas


def run(args) -> Dict[str, Any]:
    p = PipelineParams()
    for k, v in (("denoise_method", args.denoise), ("binarisation", args.binarisation),
                 ("inpaint", not args.no_inpaint), ("damage_sensitivity", args.sensitivity),
                 ("analysis_source", getattr(args, "analyse_on", None)),
                 ("deskew", not args.deskew_off), ("rectify", args.rectify)):
        if v is not None:
            setattr(p, k, v)
    # Segmentation / fidelity metrics need the degraded image and the ground
    # truth on the SAME pixel grid, so all geometric resampling is disabled for
    # this run; the geometric transforms are measured separately with
    # --geometry (angle error, corner error) where their own reference exists.
    p.deskew = False
    p.rectify = False
    p.auto_crop = False
    pipe = HeritagePipeline(p)
    cases = find_test_cases(args.dataset)
    if getattr(args, "filter", ""):
        cases = [c for c in cases if args.filter in c.name]
    if getattr(args, "limit", 0):
        cases = cases[:args.limit]
    if not cases:
        print(f"[evaluate] no *.gt.json under {args.dataset} - run "
              "tools/make_dataset.py --degrade first")
        return {}
    os.makedirs(args.out, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        row = _evaluate_one(pipe, case, args)
        if args.geometry:
            row.update(evaluate_geometry(pipe, case))
        rows.append(row)
        if not args.quiet:
            print(f"[{i:>3}/{len(cases)}] {row.get('name','')[:44]:44s} "
                  f"IoU {row.get('damage_iou', float('nan')):.3f} "
                  f"F1 {row.get('damage_f1', float('nan')):.3f} "
                  f"areaErr {row.get('area_rel_error_pct', float('nan')):+6.1f}% "
                  f"dPSNR {row.get('psnr_gain_db', float('nan')):+5.2f}dB "
                  f"{row.get('time_ms', 0):7.0f} ms")
    agg = M.summarise(rows, PRIMARY_METRICS + ["skew_abs_error_deg",
                                               "corner_mean_error_px"])
    per_sev = _group(rows, "severity")
    payload = {"n": len(rows), "aggregate": agg, "per_severity": per_sev,
               "params": pipe.params.as_dict(), "dataset": args.dataset,
               "failed": [r for r in rows if r.get("status") != "OK"]}
    write_csv(rows, os.path.join(args.out, "per_image_metrics.csv"))
    write_json(payload, os.path.join(args.out, "evaluation.json"))
    md = _markdown(rows, agg, per_sev, pipe, args.dataset)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n[evaluate] aggregate over %d controlled test images" % len(rows))
    for k in PRIMARY_METRICS + ["skew_abs_error_deg", "corner_mean_error_px"]:
        key = f"{k}_mean"
        if key in agg:
            print(f"   {k:28s} mean {agg[key]:7.3f}  med {agg.get(k+'_median',0):7.3f}"
                  f"  iqr {agg.get(k+'_iqr',0):6.3f}  std {agg.get(k+'_std',0):6.3f}"
                  f"  min {agg.get(k+'_min',0):7.3f}  max {agg.get(k+'_max',0):7.3f}"
                  f"  n {int(agg.get(k+'_n',0))}")
    print(f"[evaluate] wrote per_image_metrics.csv, evaluation.json, results.md "
          f"in {args.out}")
    return payload


def _evaluate_one(pipe: HeritagePipeline, case: TestCase,
                  args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """One test case -> metrics row (segmentation, fidelity, geometry-free)."""
    import time
    row: Dict[str, Any] = {"name": case.name, "case": case.meta.get("case_id", ""),
                           "severity": case.meta.get("severity", "")}
    t0 = time.perf_counter()
    res = pipe.process_file(case.degraded_path)
    row["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    if not res.ok:
        row.update({"status": "ERROR", "error": res.error})
        return row
    row["status"] = "OK"
    pred = res.damage_mask
    gt = case.gt_damage
    if pred is None or gt is None:
        row.update({"status": "ERROR", "error": "missing mask"})
        return row
    if pred.shape[:2] != gt.shape[:2]:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    ov = M.overlap_stats(pred, gt)
    row.update({f"damage_{k}": round(float(v), 4) for k, v in ov.items()})
    row["gt_damage_pct"] = round(float(np.count_nonzero(gt)) / gt.size * 100.0, 3)
    row["pred_damage_pct"] = round(float(np.count_nonzero(pred)) / pred.size * 100.0, 3)
    # per-layer scores, so a weak stain detector is not hidden by a good crack one
    for key, field in (("stain", "stain"), ("crack", "crack"), ("hole", "missing")):
        gp = os.path.join(os.path.dirname(case.degraded_path), "gt",
                          f"{os.path.splitext(case.name)[0]}.{key}_mask.png")
        if os.path.exists(gp) and res.damage and field in res.damage.masks:
            g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            pm = res.damage.masks[field]
            if g is not None and g.shape[:2] != pm.shape[:2]:
                pm = cv2.resize(pm, (g.shape[1], g.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            if g is not None:
                o = M.overlap_stats(pm, g)
                row[f"{key}_iou"] = round(o["iou"], 4)
                row[f"{key}_recall"] = round(o["recall"], 4)
                row[f"{key}_precision"] = round(o["precision"], 4)
                if float(np.count_nonzero(g)) == 0:
                    row[f"{key}_iou"] = float("nan")   # nothing to find
    # region count error
    n_gt, _l, stats, _ = cv2.connectedComponentsWithStats(
        (gt > 0).astype(np.uint8), 8)
    gt_regions = sum(1 for i in range(1, n_gt) if stats[i, cv2.CC_STAT_AREA] >= 120)
    row.update({f"region_{k}": v for k, v in
                M.count_error(res.damage.n_damage_total, gt_regions).items()})
    dpi = pipe.params.target_dpi
    row["pred_area_mm2"] = round(res.damage.damage_mm2, 3)
    row["gt_area_mm2"] = round(ov["truth_area_px"] * (25.4 / dpi) ** 2, 3)
    row.update(M.measurement_error(row["pred_area_mm2"], row["gt_area_mm2"], "area"))
    # text preservation
    if case.gt_text is not None and res.text_mask is not None:
        tm = res.text_mask
        if tm.shape[:2] != case.gt_text.shape[:2]:
            tm = cv2.resize(tm, (case.gt_text.shape[1], case.gt_text.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        o = M.overlap_stats(tm, case.gt_text)
        row["text_iou"] = round(o["iou"], 4)
        row["text_recall"] = round(o["recall"], 4)
        row["text_precision"] = round(o["precision"], 4)
    # fidelity
    try:
        deg = load_image(case.degraded_path)
    except Exception:
        deg = None
    clean = _reference_canvas(case, args) if args is not None else None
    if deg is not None:
        enh = res.stages.get("enhanced", deg)
        row.update({k: round(v, 4) for k, v in M.contrast_improvement(deg, enh).items()})
        row["readability_gain_pct"] = res.readability.improvement_pct if res.readability else 0.0
    if clean is not None:
        for tag, im in (("degraded", deg), ("enhanced", res.stages.get("enhanced")),
                        ("restored", res.processed)):
            if im is None:
                continue
            if im.shape[:2] != clean.shape[:2]:
                im = cv2.resize(im, (clean.shape[1], clean.shape[0]))
            row[f"psnr_{tag}"] = round(M.psnr(clean, im), 2)
            row[f"ssim_{tag}"] = round(M.ssim(clean, im), 4)
            row[f"rmse_{tag}"] = round(M.rmse(clean, im), 2)
        if "psnr_restored" in row and "psnr_degraded" in row:
            row["psnr_gain_db"] = round(row["psnr_restored"] - row["psnr_degraded"], 2)
            row["ssim_gain"] = round(row["ssim_restored"] - row["ssim_degraded"], 4)
        # --- restoration quality where it is claimed: inside the damage mask ---
        if res.damage_mask is not None:
            dm = (case.gt_damage > 0)
            for tag, im in (("degraded", deg), ("enhanced", res.stages.get("enhanced")),
                            ("inpainted", res.stages.get("inpaint_only")),
                            ("restored", res.processed)):
                if im is None:
                    continue
                if im.shape[:2] != clean.shape[:2]:
                    im = cv2.resize(im, (clean.shape[1], clean.shape[0]))
                row["psnr_in_damage_" + tag] = round(M.psnr_masked(clean, im, dm), 2)
                row["rmse_in_damage_" + tag] = round(
                    float(np.sqrt(max(M.mse_masked(clean, im, dm), 0.0))), 3)
            # restoration_gain_db: inpainting only, inside the damage pixels.
            # end_to_end_gain_db: the full processed output, which also carries
            # the deliberate contrast redistribution - reported so the two are
            # never confused.
            # a *gain* is signed after-minus-before, so a negative value means the
            # output moved away from the clean page inside the damage.  It does,
            # and that is expected and explained in report sec. 7.3: inpainting
            # replaces the patch with plausible paper, while the degraded input
            # still carries the occluded ink under a translucent stain, so
            # pixel-wise fidelity to the clean render drops even as the damage
            # evidence is removed (residue_reduction_pct is the measure that
            # rewards it).
            if "psnr_in_damage_degraded" in row and "psnr_in_damage_inpainted" in row:
                row["restoration_gain_db"] = round(
                    row["psnr_in_damage_inpainted"] - row["psnr_in_damage_degraded"], 2)
            if "psnr_in_damage_degraded" in row and "psnr_in_damage_restored" in row:
                row["end_to_end_gain_db"] = round(
                    row["psnr_in_damage_restored"] - row["psnr_in_damage_degraded"], 2)
            for mk in ("residue_reduction_pct", "over_reach_pct",
                       "residue_before", "residue_after"):
                if res.metrics.get("hq_" + mk) is not None:
                    row[mk] = res.metrics["hq_" + mk]
            # outside the mask the output must stay close to the input: proof
            # that the restoration is local and reversible
            base = res.processed if res.processed is not None else deg
            row["psnr_outside_damage_restored"] = round(
                M.psnr_masked(clean, base, ~(case.gt_damage > 0)), 2)
    # decisions
    row["cond_score"] = res.condition.score if res.condition else float("nan")
    row["quality_index"] = res.quality.score if res.quality else float("nan")
    return row


def _group(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "?")), []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for g, rs in groups.items():
        acc: Dict[str, float] = {"n": float(len(rs))}
        for k in ("damage_iou", "damage_f1", "stain_iou", "crack_iou", "text_iou",
                  "psnr_degraded", "psnr_restored", "psnr_gain_db",
                  "area_rel_error_pct", "time_ms"):
            vals = [float(r[k]) for r in rs if k in r and np.isfinite(float(r[k]))]
            if vals:
                acc[k] = round(float(np.mean(vals)), 4)
        out[g] = acc
    return out


def _markdown(rows, agg, per_sev, pipe, dataset) -> str:
    L: List[str] = []
    L.append(f"# Evaluation - {os.path.basename(dataset.rstrip(os.sep))}")
    L.append("")
    L.append(f"* {len(rows)} controlled test images "
             f"({sum(1 for r in rows if r.get('status')=='OK')} processed OK)  ")
    L.append(f"* pipeline: denoise={pipe.params.denoise_method}, "
             f"binarisation={pipe.params.binarisation}, "
             f"inpaint={'on' if pipe.params.inpaint else 'off'}, "
             f"sensitivity={pipe.params.damage_sensitivity}, "
             f"dpi={pipe.params.target_dpi}  ")
    L.append(f"* generated by `python -m heritage_vision.tools.evaluate --dataset {dataset}`")
    L.append("")
    L.append("## Aggregate measures")
    L.append("")
    L.append("Mean and median are both given: several measures are ratios of a small "
             "ground-truth area, so a lightly damaged page can produce an unbounded "
             "relative error and the mean alone would misrepresent the system.")
    L.append("")
    L.append("| measure | mean | median | IQR | std | min | max | n |")
    L.append("|---|---|---|---|---|---|---|---|")
    nice = {"psnr_in_damage_inpainted": "PSNR in damage, inpaint only (dB)",
            "end_to_end_gain_db": "end-to-end gain in damage (dB)",
            "residue_reduction_pct": "stain residue removed (%)",
            "over_reach_pct": "over-reach outside damage (% of local noise sigma)",
            "psnr_in_damage_degraded": "PSNR inside damage, before (dB)",
            "psnr_in_damage_restored": "PSNR inside damage, after (dB)",
            "restoration_gain_db": "restoration gain inside damage (dB)",
            "rmse_in_damage_degraded": "RMSE inside damage, before",
            "rmse_in_damage_restored": "RMSE inside damage, after",
            "psnr_outside_damage_restored": "PSNR outside damage (locality check)",
            "damage_iou": "IoU of the damage mask", "damage_iou_tol2": "IoU (2 px tolerance)",
            "damage_dice": "Dice coefficient", "damage_f1": "F1 of the damage mask",
            "damage_precision": "precision", "damage_recall": "recall",
            "damage_area_error_pct": "area error (%)", "stain_iou": "IoU stains only",
            "crack_iou": "IoU cracks only", "hole_iou": "IoU material loss only",
            "text_iou": "IoU of the preserved text layer",
            "region_count_abs_error": "region count error (abs)",
            "region_count_rel_error_pct": "region count error (%)",
            "area_rel_error_pct": "damaged-area error (%)",
            "psnr_degraded": "PSNR input vs clean (dB)",
            "psnr_enhanced": "PSNR after enhancement (dB)",
            "psnr_restored": "PSNR after restoration (dB)",
            "psnr_gain_db": "PSNR gain (dB)", "ssim_degraded": "SSIM input vs clean",
            "ssim_restored": "SSIM after restoration", "ssim_gain": "SSIM gain",
            "contrast_gain_pct": "contrast improvement (%)",
            "umichelson_gain_pct": "Michelson contrast improvement (%)",
            "readability_gain_pct": "readability gain (%)",
            "skew_abs_error_deg": "deskew error (deg)",
            "corner_mean_error_px": "rectification corner error (px)",
            "time_ms": "processing time (ms)"}
    for k, label in nice.items():
        if f"{k}_mean" not in agg:
            continue
        m = agg[f"{k}_mean"]
        if not np.isfinite(m):
            continue
        L.append(f"| {label} | {m:.3f} | {agg.get(k+'_median',0):.3f} | "
                 f"{agg.get(k+'_iqr',0):.3f} | {agg.get(k+'_std',0):.3f} | "
                 f"{agg.get(k+'_min',0):.3f} | {agg.get(k+'_max',0):.3f} | "
                 f"{int(agg.get(k+'_n',0))} |")
    L.append("")
    L.append("## By severity of the applied degradation")
    L.append("")
    cols = ("damage_iou", "damage_f1", "stain_iou", "crack_iou", "text_iou",
            "psnr_degraded", "psnr_restored", "psnr_gain_db",
            "area_rel_error_pct", "time_ms")
    L.append("| severity | n | " + " | ".join(cols) + " |")
    L.append("|---" * (len(cols) + 2) + "|")
    for g in sorted(per_sev):
        d = per_sev[g]
        L.append(f"| {g} | {int(d['n'])} | " + " | ".join(
            (f"{d[c]:.3f}" if c in d else "-") for c in cols) + " |")
    L.append("")
    L.append("## Per image")
    L.append("")
    L.append("| image | sev | GT area % | pred area % | IoU | F1 | prec | rec | "
             "count err | area err % | PSNR dB | SSIM | t (ms) |")
    L.append("|---" * 13 + "|")
    for r in rows:
        if r.get("status") != "OK":
            L.append(f"| {r.get('name','')} | {r.get('severity','')} | ERROR: "
                     f"{r.get('error','')} |" + " |" * 11)
            continue

        def g(k, f="{:.3f}"):
            v = r.get(k)
            return f.format(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else "-"
        L.append("| " + " | ".join([
            os.path.splitext(r["name"])[0][:26], str(r.get("severity", "")),
            g("gt_damage_pct", "{:.2f}"), g("pred_damage_pct", "{:.2f}"),
            g("damage_iou"), g("damage_f1"),
            g("damage_precision"), g("damage_recall"),
            g("region_count_rel_error_pct", "{:+.1f}"), g("area_rel_error_pct", "{:+.1f}"),
            g("psnr_restored", "{:.2f}"), g("ssim_restored", "{:.4f}"),
            g("time_ms", "{:.0f}")]) + " |")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HeritageVision evaluation harness")
    ap.add_argument("--dataset", default=os.path.join(_ROOT, "dataset", "unseen_test"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "evaluation"))
    ap.add_argument("--denoise", default=None)
    ap.add_argument("--binarisation", default=None)
    ap.add_argument("--sensitivity", type=float, default=None)
    ap.add_argument("--no-inpaint", action="store_true")
    ap.add_argument("--deskew-off", action="store_true",
                    help="do not deskew (keeps the GT pixel grid for IoU)")
    ap.add_argument("--rectify", action="store_true")
    ap.add_argument("--canvas", default="auto", choices=("auto", "stored-only", "off"),
                    help="reference for PSNR/SSIM: auto = stored file, else "
                         "re-generate it from the sidecar parameters")
    ap.add_argument("--analyse-on", default=None, choices=("raw", "enhanced"),
                    help="which frame the damage detectors read (PipelineParams."
                         "analysis_source); used for the A/B in report sec. 7.4")
    ap.add_argument("--geometry", action="store_true",
                    help="also measure deskew / rectification accuracy")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only")
    ap.add_argument("--filter", default="", help="only cases whose id contains this")
    a = ap.parse_args(argv)
    return 0 if run(a) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
