"""
heritage_vision.core.evaluate
=============================

Part D: "report at least two suitable evaluation measures".  For Heritage &
Culture the marking sheet names PSNR, SSIM, IoU, contrast improvement and
restoration quality, so all of them are implemented here, plus the
classification measures (precision / recall / F1) that make a segmentation
claim testable, and two domain errors (count error, area error) that convert
the numbers back to physical units.

Ground truth
------------
A real archival sheet has no "true damage mask", which is why the project
creates *controlled* test images: a clean scan is degraded with a documented
synthetic degradation model and the exact masks/parameters used are recorded in
a sidecar ``*.gt.json`` (see ``tools/make_dataset.py``).  That yields honest
IoU / F1 / count-error / PSNR numbers for the synthetic set, while the real
scans are evaluated only with reference-free measures.  The two groups are
always reported as separate tables, never averaged together.

Measures computed per test case
    damage_iou / damage_dice / damage_f1 / precision / recall / area_error_pct
    count_abs_error / count_rel_error_pct      (region count vs GT regions)
    text_iou                                    (did we keep the writing?)
    psnr_degraded_vs_clean / psnr_restored_vs_clean / psnr_gain_db
    ssim_degraded / ssim_restored / ssim_gain
    contrast_gain_pct / umichelson_gain_pct
    time_ms
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import metrics as M
from .pipeline import HeritagePipeline, ProcessResult
from .utils import load_image, to_gray

GT_SUFFIX = ".gt.json"


@dataclass
class TestCase:
    name: str
    degraded_path: str
    clean_path: Optional[str]
    gt_damage: np.ndarray                     #: uint8 0/255
    gt_text: Optional[np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)


def load_test_case(gt_json_path: str) -> TestCase:
    """Read one sidecar ground-truth record written by tools/make_dataset.py."""
    with open(gt_json_path, "r", encoding="utf-8") as fh:
        j = json.load(fh)
    deg = j["degraded_image"]
    if not os.path.isabs(deg):
        deg = os.path.join(os.path.dirname(gt_json_path), deg)
    dmg_mask = j["gt_damage_mask"]
    if not os.path.isabs(dmg_mask):
        dmg_mask = os.path.join(os.path.dirname(gt_json_path), dmg_mask)
    mask = cv2.imread(dmg_mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"missing GT mask: {dmg_mask}")
    gt_text = None
    if j.get("gt_text_mask"):
        tp = j["gt_text_mask"]
        if not os.path.isabs(tp):
            tp = os.path.join(os.path.dirname(gt_json_path), tp)
        gt_text = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
    clean = j.get("clean_image")
    if clean and not os.path.isabs(clean):
        clean = os.path.join(os.path.dirname(gt_json_path), clean)
    return TestCase(name=os.path.basename(deg), degraded_path=deg,
                    clean_path=clean if clean and os.path.exists(clean) else None,
                    gt_damage=mask, gt_text=gt_text, meta=j)


def find_test_cases(root: str) -> List[TestCase]:
    """Any ``*.gt.json`` under *root*, sorted; broken records are skipped."""
    out: List[TestCase] = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*" + GT_SUFFIX), recursive=True)):
        try:
            out.append(load_test_case(p))
        except Exception as exc:
            print(f"[evaluate] skipping {os.path.basename(p)}: {exc}")
    return out


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def evaluate_case(pipe: HeritagePipeline, case: TestCase,
                  with_restoration: bool = True) -> Dict[str, Any]:
    """Run the pipeline on one degraded image and score every output."""
    row: Dict[str, Any] = {"name": case.name, "case": case.meta.get("case_id", ""),
                           "severity": case.meta.get("severity", "")}
    t0 = time.perf_counter()
    res = pipe.process_file(case.degraded_path)
    row["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    if not res.ok:
        row["status"] = "ERROR"
        row["error"] = res.error
        return row
    row["status"] = "OK"

    # ---- 1. segmentation quality vs GT damage mask ----------------------
    pred_damage = res.damage_mask if res.damage_mask is not None else \
        np.zeros_like(case.gt_damage)
    ov = M.overlap_stats(pred_damage, case.gt_damage)
    row.update({f"damage_{k}": (round(v, 4) if isinstance(v, float) else v)
                for k, v in ov.items()})
    # region-count error: compare predicted regions to GT connected components
    n_gt, _lab, stats, _ = cv2.connectedComponentsWithStats(
        (case.gt_damage > 0).astype(np.uint8), 8)
    gt_regions = 0
    min_area = 40
    for i in range(1, n_gt):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            gt_regions += 1
    ce = M.count_error(res.damage.n_damage_total, gt_regions)
    row.update({f"region_{k}": v for k, v in ce.items()})
    row["pred_area_mm2"] = round(float(res.damage.damage_mm2), 3)
    dpi = pipe.params.target_dpi
    row["gt_area_mm2"] = round(ov["truth_area_px"] * (25.4 / dpi) ** 2, 3)
    row.update(M.measurement_error(row["pred_area_mm2"], row["gt_area_mm2"], "area"))

    # ---- 2. did we preserve the writing? --------------------------------
    if case.gt_text is not None and res.text_mask is not None:
        tv = M.overlap_stats(res.text_mask, case.gt_text)
        row["text_iou"] = round(tv["iou"], 4)
        row["text_recall"] = round(tv["recall"], 4)
        row["text_precision"] = round(tv["precision"], 4)

    # ---- 3. enhancement / restoration fidelity --------------------------
    deg = to_gray(load_image(case.degraded_path))
    row.update(M.contrast_improvement(deg, to_gray(res.stages.get("enhanced", deg))))
    if case.clean_path:
        clean = to_gray(load_image(case.clean_path))
        enh = to_gray(res.stages.get("enhanced", deg))
        restored = to_gray(res.processed if with_restoration else
                           res.stages.get("enhanced", deg))
        p_deg, p_enh, p_res = (M.psnr(clean, deg), M.psnr(clean, enh),
                               M.psnr(clean, restored))
        s_deg, s_enh, s_res = (M.ssim(clean, deg), M.ssim(clean, enh),
                               M.ssim(clean, restored))
        row.update({
            "psnr_degraded": round(p_deg, 2), "psnr_enhanced": round(p_enh, 2),
            "psnr_restored": round(p_res, 2),
            "psnr_gain_db": round(p_res - p_deg, 2),
            "ssim_degraded": round(s_deg, 4), "ssim_enhanced": round(s_enh, 4),
            "ssim_restored": round(s_res, 4),
            "ssim_gain": round(s_res - s_deg, 4),
            "rmse_restored": round(M.rmse(clean, restored), 2),
            "psnr_reintroduces_clean_only": bool(p_res > p_deg),
        })
        # "how close to the original paper is our restored output vs how close
        # the degraded scan was" - the honest way to report restoration quality.
        row["restoration_improvement_pct"] = round(
            100.0 * (M.rmse(clean, deg) - M.rmse(clean, restored)) /
            max(M.rmse(clean, deg), 1e-9), 2)
    return row


# ---------------------------------------------------------------------------
# a whole set
# ---------------------------------------------------------------------------

PRIMARY_METRICS = [
    "damage_iou", "damage_iou_tol2", "damage_dice", "damage_f1",
    "damage_precision", "damage_recall", "damage_area_error_pct",
    "region_count_abs_error", "region_count_rel_error_pct", "text_iou",
    "psnr_degraded", "psnr_enhanced", "psnr_restored", "psnr_gain_db",
    "ssim_degraded", "ssim_restored", "ssim_gain", "contrast_gain_pct",
    "umichelson_gain_pct", "psnr_in_damage_degraded", "psnr_in_damage_inpainted",
    "psnr_in_damage_restored", "restoration_gain_db", "residue_reduction_pct",
    "over_reach_pct", "stain_iou", "crack_iou", "hole_iou", "time_ms",
]


def evaluate_folder(root: str, params=None, progress=None) -> Dict[str, Any]:
    """Evaluate every ``*.gt.json`` found under *root*."""
    from .config import PipelineParams
    pipe = HeritagePipeline(params or PipelineParams())
    cases = find_test_cases(root)
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(cases):
        r = evaluate_case(pipe, c)
        rows.append(r)
        if progress:
            progress(i + 1, len(cases), r)
    return {"n_cases": len(rows), "rows": rows,
            "aggregate": M.summarise(rows, PRIMARY_METRICS),
            "params": pipe.params.as_dict(),
            "errors": [r for r in rows if r.get("status") != "OK"]}


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
                   header: Optional[Sequence[str]] = None,
                   floatfmt: str = "{:.3f}") -> str:
    """Markdown table for the report (also pasted into results/tables.md)."""
    head = header or columns
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(floatfmt.format(v) if isinstance(v, (int, float)) and
                         not isinstance(v, bool) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_class_score(rows: Sequence[Dict[str, Any]], by: str = "severity",
                     keys: Sequence[str] = ("damage_iou", "damage_f1", "text_iou",
                                            "psnr_gain_db", "time_ms")
                     ) -> Dict[str, Dict[str, float]]:
    """Group by a categorical field ('severity' or 'degradation') so the
    report can show *where* the method degrades, not just an average."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(by, "?")), []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for g, rs in groups.items():
        acc: Dict[str, float] = {"n": float(len(rs))}
        for k in keys:
            vals = [float(r[k]) for r in rs if k in r and
                    isinstance(r[k], (int, float)) and math.isfinite(float(r[k]))]
            if vals:
                acc[k] = round(float(np.mean(vals)), 4)
        out[g] = acc
    return out
