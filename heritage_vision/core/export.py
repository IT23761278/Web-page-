"""
heritage_vision.core.export
===========================

Part D requirement: "Save processed images and a CSV or text summary of
results".  One writer per artefact, all of them idempotent and safe on a
non-ASCII path (Windows lab machines are the target).

    results.csv    one row per image, ~60 columns (all measurements + timing)
    results.json   machine-readable, includes the parameter block used
    summary.txt    human-readable, per-image block + aggregate statistics
    dataset_manifest.csv  (built by tools/build_manifest.py) dataset contract
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

CSV_FIELDS_ORDER = [
    "name", "file", "status", "error", "format", "width", "height", "channels",
    "bit_depth", "file_size_kb", "roi_area_pct", "roi_area_mm2",
    "damage_pct", "damage_mm2", "damage_total_px", "n_damage_total", "n_stain",
    "n_crack", "n_missing", "largest_damage_mm2", "damage_density_per_cm2",
    "top_classes", "condition_score", "severity_class", "pass_fail", "warning",
    "quality_index", "quality_grade", "quality_flags", "quality_advice",
    "readability_before", "readability_after", "readability_gain_pct",
    "text_lines", "text_words", "text_components", "text_ink_pct",
    "denoise_method", "t_preprocess", "t_enhancement", "t_denoise",
    "t_segmentation", "t_analysis", "t_scoring", "t_annot", "total_time_ms",
]


def write_csv(rows: Sequence[Dict[str, Any]], path: str,
              field_order: Optional[Sequence[str]] = None) -> str:
    """Write every key present in any row (union), ordered by *field_order*."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    order = [k for k in (field_order or CSV_FIELDS_ORDER) if k in keys]
    rest = [k for k in keys if k not in order]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=order + sorted(rest))
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k, "")) for k in (order + rest)})
    return path


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (list, tuple, set)):
        return "|".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return v


def write_json(data: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_jsonable)
    return path


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return {"__ndarray_shape__": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def write_text_report(results: List[Any], path: str, title: str =
                      "Heritage Vision - processing summary",
                      params: Optional[Dict[str, Any]] = None,
                      evaluation: Optional[Dict[str, Any]] = None) -> str:
    """The text summary: header, per-image block, aggregate, evaluation."""
    lines: List[str] = []
    A = lines.append
    A("=" * 78)
    A(title)
    A(f"generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    A(f"images    : {len(results)}   (ok {sum(1 for r in results if r.ok)}, "
      f"failed {sum(1 for r in results if not r.ok)})")
    A("=" * 78)
    if params:
        A("")
        A("PARAMETERS")
        A("-" * 78)
        for k in sorted(params):
            A(f"  {k:26s} = {params[k]}")
    A("")
    A("PER-IMAGE RESULTS")
    A("-" * 78)
    for r in results:
        if not r.ok:
            A(f"  [ERROR] {r.name}: {r.error}")
            continue
        A("")
        A(f"  {r.name}")
        A("    " + "-" * 70)
        if r.metadata:
            for ln in r.metadata.pretty().splitlines():
                A(f"      {ln}")
        d, c, q, rd = r.damage, r.condition, r.quality, r.readability
        if d:
            A(f"      damage        : {d.damage_pct:.2f} % of sheet  |  "
              f"{d.damage_mm2:.1f} mm2  |  {d.n_damage_total} regions")
            A(f"                      stain {d.stain_px} px  crack {d.crack_px} px  "
              f"missing {d.missing_px} px")
            A(f"      classes       : " + (", ".join(f"{k}:{v}" for k, v in
                                                      sorted(d.class_counts.items(),
                                                             key=lambda kv: -kv[1]))
                                          or "none"))
            for reg in d.regions[:5]:
                A(f"        - {reg.class_name:14s} area {reg.area_mm2:8.2f} mm2  "
                  f"circ {reg.circularity:.2f} elong {reg.elongation:5.2f} "
                  f"orient {reg.orientation_deg:+6.1f} deg  sev {reg.severity:.2f}")
                if reg.overlaps_text:
                    A(f"          ! overlaps writing -> reconstruction candidate")
        if c:
            A(f"      condition     : {c.score:.1f}/100 -> {c.severity_class} "
              f"[{c.pass_fail}]")
            if c.warning:
                A(f"      WARNING       : {c.warning}")
            for a in c.actions:
                A(f"      action        : {a}")
        if q:
            A(f"      quality index : {q.score:.1f}/100 ({q.grade})  "
              f"factors: " + ", ".join(f"{k}={v:.2f}" for k, v in q.factors.items()))
            for f in q.flags:
                A(f"      flag          : {f}")
        if rd:
            A(f"      readability   : {rd.before.get('score',0):.1f} -> "
              f"{rd.after.get('score',0):.1f} ({rd.improvement_pct:+.1f} %) "
              f"[{rd.verdict}]")
            A(f"      layout        : {rd.lines} lines, {rd.words} words, "
              f"{rd.characters} components")
        A(f"      timing        : " + ", ".join(f"{k} {v:.0f}ms" for k, v in
                                                sorted(r.timings.items())))
        if r.denoise:
            A(f"      denoise       : {r.denoise.method} " + ", ".join(
                f"{k}={v}" for k, v in r.denoise.metrics.items()))
        if r.geometry:
            A(f"      geometry      : " + ", ".join(
                f"{k}={v}" for k, v in r.geometry.items() if k != "log_lines"))
    agg = aggregate(results)
    A("")
    A("AGGREGATE")
    A("-" * 78)
    for k, v in agg.items():
        A(f"  {k:28s} {v}")
    if evaluation:
        A("")
        A("EVALUATION (against ground truth)")
        A("-" * 78)
        for k, v in evaluation.items():
            A(f"  {k:28s} {v}")
    A("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def aggregate(results: List[Any]) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    if not ok:
        return {"images": 0, "note": "no successful runs"}
    dmg = [r.damage.damage_pct for r in ok if r.damage]
    cond = [r.condition.score for r in ok if r.condition]
    qual = [r.quality.score for r in ok if r.quality]
    gains = [r.readability.improvement_pct for r in ok if r.readability]
    tm = [r.timings.get("TOTAL", 0.0) for r in ok]
    fails = sum(1 for r in ok if r.condition and r.condition.pass_fail != "PASS")
    return {
        "images": len(results),
        "images_ok": len(ok),
        "images_rejected": len(results) - len(ok),
        "mean_damage_pct": round(float(np.mean(dmg)), 3) if dmg else 0.0,
        "max_damage_pct": round(float(np.max(dmg)), 3) if dmg else 0.0,
        "mean_condition_score": round(float(np.mean(cond)), 2) if cond else 0.0,
        "mean_quality_index": round(float(np.mean(qual)), 2) if qual else 0.0,
        "mean_readability_gain_pct": round(float(np.mean(gains)), 2) if gains else 0.0,
        "images_flagged_warn_or_fail": fails,
        "total_time_ms": round(float(np.sum(tm)), 1),
        "mean_time_ms": round(float(np.mean(tm)), 1) if tm else 0.0,
        "median_time_ms": round(float(np.median(tm)), 1) if tm else 0.0,
    }
