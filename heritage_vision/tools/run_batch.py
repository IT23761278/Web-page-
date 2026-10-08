"""
tools/run_batch.py
==================

Command-line front end: single image, folder batch, parameter overrides, and
the CSV / text / JSON exports required by Part D.  It uses exactly the same
:class:`HeritagePipeline` as the GUI, so a result produced in the interface is
reproducible from the terminal (that is also how the report tables were made).

Examples
--------
    # one image, show everything, write outputs
    python -m heritage_vision.tools.run_batch -i dataset/student_created/003_clean_manuscript_a_severe.jpg -o outputs/run1 -v

    # whole folder with a different binarisation and stronger inpainting
    python -m heritage_vision.tools.run_batch -i dataset/unseen_test -o outputs/run2 \
        --binarisation adaptive --inpaint-radius 5 --clahe-clip 3.0

    # compare two denoisers on a folder and export the comparison
    python -m heritage_vision.tools.run_batch -i dataset/real_scans --denoise median --csv only
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heritage_vision.core.config import PipelineParams          # noqa: E402
from heritage_vision.core.export import (aggregate, write_csv,  # noqa: E402
                                         write_json, write_text_report)
from heritage_vision.core.pipeline import HeritagePipeline      # noqa: E402
from heritage_vision.core.utils import list_image_files         # noqa: E402

PARAM_FLAGS = {
    "--clahe-clip": ("clahe_clip_limit", float),
    "--clahe-tiles": ("clahe_tile_grid", int),
    "--gamma": ("gamma", float),
    "--stretch": ("percentile_stretch", float),
    "--bg-method": ("bg_normalisation", str),
    "--bg-kernel-mm": ("bg_kernel_mm", float),
    "--denoise": ("denoise_method", str),
    "--denoise-ksize": ("denoise_ksize", int),
    "--nlm-h": ("nlm_h", float),
    "--binarisation": ("binarisation", str),
    "--adaptive-block": ("adaptive_block", int),
    "--adaptive-c": ("adaptive_c", int),
    "--morph-open": ("morph_open_px", int),
    "--morph-close": ("morph_close_px", int),
    "--min-text-area": ("min_text_area_px", int),
    "--sensitivity": ("damage_sensitivity", float),
    "--min-damage-mm2": ("damage_min_area_mm2", float),
    "--damage-space": ("damage_space", str),
    "--inpaint-radius": ("inpaint_radius", float),
    "--inpaint-method": ("inpaint_method", str),
    "--inpaint-dilate": ("inpaint_dilate_px", int),
    "--no-inpaint": ("inpaint", lambda v: False),
    "--no-sharpen": ("sharpen", lambda v: False),
    "--unsharp-amount": ("unsharp_amount", float),
    "--unsharp-sigma": ("unsharp_sigma", float),
    "--dpi": ("target_dpi", int),
    "--deskew": ("deskew", lambda v: True),
    "--no-deskew": ("deskew", lambda v: False),
    "--rectify": ("rectify", lambda v: True),
    "--auto-crop": ("auto_crop", lambda v: True),
    "--no-auto-crop": ("auto_crop", lambda v: False),
    "--weight-damage": ("weight_damage", float),
    "--weight-sharpness": ("weight_sharpness", float),
    "--weight-contrast": ("weight_contrast", float),
    "--warn-pct": ("warn_damage_pct", float),
    "--fail-pct": ("fail_damage_pct", float),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_batch", description="HeritageVision batch processor (OpenCV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("-i", "--input", required=True,
                    help="image file or folder (folders are processed recursively)")
    ap.add_argument("-o", "--out", default=os.path.join(_ROOT, "outputs", "batch"),
                    help="output folder")
    ap.add_argument("--max", type=int, default=0, help="limit the number of images")
    ap.add_argument("--downscale", type=int, default=2400,
                    help="longest side for batch input (0 = keep)")
    ap.add_argument("--no-stages", action="store_true",
                    help="do not write the per-stage debug images")
    ap.add_argument("--csv", default="", help="csv path (default: <out>/results.csv)")
    ap.add_argument("--json", dest="json_out", default="",
                    help="json path (default: <out>/results.json)")
    ap.add_argument("--report", dest="txt", default="",
                    help="text report path (default: <out>/summary.txt)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list-params", action="store_true",
                    help="print the parameter defaults and exit")
    for flag in PARAM_FLAGS:
        if flag.startswith("--no-"):
            ap.add_argument(flag, action="store_true", help="turn the feature off")
        elif "deskew" in flag or "rectify" in flag or "auto-crop" in flag:
            ap.add_argument(flag, action="store_true", help="turn the feature on")
        else:
            ap.add_argument(flag, default=None, help="override " + PARAM_FLAGS[flag][0])
    return ap


def params_from_args(a: argparse.Namespace) -> PipelineParams:
    p = PipelineParams()
    for flag, (attr, cast) in PARAM_FLAGS.items():
        raw = getattr(a, flag.replace("-", "_"), None)
        if raw is None or raw is False:
            continue
        try:
            setattr(p, attr, raw if callable(cast) else cast(raw))
        except (TypeError, ValueError) as exc:
            print(f"[run_batch] ignoring {flag}={raw}: {exc}")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_params:
        from dataclasses import asdict
        for k, v in asdict(PipelineParams()).items():
            print(f"  {k:26s} = {v}")
        return 0
    if not os.path.exists(args.input):
        print(f"[run_batch] input does not exist: {args.input}")
        return 2
    files = ([args.input] if os.path.isfile(args.input)
             else list_image_files(args.input))
    if args.max:
        files = files[:args.max]
    if not files:
        print(f"[run_batch] no supported images found in {args.input}")
        return 2
    pipe = HeritagePipeline(params_from_args(args))
    os.makedirs(args.out, exist_ok=True)
    print(f"[run_batch] {len(files)} image(s) -> {args.out}")
    print(f"[run_batch] pipeline: {' -> '.join(HeritagePipeline.STAGES)}")
    results = []
    for i, f in enumerate(files, 1):
        r = pipe.process_file(f, downscale_to=(args.downscale or None))
        results.append(r)
        if r.ok:
            sub = os.path.join(args.out, "images",
                               os.path.splitext(os.path.basename(f))[0])
            try:
                r.save(sub, include_stages=not args.no_stages)
            except Exception as exc:                # keep the batch running
                print(f"  ! export failed for {os.path.basename(f)}: {exc}")
        if not args.quiet:
            print(f"[{i:>3}/{len(files)}] " + r.summary())
    rows = [r.to_dict() for r in results]
    csv_path = args.csv or os.path.join(args.out, "results.csv")
    json_path = args.json_out or os.path.join(args.out, "results.json")
    txt_path = args.txt or os.path.join(args.out, "summary.txt")
    write_csv(rows, csv_path)
    write_json({"params": pipe.params.as_dict(), "aggregate": aggregate(results),
                "rows": rows, "n": len(rows)}, json_path)
    write_text_report(results, txt_path,
                      params=pipe.params.as_dict(),
                      title=f"HeritageVision batch run - {args.input}")
    print(f"\n[run_batch] wrote {os.path.relpath(csv_path, _ROOT)}, "
          f"{os.path.relpath(json_path, _ROOT)}, {os.path.relpath(txt_path, _ROOT)}")
    agg = aggregate(results)
    for k in ("images", "images_ok", "images_rejected", "mean_damage_pct",
              "mean_condition_score", "mean_quality_index",
              "mean_readability_gain_pct", "images_flagged_warn_or_fail",
              "mean_time_ms"):
        print(f"  {k:30s} {agg.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
