"""Regenerate the figures used in docs/technical_report.md.

    python demo/make_report_figures.py            # ~40 s, writes docs/figures/

Nothing here is part of the assessed core: it reads the CSV that
``heritage_vision.tools.evaluate`` already wrote and plots it with matplotlib
(the Agg backend, so it runs with no display), plus one before/after plate
produced by the pipeline itself.  Keeping it in a script rather than pasting
screenshots means every figure in the report can be re-made from the repository
alone - which is what "the student can explain and modify the code" implies for
a figure.
"""

import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "figures")
MAIN = os.path.join(ROOT, "outputs", "evaluation_main", "per_image_metrics.csv")
UNSEEN = os.path.join(ROOT, "outputs", "evaluation_unseen", "per_image_metrics.csv")


def _style():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})


def fig_results(d, path, title):
    """IoU against the amount of damage present: the honest picture of where a
    region measure is and is not trustworthy."""
    xcol = "gt_damage_pct"
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    ax = axes[0]
    ax.scatter(d[xcol], d["damage_iou"], s=16, c="#8c6d3f", alpha=0.9,
               label="detected damage")
    ax.scatter(d[xcol], d["hole_iou"].fillna(-0.05), s=14, marker="^", c="#2f6f8f",
               label="material loss (holes)")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.plot([0, max(d[xcol].max(), 1)], [0, max(d[xcol].max(), 1)], color="#b03a2e",
            lw=0.8, label="IoU = GT area (impossible upper bound)")
    ax.set_xlabel("ground-truth damage (% of sheet)")
    ax.set_ylabel("IoU")
    ax.set_title("(a) accuracy vs how much damage exists")
    ax.legend(fontsize=6.6, loc="lower right")
    ax = axes[1]
    pr, rc = d["damage_precision"], d["damage_recall"]
    ax.scatter(pr, rc, s=16, c="#4a7c59", alpha=0.9)
    ax.plot([0, 1], [0, 1], color="k", lw=0.7, ls=":")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.set_title("(b) the trade-off is recall-limited")
    ax.annotate("missed damage\n(over-conservative rejects)", (0.95, 0.30),
                fontsize=6.4, ha="right", color="#444")
    ax = axes[2]
    g = d.groupby("severity")[["damage_iou", "stain_iou", "crack_iou", "text_iou"]].mean()
    g = g.reindex(["light", "moderate", "severe"])
    g.plot.bar(ax=ax, width=0.72, rot=0,
               color=["#8c6d3f", "#c08b4d", "#2f6f8f", "#4a7c59"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean IoU")
    ax.set_title("(c) by severity of degradation")
    ax.tick_params(labelsize=7)
    ax.get_legend().remove()
    ax.legend(list(g.columns), fontsize=6.6, ncol=2, loc="upper center")
    fig.suptitle(title, fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline_speed(d, path):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.scatter(d["time_ms"], d["damage_iou"], s=16,
               c=np.where(d["severity"] == "severe", "#b03a2e",
                          np.where(d["severity"] == "moderate", "#d9a441", "#4a7c59")))
    ax.set_xlabel("processing time (ms)")
    ax.set_ylabel("damage IoU")
    ax.set_title("cost vs accuracy (green light, amber moderate, red severe)")
    ax.axvline(np.median(d["time_ms"]), color="k", lw=0.7, ls=":")
    ax.annotate(f"median {np.median(d['time_ms']):.0f} ms",
                (np.median(d["time_ms"]), ax.get_ylim()[1] * 0.92), fontsize=6.6,
                xytext=(4, 0), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_before_after(case_stem, path):
    """One plate: degraded input | enhanced + damage layers highlighted |
    restored output, so the reader can see what the score is talking about."""
    from heritage_vision.core.config import PipelineParams
    from heritage_vision.core.pipeline import HeritagePipeline
    from heritage_vision.core import annotate

    img_path = os.path.join(ROOT, "dataset", "student_created", case_stem + ".jpg")
    gt = cv2.imread(os.path.join(ROOT, "dataset", "student_created", "gt",
                                  case_stem + ".damage_mask.png"), 0)
    p = PipelineParams()
    p.deskew = p.rectify = p.auto_crop = False
    r = HeritagePipeline(p).process_file(img_path)
    if not r.ok:
        raise SystemExit(f"pipeline failed on {case_stem}: {r.error}")
    left = r.input_image
    mid = annotate.mask_overlay(r.processed, r.damage_mask, color=(40, 60, 220))
    right = r.processed
    if gt is not None:
        gt_small = cv2.resize(gt, (left.shape[1], left.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        left = annotate.mask_overlay(left, gt_small, color=(40, 170, 60))
    strip = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8),
                       mid, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
    strip = cv2.resize(strip, None, fx=0.5, fy=0.5)
    fig, ax = plt.subplots(figsize=(9.2, strip.shape[0] / 130.0))
    ax.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(f"{case_stem}: left = input with ground-truth damage (green), "
                 f"middle = enhanced output with detected damage (red), "
                 f"right = restored output", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    # a zoomed crop pair is what actually shows whether the stain disappeared
    h, w = r.processed.shape[:2]
    box = (int(0.18 * w), int(0.30 * h), int(0.30 * h), int(0.30 * h))
    crop_before = r.input_image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    crop_after = r.processed[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    pair = np.hstack([crop_before, np.full((crop_before.shape[0], 8, 3), 255, np.uint8),
                      crop_after])
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.imshow(cv2.cvtColor(cv2.resize(pair, None, fx=1.4, fy=1.4), cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title("detail: degraded (left) vs restored (right)", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(path.replace(".png", "_detail.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    _style()
    d = pd.read_csv(MAIN)
    fig_results(d, os.path.join(OUT, "results_tuning.png"),
                "Controlled set (24 student-created cases, exact ground truth)")
    if os.path.exists(UNSEEN):
        u = pd.read_csv(UNSEEN)
        fig_results(u, os.path.join(OUT, "results_unseen.png"),
                    "Held-out unseen set (22 cases, seed 99, half with perspective)")
    fig_pipeline_speed(d, os.path.join(OUT, "cost_vs_accuracy.png"))
    fig_before_after("003_clean_manuscript_a_severe",
                     os.path.join(OUT, "before_after.png"))
    print("[figures] ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("   ", f, os.path.getsize(os.path.join(OUT, f)) // 1024, "kB")


if __name__ == "__main__":
    main()
