# Heritage Vision — condition assessment and restoration of historical paper items

SE3071 Individual Project (Y3 S2, 2026) · allocated domain **8 — Heritage & Culture**

A Python + OpenCV system that takes a scan or photograph of a damaged paper
artefact (manuscript page, old printed sheet, photographic plate, inscription)
and returns what a conservator needs from a triage pass: **where the damage is,
how much of it there is in real millimetres, how urgent the item looks, which
text is at risk, and a proposed visual restoration** — with every number
traceable back to the pixels it came from.

```
python app.py                                    # GUI (Tk)
python app.py dataset/student_created/003_clean_manuscript_a_severe.jpg
python app.py --folder dataset/unseen_test --out outputs/batch_unseen   # batch
python app.py --selftest                         # headless end-to-end smoke test
python -m unittest discover -s heritage_vision/tests -t .              # 101 tests
```

## What it does

| stage | operations (all hand-written on numpy/OpenCV) |
|---|---|
| 1 acquisition | extension + magic-byte checks, EXIF/PNG metadata, DPI, safe refusal of corrupt files |
| 2 geometry | deskew (projection + periodicity guards), 4-point rectification, ROI crop, `scale_to_dpi` |
| 3 colour | BGR / gray / HSV / Lab / YCrCb; illumination flattening, CLAHE, stretch, gamma, optional de-yellowing |
| 4 denoise | bilateral (default) compared against gaussian, median, guided, NLM, Wiener with a printed justification |
| 5 binarise | Otsu / adaptive / Sauvola + morphology + connected components → the text layer |
| 6 analyse | page mask, chroma-excess stain map, vesselness crack map, background-colour hole map, watershed-separated regions, shape descriptors, defect classification, mm² measurements |
| 7 restore | inpainting restricted to the damage mask + text-preserving enhancement |
| 8 decide | quality index (0–100), condition index with PASS/WARN/FAIL and actions, readability before/after, per-region table |

Deep learning is deliberately **not** used: the assessed core is classic image
processing, and every operation has to be explainable and modifiable by hand.

## Decision outputs

* damage area: `px`, `% of sheet`, `mm²`, `cm²` — with the ROI it was measured over
* region list: class (`foxing/stain`, `crack/tear`, `ink-blodge`, `fold/crease`,
  `abrasion`, `stain-complex`, `missing fragment`, …), bbox, area, circularity,
  elongation, orientation, severity 0–1, `overlaps_text`
* counts: stain / crack / material-loss / total, `damage_density_per_cm2`
* `condition.score` 0–100 + `PASS|WARN|FAIL` + written warnings and suggested actions
* `quality.score` 0–100 with its factors, `readability.before/after/improvement_pct`
* exported annotated plate, restored PNG, stage masks, `*_summary.csv`,
  `*_regions.csv`, `*_report.txt`, `*_report.json`

## Evaluation (measured, not asserted)

24 controlled student-created cases with exact pixel ground truth, plus 22
held-out cases generated with a different seed and half with perspective warp.
Full tables: [`docs/technical_report.md`](docs/technical_report.md),
`outputs/evaluation_main/results.md`, `outputs/evaluation_unseen/results.md`.

| measure | controlled set (mean / median) | unseen set |
|---|---|---|
| damage IoU | 0.567 / 0.636 | 0.416 / 0.469 |
| damage F1 (=Dice) | 0.695 / 0.778 | 0.545 / 0.638 |
| precision / recall | 0.781 / 0.731 | 0.757 / 0.524 |
| area error | +48.9 % / **−0.7 %** | −12.2 % / −30.0 % |
| stain / crack / hole IoU | 0.611 / 0.319 / 0.577 | 0.507 / 0.328 / 0.492 |
| contrast improvement | +14.0 % | +19.2 % |
| stain residue removed by restoration | 52.7 % / 64.5 % | 58.0 % / 63.8 % |
| deskew error | 0.239° / 0.011° | 0.323° / 0.038° |
| rectification corner error | 1.42 px / 1.23 px | 23.3 px / 1.56 px |
| time per image (1200×900) | 2792 ms / 2521 ms | 2311 ms / 753 ms |

## Layout

```
app.py                       Part D: Tkinter GUI + headless AppController + --selftest
heritage_vision/core/        17 modules: config utils metadata colorspace enhancement denoise
                             geometry segmentation features frequency matching damage
                             restoration scoring metrics annotate export pipeline evaluate
heritage_vision/tools/       make_dataset.py (generator), evaluate.py, run_batch.py
heritage_vision/tests/       test_core.py (69 unit) test_pipeline.py (19 integration)
                             test_dataset.py (13 Part A) - plain unittest, no pytest
dataset/clean/               8 rendered reference sheets (student-created, no copyright)
dataset/student_created/     24 controlled degraded cases + 144 ground-truth masks
dataset/unseen_test/         22 held-out cases (seed 99, perspective on half)
dataset/real_scans/          10 DIBCO 2019 pages + 1 aged-paper photo (read-only; see CREDITS.csv)
dataset/dataset_manifest.csv 81 rows: source, origin, format, w×h, bit depth, purpose, GT stats
docs/technical_report.md     Part E report (figures in docs/figures/)
docs/dataset_spec/SOURCES.md provenance, licensing, ethics, reproduction commands
outputs/                     evaluation tables and sample exports
```

## Notes for the viva

* Everything the GUI shows is produced by `AppController`, which runs headless —
  `python app.py --selftest` exercises load → process → ROI → export → batch and
  prints `[selftest] PASS`.
* `CONTROLS` in `app.py` is the single source of truth for widget ranges, and it
  is asserted to agree with `PipelineParams.validated()`'s clamps, so a slider
  can never sit outside the range the model accepts.
* The system never claims authenticity or value: `scoring.authenticity_indicators`
  reports advisory texture statistics only, and the report says so in §8.
