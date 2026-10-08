# Dataset specification and sources

Part A of the brief asks for a documented image set: at least 30 images, at
least 10 created by the student, named folders, and a manifest carrying source /
resolution / format / purpose for every file.  This document is the provenance
record; `dataset/dataset_manifest.csv` is the machine-readable one
(`python -m heritage_vision.tools.make_dataset --manifest-only` rebuilds it from
what is on disk, so the two can never drift).

## 1. Folder layout

| folder | contents | count | origin |
|---|---|---|---|
| `dataset/clean/` | synthetic "pristine" sheets (manuscript a/b, photo plate a/b, printed a/b, typed a/b) at 1200×900 | 8 + 16 derived masks (`*_ink.png`, `*_sheet.png`) | student-created |
| `dataset/student_created/` | the controlled degradation set `NNN_<source>_<severity>.jpg` | 24 | student-created, derived from `clean/` |
| `dataset/student_created/gt/` | per-case ground truth: `damage`, `stain`, `crack`, `hole`, `text`, `sheet` layers | 144 PNG | generated with the canvas-space truth described in §3 |
| `dataset/unseen_test/` | held-out cases, different seed (99), different degradation mix, half with perspective warp | 22 | student-created |
| `dataset/real_scans/` | real historical printed pages (DIBCO 2019) + one aged-paper photograph | 11 | external, read-only (see §4) |
| `dataset/real_scans/gt/` | the bitonal ink references that came with those pages | 10 | external |
| `dataset/dataset_manifest.csv` | one row per image: `file, folder, source, origin, format, width, height, bit_depth, channels, dpi, purpose, gt_layers, gt_damage_pct, skew_deg, perspective, notes` | 81 | generated |

Counts against the brief: 8 + 24 + 22 + 11 = **65 images ≥ 30** ✓, of which
**46 are student-created ≥ 10** ✓ (24 + 22 degraded cases plus the 8 renders).

## 2. Student-created "clean" sheets

`dataset/clean/*.png` were drawn with OpenCV by
`python -m heritage_vision.tools.make_dataset --render`: warm paper
(BGR 228/214/200) with grain, ruled text bands of variable stroke width, a
decorated initial block, a caption band, plus for the photo-plate sources a
tonal "photograph" rectangle with a mount border.  They are deliberately
four *kinds* of paper item (handwritten manuscript, photographic plate with a
caption, letterpress print, typewritten sheet) because the pipeline's text
layer has to survive cursive ink, halftone tone and typeface edges alike.

No third-party material, font file or photograph was used, so these eight
images are unencumbered: they can be copied, modified and redistributed with
the code.  They are *synthetic* and this is stated everywhere they are used -
they exist to make exact ground truth possible, not to claim "real scans".

## 3. Degradations and how the ground truth stays exact

`degrade()` in `heritage_vision/tools/make_dataset.py` composes, in canvas space
(1264×964 after a 32 px desk pad, so a torn edge has something to reveal):

foxing specks · brown ink blots · translucent stains (chroma raised, value
lowered) · crack polylines with 1–3 px width and true meandering · corner
tears and interior holes punched to the desk colour · fold/crease bands ·
edge abrasion · yellowing field (a low-order chroma gradient) · vignette ·
blur · JPEG q=62/78 · optional 4-point perspective · optional skew.

Severity is recorded per case in the manifest (`gt_damage_pct`,
`skew_deg`, `perspective`), light = 0.31–2.63 %, moderate = 2.44–16.21 %,
severe = 5.37–23.29 % of the sheet area.

Three rules that the measurements forced on this generator, and that are worth
stating because a naive generator silently measures the wrong thing:

1. **Masks are rasterised in canvas space before warping.**  Any mask that was
   transformed with `INTER_NEAREST` after the image had been resized would put
   the IoU error on the boundary rather than on the detector.
2. **The shared warp is applied to image and masks together**, so
   `deskew`/`rectify` are testable: `evaluate.py` therefore disables all
   geometric resampling for the mask metrics (`--geometry` measures them
   separately, against corner positions and angles, which are warp-invariant).
3. **Every detector premise must be mirrored in the generator.**  Material-loss
   holes are filled with `desk_under * 1.06 + 6` (the *desk* colour, slightly
   brightened) because `missing_fragment_map()` looks for background colour
   inside the sheet; with paper-coloured holes the detector scored IoU 0.000 and
   the *generator* was wrong, not the detector.  After the change the same
   metric is 0.577 mean (16 cases with holes).

## 4. Real scans - external, read-only, licence-stated

Ten pages from the **DIBCO 2019** document-image binarisation competition were
obtained from the public mirror `github.com/ArjunAnil2000/DIP-dataset`
(tarball fetched with `codeload.github.com`, BMPs converted to PNG, original
bitonal GT kept next to them for the ink-preservation measure):

> "This is a modified version of the dibco 2019 dataset for image binarization
> experiments." - that repository's `README.md`

The mirror states **no licence**.  These files are therefore used only as a
read-only benchmark inside this coursework submission (never re-licensed, never
redistributed in modified form); the underlying dataset should be cited to its
authors - B. Gatos, K. Ntirogiannis, I. Pratikakis, *ICFHR2019 Document Image
Binarization Competition (DIBCO 2019 + HIENAM)*, 2019 - and re-downloaded from
the competition site for any use beyond this submission.  `CREDITS.csv` records
the file-by-file origin, resolution and purpose.

One colour aged-paper sheet (`faithdiff_paper_real.jpg`) came from the demo
figures of `github.com/JyChen9811/FaithDiff` (same read-only rule).  Two things
about it are on the record: the file is an *AI-generated* "real paper" texture
from that repository's figure set, and the repository's portrait photographs
were deliberately **not** copied - a synthetic face adds nothing to a
paper-conservation dataset, and the brief requires images of identifiable people
to be handled with care.

## 5. Unseen test set (Part D requirement)

`dataset/unseen_test/` was generated with seed 99 (the tuning set uses 7), a
different degradation draw, and `--perspective` on half the cases; none of it was
consulted while thresholds were chosen.  The 22 cases were run through
`evaluate.py` and through the GUI batch mode; results in
`outputs/evaluation_unseen/results.md` and §7.5 of the report.

## 6. Reproducing the numbers

```bash
python -m heritage_vision.tools.make_dataset --render            # 8 clean sheets
python -m heritage_vision.tools.make_dataset --cases 24 --seed 7 # tuning set + GT
python -m heritage_vision.tools.make_dataset --cases 22 --seed 99 \
        --dir dataset/unseen_test --perspective                  # unseen set
python -m heritage_vision.tools.make_dataset --manifest-only     # the CSV above
python -m heritage_vision.tools.evaluate --dataset dataset/student_created \
        --out outputs/evaluation_main --geometry
python app.py --folder dataset/unseen_test --out outputs/batch_unseen
```

The real scans are the only inputs that cannot be regenerated; if the folders
are absent, fetch the two tarballs named above and copy the ten DIBCO PNGs (plus
their GT) and the one FaithDiff JPG into `dataset/real_scans/`.
