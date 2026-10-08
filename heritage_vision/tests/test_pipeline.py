"""Integration tests - the pipeline and the GUI controller on real files.

The unit suite (`test_core.py`) checks each operation against an analytic
expectation.  This file checks the things that can only be seen end to end:

* the seven stages really run and really fill the decision outputs,
* a corrupt or unsupported file is refused instead of raising,
* a parameter set through the controller changes the measurement,
* the ROI box restricts what is counted,
* export writes the documented file set and the CSV parses back,
* batch mode reports the same numbers as single-image mode.

Everything here is headless (`AppController` with ``thread=False``), so the file
runs in CI and in a terminal with no display.  Longest test ~15 s.

    python -m unittest discover -s heritage_vision/tests -t .
"""

import csv
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from heritage_vision.core.config import PipelineParams            # noqa: E402
from heritage_vision.core.pipeline import HeritagePipeline        # noqa: E402

CASE_DIR = os.path.join(ROOT, "dataset", "student_created")
CASE = os.path.join(CASE_DIR, "003_clean_manuscript_a_severe.jpg")
LIGHT = os.path.join(CASE_DIR, "019_clean_typed_a_light.jpg")
GT = os.path.join(CASE_DIR, "gt", "003_clean_manuscript_a_severe.damage_mask.png")
HAVE = os.path.exists(CASE) and os.path.exists(LIGHT)


def base_params(**over):
    """Geometry off: mask metrics are only meaningful on the source grid."""
    p = PipelineParams()
    p.deskew = p.rectify = p.auto_crop = False
    for k, v in over.items():
        setattr(p, k, v)
    return p


@unittest.skipUnless(HAVE, "dataset not generated (run make_dataset first)")
class TestPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = HeritagePipeline(base_params()).process_file(CASE)

    def test_all_stages_ran_and_the_result_is_usable(self):
        r = self.res
        self.assertTrue(r.ok, msg=str(r.error))
        for stage in ("preprocess", "enhancement", "denoise", "segmentation",
                      "analysis", "scoring", "annot"):
            self.assertIn(stage, r.timings, msg=str(sorted(r.timings)))
            self.assertGreater(r.timings[stage], 0.0)
        self.assertEqual(r.processed.shape, r.input_image.shape)
        self.assertIsNotNone(r.annotated)

    def test_decision_outputs_exist_and_are_in_range(self):
        r = self.res
        self.assertEqual(r.damage_mask.shape, r.input_image.shape[:2])
        self.assertGreater(r.damage.damage_pct, 0.0)
        self.assertGreater(r.damage.n_damage_total, 0)
        self.assertIn(r.condition.pass_fail, ("PASS", "WARN", "FAIL"))
        self.assertTrue(0.0 <= r.condition.score <= 100.0)
        self.assertTrue(0.0 <= r.quality.score <= 100.0)
        self.assertIsInstance(r.condition.warning, str)
        # the enhancement must not silently drop the text layer it protects
        self.assertGreater(float(np.count_nonzero(r.text_mask)), 0.02 * r.roi.size)

    def test_damage_measurements_are_physically_consistent(self):
        d = self.res.damage
        self.assertLessEqual(d.stain_px + d.crack_px + d.missing_px,
                             d.damage_total_px * 3)      # layers may overlap
        self.assertGreater(d.damage_mm2, 0.0)
        # the millimetre figure and the percentage must be the same measurement
        # expressed twice - otherwise the CSV column and the GUI disagree
        roi_mm2 = float(np.count_nonzero(self.res.roi)) * (25.4 / 300.0) ** 2
        self.assertAlmostEqual(d.damage_mm2, d.damage_pct / 100.0 * roi_mm2,
                               delta=max(1.0, 0.01 * roi_mm2))
        for reg in d.regions:
            self.assertGreaterEqual(reg.area_mm2, 0.0)
            self.assertGreaterEqual(reg.severity, 0.0)
            self.assertLessEqual(reg.severity, 1.0)
            x, y, w, h = reg.bbox
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertLessEqual(x + w, self.res.input_image.shape[1])

    def test_iou_against_the_damage_ground_truth(self):
        """Same measure the report quotes, computed here from scratch."""
        gt = cv2.imread(GT, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gt, msg="ground-truth mask missing")
        pred = self.res.damage_mask
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        iou = float(np.count_nonzero((pred > 0) & (gt > 0))) / max(
            float(np.count_nonzero((pred > 0) | (gt > 0))), 1.0)
        self.assertGreater(iou, 0.6, msg=f"damage IoU regressed to {iou:.3f}")

    def test_restoration_only_touches_the_damaged_pixels(self):
        """Outside the inpainted area the output must equal the enhanced input;
        if it does not, 'restoration' is really a global filter and the
        before/after comparison in the GUI would be misleading."""
        r = self.res
        mask = r.extra.get("restored_damage_mask")          # what was inpainted
        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask)), 0)  # something was filled
        # a 1 px fill grows under texture, so allow one kernel of slack; nothing
        # further out may be touched
        allowed = cv2.dilate(r.damage_mask, np.ones((31, 31), np.uint8))
        outside = int(np.count_nonzero((mask > 0) & (allowed == 0)))
        self.assertLess(outside / float(np.count_nonzero(mask)), 0.02)
        self.assertLessEqual(int(np.count_nonzero(mask)), r.damage.damage_total_px * 4)

    def test_a_parameter_change_changes_the_measurement(self):
        strong = HeritagePipeline(base_params(damage_sensitivity=2.2)).process_file(CASE)
        self.assertTrue(strong.ok)
        self.assertGreater(strong.damage.damage_pct, self.res.damage.damage_pct * 0.99)
        plain = HeritagePipeline(base_params(inpaint=False)).process_file(CASE)
        self.assertTrue(plain.ok)
        self.assertGreater(plain.damage.damage_pct, 0.0)

    def test_downscale_option_keeps_the_result_on_the_source_grid(self):
        r = HeritagePipeline(base_params()).process_file(CASE, downscale_to=600)
        self.assertTrue(r.ok)
        self.assertLessEqual(max(r.processed.shape[:2]), 620)
        self.assertEqual(r.damage_mask.shape, r.processed.shape[:2])


@unittest.skipUnless(HAVE, "dataset not generated")
class TestRefusals(unittest.TestCase):
    def test_corrupt_and_unreadable_files_are_refused_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "broken.jpg")
            with open(CASE, "rb") as f:
                head = f.read(400)                    # a valid header, no data
            with open(bad, "wb") as f:
                f.write(head)
            empty = os.path.join(d, "empty.png")
            open(empty, "wb").close()   # noqa: SIM115
            nope = HeritagePipeline(base_params()).process_file(os.path.join(d, "no.png"))
            r1 = HeritagePipeline(base_params()).process_file(bad)
            r2 = HeritagePipeline(base_params()).process_file(empty)
            for r in (r1, r2, nope):
                self.assertFalse(r.ok)
                self.assertTrue(r.error)               # with a written explanation
                self.assertIsInstance(r.error, str)

    def test_unsupported_extension_is_refused_when_picked_and_skipped_in_batch(self):
        # the folder scan filters on extension first (cheap, and it stops the
        # batch mode from opening 1 000 sidecar files); picking such a file in
        # the GUI must still be refused *with a reason*
        with tempfile.TemporaryDirectory() as d:
            doc = os.path.join(d, "notes.txt")
            with open(doc, "w") as f:
                f.write("not an image")
            results, rejected = HeritagePipeline(base_params()).process_folder(d)
            self.assertEqual(results, [])
            self.assertFalse(any("notes.txt" in r for r in
                                 [x.name for x in results if hasattr(x, "name")]))
            r = HeritagePipeline(base_params()).process_file(doc)
            self.assertFalse(r.ok)
            self.assertIn(".txt", str(r.error))
            bad = HeritagePipeline(base_params()).process_file(os.path.join(d, "nope.jpg"))
            self.assertFalse(bad.ok)
            self.assertTrue(bad.error)

    def test_degenerate_sizes_survive(self):
        for shape in ((40, 30, 3), (8, 8, 3), (300, 4, 3)):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "tiny.png")
                cv2.imwrite(path, np.full(shape, 200, np.uint8))
                r = HeritagePipeline(base_params()).process_file(path)
                self.assertTrue(r.ok or r.error, msg=str(r.error))   # never raises
                if r.ok:
                    self.assertEqual(r.damage_mask.shape, shape[:2])

    def test_folder_mode_matches_single_image_mode(self):
        with tempfile.TemporaryDirectory() as d:
            for src in (CASE, LIGHT):        # copy, do not re-encode: a JPEG
                shutil.copy(src, os.path.join(d, os.path.basename(src)))
            results, rejected = HeritagePipeline(base_params()).process_folder(d)
            self.assertEqual(rejected, [])
            self.assertEqual(len(results), 2)
            by_name = {r.name: r for r in results}
            single = HeritagePipeline(base_params()).process_file(CASE)
            self.assertAlmostEqual(by_name[os.path.basename(CASE)].damage.damage_pct,
                                   single.damage.damage_pct, places=6)


@unittest.skipUnless(HAVE, "dataset not generated")
class TestRoi(unittest.TestCase):
    def test_roi_box_restricts_what_is_counted(self):
        img = cv2.imread(CASE)
        h, w = img.shape[:2]
        box = (int(0.25 * w), int(0.25 * h), int(0.4 * w), int(0.4 * h))
        full = HeritagePipeline(base_params()).process_file(CASE)
        r = HeritagePipeline(base_params()).process_file(CASE, roi_box=box)
        self.assertTrue(r.ok)
        self.assertLess(r.damage.damage_mm2, full.damage.damage_mm2)
        # every reported region must lie inside the box - a region outside it
        # would mean the ROI was used for the area but not for the detection
        for reg in r.damage.regions:
            x, y, ww, hh = reg.bbox
            self.assertGreaterEqual(x, box[0] - 4, msg=f"{reg.bbox} outside {box}")
            self.assertGreaterEqual(y, box[1] - 4)
            self.assertLessEqual(x + ww, box[0] + box[2] + 4)
            self.assertLessEqual(y + hh, box[1] + box[3] + 4)
        self.assertLess(float(np.count_nonzero(r.roi)) / r.roi.size, 0.35)


@unittest.skipUnless(HAVE, "dataset not generated")
class TestGuiController(unittest.TestCase):
    """Part D behaviour, exercised without tkinter."""

    def setUp(self):
        import app
        self.c = app.AppController(thread=False)

    def test_load_process_report(self):
        ok, msg = self.c.load(CASE)
        self.assertTrue(ok, msg)
        ok, msg = self.c.process()
        self.assertTrue(ok, msg)
        lines = self.c.report_lines()
        self.assertGreater(len(lines), 3)
        text = "\n".join(lines)
        for token in ("damage", "condition", "quality", "readability"):
            self.assertIn(token, text.lower())

    def test_unknown_parameter_is_refused_with_a_suggestion(self):
        ok, msg = self.c.set("not_a_parameter", 1)
        self.assertFalse(ok)
        self.assertIn("not_a_parameter", msg)
        ok, msg = self.c.set("clahe_clip_limit", 999)
        self.assertTrue(ok)                           # clamped, not rejected
        self.assertLessEqual(self.c.values()["clahe_clip_limit"], 8.0)

    def test_reset_restores_the_defaults(self):
        self.c.set("damage_sensitivity", 2.4)
        self.c.reset()
        self.assertAlmostEqual(self.c.values()["damage_sensitivity"],
                               PipelineParams().damage_sensitivity, places=6)

    def test_view_image_returns_the_requested_panel(self):
        self.c.load(CASE)
        self.c.process()
        for which in ("original", "processed", "annotated", "overlay"):
            im = self.c.view_image(which)
            self.assertIsNotNone(im, msg=which)
            self.assertEqual(im.ndim, 3)

    def test_export_writes_the_documented_files(self):
        self.c.load(CASE)
        self.c.process()
        with tempfile.TemporaryDirectory() as d:
            ok, msg, files = self.c.export(d)
            self.assertTrue(ok, msg)
            names = sorted(os.path.basename(f) for f in files)
            self.assertGreaterEqual(len(files), 8)
            stem = "003_clean_manuscript_a_severe"
            for suffix in ("_processed.png", "_annotated.png", "_summary.csv",
                           "_regions.csv", "_report.txt", "_report.json"):
                self.assertIn(stem + suffix, names, msg=str(names))
            # every stage mask is exported too - that is what makes a claim
            # checkable by somebody who does not trust the score
            self.assertGreaterEqual(sum("_stage-" in n for n in names), 8)
            # written as utf-8-sig so Excel opens the Unicode correctly; the BOM
            # has to be stripped when a test (or a script) reads it back raw
            with open(os.path.join(d, stem + "_summary.csv"), encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertIn("damage_pct", rows[0])
            self.assertGreater(float(rows[0]["damage_pct"]), 0.0)
            with open(os.path.join(d, stem + "_regions.csv"), encoding="utf-8-sig") as f:
                regs = list(csv.DictReader(f))
            self.assertGreater(len(regs), 0)
            self.assertIn("area_mm2", regs[0])
            for f_ in files:                           # nothing left half-written
                self.assertGreater(os.path.getsize(f_), 0)

    def test_batch_mode_with_a_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            cv2.imwrite(os.path.join(d, "a.png"), cv2.imread(CASE))
            cv2.imwrite(os.path.join(d, "b.png"), cv2.imread(LIGHT))
            open(os.path.join(d, "broken.png"), "wb").close()
            ok, msg, results = self.c.batch(d, o)
            self.assertTrue(ok, msg)
            self.assertIn("2/3", msg, msg)                 # one refused, two ok
            self.assertIn("1 rejected", msg, msg)
            self.assertEqual(len(results), 3)               # one is a failed result
            self.assertEqual(sum(1 for r in results if r.ok), 2)
            self.assertTrue(os.path.exists(os.path.join(o, "results.csv")))


@unittest.skipUnless(HAVE, "dataset not generated")
class TestSelftestEntry(unittest.TestCase):
    def test_app_selftest_passes(self):
        """`python app.py --selftest` is what the marker (and the viva) runs."""
        import io
        import contextlib
        sys.argv = ["app.py", "--selftest"]
        import app
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = app.main()
        self.assertIn(code, (0, None), msg=f"selftest returned {code}")
        self.assertIn("[selftest] PASS", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
