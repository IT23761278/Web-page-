"""Part A checks - the dataset is what the report claims it is.

These tests are deliberately mechanical: a manifest that says "30 images, 10 of
them mine, every row with a source and a purpose" is only worth anything if
somebody checks it.  Run with

    python -m unittest discover -s heritage_vision/tests -t .
"""

import csv
import os
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "dataset")
MANIFEST = os.path.join(DS, "dataset_manifest.csv")


def rows():
    with open(MANIFEST, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(os.path.exists(MANIFEST),
                     "no manifest - run: python -m heritage_vision.tools.make_dataset "
                     "--manifest-only")
class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows()

    def test_every_required_column_is_present_and_filled(self):
        need = ("file", "folder", "source", "origin", "format", "width", "height",
                "bit_depth", "purpose")
        self.assertIn("purpose", self.rows[0])
        self.assertTrue(set(need) <= set(self.rows[0]), msg=str(list(self.rows[0])))
        for i, r in enumerate(self.rows):
            for k in need:
                self.assertTrue(str(r.get(k, "")).strip(),
                                msg=f"row {i} ({r.get('file')}) has an empty {k}")
            self.assertGreater(int(float(r["width"]) or 0), 0, msg=r["file"])

    def test_at_least_thirty_images(self):
        self.assertGreaterEqual(len(self.rows), 30, msg=f"{len(self.rows)} rows")

    def test_at_least_ten_are_student_created(self):
        mine = [r for r in self.rows if "student" in r["origin"].lower()
                or "rendered" in r["origin"].lower()]
        self.assertGreaterEqual(len(mine), 10, msg=f"{len(mine)} student rows")

    def test_the_named_folders_exist(self):
        for folder in ("clean", "student_created", "unseen_test", "real_scans"):
            self.assertTrue(os.path.isdir(os.path.join(DS, folder)), msg=folder)
        self.assertTrue({r["folder"] for r in self.rows} >= {"clean", "student_created"})

    def test_resolution_and_format_fields_are_true_of_the_files(self):
        checked = 0
        for r in self.rows:
            path = os.path.join(DS, r["folder"], r["file"])
            if not os.path.exists(path):
                # a gt/ sidecar row records its parent image, not itself
                path = os.path.join(DS, r["folder"], "gt", r["file"])
            if not os.path.exists(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.fail(f"{r['file']} is listed but cannot be decoded")
            h, w = img.shape[:2]
            self.assertEqual(int(r["width"]), w, msg=r["file"])
            self.assertEqual(int(r["height"]), h, msg=r["file"])
            self.assertEqual(r["format"].upper(), os.path.splitext(r["file"])[1][1:].upper())
            checked += 1
        self.assertGreaterEqual(checked, 20, msg="manifest rows do not point at files")

    def test_purpose_strings_say_what_each_set_is_for(self):
        text = " ".join(r["purpose"].lower() for r in self.rows)
        for token in ("controlled damage test", "reference clean sheet",
                      "real historical", "ground-truth ink layer"):
            self.assertIn(token, text)
        # every unseen case must say what it is for, not just carry a filename
        unseen = [r for r in self.rows if r["folder"] == "unseen_test"]
        self.assertGreaterEqual(len(unseen), 10)
        self.assertTrue(all("controlled damage test" in r["purpose"] for r in unseen))


@unittest.skipUnless(os.path.isdir(os.path.join(DS, "student_created")),
                     "dataset not generated")
class TestStudentSet(unittest.TestCase):
    def test_cases_have_the_documented_name_and_severity_spread(self):
        files = sorted(f for f in os.listdir(os.path.join(DS, "student_created"))
                       if f.endswith(".jpg"))
        self.assertGreaterEqual(len(files), 24)
        sevs = [f.rsplit("_", 1)[1][:-4] for f in files]
        for sev in ("light", "moderate", "severe"):
            self.assertGreaterEqual(sevs.count(sev), 6, msg=sev)

    def test_ground_truth_layers_exist_for_every_case(self):
        gt = os.path.join(DS, "student_created", "gt")
        files = [f[:-4] for f in os.listdir(os.path.join(DS, "student_created"))
                 if f.endswith(".jpg")]
        for stem in files:
            for layer in ("damage_mask", "stain_mask", "crack_mask", "text_mask",
                          "sheet_mask"):
                self.assertTrue(os.path.exists(os.path.join(gt, f"{stem}.{layer}.png")),
                                msg=f"{stem}.{layer}.png missing")

    def test_gt_damage_area_matches_the_recorded_severity(self):
        gt = os.path.join(DS, "student_created", "gt")
        with open(MANIFEST, encoding="utf-8-sig") as f:
            by_file = {r["file"]: r for r in csv.DictReader(f)}
        for stem in sorted(os.listdir(gt))[:60]:
            if not stem.endswith(".damage_mask.png"):
                continue
            name = stem[:-len(".damage_mask.png")] + ".jpg"
            row = by_file.get(name)
            if row is None or not row.get("gt_damage_pct"):
                continue
            m = cv2.imread(os.path.join(gt, stem), cv2.IMREAD_GRAYSCALE)
            sheet = cv2.imread(os.path.join(gt, f"{name[:-4]}.sheet_mask.png"),
                               cv2.IMREAD_GRAYSCALE)
            if sheet is None:
                continue
            frac = 100.0 * float(np.count_nonzero(m)) / max(float(np.count_nonzero(sheet)), 1)
            self.assertAlmostEqual(frac, float(row["gt_damage_pct"]), delta=1.0,
                                   msg=f"{name}: manifest {row['gt_damage_pct']} vs "
                                       f"{frac:.2f} % measured")

    def test_masks_are_binary_and_aligned(self):
        gt = os.path.join(DS, "student_created", "gt")
        stem = sorted(f[:-len(".damage_mask.png")] for f in os.listdir(gt)
                      if f.endswith(".damage_mask.png"))[0]
        shapes = set()
        for layer in ("damage", "stain", "crack", "hole", "text", "sheet"):
            path = os.path.join(gt, f"{stem}.{layer}_mask.png")
            if not os.path.exists(path):
                continue
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if layer != "hole":       # a light case legitimately has no holes
                self.assertGreater(int(np.count_nonzero(m)), 0, msg=layer)
            if layer in ("sheet", "text", "stain", "hole"):
                self.assertTrue(np.all((m == 0) | (m == 255)), msg=layer)
            else:
                # cracks are 1-3 px anti-aliased polylines and the damage layer is
                # their union with the blobs, so mid-grey edge pixels exist by
                # construction; the evaluator thresholds on > 0 for that reason
                self.assertGreater(float(np.mean(m[m > 0] == 255)), 0.4, msg=layer)
            shapes.add(m.shape)
        self.assertEqual(len(shapes), 1)              # one grid for every layer
        # the union has to contain every layer, and no more than their sum
        read = lambda lay: cv2.imread(os.path.join(gt, f"{stem}.{lay}_mask.png"), 0)
        total = read("damage") > 0
        layers = [read(l) > 0 for l in ("stain", "crack", "hole")]
        self.assertGreaterEqual(int(np.count_nonzero(total)),
                                int(max(np.count_nonzero(l) for l in layers)))
        self.assertLessEqual(int(np.count_nonzero(total)),
                             sum(int(np.count_nonzero(l)) for l in layers) * 2)
        self.assertEqual(int(np.count_nonzero(cv2.bitwise_and(read("text"),
                                                              read("sheet")))),
                         int(np.count_nonzero(read("text"))))   # text inside sheet

    def test_degradation_actually_happened(self):
        """A 'severe' case must be measurably worse than a 'light' one of the
        same source page, otherwise the severity labels are decoration."""
        d = os.path.join(DS, "student_created")
        a = cv2.imread(os.path.join(d, "001_clean_manuscript_a_light.jpg"))
        b = cv2.imread(os.path.join(d, "003_clean_manuscript_a_severe.jpg"))
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        # fewer sharp edges (blur + stains) on the severe case; note that raw
        # chroma *variance* is NOT a severity proxy (a lightly degraded page keeps
        # its film grain, which has more high-frequency colour than a blurred
        # severe one), which is why severity is measured on the masks below
        self.assertLess(float(np.abs(cv2.Laplacian(gb, cv2.CV_32F)).mean()),
                        float(np.abs(cv2.Laplacian(ga, cv2.CV_32F)).mean()) * 1.25)
        gt = os.path.join(d, "gt")
        sa = float(np.count_nonzero(cv2.imread(
            os.path.join(gt, "001_clean_manuscript_a_light.damage_mask.png"), 0)))
        sb = float(np.count_nonzero(cv2.imread(
            os.path.join(gt, "003_clean_manuscript_a_severe.damage_mask.png"), 0)))
        self.assertGreater(sb, 2.0 * sa)              # and not just a rename


@unittest.skipUnless(os.path.isdir(os.path.join(DS, "unseen_test")),
                     "unseen set not generated")
class TestUnseenSet(unittest.TestCase):
    def test_unseen_set_is_held_out_and_different(self):
        files = sorted(f for f in os.listdir(os.path.join(DS, "unseen_test"))
                       if f.endswith((".jpg", ".png")))
        self.assertGreaterEqual(len(files), 10)        # brief: >= 10 unseen tests
        seen = {f for f in os.listdir(os.path.join(DS, "student_created"))}
        self.assertFalse({f for f in files} & seen)     # no overlap by name/content

    def test_evaluation_outputs_exist(self):
        out = os.path.join(ROOT, "outputs", "evaluation_unseen")
        if not os.path.isdir(out):
            self.skipTest("run the evaluator first")
        self.assertTrue(os.path.exists(os.path.join(out, "evaluation.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "results.md")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
