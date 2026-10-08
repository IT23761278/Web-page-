"""Unit tests for the HeritageVision core (run with ``python -m unittest``).

Every test is built on a tiny synthetic image with a known answer, so it checks
something physical ("a 60x60 square has an area of 3600 px", "a page rotated by
2 deg is recovered to within 0.4 deg", "a mask that is 25 rows short of the
truth has IoU 0.5") rather than comparing against a golden file nobody can
explain.  The whole file runs in a few seconds.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heritage_vision.core import (annotate, colorspace, config, damage,     # noqa: E402
                                  denoise, enhancement, features, frequency,
                                  geometry, matching, metrics, restoration,
                                  scoring, segmentation, utils)
from heritage_vision.core.config import PipelineParams                       # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def page(width: int = 480, height: int = 640, ink: bool = True) -> np.ndarray:
    """A small fake scan: warm paper, six text lines, a stamp."""
    rng = np.random.default_rng(11)
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 214
    img[:, :, 2] = 228                  # warm (yellowish) paper
    if ink:
        # neutral, thin strokes: real ink is dark and near-grey, which is what
        # makes "chromatic damage" separable from writing at all
        for i in range(6):
            y = 90 + i * 60
            cv2.rectangle(img, (50, y), (width - 50, y + 5), (34, 33, 32), -1)
        cv2.circle(img, (width // 2, height - 70), 16, (36, 33, 30), -1)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6.0, img.shape),
                  0, 255).astype(np.uint8)
    return img


def noisy(img: np.ndarray, sigma: float = 14.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.int16) + rng.normal(0, sigma, img.shape),
                   0, 255).astype(np.uint8)


def square(side: int = 60, size=(120, 120)) -> np.ndarray:
    m = np.zeros(size, np.uint8)
    o = (size[0] - side) // 2
    m[o:o + side, o:o + side] = 255
    return m


# ---------------------------------------------------------------------------
class TestIOAndConfig(unittest.TestCase):
    def test_odd_and_uint8_helpers(self):
        self.assertEqual(utils.odd(4), 5)
        self.assertEqual(utils.odd(3), 3)
        self.assertEqual(utils.ensure_uint8(np.array([[0.5, 1.0]])).dtype, np.uint8)

    def test_mm_px_roundtrip(self):
        self.assertAlmostEqual(utils.px_to_mm(utils.mm_to_px(10, 300), 300), 10.0,
                               places=6)

    def test_missing_file_raises_image_load_error(self):
        with self.assertRaises(utils.ImageLoadError):
            utils.load_image(os.path.join(_ROOT, "not-here.png"))

    def test_corrupt_file_raises_image_load_error(self):
        d = os.path.join(_ROOT, "outputs", "_tmp_corrupt")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "broken.png")
        with open(p, "wb") as fh:
            fh.write(b"definitely not a png")
        try:
            with self.assertRaises(utils.ImageLoadError):
                utils.load_image(p)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_defaults_are_already_valid(self):
        p = PipelineParams()
        self.assertEqual(p.validated().as_dict(), p.as_dict())

    def test_validated_clamps_every_slider(self):
        p = PipelineParams()
        p.clahe_clip_limit = 99.0
        p.target_dpi = 10
        p.damage_sensitivity = -3.0
        p.bg_kernel_mm = 1e6
        p.damage_min_area_mm2 = 1e9
        p.inpaint_radius = 1e6
        q = p.validated()
        self.assertLessEqual(q.clahe_clip_limit, 8.0)
        self.assertGreaterEqual(q.target_dpi, 72)
        self.assertGreaterEqual(q.damage_sensitivity, 0.4)
        self.assertLessEqual(q.bg_kernel_mm, 80.0)
        self.assertLessEqual(q.damage_min_area_mm2, 50.0)
        self.assertLessEqual(q.inpaint_radius, 15.0)

    def test_weights_are_renormalised(self):
        p = PipelineParams()
        p.weight_damage, p.weight_sharpness = 0.0, 0.0
        q = p.validated()
        self.assertAlmostEqual(sum(getattr(q, w) for w in
                                   ("weight_damage", "weight_sharpness",
                                    "weight_contrast", "weight_lighting",
                                    "weight_noise")), 1.0, places=6)
        json.dumps(p.as_dict())                    # the export must be json-safe


class TestColourSpace(unittest.TestCase):
    def test_decompose_yields_every_requested_space(self):
        img = page()
        d = colorspace.decompose(img, spaces=("HSV", "Lab", "YCrCb"))
        self.assertEqual(d.bgr.shape[:2], img.shape[:2])
        self.assertEqual(d.gray.ndim, 2)              # the gray view is free
        self.assertTrue(np.array_equal(d.gray, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        for space in ("hsv", "lab", "ycrcb"):
            self.assertEqual(getattr(d, space).shape[:2], img.shape[:2], msg=space)
            self.assertEqual(getattr(d, space).shape[2], 3)
        self.assertIsNone(d.binary)                   # the caller supplies it
        mask = np.zeros(img.shape[:2], np.uint8)
        d2 = colorspace.decompose(img, spaces=("GRAY",), binary=mask)
        self.assertIs(d2.binary, mask)
        self.assertIsNone(d2.lab)                     # only what was asked for
        self.assertIsNone(d2.hsv)
        self.assertEqual(d2.gray.ndim, 2)

    def test_deyellow_moves_the_paper_toward_the_warm_neutral_target(self):
        # a strongly yellowed sheet: Lab b* far above the b*=12 target
        img = np.zeros((200, 200, 3), np.uint8)
        img[..., 0] = 150
        img[..., 1] = 190
        img[..., 2] = 235
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        before = float(np.median(lab[..., 2]))
        out = colorspace.deyellow(img, 1.0)
        after = float(np.median(cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[..., 2]))
        self.assertLess(abs(after - 12.0), abs(before - 12.0))
        self.assertGreater(after, 0.0)

    def test_yellowness_of_paper_is_above_ink(self):
        img = page()
        y = colorspace.yellowness_map(img)
        self.assertGreater(float(y.mean()), float(y[95:104, 60:120].mean()))

    def test_estimate_paper_colour_is_warm(self):
        b, g, r = colorspace.estimate_paper_colour(page())
        self.assertGreater(r, b)                   # paper: red channel highest


class TestEnhancement(unittest.TestCase):
    def test_clahe_lifts_a_washed_out_page_without_eating_a_good_one(self):
        g = utils.to_gray(page(320, 400))
        # the case CLAHE exists for: a faded sheet whose dynamic range collapsed
        low = np.clip(150 + (g.astype(np.float32) - 150.0) * 0.3, 0, 255).astype(np.uint8)
        c = enhancement.clahe_gray(low, 2.5, 8)
        self.assertGreater(float(c.std()), float(low.std()))
        # on an already-contrasty page it may shave a little global std (that is
        # clipping, by design) but must not flatten the strokes
        c2 = enhancement.clahe_gray(g, 2.0, 8)
        self.assertGreaterEqual(float(c2.std()), float(g.std()) * 0.90)
        self.assertEqual(c2.shape, g.shape)

    def test_gamma_brightens_when_below_one(self):
        g = utils.to_gray(page(320, 400))
        self.assertGreater(float(enhancement.gamma_correction(g, 1.6).mean()),
                           float(g.mean()))

    def test_stretch_is_monotone_and_robust(self):
        g = utils.to_gray(page(320, 400))
        s = enhancement.stretch_contrast(g, 1.0)
        self.assertGreaterEqual(int(s.max()), int(g.max()) - 1)
        self.assertLessEqual(int(s.min()), int(g.min()) + 1)

    def test_flatten_illumination_removes_a_gradient(self):
        h, w = 300, 300
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = np.clip(120 + 80 * (xx / w) + 40 * (yy / h), 0, 255).astype(np.uint8)
        bg = enhancement.background_estimate_morph(grad, 61)
        flat = enhancement.flatten_illumination(grad, bg)
        self.assertLess(float(np.std(flat)), float(np.std(grad)))

    def test_measure_quality_reports_the_documented_keys(self):
        q = enhancement.measure_quality(utils.to_gray(page()))
        for k in ("contrast_std", "umichelson", "sharpness_lap_var", "entropy",
                  "bg_uniformity_std", "mean_intensity"):
            self.assertIn(k, q)

    def test_compare_variants_labels_each_row(self):
        g = utils.to_gray(page(240, 300))
        rows = enhancement.compare_variants(
            g, {"none": lambda x: x,
                "clahe": lambda x: enhancement.clahe_gray(x, 2.5, 8),
                "gamma": lambda x: enhancement.gamma_correction(x, 0.7)})
        self.assertEqual(sorted(r.name for r in rows), ["clahe", "gamma", "none"])
        self.assertIn("umichelson", rows[0].metrics)


class TestDenoise(unittest.TestCase):
    def test_every_named_method_reduces_grain(self):
        flat = np.full((120, 160, 3), 180, np.uint8)
        nz = noisy(flat, 18.0)
        for method in denoise.METHODS:
            if method in ("none", "auto"):
                continue
            out = denoise.apply_method(nz, method, {"ksize": 5, "h": 10.0})
            self.assertEqual(out.shape, nz.shape, msg=method)
            self.assertLess(float(out.std()), float(nz.std()),
                            msg=f"{method} did not reduce noise")

    def test_wiener_keeps_a_strong_edge(self):
        img = np.zeros((120, 120), np.uint8)
        img[:, 60:] = 160
        nz = noisy(img, 12.0)
        w = denoise.wiener_local(nz, ksize=5)
        before = float(np.abs(np.diff(nz[:, 58:63].astype(np.int16), axis=1)).mean())
        after = float(np.abs(np.diff(w[:, 58:63].astype(np.int16), axis=1)).mean())
        self.assertGreater(after, 0.5 * before)

    def test_estimate_noise_sigma_increases_with_grain(self):
        clean = np.full((120, 160), 180, np.uint8)
        self.assertLess(denoise.estimate_noise_sigma(clean),
                        denoise.estimate_noise_sigma(noisy(clean, 20.0)))

    def test_compare_filters_returns_one_report_per_method(self):
        rep = denoise.compare_filters(noisy(page(200, 260), 16.0),
                                      methods=("median", "bilateral", "gaussian"),
                                      params={"ksize": 3})
        self.assertEqual(len(rep), 3)
        self.assertTrue(all(r.method in ("median", "bilateral", "gaussian")
                            for r in rep))
        self.assertTrue(all(isinstance(r.metrics, dict) for r in rep))


class TestGeometry(unittest.TestCase):
    def test_crop_and_make_roi(self):
        img = page(300, 400)
        self.assertEqual(geometry.crop(img, (10, 20, 100, 80)).shape[:2], (80, 100))
        mask = geometry.make_roi(img.shape[:2], mode="box", box=(10, 20, 100, 80))
        self.assertEqual(int(np.count_nonzero(mask)), 100 * 80)
        full = geometry.make_roi(img.shape[:2])
        self.assertEqual(int(np.count_nonzero(full)), full.size)

    def test_deskew_recovers_a_known_rotation_with_the_right_sign(self):
        # rotate the page by +2 deg, then ask the estimator for the correction
        img = page(300, 400)
        turned, _ = geometry.rotate(img, 2.0)
        ang, info = geometry.estimate_skew_angle(utils.to_gray(turned), max_angle=6.0)
        self.assertAlmostEqual(ang, -2.0, delta=0.4)
        self.assertEqual(info.get("rejected"), 0.0)

    def test_a_blank_frame_is_never_rotated(self):
        ang, info = geometry.estimate_skew_angle(np.full((300, 300), 200, np.uint8))
        self.assertEqual(ang, 0.0)
        self.assertEqual(info.get("rejected"), 1.0)

    def test_four_point_transform_of_a_synthetic_photo(self):
        src = page(300, 400)
        M = cv2.getRotationMatrix2D((150, 200), 4.0, 1.0)
        turned = cv2.warpAffine(src, M, (300, 400), borderValue=(20, 20, 20))
        frame = np.full((480, 380, 3), 20, np.uint8)
        frame[40:440, 40:340] = turned
        pts, how = geometry.find_page_corners(frame)
        self.assertIsNotNone(pts, msg=how)
        rect, meta = geometry.four_point_transform(frame, pts)
        self.assertGreater(rect.shape[0], rect.shape[1])       # portrait kept
        for k in ("width_px", "height_px", "aspect_after", "skew_corner_dev_px"):
            self.assertIn(k, meta)
        self.assertAlmostEqual(meta["width_px"], 300.0, delta=26)   # 4 deg rotation
        self.assertLess(meta["aspect_after"], 0.9)      # w/h < 1 = portrait kept

    def test_scale_to_dpi_changes_only_the_size(self):
        out = geometry.scale_to_dpi(page(200, 300), src_dpi=150, dst_dpi=300)
        self.assertEqual(out.shape[:2], (600, 400))

    def test_auto_crop_removes_the_desk(self):
        frame = np.full((300, 300, 3), 15, np.uint8)
        frame[50:250, 60:240] = page(180, 200)
        out, box = geometry.auto_crop(frame, margin_px=2)
        self.assertLess(out.shape[0], 300)
        self.assertEqual(len(box), 4)


class TestSegmentation(unittest.TestCase):
    def test_otsu_separates_ink_from_paper(self):
        g = utils.to_gray(page(300, 400))
        bw, thr = segmentation.otsu(g)
        frac = float(np.count_nonzero(bw)) / bw.size
        self.assertTrue(0.02 < frac < 0.4, msg=f"ink fraction {frac}, threshold {thr}")

    def test_adaptive_and_sauvola_agree_in_size(self):
        g = utils.to_gray(page(300, 400))
        a = segmentation.adaptive_gray(g, block=31, c=10)
        s, _T = segmentation.sauvola(g, window=31, k=0.2)
        self.assertLess(abs(float(np.count_nonzero(a)) - float(np.count_nonzero(s)))
                        / a.size, 0.15)

    def test_morph_plan_and_components(self):
        mask = square(40, (60, 60))
        noise = mask.copy()
        noise[3, 3] = 255                            # an isolated speckle
        plan = segmentation.MorphPlan(open_px=1, close_px=0, kernel=3)
        cleaned = plan.apply(noise)
        self.assertLess(int(np.count_nonzero(cleaned)), int(np.count_nonzero(noise)))
        self.assertGreater(int(np.count_nonzero(cleaned)), 1400)
        cc = segmentation.connected_components(cleaned, min_area=20)
        self.assertEqual(len(cc), 1)                 # the speckle is gone
        self.assertAlmostEqual(float(cc[0]["area_px"]), float(np.count_nonzero(cleaned)))
        self.assertEqual(len(cc[0]["bbox"]), 4)

    def test_fill_holes_without_evidence_fills_any_enclosed_hole(self):
        ring = np.zeros((80, 80), np.uint8)
        cv2.circle(ring, (40, 40), 25, 255, 6)
        filled = segmentation.fill_holes(ring, max_hole_frac=0.8)
        self.assertGreater(int(np.count_nonzero(filled)), int(np.count_nonzero(ring)))

    def test_fill_holes_with_evidence_is_selective(self):
        ring = np.zeros((80, 80), np.uint8)
        cv2.circle(ring, (40, 40), 25, 255, 6)
        ev_dark = np.full((80, 80), 10.0, np.float32)
        ev_hot = np.full((80, 80), 90.0, np.float32)
        a = segmentation.fill_holes(ring, 0.8, evidence=ev_dark, evidence_min=50.0)
        b = segmentation.fill_holes(ring, 0.8, evidence=ev_hot, evidence_min=50.0)
        self.assertEqual(int(np.count_nonzero(a)), int(np.count_nonzero(ring)))
        self.assertGreater(int(np.count_nonzero(b)), int(np.count_nonzero(a)))

    def test_colour_segment_profiles(self):
        img = page()
        cv2.circle(img, (240, 200), 40, (60, 120, 175), -1)
        out = segmentation.colour_segment(img, profile="foxing / brown stain")
        self.assertIn("mask", out)
        self.assertGreater(int(np.count_nonzero(out["mask"])), 1000)

    def test_watershed_peaks_split_touching_blobs(self):
        m = np.zeros((120, 260), np.uint8)
        cv2.circle(m, (60, 60), 34, 255, -1)
        cv2.circle(m, (140, 60), 34, 255, -1)
        peaks = segmentation.marker_watershed(m, min_sep_px=24)
        self.assertGreaterEqual(peaks["regions"], 2)
        self.assertEqual(peaks["method"], "peaks")
        thr = segmentation.marker_watershed(m, mode="threshold", sure_fg_frac=0.6)
        self.assertLessEqual(thr["regions"], peaks["regions"])

    def test_mask_stats_areas(self):
        st = segmentation.mask_stats(square(40, (100, 100)), dpi=300)
        self.assertAlmostEqual(st["area_px"], 1600.0)
        self.assertAlmostEqual(st["area_mm2"], 1600.0 * (25.4 / 300.0) ** 2, places=3)
        self.assertAlmostEqual(st["area_pct"], 16.0)
        self.assertGreater(st["area_cm2"], 0.0)

    def test_largest_component(self):
        m = np.zeros((100, 100), np.uint8)
        m[5:15, 5:15] = 255
        m[50:90, 50:90] = 255
        big = segmentation.largest_component(m)
        self.assertAlmostEqual(float(np.count_nonzero(big)), 1600.0)


class TestFeatures(unittest.TestCase):
    @staticmethod
    def _contour(mask):
        c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        return c

    def test_descriptors_of_a_square_and_a_circle(self):
        cnt = self._contour(square(60, (120, 120)))
        fs = features.describe_contour(cnt, dpi=300)
        # 59*59, not 60*60: findContours traces between pixel centres, so the
        # polygon of an n x n filled square encloses (n-1)^2.  The mask pixel
        # count is what the damage layer reports, and `mask=` is what lets the
        # two agree - so both numbers must be present and stable.
        self.assertAlmostEqual(fs["area_px"], 59 * 59, delta=8)
        self.assertAlmostEqual(fs["area_mm2"], 59 * 59 * (25.4 / 300.0) ** 2, delta=1.0)
        self.assertAlmostEqual(fs["extent"], 1.0, delta=0.05)
        self.assertAlmostEqual(fs["solidity"], 1.0, delta=0.05)
        circ = features.describe_contour(
            self._contour(cv2.circle(np.zeros((160, 160), np.uint8), (80, 80), 30, 255, -1)),
            dpi=300)
        self.assertGreater(circ["circularity"], 0.85)      # pixelation caps this
        self.assertLess(circ["circularity"], 1.06)
        self.assertLess(circ["elongation"], 1.25)
        self.assertLess(fs["circularity"], circ["circularity"])
        self.assertAlmostEqual(fs["extent"], 1.0, delta=0.05)
        self.assertAlmostEqual(fs["solidity"], 1.0, delta=0.05)
        c = np.zeros((120, 120), np.uint8)
        cv2.circle(c, (60, 60), 40, 255, -1)
        fc = features.describe_contour(self._contour(c), dpi=300)
        # 0.89, not 1.0: findContours walks pixel centres, so a 40 px disc is a
        # 28-gon.  Circularity is used as a *ratio* between shapes (that is what
        # classify_defect thresholds on), never as an absolute test for "round".
        self.assertGreater(fc["circularity"], 0.85)
        self.assertLess(fs["circularity"], 0.85)
        # and circularity is exactly the definition, computed from the two
        # stored measurements (so a later refactor cannot silently redefine it)
        self.assertAlmostEqual(fc["circularity"],
                               4 * np.pi * fc["area_px"] / fc["perimeter_px"] ** 2,
                               delta=2e-3)
        self.assertAlmostEqual(fs["area_mm2"], fs["area_px"] * (25.4 / 300.0) ** 2,
                               delta=0.01)

    def test_classify_defect_is_shape_driven(self):
        crack = {"elongation": 9.0, "circularity": 0.05, "solidity": 0.9,
                 "extent": 0.2, "area_mm2": 4.0, "perimeter_mm": 40.0}
        blob = {"elongation": 1.1, "circularity": 0.8, "solidity": 0.95,
                "extent": 0.8, "area_mm2": 3.0, "perimeter_mm": 6.5}
        ragged = {"elongation": 1.2, "circularity": 0.3, "solidity": 0.4,
                  "extent": 0.5, "area_mm2": 5.0, "perimeter_mm": 14.0}
        self.assertEqual(features.classify_defect(crack), "crack/tear")
        self.assertIn(features.classify_defect(blob), ("foxing/stain", "ink-blodge"))
        self.assertEqual(features.classify_defect(ragged), "abrasion")

    def test_lbp_histogram_is_normalised(self):
        h = features.lbp_histogram(utils.to_gray(page(200, 260)))
        self.assertAlmostEqual(float(h.sum()), 1.0, places=3)

    def test_texture_bundle_and_profiles(self):
        g = utils.to_gray(page(240, 320))
        tb = features.texture_bundle(g)
        self.assertGreater(tb["glcm_contrast"], 0.0)
        # the profile projects the *supplied mask*, which is how the readability
        # measure uses it: give it the ink and it counts text bands (an all-valid
        # mask would be flat - garbage in, garbage out, on purpose).
        ink = segmentation.otsu(g)[0]
        prof = features.ink_density_profile(g, ink)
        self.assertEqual(len(prof["profile"]), 24)              # the default bins
        self.assertGreaterEqual(prof["peaks"], 2)               # 3 lines fit in 240 rows
        self.assertGreater(prof["peak"], prof["mean"])

    def test_gabor_and_dog_bands(self):
        g = utils.to_gray(page(200, 240))
        gb = features.gabor_energy(g)
        self.assertTrue(gb)
        self.assertTrue(all(np.isfinite(v) and v >= 0 for v in gb.values()))
        dd = features.dog_detail(g)
        self.assertTrue(dd)
        self.assertTrue(all(isinstance(v, float) and np.isfinite(v) for v in dd.values()))


class TestFrequency(unittest.TestCase):
    def test_high_frequency_share_drops_with_blur(self):
        g = utils.to_gray(page(300, 400))
        hi = frequency.spectrum_metrics(g)
        lo = frequency.spectrum_metrics(cv2.GaussianBlur(g, (0, 0), 4.0))
        self.assertGreater(hi["hf_power_ratio"], lo["hf_power_ratio"])
        self.assertLess(frequency.estimate_blur_std(g),
                        frequency.estimate_blur_std(cv2.GaussianBlur(g, (0, 0), 4.0)))
        for k in ("lf_power_ratio", "spectral_flatness", "half_power_radius"):
            self.assertIn(k, hi)

    def test_bandpass_notch_and_spectrum_images(self):
        g = utils.to_gray(page(200, 260))
        bp = frequency.bandpass_filter(g, 0.02, 0.4)
        nt, info = frequency.notch_suppress(g)
        sp = frequency.dft_analysis(g)
        self.assertEqual(bp.shape, g.shape)
        self.assertEqual(nt.shape, g.shape)
        self.assertEqual(frequency.log_spectrum_image(g).shape, g.shape)
        self.assertEqual(sp.magnitude.shape[:2], g.shape)
        self.assertIn("notches", info)


class TestMatching(unittest.TestCase):
    @staticmethod
    def _dense_text():
        rng = np.random.default_rng(5)
        g = np.full((240, 320), 225, np.uint8)
        for y in range(10, 230, 7):
            x = 14
            while x < 300:
                w = int(rng.integers(8, 30))
                cv2.rectangle(g, (x, y), (x + w, y + 3),
                              int(40 + rng.integers(0, 25)), -1)
                x += w + int(rng.integers(3, 9))
        return np.clip(g.astype(np.int16) + rng.normal(0, 4, g.shape),
                       0, 255).astype(np.uint8)

    def test_compare_images_reports_a_real_change(self):
        a = page(240, 320)
        b = a.copy()
        cv2.circle(b, (120, 160), 30, (10, 10, 10), -1)
        d = matching.compare_images(a, b)
        self.assertGreater(d.changed_pct, 0.5)
        self.assertGreater(d.n_changes, 0)
        self.assertEqual(d.diff_map.shape[:2], a.shape[:2])
        self.assertLess(d.psnr, 60.0)
        heat = matching.change_heatmap(d.diff_map)
        self.assertEqual(heat.ndim, 3)
        self.assertEqual(heat.shape[:2], a.shape[:2])

    def test_phase_correlation_recovers_a_shift(self):
        a = utils.to_gray(page(240, 300))
        b = cv2.warpAffine(a, np.float32([[1, 0, 12], [0, 1, -7]]),
                           (a.shape[1], a.shape[0]))
        out = matching.phase_correlate(a, b)
        self.assertAlmostEqual(out["dx"], 12.0, delta=2.0)
        self.assertAlmostEqual(out["dy"], -7.0, delta=2.0)

    def test_template_match_finds_a_stamp(self):
        img = page(320, 420)
        tpl = img[420 - 44:420 - 6, 320 // 2 - 24:320 // 2 + 24].copy()
        res = matching.match_template(img, tpl, threshold=0.8)
        self.assertGreaterEqual(res["count"], 1)

    def test_register_ecc_reduces_the_difference(self):
        a = utils.to_gray(page(240, 300))
        b = cv2.warpAffine(a, np.float32([[1, 0, 6], [0, 1, 4]]),
                           (a.shape[1], a.shape[0]))
        # dense text: ECC needs gradients everywhere, and a page that is 90%
        # blank paper is exactly the case where it must *refuse* rather than
        # invent a shift - both behaviours are part of the contract.
        g = self._dense_text()
        shift = (3.0, -2.0)
        warped = cv2.warpAffine(g, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                                (g.shape[1], g.shape[0]))
        r = matching.register_ecc(warped, g, mode="translation")
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["shift_px"][0], -shift[0], delta=0.4)
        self.assertAlmostEqual(r["shift_px"][1], -shift[1], delta=0.4)
        self.assertGreater(r["correlation"], 0.9)
        self.assertEqual(r["aligned"].shape, warped.shape)
        flat = matching.register_ecc(np.full((120, 120), 128, np.uint8),
                                     np.full((120, 120), 120, np.uint8))
        self.assertFalse(flat["ok"])
        self.assertIn("reason", flat)


class TestMetrics(unittest.TestCase):
    def test_psnr_ssim_rmse_bounds(self):
        a = page(120, 160)
        self.assertEqual(metrics.rmse(a, a), 0.0)
        self.assertTrue(np.isinf(metrics.psnr(a, a)))
        self.assertAlmostEqual(metrics.ssim(a, a), 1.0, places=3)
        nz = noisy(a, 20.0)
        self.assertTrue(10.0 < metrics.psnr(a, nz) < 45.0)
        self.assertLess(metrics.ssim(a, nz), 1.0)

    def test_overlap_stats_on_a_known_case(self):
        pred = np.zeros((100, 100), np.uint8)
        pred[:50] = 255
        truth = np.zeros((100, 100), np.uint8)
        truth[:25] = 255
        st = metrics.overlap_stats(pred, truth)
        self.assertAlmostEqual(st["iou"], 0.5, places=3)
        self.assertAlmostEqual(st["precision"], 0.5, places=3)
        self.assertAlmostEqual(st["recall"], 1.0, places=3)
        self.assertAlmostEqual(st["area_error_pct"], 100.0, places=1)

    def test_masked_metrics_only_see_the_region(self):
        a = page(120, 160)
        b = a.copy()
        b[10:20, 10:20] = 0
        inside = np.zeros(a.shape[:2], np.uint8)
        inside[10:20, 10:20] = 255
        outside = np.zeros(a.shape[:2], np.uint8)
        outside[60:70, 60:70] = 255
        self.assertFalse(np.isinf(metrics.psnr_masked(a, b, inside)))
        self.assertTrue(np.isinf(metrics.psnr_masked(a, b, outside)))
        self.assertTrue(np.isnan(metrics.psnr_masked(a, b, np.zeros_like(inside))))

    def test_error_helpers(self):
        self.assertEqual(metrics.count_error(5, 5)["count_abs_error"], 0)
        self.assertAlmostEqual(metrics.count_error(3, 5)["count_rel_error_pct"], -40.0)
        e = metrics.measurement_error(11.0, 10.0, "area")
        self.assertAlmostEqual(e["area_rel_error_pct"], 10.0, places=2)
        self.assertAlmostEqual(e["area_abs_error"], 1.0, places=4)

    def test_contrast_improvement_direction(self):
        # a washed-out, low-dynamic-range page: this is the case enhancement is
        # meant to help, so the measure must be positive here
        g = utils.to_gray(page(200, 260))
        low = np.clip(150 + (g.astype(np.float32) - 150.0) * 0.25, 0, 255).astype(np.uint8)
        out = enhancement.clahe_gray(low, 3.5, 8)
        d = metrics.contrast_improvement(low, out)
        self.assertGreater(d["contrast_gain_pct"], 0.0)
        self.assertGreater(d["contrast_after"], d["contrast_before"])

    def test_summarise_reports_median_and_iqr(self):
        rows = [{"x": float(i)} for i in range(1, 6)]
        s = metrics.summarise(rows, ["x"])
        self.assertAlmostEqual(s["x_mean"], 3.0)
        self.assertAlmostEqual(s["x_median"], 3.0)
        self.assertEqual(s["x_n"], 5.0)
        self.assertIn("x_iqr", s)
        self.assertIn("x_min", s)


class TestDamage(unittest.TestCase):
    def test_page_mask_of_a_photo_on_a_dark_desk(self):
        frame = np.full((320, 320, 3), 18, np.uint8)
        frame[40:280, 50:270] = page(220, 240)
        m = damage.page_mask(frame)
        frac = float(np.count_nonzero(m)) / m.size
        self.assertTrue(0.45 < frac < 0.85, msg=str(frac))

    def test_stain_map_finds_a_brown_spot_and_ignores_clean_paper(self):
        clean = page(320, 420)
        spot = clean.copy()
        cv2.circle(spot, (200, 210), 45, (60, 120, 175), -1)
        r = damage.stain_map(spot, dpi=300, sensitivity=1.0)
        frac = float(np.count_nonzero(r["mask"])) / r["mask"].size
        self.assertGreater(frac, 0.01)
        self.assertLess(frac, 0.25)
        self.assertIn("t_sat", r)
        self.assertLess(float(np.count_nonzero(damage.stain_map(clean, dpi=300)["mask"]))
                        / (320 * 420), 0.03)

    def test_crack_map_returns_a_mask_and_lengths(self):
        # flat paper on purpose.  The crack layer looks for ridge evidence, and
        # the pipeline hands it the *denoised* frame for exactly that reason;
        # the second half of this test pins the documented failure mode.
        img = np.zeros((420, 320, 3), np.uint8)
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = 200, 214, 228
        pts = np.array([[60, 150], [130, 205], [200, 235], [280, 320]], np.int32)
        cv2.polylines(img, [pts], False, (30, 28, 26), 2, cv2.LINE_AA)
        r = damage.crack_map(utils.to_gray(img), sensitivity=1.0, dpi=300)
        self.assertIsInstance(r["mask"], np.ndarray)
        self.assertEqual(r["mask"].shape, img.shape[:2])
        self.assertGreater(r["components"], 0)
        self.assertGreater(int(np.count_nonzero(r["mask"])), 500)
        self.assertGreater(max(seg["length_mm"] for seg in r["segments"]), 5.0)
        self.assertTrue(all(seg["length_px"] > 0 for seg in r["segments"]))
        # an explicit ROI identical to the frame must change nothing, and the
        # ROI must act as a hard restriction (a crack clipped by it stops being
        # recognisable as one - the blunt cut fails the width test, which is the
        # correct conservative answer when the sheet edge is unknown)
        full = np.full(img.shape[:2], 255, np.uint8)
        self.assertEqual(int(np.count_nonzero(
            damage.crack_map(utils.to_gray(img), roi=full, dpi=300)["mask"])),
            int(np.count_nonzero(r["mask"])))
        half = np.zeros(img.shape[:2], np.uint8)
        half[:200] = 255
        clipped = damage.crack_map(utils.to_gray(img), roi=half, dpi=300)
        self.assertEqual(int(np.count_nonzero(clipped["mask"][200:])), 0)

    def test_missing_fragment_needs_a_visible_background(self):
        img = page(320, 420)
        roi = np.full(img.shape[:2], 255, np.uint8)
        mask, info = damage.missing_fragment_map(img, roi)
        self.assertEqual(int(mask.sum()), 0)
        self.assertFalse(info["available"])
        self.assertIn("fills the frame", str(info["reason"]))

    def test_missing_fragment_finds_a_hole_over_the_mount_board(self):
        # the board is mid-grey on purpose: a *dark* desk and dark ink are the
        # same colour, and no colour-based test can separate a hole from a
        # letter in that situation - so the test uses a distinguishable mount
        frame = np.full((560, 460, 3), 118, np.uint8)
        frame[20:540, 20:440] = page(420, 520)
        cv2.circle(frame, (230, 280), 26, (120, 121, 118), -1)
        roi = np.zeros(frame.shape[:2], np.uint8)
        roi[20:540, 20:440] = 255
        mask, info = damage.missing_fragment_map(frame, roi, min_area_mm2=0.8)
        self.assertTrue(info["available"], msg=info.get("reason"))
        self.assertGreater(int(np.count_nonzero(mask)), 1200)
        self.assertLess(float(np.count_nonzero(mask)) / mask.size, 0.03)
        self.assertTrue(info["separation"] > 45.0)

    def test_analyse_damage_produces_the_decision_fields(self):
        img = page(360, 480)
        cv2.circle(img, (180, 150), 40, (60, 120, 175), -1)
        cv2.circle(img, (250, 380), 18, (55, 110, 165), -1)
        p = PipelineParams()
        text = segmentation.otsu(utils.to_gray(img))[0]
        roi = np.full(img.shape[:2], 255, np.uint8)
        rep = damage.analyse_damage(img, p, text, roi, name="t.png")
        self.assertEqual(rep.image_name, "t.png")
        for layer in ("damage", "stain", "crack", "missing"):
            self.assertIn(layer, rep.masks)
        self.assertGreaterEqual(rep.damage_pct, 1.0)
        self.assertGreater(rep.n_damage_total, 0)
        self.assertGreater(rep.n_stain, 0)
        self.assertIn("damage_pct", rep.as_row())
        self.assertTrue(all(0.0 <= r.severity <= 1.0 for r in rep.regions))

    def test_hole_fill_is_evidence_bounded(self):
        # a ring of stain with clean-looking centre must not swallow the page
        img = page(360, 480)
        cv2.circle(img, (180, 240), 90, (60, 120, 175), 24)
        r = damage.stain_map(img, dpi=300, sensitivity=0.6)
        self.assertLess(float(np.count_nonzero(r["mask"])) / r["mask"].size, 0.4)


class TestRestorationAndScoring(unittest.TestCase):
    def test_inpaint_repairs_a_mark_and_keeps_the_frame(self):
        img = page(240, 320)
        mark = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mark, (120, 160), 18, 255, -1)
        img[mark > 0] = (10, 10, 10)
        out = restoration.inpaint_damage(img, mark, radius=4, method="telea")
        self.assertEqual(out.restored.shape, img.shape)
        self.assertEqual(out.original.shape, img.shape)
        ring = cv2.dilate(mark, np.ones((17, 17), np.uint8))
        ring = cv2.subtract(ring, mark)
        inside_after = float(out.restored[mark > 0].astype(np.float32).mean())
        paper = float(out.restored[ring > 0].astype(np.float32).mean())
        self.assertGreater(inside_after, 0.55 * paper)      # pulled toward paper

    def test_restore_lists_the_operations_it_applied(self):
        img = page(240, 320)
        mark = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mark, (60, 60), 12, 255, -1)
        rr = restoration.restore(noisy(img, 12.0), mark, None, None, PipelineParams())
        self.assertEqual(rr.restored.shape, img.shape)
        self.assertTrue(rr.operations)
        self.assertIn("inpaint_area_px", rr.stats)

    def test_despeckle_removes_isolated_specks_only(self):
        img = page(240, 320, ink=False)
        for (x, y) in ((30, 30), (200, 250), (120, 80)):
            img[y:y + 2, x:x + 2] = (10, 10, 10)
        out, removed = restoration.despeckle(img, open_px=2, max_speckle_area_px=12)
        self.assertGreater(int(np.count_nonzero(removed)), 0)
        self.assertEqual(out.shape, img.shape)

    def test_quality_index_reacts_to_blur(self):
        p = PipelineParams()
        g = utils.to_gray(page(300, 400))
        q1 = scoring.assess_quality(g, None, 300, p)
        q2 = scoring.assess_quality(cv2.GaussianBlur(g, (0, 0), 5.0), None, 300, p)
        self.assertTrue(0.0 <= q1.score <= 100.0)
        self.assertLess(q2.score, q1.score)
        self.assertTrue(q1.factors)
        self.assertIn("q_sharpness", q1.as_dict())

    def test_condition_flag_follows_the_damage_percentage(self):
        p = PipelineParams(warn_damage_pct=1.0, fail_damage_pct=4.0)
        c0 = scoring.assess_condition(damage.DamageReport(damage_pct=0.2), p, 1e5)
        c1 = scoring.assess_condition(
            damage.DamageReport(damage_pct=2.0, n_damage_total=3), p, 1e5)
        c2 = scoring.assess_condition(
            damage.DamageReport(damage_pct=25.0, n_damage_total=40, n_text_loss=3,
                                largest_damage_mm2=90.0), p, 1e5)
        self.assertEqual(c0.pass_fail, "PASS")
        self.assertEqual(c1.pass_fail, "WARN")
        self.assertEqual(c2.pass_fail, "FAIL")
        self.assertTrue(c2.warning)
        self.assertTrue(c2.actions)
        self.assertGreater(c2.score, c1.score)
        self.assertEqual(c2.severity_class, "SEVERE")

    def test_readability_and_text_statistics(self):
        g = utils.to_gray(page(300, 400))
        roi = np.full(g.shape, 255, np.uint8)
        bw = segmentation.otsu(g)[0]
        rd = scoring.measure_readability(g, bw, roi, 300)
        self.assertIn("score", rd)
        self.assertTrue(0.0 <= rd["score"] <= 100.0)
        ts = scoring.text_statistics(bw, roi, 300)
        self.assertGreaterEqual(ts["lines"], 4)
        self.assertGreater(ts["components"], 4)
        self.assertGreater(ts["words"], 0)

    def test_authenticity_indicators_are_advisory_only(self):
        ind = scoring.authenticity_indicators(utils.to_gray(page(300, 400)), 300)
        for k in ("ind_halftone_periodicity", "ind_anisotropy", "ind_micro_texture",
                  "ind_blur_sigma_px", "ind_glcm_contrast", "ind_lbp_entropy"):
            self.assertIn(k, ind)
        self.assertGreaterEqual(ind["ind_lbp_entropy"], 0.0)
        self.assertLessEqual(ind["ind_lbp_entropy"], 8.0)     # entropy in bits
        # a flat vector graphic has no paper texture at all -> the indicator moves
        flat = np.full((300, 400), 200, np.uint8)
        self.assertLess(scoring.authenticity_indicators(flat, 300)["ind_micro_texture"],
                        ind["ind_micro_texture"])


class TestAnnotateAndExport(unittest.TestCase):
    def test_overlay_and_side_by_side_shapes(self):
        img = page(300, 400)
        mask = square(60, img.shape[:2])
        ov = annotate.mask_overlay(img, mask)
        self.assertEqual(ov.shape, img.shape)
        self.assertFalse(np.array_equal(ov[mask > 0], img[mask > 0]))
        sb = annotate.side_by_side(img, ov)
        self.assertGreater(sb.shape[1], 2 * img.shape[1] - 30)   # both panels + gap
        self.assertGreater(sb.shape[0], img.shape[0])             # + the label band

    def test_class_colours_cover_every_class(self):
        rep = damage.DamageReport(image_name="x")
        self.assertTrue(annotate.CLASS_COLORS)
        self.assertIsInstance(rep, damage.DamageReport)


if __name__ == "__main__":
    unittest.main(verbosity=1)
