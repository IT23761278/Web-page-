"""
heritage_vision.core.config
==========================

Central place for every default parameter, colour constant, file rule and
folder name used by the HeritageVision application.

Why a config module?
    The VIVA asks "why did you choose this value?".  Keeping every tunable
    value in one file (with the justification as a comment next to it) makes
    the answer traceable and lets the GUI/CLI share exactly the same defaults,
    so a result produced by the GUI can be reproduced from the command line.

Domain assumptions that drive these defaults
    * Input material: flatbed/phone scans of paper heritage items (manuscript
      leaves, photographs, inscriptions, paintings).  They are usually
      8-bit, unevenly illuminated, yellowed by lignin oxidation and covered
      with foxing/stains, dust specks and hairline cracks.
    * Information carrier: dark ink/pigment on a light, low-saturation
      cellulosic background.  Therefore dark, compact, high-saturation blobs
      are candidates for damage, and the light background is what the
      illumination model has to estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

# --------------------------------------------------------------------------
# 1. Acquisition rules
# --------------------------------------------------------------------------
#: Extensions accepted by the loader.  Anything else is rejected *before*
#: OpenCV is called, which gives the user a clearer message.
SUPPORTED_EXTENSIONS: List[str] = [
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
]

#: Hard cap for a decoded frame (pixels).  A 600 dpi A4 scan is ~200 Mpx;
#: decoding such a file inside a UI thread would freeze the application.
MAX_IMAGE_PIXELS: int = 60_000_000

#: Longest side used when a folder is processed in batch mode.
BATCH_LONGEST_SIDE: int = 2400

# --------------------------------------------------------------------------
# 2. Default pipeline parameters
# --------------------------------------------------------------------------


@dataclass
class PipelineParams:
    """Every user-tunable knob of the processing pipeline.

    The GUI edits an instance of this class, the CLI builds one from
    argparse values and ``PipelineConfig.run`` consumes it.  ``validate()``
    clamps values into the safe range so a bad edit can never crash OpenCV.
    """

    # --- 2.1 colour space ------------------------------------------------
    damage_space: str = "HSV"            #: HSV | Lab | YCrCb
    analysis_source: str = "raw"         #: raw | enhanced - which frame the
                                         #: damage detectors look at.  Stains are
                                         #: a *colour* property of the original
                                         #: scan, and the enhancement step
                                         #: (de-yellowing, background flattening)
                                         #: deliberately compresses exactly that
                                         #: range, so detecting on the enhanced
                                         #: image costs recall.  Measured on the
                                         #: controlled set in sec. 7.4 of the
                                         #: report; "enhanced" stays available
                                         #: because it is the right choice when
                                         #: the enhancement *is* the inspection.
    use_grayscale_for_text: bool = True

    # --- 2.2 enhancement -------------------------------------------------
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0        #: 1.0-4.0; >3 amplifies paper grain noise
    clahe_tile_grid: int = 8             #: 8x8 tiles for A4 aspect-ratio scans
    gamma: float = 1.0                   #: 1.0 = off, <1 brightens faded paper
    percentile_stretch: float = 1.0      #: robust min/max stretch, 1% each tail
    bg_normalisation: str = "morph"      #: none | morph | gaussian | none
    bg_kernel_mm: float = 12.0           #: structuring size of the background
                                         #: estimate; must be >> stroke width
    bg_smooth: int = 21                  #: Gaussian sigma (px) of bg estimate

    # --- 2.3 denoising ---------------------------------------------------
    denoise_method: str = "bilateral"    #: none|gaussian|median|bilateral|nlm|guided
    denoise_ksize: int = 5               #: odd, 3-9
    bilateral_sigma_color: float = 35.0  #: px-intensity sigma, keeps ink edges
    bilateral_sigma_space: float = 5.0
    nlm_h: float = 10.0                  #: NLM filter strength for 8-bit input

    # --- 2.4 geometry ----------------------------------------------------
    rectify: bool = False
    deskew: bool = True
    deskew_max_angle: float = 6.0        #: search window; a scan is never worse
                                         #: than a few degrees, and a wide window
                                         #: lets textureless photos "find" 13 deg
    target_dpi: int = 300                #: used for mm <-> px conversion
    auto_crop: bool = False
    crop_margin_px: int = 12             #: kept border after auto crop

    # --- 2.5 segmentation / damage detection -----------------------------
    binarisation: str = "otsu"           #: otsu | adaptive | sauvola-like
    adaptive_block: int = 51             #: odd, ~3x max stroke width
    adaptive_c: int = 10
    morph_open_px: int = 2               #: removes salt noise smaller than this
    morph_close_px: int = 3              #: rejoins broken strokes
    min_text_area_px: int = 25           #: connected component area filter
    max_text_area_frac: float = 0.60     #: reject page-wide components
    damage_sensitivity: float = 1.0      #: 1/sigma multiplier of the top-hat
                                         #: threshold; 1.5 = more lesions,
                                         #: 0.7 = only obvious ones
    damage_min_area_mm2: float = 0.8     #: physical size filter, dpi-proof
    damage_close_px: int = 5
    region_merge_mm: float = 2.0       #: gaps up to this are treated as one
                                       #: object when the region *count* is
                                       #: formed (areas are still measured on the
                                       #: unclosed mask)             #: merge stain fragments (px)
    hole_fill_max_frac: float = 0.45     #: fill the enclosed core of a tide ring

    # --- 2.6 restoration -------------------------------------------------
    inpaint: bool = True
    inpaint_radius: float = 3.0
    inpaint_method: str = "telea"        #: telea | ns
    inpaint_dilate_px: int = 2           #: safety margin around the damage mask
    sharpen: bool = True
    unsharp_amount: float = 0.6
    unsharp_sigma: float = 2.0
    despeckle_open_px: int = 2           #: removes isolated light/dark specks

    # --- 2.7 scoring ------------------------------------------------------
    weight_damage: float = 0.30
    weight_sharpness: float = 0.25
    weight_contrast: float = 0.20
    weight_lighting: float = 0.15
    weight_noise: float = 0.10
    warn_damage_pct: float = 1.0         #: % of ROI area -> WARNING flag
    fail_damage_pct: float = 4.0         #: % of ROI area -> CONSERVATION FLAG

    def validated(self) -> "PipelineParams":
        """Return a *clamped* copy; never raises, so the UI stays usable."""
        p = asdict(self)
        p["clahe_clip_limit"] = float(min(max(self.clahe_clip_limit, 0.5), 8.0))
        p["clahe_tile_grid"] = int(min(max(self.clahe_tile_grid, 2), 32))
        p["gamma"] = float(min(max(self.gamma, 0.4), 2.5))
        p["percentile_stretch"] = float(min(max(self.percentile_stretch, 0.0), 10.0))
        p["denoise_ksize"] = int(min(max(self.denoise_ksize, 3), 9))
        if p["denoise_ksize"] % 2 == 0:
            p["denoise_ksize"] += 1
        p["adaptive_block"] = int(min(max(self.adaptive_block, 9), 201))
        if p["adaptive_block"] % 2 == 0:
            p["adaptive_block"] += 1
        p["morph_open_px"] = int(min(max(self.morph_open_px, 0), 9))
        p["morph_close_px"] = int(min(max(self.morph_close_px, 0), 15))
        p["target_dpi"] = int(min(max(self.target_dpi, 72), 1200))
        p["deskew_max_angle"] = float(min(max(self.deskew_max_angle, 1.0), 15.0))
        p["damage_sensitivity"] = float(min(max(self.damage_sensitivity, 0.4), 2.5))
        p["region_merge_mm"] = float(min(max(self.region_merge_mm, 0.0), 6.0))
        if p.get("analysis_source") not in ("raw", "enhanced"):
            p["analysis_source"] = "raw"
        # every remaining field that a slider or a CLI flag can move; without
        # these, "--damage_min_area_mm2 1e9" would silently disable detection
        p["bg_kernel_mm"] = float(min(max(self.bg_kernel_mm, 2.0), 80.0))
        p["bg_smooth"] = int(min(max(self.bg_smooth, 1), 101))
        p["damage_min_area_mm2"] = float(min(max(self.damage_min_area_mm2, 0.05), 50.0))
        p["damage_close_px"] = int(min(max(self.damage_close_px, 0), 31))
        p["hole_fill_max_frac"] = float(min(max(self.hole_fill_max_frac, 0.0), 0.9))
        p["bilateral_sigma_color"] = float(min(max(self.bilateral_sigma_color, 1.0), 200.0))
        p["bilateral_sigma_space"] = float(min(max(self.bilateral_sigma_space, 1.0), 51.0))
        p["nlm_h"] = float(min(max(self.nlm_h, 1.0), 80.0))
        p["adaptive_c"] = int(min(max(self.adaptive_c, -30), 60))
        p["min_text_area_px"] = int(min(max(self.min_text_area_px, 1), 5000))
        p["max_text_area_frac"] = float(min(max(self.max_text_area_frac, 0.05), 0.98))
        p["crop_margin_px"] = int(min(max(self.crop_margin_px, 0), 200))
        p["despeckle_open_px"] = int(min(max(self.despeckle_open_px, 0), 9))
        p["warn_damage_pct"] = float(min(max(self.warn_damage_pct, 0.0), 100.0))
        p["fail_damage_pct"] = float(min(max(self.fail_damage_pct, 0.0), 100.0))
        if p["fail_damage_pct"] < p["warn_damage_pct"]:
            p["fail_damage_pct"] = p["warn_damage_pct"]
        p["inpaint_radius"] = float(min(max(self.inpaint_radius, 1.0), 15.0))
        p["inpaint_dilate_px"] = int(min(max(self.inpaint_dilate_px, 0), 9))
        p["unsharp_amount"] = float(min(max(self.unsharp_amount, 0.0), 3.0))
        p["unsharp_sigma"] = float(min(max(self.unsharp_sigma, 0.3), 10.0))
        for w in ("weight_damage", "weight_sharpness", "weight_contrast",
                  "weight_lighting", "weight_noise"):
            p[w] = float(min(max(p[w], 0.0), 1.0))
        s = sum(p[w] for w in ("weight_damage", "weight_sharpness",
                               "weight_contrast", "weight_lighting", "weight_noise"))
        if s <= 0:                                   # all weights zero -> fall back
            p.update(self.__class__().__dict__)
            s = 1.0
        for w in ("weight_damage", "weight_sharpness", "weight_contrast",
                  "weight_lighting", "weight_noise"):
            p[w] = p[w] / s                          # normalise -> score stays 0..100
        return PipelineParams(**p)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# 3. Visualisation constants
# --------------------------------------------------------------------------
COLOR_DAMAGE_OUTLINE = (0, 0, 220)     #: BGR red-ish for cracks/stains
COLOR_DAMAGE_FILL = (0, 0, 160)
COLOR_TEXT_BOX = (0, 160, 0)           #: green = detected writing
COLOR_REGION_BOX = (255, 140, 0)      #: blue = region of interest
ANNOTATION_ALPHA = 0.32                #: blend factor of the damage overlay

#: Severity classes reported to the user (Part C "decision output").
SEVERITY_NONE = "NONE"
SEVERITY_MINOR = "MINOR"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_SEVERE = "SEVERE"

#: Default locations, relative to the project root.
DIR_DATASET = "dataset"
DIR_STUDENT = "dataset/student_created"
DIR_REAL = "dataset/real_scans"
DIR_UNSEEN = "dataset/unseen_test"
DIR_GT = "dataset/ground_truth"
DIR_OUTPUT = "outputs"


@dataclass
class FolderLayout:
    """Folder contract of the project (Part A requirement)."""
    real: str = DIR_REAL
    student: str = DIR_STUDENT
    unseen: str = DIR_UNSEEN
    ground_truth: str = DIR_GT
    outputs: str = DIR_OUTPUT

    @property
    def all(self) -> List[str]:
        return [self.real, self.student, self.unseen, self.ground_truth, self.outputs]

    @property
    def dataset_root(self) -> str:
        return DIR_DATASET

    def manifest_name(self) -> str:
        return "dataset_manifest.csv"


LAYOUT = FolderLayout()
