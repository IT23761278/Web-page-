"""Core image-processing modules of HeritageVision (OpenCV + NumPy only).

Module map (order of use in the pipeline):

    utils         validation, safe I/O, colour/number helpers
    metadata      dimensions, channels, bit depth, file size, region info
    colorspace    Grayscale / HSV / Lab / YCrCb / Binary handling
    enhancement   CLAHE, stretch, gamma, illumination correction, unsharp
    denoise       6 filters + automatic comparison and selection
    geometry      ROI, crop, deskew, perspective rectification, dpi scaling
    segmentation  Otsu / adaptive / Sauvola, morphology, CCL, watershed
    damage        domain analysis: stains, cracks, material loss, regions
    features      shape + texture descriptors (LBP, GLCM, Gabor, DoG)
    frequency     DFT analysis, notch / band-pass filtering, blur estimate
    matching      difference detection, template matching, ECC registration
    restoration   inpainting, despeckle, tone recovery
    scoring       quality index, condition score, readability, decisions
    metrics       PSNR / SSIM / IoU / Dice / F1 / errors
    evaluate      ground-truth evaluation harness for controlled test images
    annotate      visual decision overlay
    export        CSV / JSON / text summary writers
    pipeline      integration of all of the above
"""

from .config import PipelineParams  # noqa: F401
from .pipeline import HeritagePipeline, ProcessResult  # noqa: F401

__all__ = ["HeritagePipeline", "ProcessResult", "PipelineParams"]
