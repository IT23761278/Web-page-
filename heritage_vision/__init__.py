"""
HeritageVision - Digital Image Processing final project (SE3071)
=================================================================

Domain (allocated by the last digit of the IT number = 8):
**Heritage & Culture** - restoration / enhancement / damage analysis of old
photographs, historical documents, paintings and inscriptions.

Specific problem solved
-----------------------
Quantify and reduce *surface damage and tonal fading* on scans/photographs of
paper-based heritage items (manuscript leaves, archival photographs, printed
documents) in a single reproducible pass, producing a damage mask, an area in
mm2, a per-region severity list, an archive-readiness quality index and a
conservation decision (PASS / WARN / FAIL).

Package layout
--------------
``core``     the image-processing pipeline (no GUI imports at all)
``tools``    dataset generator, CLI batch runner, evaluation, demo recorder
``app.py``   Tkinter interface (the only GUI file)
``tests``    unit tests for every core stage

The core deliberately has no dependency on Tkinter/PyQt so it can run headless
in the batch and evaluation tools (and in the unit tests).
"""

__version__ = "1.0.0"
__author__ = "IT23761278"
__module__ = "SE3071 Digital Image Processing - Year 3 Semester 2 (2026) Final Project"

from .core.config import PipelineParams  # noqa: F401
from .core.pipeline import HeritagePipeline, ProcessResult  # noqa: F401

__all__ = ["HeritagePipeline", "ProcessResult", "PipelineParams",
           "__version__"]
