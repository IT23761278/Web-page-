"""
heritage_vision.core.frequency
==============================

Part C technique 8 - frequency-domain analysis / filtering.

Two genuine uses in this domain (both are things spatial filters cannot do):

1) Blur and sampling diagnosis.  The radial power spectrum of a sharp scan
   falls off smoothly; a defocused or heavily upsampled scan collapses early.
   Reporting the *half-power radius* and the 0.1/0.5 power ratio tells the
   archivist "this scan is soft, re-scan at higher dpi" without any reference
   image, and it is what our 'sharpness' score is cross-checked against.

2) Periodic-noise removal.  Woven canvas, engraving hatching, halftone dots of
   a 19th century printed page and scanner moire all appear as *isolated peaks*
   away from the DC axis.  Notch-filtering them (while keeping every other
   frequency) recovers legibility without the edge loss of a low-pass blur -
   precisely the trade-off a document restorer faces.

Everything is float32 in log-magnitude space, with an fftshift-free
implementation (we build the frequency grid directly), so it is fast and exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import to_gray


@dataclass
class Spectrum:
    magnitude: np.ndarray          #: log magnitude, centred (fftshift applied)
    power: np.ndarray              #: |F|^2, centred
    fy: np.ndarray
    fx: np.ndarray
    rows: int
    cols: int

    def radial_profile(self, bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Mean power vs normalised radial frequency (0 = DC, 1 = Nyquist).

        Nyquist for a unit-sampled image is 0.5 cycles/pixel on each axis, so
        the normalisation constant is 0.5 and 1.0 means 'one dark/light pair
        per two pixels'.
        """
        r = np.sqrt(self.fx ** 2 + self.fy ** 2)
        idx = np.clip((r / 0.5 * bins).astype(np.int32), 0, bins - 1)
        prof = np.bincount(idx.ravel(), weights=self.power.ravel(), minlength=bins)
        cnt = np.bincount(idx.ravel(), minlength=bins)
        prof = prof / np.maximum(cnt, 1)
        axis = (np.arange(bins) + 0.5) / bins
        return axis, prof


def dft_analysis(gray: np.ndarray, window: bool = True) -> Spectrum:
    """2-D DFT with optional Hann window (removes the frame-edge leakage that
    would otherwise create false peaks along both axes)."""
    g = to_gray(gray).astype(np.float32)
    g = g - float(np.mean(g))
    if window:
        wy = np.hanning(g.shape[0])[:, None]
        wx = np.hanning(g.shape[1])[None, :]
        g = g * wy * wx
    F = np.fft.fft2(g)
    Fs = np.fft.fftshift(F)
    mag = np.abs(Fs)
    power = mag ** 2
    rows, cols = g.shape
    # frequency grid (cycles/sample) aligned with the *shifted* spectrum
    fy = ((np.arange(rows) - rows // 2) / float(rows)).reshape(rows, 1)
    fx = ((np.arange(cols) - cols // 2) / float(cols)).reshape(1, cols)
    logmag = np.log1p(mag)
    logmag = (logmag - logmag.min()) / max(logmag.max() - logmag.min(), 1e-9) * 255
    return Spectrum(magnitude=logmag.astype(np.uint8), power=power,
                    fy=np.broadcast_to(fy, (rows, cols)).copy(),
                    fx=np.broadcast_to(fx, (rows, cols)).copy(),
                    rows=rows, cols=cols)


def _reduce(gray: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Downsample for spectral work and return the factor back.

    A radial profile only needs ~64 bins, so a 640 px frame is more than
    enough; the per-pixel quantities (blur sigma) are converted back to the
    original scale by multiplying with this factor.
    """
    g = to_gray(gray)
    m = max(g.shape[:2])
    if max_side and m > max_side:
        f = m / float(max_side)
        small = cv2.resize(g, (max(int(g.shape[1] / f), 8), max(int(g.shape[0] / f), 8)),
                           interpolation=cv2.INTER_AREA)
        return small, f
    return g, 1.0


def spectrum_metrics(gray: np.ndarray, max_side: int = 640) -> Dict[str, float]:
    """Reference-free blur / moire descriptors from the radial power profile."""
    g, _ = _reduce(gray, max_side)
    spec = dft_analysis(g)
    axis, prof = spec.radial_profile(bins=64)
    total = prof.sum() + 1e-12
    lo = prof[axis <= 0.15].sum()
    mid = prof[(axis > 0.15) & (axis <= 0.5)].sum()
    hi = prof[axis > 0.5].sum()
    # half-power radius: frequency where the profile drops to 50% of the peak
    peak = prof.max()
    below = np.where(prof <= 0.5 * peak)[0]
    half_power_radius = float(axis[below[0]]) if below.size else 1.0
    # periodic noise detector: a bin far above its neighbours (not DC-adjacent)
    ring = prof.copy()
    ring[:3] = 0.0
    med = np.median(ring[ring > 0]) if np.any(ring > 0) else 0.0
    ratio = float(ring.max() / (med + 1e-9)) if med > 0 else 0.0
    return {
        "hf_power_ratio": round(float(hi / total), 5),
        "mf_power_ratio": round(float(mid / total), 5),
        "lf_power_ratio": round(float(lo / total), 5),
        "half_power_radius": round(half_power_radius, 4),
        "spectral_peak_ratio": round(ratio, 2),
        "spectral_flatness": round(float(np.exp(np.log(ring + 1e-12).mean()) /
                                          (ring.mean() + 1e-12)), 5),
    }


def bandpass_filter(gray: np.ndarray, low: float = 0.0, high: float = 1.0,
                    order: float = 2.0, shift_out: bool = False) -> np.ndarray:
    """Butterworth-style radial band-pass in the frequency domain.

    ``low``/``high`` are normalised radii (0..1 of Nyquist).  Used to isolate
    the ink-stroke band (0.05..0.35) or to inspect a defect band.
    """
    spec = dft_analysis(gray, window=False)
    r = np.sqrt(spec.fx ** 2 + spec.fy ** 2)
    rn = r / 0.5                                   # 1.0 == Nyquist
    hp = 1.0 / (1.0 + (max(low, 1e-6) / np.maximum(rn, 1e-9)) ** (2 * order))
    lp = 1.0 / (1.0 + (rn / max(high, 1e-6)) ** (2 * order))
    H = hp * lp
    F = np.fft.fftshift(np.fft.fft2(to_gray(gray).astype(np.float32)))
    out = np.fft.ifft2(np.fft.ifftshift(F * H)).real
    out = np.clip(out - out.min(), 0, 255).astype(np.uint8)
    return out


def notch_suppress(gray: np.ndarray, sigma_px: float = 3.0,
                   min_ratio: float = 12.0, keep_axes: float = 0.06
                   ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Detect strong off-axis spectral peaks and remove them with Gaussian
    notches, then inverse-transform.  This is the 'de-screening' step for
    halftone / hatching / moire patterns.

    ``keep_axes`` protects the horizontal and vertical frequency axes: energy
    sitting exactly on an axis is usually real content (text baselines, ruled
    lines), not periodic noise, so we must not notch it away.
    """
    g = to_gray(gray).astype(np.float32)
    g = g - float(np.mean(g))
    F = np.fft.fftshift(np.fft.fft2(g))
    P = np.abs(F) ** 2
    rows, cols = P.shape
    cy, cx = rows // 2, cols // 2
    smooth = cv2.GaussianBlur(P.astype(np.float64), (0, 0), max(sigma_px, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(smooth > 0, P / smooth, 0.0)
    yy, xx = np.mgrid[0:rows, 0:cols]
    off_dc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) > max(rows, cols) * 0.05
    off_axis = (np.abs(yy - cy) > cols * keep_axes) & (np.abs(xx - cx) > cols * keep_axes)
    peaks = (ratio > min_ratio) & off_dc & off_axis
    H = np.ones_like(P, np.float32)
    ys, xs = np.nonzero(peaks)
    n = int(ys.size)
    if n:
        # mirror pairs already included by fftshift symmetry; notch each peak
        r2 = (2 * max(sigma_px, 1.5)) ** 2
        dist2 = np.zeros_like(P, np.float32)
        for y, x in list(zip(ys, xs))[:60]:            # bound the cost
            H *= 1.0 - np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / r2))
            dist2 += 0
        filtered = np.fft.ifft2(np.fft.ifftshift(F * H)).real
        out = np.clip(filtered, 0, 255).astype(np.uint8)
    else:
        out = np.clip(g, 0, 255).astype(np.uint8)
    return out, {"notches": n, "max_ratio": round(float(ratio[peaks].max())
                                                  if n else 0.0, 2)}


def lowpass(gray: np.ndarray, cutoff_norm: float = 0.25, order: float = 2.0
            ) -> np.ndarray:
    return bandpass_filter(gray, low=0.0, high=cutoff_norm, order=order)


def highpass(gray: np.ndarray, cutoff_norm: float = 0.05, order: float = 2.0
             ) -> np.ndarray:
    return bandpass_filter(gray, low=cutoff_norm, high=1.0, order=order)


def log_spectrum_image(gray: np.ndarray) -> np.ndarray:
    """Visualisation for the GUI/report (centred log magnitude, 8-bit)."""
    spec = dft_analysis(gray)
    return spec.magnitude


def estimate_blur_std(gray: np.ndarray, max_side: int = 640) -> float:
    """Rough Gaussian-blur sigma estimate from the high-frequency decay.

    Method: fit log(power) vs f^2 over the mid band; the slope of a Gaussian
    PSF is -8*pi^2*sigma^2 (Pech-Pacheco et al.).  Reported in pixels of the
    *original* frame (the fit is done on a reduced frame for speed and the
    result multiplied by the reduction factor).
    """
    g, factor = _reduce(gray, max_side)
    spec = dft_analysis(g)
    axis, prof = spec.radial_profile(bins=96)
    band = (axis > 0.08) & (axis < 0.45)
    if band.sum() < 6:
        return 0.0
    f = axis[band]
    p = np.maximum(prof[band], 1e-12)
    y = np.log(p)
    A = np.vstack([f ** 2, np.ones_like(f)]).T
    try:
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    if slope >= 0:
        return 0.0
    sigma = np.sqrt(max(-slope / (8.0 * np.pi ** 2), 0.0)) * float(factor)
    return round(float(sigma), 4)
