"""
vision/retina.py
=================

Converts a video frame into a simplified array of "photoreceptor" responses.

BIOLOGY (real): the Drosophila compound eye has ~750-800 ommatidia per eye
arranged in a hexagonal lattice; each ommatidium contains 8 photoreceptors
(R1-R8) with different spectral sensitivities and downstream targets
(R1-R6 -> lamina, achromatic, motion pathway; R7-R8 -> medulla directly,
implicated in color/UV vision) (Hardie 1985; Ready, Hanson & Benzer 1976).
Fly photoreceptors are also known to have high temporal bandwidth and
adapt their gain to ambient light (Juusola & Hardie 2001) - i.e. they are
tuned for detecting fast motion, not for spatial acuity.

ASSUMPTIONS made here (project-specific, not biological claims):
  - A regular rectangular/hex-ish sampling grid stands in for the true
    hexagonal ommatidial lattice, at a resolution chosen for performance
    (see simulation/config.py), not measured acuity.
  - A conventional RGB video frame stands in for the fly's natural visual
    world. Real Drosophila photoreceptors are most sensitive in the UV/blue/
    green range and are NOT sensitive to red the way human cones are
    (Salcedo et al. 1999); we do not attempt to re-map RGB video into a
    fly-realistic spectral response - we simply use luminance for the
    achromatic (R1-R6-like) channel and a simple chroma proxy for the
    R7/R8-like channel, both clearly approximate.
  - Photoreceptor adaptation is modeled as a simple exponential moving
    average of local luminance (a standard, generic gain-control
    simplification), not Juusola & Hardie's biophysical adaptation model.
"""

from __future__ import annotations

import cv2
import numpy as np


class RetinaModel:
    def __init__(self, grid_shape: tuple[int, int], adapt_rate: float = 0.05):
        self.rows, self.cols = grid_shape
        self.adapt_rate = adapt_rate
        self._baseline = None  # running local-luminance baseline for adaptation

    def _sample_grid(self, frame_gray: np.ndarray) -> np.ndarray:
        """Resize the frame to the ommatidial sampling grid. Using area
        interpolation approximates each 'ommatidium' pooling light over a
        small patch, which is at least qualitatively similar to a real
        photoreceptor's finite acceptance angle (a real biological fact),
        even though the exact optics are not modeled."""
        resized = cv2.resize(
            frame_gray, (self.cols, self.rows), interpolation=cv2.INTER_AREA
        )
        return resized.astype(np.float32) / 255.0

    def process(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Returns a dict with:
            'r1_6' : (rows*cols,) achromatic, adapted, motion-relevant channel
            'r7_8' : (rows*cols,) simple chroma-proxy channel
            'raw_luminance' : (rows, cols) for visualization
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lum = self._sample_grid(gray)

        if self._baseline is None:
            self._baseline = lum.copy()
        else:
            self._baseline = (
                1 - self.adapt_rate
            ) * self._baseline + self.adapt_rate * lum

        # local-contrast signal relative to adapted baseline - a crude stand-in
        # for real photoreceptor light/dark adaptation (Laughlin 1989 discusses
        # contrast-coding in fly photoreceptors; the specific formula below is
        # our own simplification, not his model).
        contrast = np.clip((lum - self._baseline) * 4.0 + 0.5, 0.0, 1.0)

        b, g, r = cv2.split(frame_bgr.astype(np.float32) / 255.0)
        # crude chroma proxy: emphasize blue/UV-ish and green channels since
        # real fly photoreceptors are UV/blue/green sensitive and not
        # red-sensitive (Salcedo et al. 1999); still a coarse ASSUMPTION.
        chroma = 0.6 * b + 0.4 * g - 0.2 * r
        chroma_grid = self._sample_grid(
            cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        )

        return {
            "r1_6": contrast.flatten(),
            "r7_8": chroma_grid.flatten(),
            "raw_luminance": lum,
        }
