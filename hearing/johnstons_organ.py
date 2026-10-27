"""
hearing/johnstons_organ.py
==========================

Turns a window of audio samples into simplified auditory-receptor activity.

BIOLOGY (real): Drosophila hear with Johnston's organ, a chordotonal organ
in the antenna. Its mechanosensory JO neurons are the auditory receptor
neurons, and they project to the antennal mechanosensory and motor center
(AMMC). JO neurons fall into groups/subtypes (JO-A..JO-F and many
subtypes), and different groups are tuned to different components of
antennal motion; the male courtship song's ~200 Hz pulse component is the
classic behaviorally relevant sound (Kamikouchi et al. 2006, 2009;
Albert & Goepfert 2015). Fly auditory receptors adapt to sustained sound
and are especially sensitive to transients.

SIMULATION (this project's implementation):
  - A three-band FFT filterbank (low / mid / high) estimates the amplitude
    of the audio in fixed frequency ranges from a Hann-windowed short-time
    window. This is a generic spectral decomposition, not a model of the
    biomechanics of the antenna.
  - Each band has an exponential moving-average baseline (adaptation); the
    receptor response is a transient/contrast component relative to that
    baseline plus a small sustained component.
  - An "onset" signal is the rectified frame-to-frame increase in broadband
    RMS, standing in for the strong transient sensitivity of JO neurons.

ASSUMPTION (project-specific, not a biological claim):
  - The three bands are assigned to the fly's hearing range only loosely:
    they span ~80-700 Hz, which contains Drosophila's behaviorally relevant
    song frequencies but is not a measured tuning curve.
  - All JO neurons in the real-connectome path receive the same weighted
    combination of bands; we do NOT claim to reproduce the measured
    frequency tuning of individual JO subtypes. Splitting bands across
    JO-A..JO-F would require measured per-subtype tuning data.
  - The gains, time constants and normalization below are chosen so the
    audio visibly drives the network; they are not fitted to any recording.
"""

from __future__ import annotations

import numpy as np

# ASSUMPTION: coarse bands spanning the range that contains Drosophila
# courtship-song frequencies (pulse song ~200 Hz, sine song ~160 Hz) plus
# neighboring low/high components. Not measured JO tuning curves.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "low": (80.0, 180.0),
    "mid": (180.0, 350.0),
    "high": (350.0, 700.0),
}


class AuditoryReceptorModel:
    """Simplified JO-neuron response to successive audio windows."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        bands: dict[str, tuple[float, float]] | None = None,
        *,
        adapt_rate: float = 0.15,
        transient_gain: float = 4.0,
        sustained_gain: float = 0.35,
    ):
        self.sample_rate = int(sample_rate)
        self.bands = dict(bands or DEFAULT_BANDS)
        self.adapt_rate = float(adapt_rate)
        self.transient_gain = float(transient_gain)
        self.sustained_gain = float(sustained_gain)
        self._baseline = np.zeros(len(self.bands), dtype=np.float32)
        self._prev_rms = 0.0

    def reset(self) -> None:
        self._baseline[:] = 0.0
        self._prev_rms = 0.0

    def _band_amplitudes(self, samples: np.ndarray) -> np.ndarray:
        n = samples.size
        window = np.hanning(n).astype(np.float32)
        centred = (samples - float(samples.mean())) * window
        spectrum = np.fft.rfft(centred)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        magnitude = np.abs(spectrum)
        wsum = float(window.sum())
        amplitudes = np.zeros(len(self.bands), dtype=np.float32)
        for i, (lo, hi) in enumerate(self.bands.values()):
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                continue
            # Approximate peak amplitude in the band from windowed FFT bins.
            amplitudes[i] = 2.0 * float(np.sqrt(np.sum(magnitude[mask] ** 2))) / max(
                wsum, 1e-9
            )
        return amplitudes

    def process(self, samples: np.ndarray) -> dict[str, float]:
        """Return receptor responses for one audio window.

        Keys: one entry per band (``low``/``mid``/``high`` by default), plus
        ``rms`` (broadband level), ``onset`` (transient) and ``total`` (a
        courtship-song-weighted summary, mid band emphasized).
        """
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size < 8 or self.sample_rate <= 0:
            self._baseline *= 1.0 - self.adapt_rate
            self._prev_rms *= 0.5
            out = {name: 0.0 for name in self.bands}
            out.update(rms=0.0, onset=0.0, total=0.0)
            return out

        amplitudes = np.clip(self._band_amplitudes(samples), 0.0, 4.0)
        responses = np.clip(
            (amplitudes - self._baseline) * self.transient_gain
            + amplitudes * self.sustained_gain,
            0.0,
            1.0,
        )
        self._baseline = (
            1.0 - self.adapt_rate
        ) * self._baseline + self.adapt_rate * amplitudes

        rms = float(np.sqrt(np.mean(samples**2)))
        onset = float(np.clip((rms - self._prev_rms) * 8.0, 0.0, 1.0))
        self._prev_rms = rms

        out = {
            name: float(value) for name, value in zip(self.bands, responses)
        }
        low = out.get("low", 0.0)
        mid = out.get("mid", 0.0)
        high = out.get("high", 0.0)
        out["rms"] = float(min(rms, 1.0))
        out["onset"] = onset
        out["total"] = float(np.clip(0.25 * low + 0.50 * mid + 0.25 * high, 0.0, 1.0))
        return out
