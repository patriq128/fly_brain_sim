"""
hearing/audio_source.py
=======================

Decodes the audio track of the input video so it can drive the simulated
auditory pathway, and exposes it as a time-indexed mono waveform.

BIOLOGY (real): adult Drosophila hear through Johnston's organ in the
antennae; the relevant natural stimuli are near-field sounds/vibrations,
including the ~200 Hz pulse component of the male courtship song
(Kamikouchi et al. 2006, 2009; Albert & Goepfert 2015).

ASSUMPTION: using the audio track of a normal video file as the fly's
acoustic world is a project convenience, exactly as using video frames as
its visual world is. We do not model the physics of near-field sound,
antennal mechanics, or the fly's frequency-dependent sensitivity. We only
require that the soundtrack is time-aligned with the video timeline.

Implementation notes:
  - ffmpeg/ffprobe are used only as decoders (they are not needed to run a
    vision-only simulation, and `--no-hearing` bypasses this module).
  - The whole track is decoded to a mono float32 array at a fixed sample
    rate. For typical clips this is small (16 kHz mono float32 = 64 KB/s).
  - `AudioTrack.window(t, duration)` returns the samples immediately
    preceding video time `t`, so the auditory input stays in lockstep with
    the displayed frame, including after pause/seek/restart (the engine
    derives time from the frame index; no audio clock is needed).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
_FFMPEG_TIMEOUT_S = 300.0
_FFPROBE_TIMEOUT_S = 30.0


def _which(binary: str) -> str | None:
    return shutil.which(binary)


def has_audio_stream(path: str) -> bool:
    """True when ffprobe finds at least one audio stream in `path`."""
    ffprobe = _which("ffprobe")
    if ffprobe is None:
        return False
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            timeout=_FFPROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    return b"audio" in proc.stdout.lower()


class AudioTrack:
    """A decoded mono audio track addressable by video time."""

    def __init__(self, samples: np.ndarray, sample_rate: int, source: str = ""):
        self.samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.sample_rate = int(sample_rate)
        self.source = source

    @classmethod
    def from_video(
        cls, path: str, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> "AudioTrack | None":
        """Decode the first audio stream of `path`, or return None.

        Returns None (rather than raising) when the file has no audio track,
        ffmpeg is not installed, or decoding fails, so hearing input degrades
        gracefully to silence.
        """
        ffmpeg = _which("ffmpeg")
        if ffmpeg is None or not has_audio_stream(path):
            return None
        try:
            proc = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(path),
                    "-vn",                 # no video
                    "-ac", "1",            # mono
                    "-ar", str(int(sample_rate)),
                    "-f", "f32le",         # raw little-endian float32
                    "pipe:1",
                ],
                capture_output=True,
                timeout=_FFMPEG_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout:
            return None
        samples = np.frombuffer(proc.stdout, dtype="<f4").copy()
        if samples.size == 0:
            return None
        return cls(samples, sample_rate, source=str(path))

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.samples.size / float(self.sample_rate)

    def window(self, time_s: float, duration_s: float) -> np.ndarray:
        """Samples in the interval ending at `time_s`, length ~duration_s.

        Samples before the start of the track are zero (the simulation starts
        from silence) and samples past the end wrap around, matching the
        video's loop-by-default transport. Returns an empty array for an
        empty track.
        """
        n = max(1, int(round(float(duration_s) * self.sample_rate)))
        out = np.zeros(n, dtype=np.float32)
        if self.samples.size == 0 or self.sample_rate <= 0:
            return out

        end = int(round(float(time_s) * self.sample_rate))
        start = end - n
        src_start = max(start, 0)
        src_end = min(end, self.samples.size)
        if src_end <= src_start:
            return out

        dst_offset = src_start - start
        if end <= self.samples.size:
            out[dst_offset : dst_offset + (src_end - src_start)] = self.samples[
                src_start:src_end
            ]
        else:
            # Window crosses the end of the track: wrap to the beginning.
            first = self.samples[src_start:]
            out[dst_offset : dst_offset + first.size] = first
            rest = n - dst_offset - first.size
            if rest > 0:
                out[dst_offset + first.size :] = np.resize(self.samples, rest)
        return out
