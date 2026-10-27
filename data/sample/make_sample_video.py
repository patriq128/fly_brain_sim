"""
data/sample/make_sample_video.py
==================================

Generates a small synthetic .mp4 with moving shapes (a bright disk sweeping
left-right, a bar sweeping up-down, and drifting noise) plus an optional
synthetic soundtrack (a ~220 Hz pulse train reminiscent of the courtship
song, plus a slow sweep and low noise) so the whole visual *and auditory*
pipeline can be exercised end-to-end without requiring the user to supply
footage first. This is a convenience/testing tool only - it is not a
substitute for real stimuli and is clearly synthetic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave

import cv2
import numpy as np


def _write_soundtrack(path: str, seconds: float, sample_rate: int = 44_100) -> None:
    """Write a mono 16-bit WAV with courtship-song-like pulses."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    rng = np.random.default_rng(1)

    # ~3.5 Hz train of short 220 Hz pulses (rough stand-in for pulse song).
    pulse_env = (0.5 * (1.0 + np.sin(2.0 * np.pi * 3.5 * t - np.pi / 2.0))) ** 4
    carrier = np.sin(2.0 * np.pi * 220.0 * t)
    # Slower frequency sweep so the low/high bands also move.
    sweep_freq = 140.0 + 55.0 * np.sin(2.0 * np.pi * 0.2 * t)
    sweep = np.sin(2.0 * np.pi * np.cumsum(sweep_freq) / sample_rate)
    noise = rng.normal(0.0, 0.01, t.size)

    signal = 0.45 * carrier * pulse_env + 0.25 * sweep + noise
    fade = max(1, int(0.05 * sample_rate))
    signal[:fade] *= np.linspace(0.0, 1.0, fade)
    signal[-fade:] *= np.linspace(1.0, 0.0, fade)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")

    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _mux_audio(video_path: str, wav_path: str, out_path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-i",
            wav_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            out_path,
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    return proc.returncode == 0 and os.path.exists(out_path)


def make_sample_video(
    out_path: str = None,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
    seconds: int = 8,
    with_audio: bool = True,
) -> str:
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(__file__))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "synthetic_demo.mp4")
    out_path = os.path.abspath(out_path)

    # OpenCV cannot mux audio, so write the video to a temporary file and
    # let ffmpeg add the synthesized soundtrack. If ffmpeg is unavailable,
    # fall back to a silent (vision-only) sample.
    add_audio = bool(with_audio and shutil.which("ffmpeg"))
    video_path = out_path + ".video_tmp.mp4" if add_audio else out_path
    wav_path = out_path + ".audio_tmp.wav"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    n_frames = fps * seconds
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (10, 8, 6)

        # horizontally sweeping bright disk (drives horizontal motion detectors)
        cx = int((0.5 + 0.4 * np.sin(t * 0.8)) * width)
        cy = int(height * 0.35)
        cv2.circle(frame, (cx, cy), 18, (60, 200, 255), -1)

        # vertically sweeping bar (drives vertical motion detectors)
        by = int((0.5 + 0.4 * np.sin(t * 1.3 + 1.0)) * height)
        cv2.rectangle(frame, (0, by - 6), (width, by + 6), (255, 140, 60), -1)

        # a looming (expanding) circle every ~4 seconds, bottom right
        loom_phase = (t % 4.0) / 4.0
        radius = int(4 + 40 * loom_phase)
        cv2.circle(frame, (int(width * 0.8), int(height * 0.75)), radius, (120, 255, 120), -1)

        noise = rng.normal(0, 6, size=(height, width, 3))
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()

    if add_audio:
        try:
            _write_soundtrack(wav_path, float(seconds))
            if not _mux_audio(video_path, wav_path, out_path):
                os.replace(video_path, out_path)
        finally:
            for tmp in (video_path, wav_path):
                if os.path.exists(tmp) and os.path.abspath(tmp) != out_path:
                    os.remove(tmp)

    return out_path


if __name__ == "__main__":
    path = make_sample_video()
    print(f"Wrote {path}")
