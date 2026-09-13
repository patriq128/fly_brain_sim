"""
data/sample/make_sample_video.py
==================================

Generates a small synthetic .mp4 with moving shapes (a bright disk sweeping
left-right, a bar sweeping up-down, and drifting noise) so the whole
pipeline can be exercised end-to-end without requiring the user to supply
footage first. This is a convenience/testing tool only - it is not a
substitute for real visual stimuli and is clearly synthetic.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def make_sample_video(
    out_path: str = None, width: int = 320, height: int = 240, fps: int = 30, seconds: int = 8
) -> str:
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(__file__))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "synthetic_demo.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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
    return out_path


if __name__ == "__main__":
    path = make_sample_video()
    print(f"Wrote {path}")
