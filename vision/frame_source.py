"""
vision/frame_source.py
=======================

Thin wrapper around OpenCV's VideoCapture providing the transport controls
the project requires (play/pause/seek/restart) and frame-rate throttling so
the visual pathway is fed frames at a controlled, configurable rate
regardless of the source video's native frame rate. ASSUMPTION: feeding a
pre-recorded video file into a simulated fly visual system is a project
convenience for offline/reproducible experimentation - real flies see a
continuous, self-motion-coupled visual stream, not a fixed video file.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class VideoFrameSource:
    def __init__(self, path: str, target_fps: float = 15.0):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.target_fps = target_fps
        self.paused = False
        self._frame_idx = 0

    @property
    def duration_seconds(self) -> float:
        if self.native_fps <= 0:
            return 0.0
        return self.frame_count / self.native_fps

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def progress(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return self._frame_idx / self.frame_count

    def toggle_pause(self):
        self.paused = not self.paused

    def restart(self):
        self._frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek_relative(self, delta_frames: int):
        target = int(np.clip(self._frame_idx + delta_frames, 0, max(self.frame_count - 1, 0)))
        self._frame_idx = target
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Returns (ok, frame_bgr). If paused, re-returns the last frame
        without advancing (ok=True) so downstream visualization can keep
        rendering a frozen frame while the sim is paused."""
        if self.paused:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - 1, 0))
            ok, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            return ok, frame

        ok, frame = self.cap.read()
        if ok:
            self._frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            # loop by default so the demo keeps running
            self.restart()
            ok, frame = self.cap.read()
            if ok:
                self._frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return ok, frame

    def release(self):
        self.cap.release()


class FrameRateLimiter:
    """Simple wall-clock throttle so we process at `target_fps` regardless
    of how fast the simulation loop could otherwise spin."""

    def __init__(self, target_fps: float):
        self.period = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last = time.perf_counter()

    def wait(self):
        if self.period <= 0:
            return
        now = time.perf_counter()
        elapsed = now - self._last
        remaining = self.period - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.perf_counter()
