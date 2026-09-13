"""
simulation/engine.py
=====================

Orchestrates: video frame source -> FlyBrain.step() -> stats, at a
controlled rate, exposing play/pause/seek/restart. This is the piece the
UI/visualization layer drives; it has no drawing code in it so it can be
reused headlessly (e.g. in tests, or a future web backend).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.network import FlyBrain, StepStats
from simulation.config import SimulationConfig
from vision.frame_source import VideoFrameSource


@dataclass
class EngineFrameResult:
    frame_bgr: np.ndarray | None
    stats: StepStats | None
    video_progress: float
    video_time_s: float
    ok: bool


class SimulationEngine:
    def __init__(self, config: SimulationConfig, video_path: str):
        self.config = config
        self.brain = FlyBrain(config)
        self.video = VideoFrameSource(video_path, target_fps=config.video_target_fps)

    # -- transport controls -------------------------------------------------
    def play(self):
        self.video.paused = False

    def pause(self):
        self.video.paused = True

    def toggle_pause(self):
        self.video.toggle_pause()

    def restart(self):
        self.video.restart()
        self.brain.reset()

    def seek(self, delta_frames: int):
        self.video.seek_relative(delta_frames)

    @property
    def is_paused(self) -> bool:
        return self.video.paused

    # -- main tick ------------------------------------------------------------
    def tick(self) -> EngineFrameResult:
        ok, frame = self.video.read()
        if not ok or frame is None:
            return EngineFrameResult(None, None, self.video.progress, 0.0, False)

        stats = self.brain.step(frame, sub_steps=self.config.steps_per_video_frame)
        video_time_s = (
            self.video.frame_idx / self.video.native_fps if self.video.native_fps else 0.0
        )
        return EngineFrameResult(frame, stats, self.video.progress, video_time_s, True)

    def close(self):
        self.video.release()
