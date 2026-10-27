"""
simulation/engine.py
=====================

Orchestrates: video frame source -> FlyBrain.step() -> stats, at a
controlled rate, exposing play/pause/seek/restart. This is the piece the
UI/visualization layer drives; it has no drawing code in it so it can be
reused headlessly (e.g. in tests, or a future web backend).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from brain.network import FlyBrain, StepStats
from hearing.audio_source import AudioTrack
from simulation.behavior_log import BehaviorLog
from simulation.config import SimulationConfig
from vision.frame_source import VideoFrameSource


@dataclass
class EngineFrameResult:
    frame_bgr: np.ndarray | None
    stats: StepStats | None
    video_progress: float
    video_time_s: float
    ok: bool
    audio_window: np.ndarray | None = None
    # True on the tick where the clip reached its end and looped.
    reached_video_end: bool = False


class SimulationEngine:
    def __init__(self, config: SimulationConfig, video_path: str):
        self.config = config
        self.brain = FlyBrain(config)
        self.video = VideoFrameSource(video_path, target_fps=config.video_target_fps)
        # One snapshot per displayed video frame, used by the end-of-video
        # behavior summary. See simulation/behavior_log.py. Nothing is logged
        # when the summary is disabled, and logging stops once a summary has
        # been written (so a looping session does not grow without bound).
        self.history = BehaviorLog()
        self._log_history = bool(config.summary_graph)
        self._last_logged_frame_idx = -1
        # ASSUMPTION: the video soundtrack is the simulated fly's acoustic
        # world (see hearing/audio_source.py). None means silent/vision-only.
        self.audio = (
            AudioTrack.from_video(video_path, sample_rate=config.audio_sample_rate)
            if config.hearing_enabled
            else None
        )
        if config.hearing_enabled and self.audio is None:
            print(
                "[hearing] No soundtrack could be decoded from "
                f"{video_path!r} (video has no audio, or ffmpeg/ffprobe is "
                "missing). Auditory input will be silent; use --no-hearing "
                "to suppress this pathway entirely.",
                file=sys.stderr,
            )

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
        self.history.clear()
        self._log_history = bool(self.config.summary_graph)
        self._last_logged_frame_idx = -1

    def stop_history_logging(self):
        """Freeze and clear the per-frame log (after a summary is written)."""
        self._log_history = False
        self.history.clear()

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

        reached_end = bool(self.video.looped)
        video_time_s = (
            self.video.frame_idx / self.video.native_fps if self.video.native_fps else 0.0
        )
        audio_window = None
        if self.audio is not None:
            # Use a slightly longer analysis window than one video frame so
            # the FFT filterbank has enough frequency resolution (and so
            # high-frame-rate videos still yield stable band estimates).
            window_s = max(1.0 / (self.video.native_fps or 30.0), 0.075)
            audio_window = self.audio.window(video_time_s, window_s)

        stats = self.brain.step(
            frame,
            sub_steps=self.config.steps_per_video_frame,
            audio_window=audio_window,
        )

        # Log only frames from the pass that just finished: the first frame
        # of the new loop must not be appended to the previous pass's summary,
        # and paused/held frames must not inflate the history.
        if (
            self._log_history
            and not reached_end
            and self.video.frame_idx != self._last_logged_frame_idx
        ):
            self.history.add(video_time_s, stats)
            self._last_logged_frame_idx = self.video.frame_idx

        return EngineFrameResult(
            frame,
            stats,
            self.video.progress,
            video_time_s,
            True,
            audio_window,
            reached_end,
        )

    def close(self):
        self.video.release()
