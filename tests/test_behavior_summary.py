"""Tests for the per-frame behavior log and end-of-video summary graph."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from brain.network import FlyBrain
from simulation.behavior_log import BehaviorLog
from simulation.config import SimLevel, SimulationConfig
from simulation.engine import SimulationEngine
from visualization.behavior_summary import write_behavior_summary


def test_behavior_log_records_step_stats():
    config = SimulationConfig.for_level(SimLevel.DEMO)
    brain = FlyBrain(config)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    log = BehaviorLog()
    for i in range(3):
        stats = brain.step(frame, sub_steps=1)
        log.add(i / 15.0, stats)

    assert len(log) == 3
    sample = log.samples[-1]
    assert sample.video_time_s == pytest.approx(2 / 15.0)
    assert 0.0 <= sample.active_fraction <= 1.0
    assert sample.mood
    assert "antenna" in sample.region_activity
    assert set(sample.audio_band_levels) == {"low", "mid", "high"}


def test_engine_detects_video_end_and_logs_frames(tmp_path):
    path = str(tmp_path / "tiny.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32)
    )
    for i in range(10):
        frame = np.full((32, 32, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    config = SimulationConfig.for_level(SimLevel.DEMO)
    config.hearing_enabled = False  # no soundtrack in this fixture
    config.video_target_fps = 1000.0
    config.steps_per_video_frame = 1
    engine = SimulationEngine(config, path)

    reached = False
    for _ in range(60):
        result = engine.tick()
        if not result.ok:
            break
        if result.reached_video_end:
            reached = True
            break

    assert reached
    # One snapshot per frame of the pass that just ended (the restarted first
    # frame is deliberately not appended).
    assert 5 <= len(engine.history) <= 10
    assert engine.history.samples[0].video_time_s > 0.0
    engine.close()


def test_summary_png_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    config = SimulationConfig.for_level(SimLevel.DEMO)
    brain = FlyBrain(config)
    rng = np.random.default_rng(0)
    log = BehaviorLog()
    for i in range(30):
        frame = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        stats = brain.step(frame, sub_steps=1)
        log.add(i / 15.0, stats)

    path = write_behavior_summary(
        log,
        video_path="clip.mp4",
        level="demo",
        source="synthetic",
        n_neurons=brain.graph.n_neurons,
        output_dir=str(tmp_path),
    )
    assert path is not None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[0] > 500 and image.shape[1] > 500


def test_summary_skips_empty_log(tmp_path):
    assert (
        write_behavior_summary(
            BehaviorLog(),
            video_path="clip.mp4",
            level="demo",
            source="synthetic",
            n_neurons=10,
            output_dir=str(tmp_path),
        )
        is None
    )
