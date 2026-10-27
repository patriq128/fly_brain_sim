"""Tests for the video-soundtrack -> Johnston's-organ auditory pathway."""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from brain.network import FlyBrain
from hearing.audio_source import AudioTrack
from hearing.johnstons_organ import AuditoryReceptorModel
from simulation.config import SimLevel, SimulationConfig


def _tone(freq_hz: float, seconds: float = 0.2, sample_rate: int = 16_000, amp: float = 0.5):
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_audio_track_window_is_time_indexed():
    track = AudioTrack(np.arange(100, dtype=np.float32), sample_rate=100)
    window = track.window(0.5, 0.2)
    assert window.shape == (20,)
    assert np.array_equal(window, np.arange(30, 50, dtype=np.float32))


def test_audio_track_pads_start_and_wraps_end():
    track = AudioTrack(np.arange(100, dtype=np.float32), sample_rate=100)
    # Before the track starts: zero pre-roll, then real samples.
    start = track.window(0.05, 0.2)
    assert np.all(start[:15] == 0.0)
    assert np.array_equal(start[15:], np.arange(0, 5, dtype=np.float32))
    # Past the end: wraps to the beginning, matching the video loop.
    wrapped = track.window(1.1, 0.2)
    expected = np.concatenate([np.arange(90, 100), np.arange(0, 10)]).astype(np.float32)
    assert np.array_equal(wrapped, expected)


def test_audio_track_empty_is_silent():
    track = AudioTrack(np.empty(0, dtype=np.float32), sample_rate=16_000)
    assert track.duration_s == 0.0
    assert np.all(track.window(1.0, 0.1) == 0.0)


def test_receptor_model_is_band_selective():
    model = AuditoryReceptorModel()
    low = model.process(_tone(130.0))
    assert low["low"] > low["mid"] and low["low"] > low["high"]

    model.reset()
    mid = model.process(_tone(260.0))
    assert mid["mid"] > mid["low"] and mid["mid"] > mid["high"]

    model.reset()
    high = model.process(_tone(520.0))
    assert high["high"] > high["low"] and high["high"] > high["mid"]


def test_receptor_model_adapts_to_sustained_sound_and_resets():
    model = AuditoryReceptorModel()
    tone = _tone(220.0)
    first = model.process(tone)["total"]
    for _ in range(12):
        last = model.process(tone)["total"]
    assert first > last
    model.reset()
    assert model.process(tone)["total"] == pytest.approx(first)


def test_receptor_model_reports_onset_after_silence():
    model = AuditoryReceptorModel()
    silence = np.zeros(1600, dtype=np.float32)
    silence_out = model.process(silence)
    assert silence_out["onset"] == 0.0
    onset = model.process(_tone(220.0))
    assert onset["onset"] > 0.0
    assert onset["rms"] > 0.0


def test_demo_brain_injects_audio_into_antenna_region():
    config = SimulationConfig.for_level(SimLevel.DEMO)
    brain = FlyBrain(config)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    stats = brain.step(frame, sub_steps=1, audio_window=_tone(220.0))

    antenna = brain.graph.region_slices["antenna"]
    assert np.any(brain._external_input[antenna] > 0.0)
    assert stats.hearing_active
    assert stats.audio_band_levels["mid"] > 0.0
    assert stats.audio_rms > 0.0


def test_disabling_hearing_leaves_auditory_region_silent():
    config = SimulationConfig.for_level(SimLevel.DEMO)
    config.hearing_enabled = False
    brain = FlyBrain(config)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    stats = brain.step(frame, sub_steps=1, audio_window=_tone(220.0))

    antenna = brain.graph.region_slices["antenna"]
    assert not np.any(brain._external_input[antenna] > 0.0)
    assert not stats.hearing_active
    assert all(value == 0.0 for value in stats.audio_band_levels.values())
    assert stats.audio_rms == 0.0


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)
def test_from_video_decodes_sample_soundtrack(tmp_path):
    from data.sample.make_sample_video import make_sample_video

    video = make_sample_video(
        out_path=str(tmp_path / "with_audio.mp4"), seconds=1, with_audio=True
    )
    track = AudioTrack.from_video(video)
    assert track is not None
    assert track.duration_s == pytest.approx(1.0, abs=0.15)
    assert float(np.sqrt(np.mean(track.samples**2))) > 0.01


def test_from_video_without_audio_returns_none(tmp_path):
    path = str(tmp_path / "silent.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32)
    )
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert AudioTrack.from_video(path) is None
