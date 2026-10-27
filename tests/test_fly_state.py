"""Tests for the illustrative activity-derived "fly state" readout."""

from __future__ import annotations

import numpy as np
import pytest

from brain.fly_state import (
    EMOTION_ORDER,
    EmotionTracker,
    FlyState,
    compute_fly_state,
)
from brain.network import FlyBrain
from simulation.config import SimLevel, SimulationConfig
from visualization.emotion_chart import CHART_HEIGHT, render_emotion_chart
from visualization.stats_panel import format_stats


def _activity(**overrides):
    values = {
        "antenna": 0.0,
        "retina": 0.0,
        "lamina": 0.0,
        "medulla": 0.0,
        "lobula": 0.0,
        "lobula_plate": 0.0,
        "central_complex": 0.0,
        "mushroom_body": 0.0,
    }
    values.update(overrides)
    return values


def test_quiet_network_is_calm():
    state = compute_fly_state(
        _activity(), active_fraction=0.0, motion_energy_mean=0.0
    )
    assert state.mood == "Calm"
    assert state.arousal == 0.0
    assert state.salience == 0.0
    assert state.escape == 0.0
    assert state.curiosity == 0.0
    assert state.valence == 0.0


def test_motion_and_activity_produce_startle():
    state = compute_fly_state(
        _activity(lobula_plate=0.7, retina=0.6, medulla=0.4),
        active_fraction=0.6,
        motion_energy_mean=0.2,
    )
    assert state.mood == "Startled"
    assert state.escape >= 0.7
    assert -1.0 <= state.valence <= 1.0


def test_valence_is_approach_minus_avoidance():
    state = compute_fly_state(
        _activity(central_complex=0.5, mushroom_body=0.5),
        active_fraction=0.2,
        motion_energy_mean=0.0,
    )
    assert state.valence == pytest.approx(state.curiosity - state.escape)


def test_hearing_raises_arousal_and_escape():
    quiet = compute_fly_state(
        _activity(), active_fraction=0.1, motion_energy_mean=0.0
    )
    loud = compute_fly_state(
        _activity(),
        active_fraction=0.1,
        motion_energy_mean=0.0,
        audio_level=0.9,
        audio_onset=0.8,
    )
    assert loud.arousal > quiet.arousal
    assert loud.salience > quiet.salience
    assert loud.escape > quiet.escape


def test_all_axes_stay_in_range():
    for active_fraction in (0.0, 0.5, 1.0):
        for motion in (0.0, 0.05, 0.5):
            for value in (0.0, 0.5, 1.0):
                state = compute_fly_state(
                    _activity(retina=value, medulla=value, lobula_plate=value,
                              central_complex=value, mushroom_body=value),
                    active_fraction=active_fraction,
                    motion_energy_mean=motion,
                )
                assert 0.0 <= state.arousal <= 1.0
                assert 0.0 <= state.salience <= 1.0
                assert 0.0 <= state.escape <= 1.0
                assert 0.0 <= state.curiosity <= 1.0
                assert -1.0 <= state.valence <= 1.0
                assert state.mood
                assert state.note


def test_step_stats_and_overlay_include_fly_state():
    brain = FlyBrain(SimulationConfig.for_level(SimLevel.DEMO))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    stats = brain.step(frame, sub_steps=1)

    assert stats.fly_state is not None
    text = format_stats(
        stats,
        fps=30.0,
        video_time_s=0.0,
        video_progress=0.0,
        paused=False,
        level="demo",
        n_neurons=brain.graph.n_neurons,
        n_edges=brain.graph.adjacency.nnz,
    )
    assert 'Simulated "feeling"' in text
    assert "ILLUSTRATIVE" in text
    assert stats.fly_state.mood in text
    assert "valence" in text


def _state(mood: str) -> FlyState:
    return FlyState(0.0, 0.0, 0.0, 0.0, 0.0, mood, "")


def test_emotion_tracker_is_a_rolling_histogram():
    tracker = EmotionTracker(window=3)
    tracker.update(_state("Calm"))
    mix = tracker.update(_state("Curious"))
    assert mix["Calm"] == pytest.approx(0.5)
    assert mix["Curious"] == pytest.approx(0.5)
    assert sum(mix.values()) == pytest.approx(1.0)

    mix = tracker.update(_state("Curious"))
    assert mix["Curious"] == pytest.approx(2.0 / 3.0)
    # The oldest sample (Calm) was in the window; no mood is "Calm" now?
    # It is still the first sample, so Calm remains until the window rolls.
    assert mix["Calm"] == pytest.approx(1.0 / 3.0)

    # Adding a fourth sample evicts "Calm" from the 3-frame window.
    mix = tracker.update(_state("Wary"))
    assert mix["Calm"] == 0.0
    assert mix["Curious"] == pytest.approx(2.0 / 3.0)
    assert mix["Wary"] == pytest.approx(1.0 / 3.0)
    assert set(mix) == set(EMOTION_ORDER)

    tracker.reset()
    assert tracker.sample_count == 0


def test_step_stats_expose_emotion_mix_and_overlay_graph():
    brain = FlyBrain(SimulationConfig.for_level(SimLevel.DEMO))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for _ in range(3):
        stats = brain.step(frame, sub_steps=1)

    assert stats.emotion_mix_frames == 3
    assert set(stats.emotion_mix) == set(EMOTION_ORDER)
    assert sum(stats.emotion_mix.values()) == pytest.approx(1.0)

    text = format_stats(
        stats,
        fps=30.0,
        video_time_s=0.0,
        video_progress=0.0,
        paused=False,
        level="demo",
        n_neurons=brain.graph.n_neurons,
        n_edges=brain.graph.adjacency.nnz,
    )
    assert "Emotion mix over last 3 frames" in text
    assert "%" in text

    # Restart clears the rolling window.
    brain.reset()
    stats = brain.step(frame, sub_steps=1)
    assert stats.emotion_mix_frames == 1


def test_emotion_chart_renders_all_labels():
    mix = {mood: 0.0 for mood in EMOTION_ORDER}
    mix["Curious"] = 0.75
    mix["Wary"] = 0.25
    image = render_emotion_chart(480, mix, frames=42, current_mood="Curious")
    assert image.shape == (CHART_HEIGHT, 480, 3)
    assert image.dtype == np.uint8
    # Some pixels are colored (the filled bars), not just the background.
    assert int(image.max()) > 100

