"""
simulation/behavior_log.py
==========================

Records one lightweight snapshot per displayed video frame so the app can
summarize "what the fly was doing" when the video reaches its end.

This is deliberately just a log of the values already shown in the UI
(`StepStats`): active fraction, per-region activity, the illustrative
fly-state axes, motion energy and auditory band levels. It adds no new
modeling and makes no new biological claim. The end-of-video chart built
from it (`visualization/behavior_summary.py`) is a descriptive summary of
the simulation, not a reconstruction of real fly behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorSample:
    video_time_s: float
    sim_time_s: float
    active_fraction: float
    active_neuron_count: int
    region_activity: dict[str, float]
    mood: str
    arousal: float
    salience: float
    escape: float
    curiosity: float
    valence: float
    motion_energy: float
    audio_band_levels: dict[str, float]
    audio_rms: float


class BehaviorLog:
    """Append-only per-frame history; cleared on restart."""

    def __init__(self):
        self._samples: list[BehaviorSample] = []

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list[BehaviorSample]:
        return self._samples

    def clear(self) -> None:
        self._samples.clear()

    def add(self, video_time_s: float, stats) -> None:
        state = stats.fly_state
        self._samples.append(
            BehaviorSample(
                video_time_s=float(video_time_s),
                sim_time_s=float(stats.sim_time_s),
                active_fraction=float(stats.active_fraction),
                active_neuron_count=int(stats.active_neuron_count),
                region_activity=dict(stats.region_activity),
                mood=state.mood if state is not None else "Unknown",
                arousal=float(state.arousal) if state is not None else 0.0,
                salience=float(state.salience) if state is not None else 0.0,
                escape=float(state.escape) if state is not None else 0.0,
                curiosity=float(state.curiosity) if state is not None else 0.0,
                valence=float(state.valence) if state is not None else 0.0,
                motion_energy=float(stats.motion_energy_mean),
                audio_band_levels=dict(stats.audio_band_levels),
                audio_rms=float(stats.audio_rms),
            )
        )
