"""
visualization/stats_panel.py
==============================

Formats the StepStats produced by brain/network.py into a compact text
block for overlay in the 3D view (and could equally be piped to a log file,
a dashboard, or a websocket for a future web UI - kept decoupled from any
particular rendering technology on purpose).

The "Simulated feeling" block is formatted here but computed in
brain/fly_state.py. It is an illustrative projection of the activity axes,
not a claim that the simulation experiences anything.
"""

from __future__ import annotations

from brain.fly_state import EMOTION_ORDER
from brain.network import StepStats
from brain.regions import REGIONS

REGION_ORDER_DISPLAY = [
    "antenna",
    "retina",
    "lamina",
    "medulla",
    "lobula",
    "lobula_plate",
    "central_complex",
    "mushroom_body",
]


def _bar(value: float, width: int = 8) -> str:
    """ASCII meter for a 0..1 value (caller maps signed axes to 0..1 first)."""
    filled = int(round(min(1.0, max(0.0, float(value))) * width))
    return "#" * filled + "-" * (width - filled)


def format_stats(
    stats: StepStats,
    fps: float,
    video_time_s: float,
    video_progress: float,
    paused: bool,
    level: str,
    n_neurons: int,
    n_edges: int,
    source: str = "",
) -> str:
    lines = []
    source_tag = f" [{source}]" if source and source != "synthetic" else ""
    lines.append(
        f"Level: {level}{source_tag}   Neurons: {n_neurons:,}   Synapses: {n_edges:,}"
    )
    state = "PAUSED" if paused else "PLAYING"
    lines.append(
        f"[{state}]  video t={video_time_s:6.2f}s ({video_progress*100:5.1f}%)   "
        f"sim t={stats.sim_time_s:7.2f}s"
    )
    lines.append(f"App FPS: {fps:5.1f}   frame process: {stats.frame_process_ms:6.2f} ms")
    lines.append(
        f"Active neurons: {stats.active_neuron_count:,} "
        f"({stats.active_fraction*100:4.1f}%)   mean rate ~{stats.mean_firing_rate_hz:5.1f} Hz*"
    )
    lines.append(f"Motion energy (mean): {stats.motion_energy_mean:.4f}")
    if stats.hearing_active:
        bands = stats.audio_band_levels
        lines.append(
            "Hearing (Johnston's organ): "
            f"low {_bar(bands.get('low', 0.0))} "
            f"mid {_bar(bands.get('mid', 0.0))} "
            f"high {_bar(bands.get('high', 0.0))}"
        )
        lines.append(
            f"  audio rms {stats.audio_rms:.3f}   onset {stats.audio_onset:.3f}"
        )
    else:
        lines.append("Hearing: disabled (--no-hearing)")
    if stats.fly_state is not None:
        fs = stats.fly_state
        lines.append(
            'Simulated "feeling" (ILLUSTRATIVE - not a real experience):'
        )
        lines.append(f"  mood: {fs.mood} - {fs.note}")
        lines.append(
            f"  arousal {_bar(fs.arousal)} {fs.arousal:.2f}   "
            f"vision    {_bar(fs.salience)} {fs.salience:.2f}"
        )
        lines.append(
            f"  fear    {_bar(fs.escape)} {fs.escape:.2f}   "
            f"curiosity {_bar(fs.curiosity)} {fs.curiosity:.2f}"
        )
        lines.append(
            f"  valence {_bar((fs.valence + 1.0) / 2.0)} {fs.valence:+.2f}"
        )
    if stats.emotion_mix:
        frame_word = "frame" if stats.emotion_mix_frames == 1 else "frames"
        lines.append(
            f"Emotion mix over last {stats.emotion_mix_frames} {frame_word} "
            "(share of frames):"
        )
        for mood in EMOTION_ORDER:
            fraction = stats.emotion_mix.get(mood, 0.0)
            lines.append(
                f"  {mood:<11s} {_bar(fraction, 18)} {fraction * 100.0:5.1f}%"
            )
    lines.append("Activity by region:")
    display_order = list(REGION_ORDER_DISPLAY)
    if "other" in stats.region_activity:
        display_order.append("other")
    for r in display_order:
        if r not in stats.region_activity:
            continue
        name = REGIONS[r].display_name
        lines.append(
            f"  {name:<24s} act={stats.region_activity[r]:.3f}  "
            f"active_n={stats.region_active_count.get(r, 0)}"
        )
    if stats.strongest_connections:
        lines.append("Strongest active connections (src -> dst : weight):")
        for i, (src, dst, w) in enumerate(stats.strongest_connections[:5]):
            label = (
                stats.strongest_connection_labels[i]
                if i < len(stats.strongest_connection_labels)
                else f"{src} -> {dst}"
            )
            lines.append(f"  {label} : {w:+.2f}")
    lines.append("*illustrative rescaling, not a measured firing rate - see docs/SIMULATION_NOTES.md")
    return "\n".join(lines)
