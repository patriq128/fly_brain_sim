"""
visualization/stats_panel.py
==============================

Formats the StepStats produced by brain/network.py into a compact text
block for overlay in the 3D view (and could equally be piped to a log file,
a dashboard, or a websocket for a future web UI - kept decoupled from any
particular rendering technology on purpose).
"""

from __future__ import annotations

from brain.network import StepStats
from brain.regions import REGIONS

REGION_ORDER_DISPLAY = [
    "retina",
    "lamina",
    "medulla",
    "lobula",
    "lobula_plate",
    "central_complex",
    "mushroom_body",
]


def format_stats(
    stats: StepStats,
    fps: float,
    video_time_s: float,
    video_progress: float,
    paused: bool,
    level: str,
    n_neurons: int,
    n_edges: int,
) -> str:
    lines = []
    lines.append(f"Level: {level}   Neurons: {n_neurons:,}   Synapses: {n_edges:,}")
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
    lines.append("Activity by region:")
    for r in REGION_ORDER_DISPLAY:
        if r not in stats.region_activity:
            continue
        name = REGIONS[r].display_name
        lines.append(
            f"  {name:<24s} act={stats.region_activity[r]:.3f}  "
            f"active_n={stats.region_active_count.get(r, 0)}"
        )
    if stats.strongest_connections:
        lines.append("Strongest active connections (src -> dst : weight):")
        for src, dst, w in stats.strongest_connections[:5]:
            lines.append(f"  {src} -> {dst} : {w:+.2f}")
    lines.append("*illustrative rescaling, not a measured firing rate - see docs/SIMULATION_NOTES.md")
    return "\n".join(lines)
