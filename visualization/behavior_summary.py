"""
visualization/behavior_summary.py
==================================

Builds the end-of-video "what was the fly doing?" summary figure from a
`simulation.behavior_log.BehaviorLog` and writes it to a PNG.

The chart is a descriptive view of values the simulation already reports:
  - a time-aligned mood strip (illustrative UI labels, not real emotions),
  - a heatmap of mean activity per brain region over time,
  - the illustrative fly-state axes over time,
  - the share of monitored frames spent in each mood,
  - a small "at a glance" statistics block.

Everything here is SIMULATION/ASSUMPTION display: the log records the
generic leaky-rate network's activity and the project's playful mood
projection (`brain/fly_state.py`), not measured fly behavior.

Uses matplotlib's non-interactive Agg backend so it never interferes with
the PyVista/OpenCV windows; the caller displays the resulting PNG.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

# Render off-screen; the caller displays the PNG in an OpenCV window, so the
# summary never fights the PyVista/OpenCV event loops for a GUI backend.
matplotlib.use("Agg")

from brain.fly_state import EMOTION_ORDER
from brain.regions import OTHER_REGION, REGION_ORDER, REGIONS
from simulation.behavior_log import BehaviorLog

# --- Reference palette (validated default; see the project's dataviz notes) --
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
MOOD_COLORS = {mood: SERIES[i] for i, mood in enumerate(EMOTION_ORDER)}
BEHAVIOR_AXES = (
    ("arousal", "Arousal", SERIES[0]),
    ("salience", "Salience", SERIES[1]),
    ("escape", "Escape / fear", SERIES[2]),
    ("curiosity", "Curiosity", SERIES[3]),
)
SHORT_REGION_NAMES = {
    "antenna": "Hearing (JO)",
    "retina": "Retina",
    "lamina": "Lamina",
    "medulla": "Medulla",
    "lobula": "Lobula",
    "lobula_plate": "Lobula plate",
    "central_complex": "Central complex",
    "mushroom_body": "Mushroom body",
    "other": "Other",
}


def _sequential_blue():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "fly_seq_blue",
        ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    )


def _ink_on(color: str) -> str:
    """Black or white label ink, whichever contrasts more with a fill."""
    hexcolor = color.lstrip("#")
    rgb = [int(hexcolor[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    # Contrast against white vs black; choose the better one.
    white_contrast = 1.05 / (luminance + 0.05)
    black_contrast = (luminance + 0.05) / 0.05
    return "#ffffff" if white_contrast >= black_contrast else INK


def _bin_samples(samples, n_bins: int):
    """Bin samples onto a time axis; returns centres, edges, indices."""
    times = np.array([s.video_time_s for s in samples], dtype=np.float64)
    t_start = float(times.min())
    t_end = float(times.max())
    if t_end <= t_start:
        t_end = t_start + 1.0
    edges = np.linspace(t_start, t_end, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    indices = np.clip(np.digitize(times, edges) - 1, 0, n_bins - 1)
    return times, centres, edges, indices


def _region_matrix(samples, indices: list[int], n_bins: int):
    keys = [r for r in REGION_ORDER]
    if any(OTHER_REGION in s.region_activity for s in samples):
        keys.append(OTHER_REGION)
    matrix = np.zeros((len(keys), n_bins), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int32)
    for sample, b in zip(samples, indices):
        for i, key in enumerate(keys):
            matrix[i, b] += float(sample.region_activity.get(key, 0.0))
        counts[b] += 1
    nonempty = counts > 0
    matrix[:, nonempty] /= counts[nonempty]
    # Carry the last populated bin forward so gaps are not drawn as zero.
    for i in range(len(keys)):
        last = 0.0
        for b in range(n_bins):
            if counts[b] > 0:
                last = matrix[i, b]
            else:
                matrix[i, b] = last
    return keys, matrix


def _mood_indices(samples, indices: list[int], n_bins: int) -> np.ndarray:
    mood_rank = {mood: i for i, mood in enumerate(EMOTION_ORDER)}
    per_bin: list[Counter] = [Counter() for _ in range(n_bins)]
    for sample, b in zip(samples, indices):
        per_bin[b][sample.mood] += 1
    result = np.zeros(n_bins, dtype=np.int32)
    last = mood_rank.get("Calm", 0)
    for b, counter in enumerate(per_bin):
        if counter:
            last = mood_rank.get(counter.most_common(1)[0][0], 0)
        result[b] = last
    return result


def _build_figure(
    log: BehaviorLog,
    *,
    video_name: str,
    level: str,
    source: str,
    n_neurons: int,
):
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    samples = list(log.samples)
    n = len(samples)
    duration = float(max(s.video_time_s for s in samples))
    n_bins = int(min(240, max(10, n)))

    times, centres, edges, indices = _bin_samples(samples, n_bins)
    region_keys, region_matrix = _region_matrix(samples, indices, n_bins)
    mood_bins = _mood_indices(samples, indices, n_bins)

    mood_counts = Counter(s.mood for s in samples)
    mood_share = {mood: mood_counts.get(mood, 0) / n for mood in EMOTION_ORDER}
    dominant_mood, dominant_count = mood_counts.most_common(1)[0]
    peak_arousal = max(s.arousal for s in samples)
    peak_escape = max(s.escape for s in samples)
    mean_active = float(np.mean([s.active_fraction for s in samples]))
    mean_motion = float(np.mean([s.motion_energy for s in samples]))
    mean_audio = float(np.mean([s.audio_rms for s in samples]))
    hearing_active = any(sum(s.audio_band_levels.values()) > 0.0 for s in samples)

    fig = plt.figure(figsize=(14, 10.2), facecolor=SURFACE)
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.42, 3.0, 2.5, 1.0],
        width_ratios=[4.0, 1.15],
        left=0.115,
        right=0.975,
        top=0.90,
        bottom=0.075,
        hspace=0.42,
        wspace=0.06,
    )
    ax_mood = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_axes = fig.add_subplot(gs[2, 0])
    ax_share = fig.add_subplot(gs[3, 0])
    ax_stats = fig.add_subplot(gs[:, 1])

    t_start = float(edges[0])
    t_end = float(edges[-1])
    # Assumption: leave room at the right for direct series labels.
    x_right = t_end + 0.20 * max(t_end - t_start, 1e-6)

    source_tag = f"  [{source}]" if source and source != "synthetic" else ""
    fig.text(
        0.115,
        0.965,
        "What the simulated fly was doing",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.115,
        0.932,
        f"{video_name}  •  {duration:.1f}s of video  •  {level}{source_tag}  "
        f"•  {n_neurons:,} neurons  •  {n:,} frames logged",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK_SECONDARY,
    )

    # --- mood timeline -----------------------------------------------------
    mood_cmap = ListedColormap([MOOD_COLORS[m] for m in EMOTION_ORDER])
    ax_mood.pcolormesh(
        edges,
        [0.0, 1.0],
        mood_bins[None, :],
        cmap=mood_cmap,
        vmin=-0.5,
        vmax=len(EMOTION_ORDER) - 0.5,
        shading="flat",
    )
    ax_mood.set_xlim(t_start, x_right)
    ax_mood.set_ylim(0.0, 1.0)
    ax_mood.set_yticks([0.5])
    ax_mood.set_yticklabels(["Mood"], fontsize=8.5, color=INK_SECONDARY)
    ax_mood.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_mood.spines.values():
        spine.set_visible(False)

    # --- region activity heatmap ------------------------------------------
    short_names = [SHORT_REGION_NAMES.get(key, REGIONS[key].display_name) for key in region_keys]
    y_edges = np.arange(len(region_keys) + 1)
    mesh = ax_heat.pcolormesh(
        edges,
        y_edges,
        region_matrix[::-1],
        cmap=_sequential_blue(),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    ax_heat.set_yticks(np.arange(len(region_keys)) + 0.5)
    ax_heat.set_yticklabels(short_names[::-1], fontsize=8.5, color=INK_SECONDARY)
    ax_heat.set_xlim(t_start, x_right)
    ax_heat.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(mesh, ax=ax_heat, orientation="horizontal", fraction=0.045, pad=0.02)
    cbar.set_label("mean region activity (0-1)", fontsize=8, color=MUTED)
    cbar.ax.tick_params(labelsize=7.5, colors=MUTED)
    cbar.outline.set_edgecolor(BASELINE)

    # --- behavior axes -----------------------------------------------------
    order = np.argsort(times)
    sorted_t = times[order]
    ax_axes.set_facecolor(SURFACE)
    for key, label, color in BEHAVIOR_AXES:
        values = np.array([getattr(s, key) for s in samples], dtype=np.float64)[order]
        ax_axes.plot(sorted_t, values, color=color, linewidth=2.0, label=label)
    ax_axes.set_xlim(t_start, x_right)
    ax_axes.set_ylim(0.0, 1.02)
    ax_axes.set_xlabel("video time (seconds)", fontsize=9, color=INK_SECONDARY, labelpad=6)
    ax_axes.set_ylabel("illustrative axis (0-1)", fontsize=9, color=INK_SECONDARY)
    ax_axes.grid(axis="y", color=GRID, linewidth=0.8)
    ax_axes.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_axes.spines[spine].set_visible(False)
    ax_axes.spines["bottom"].set_color(BASELINE)
    ax_axes.tick_params(colors=MUTED, labelsize=8)
    ax_axes.legend(
        loc="upper left",
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        columnspacing=1.6,
    )

    # Direct labels at the line ends, nudged apart when values are close.
    last_idx = int(np.argmax(times))
    end_positions = []
    for key, label, color in BEHAVIOR_AXES:
        end_positions.append((float(getattr(samples[last_idx], key)), label))
    end_positions.sort(key=lambda item: item[0])
    min_gap = 0.065
    placed = []
    for value, label in end_positions:
        y = value
        if placed and y < placed[-1] + min_gap:
            y = placed[-1] + min_gap
        placed.append(min(y, 1.0))
    label_x = t_end + 0.025 * max(t_end - t_start, 1e-6)
    for (value, label), y in zip(end_positions, placed):
        ax_axes.text(
            label_x,
            y,
            label,
            fontsize=8,
            color=INK_SECONDARY,
            va="center",
            ha="left",
        )

    # --- mood share --------------------------------------------------------
    left = 0.0
    for mood in EMOTION_ORDER:
        share = mood_share[mood]
        if share <= 0.0:
            continue
        ax_share.barh(
            [0],
            [share],
            left=left,
            height=0.5,
            color=MOOD_COLORS[mood],
            edgecolor=SURFACE,
            linewidth=2.0,
        )
        if share >= 0.06:
            ax_share.text(
                left + share / 2.0,
                0.0,
                f"{share * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=_ink_on(MOOD_COLORS[mood]),
            )
        left += share
    ax_share.set_xlim(0.0, 1.0)
    ax_share.set_ylim(-0.55, 0.55)
    ax_share.set_yticks([])
    ax_share.set_xlabel("share of monitored frames", fontsize=9, color=INK_SECONDARY, labelpad=6)
    ax_share.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right", "left"):
        ax_share.spines[spine].set_visible(False)
    ax_share.spines["bottom"].set_color(BASELINE)

    # --- at-a-glance stats + mood key -------------------------------------
    ax_stats.axis("off")
    y = 0.99
    ax_stats.text(
        0.0,
        y,
        "AT A GLANCE",
        transform=ax_stats.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="top",
    )
    stats_lines = [
        ("Video time", f"{duration:.1f} s"),
        ("Frames logged", f"{n:,}"),
        ("Dominant mood", f"{dominant_mood} ({dominant_count / n * 100:.0f}%)"),
        ("Mean active neurons", f"{mean_active * 100:.1f}%"),
        ("Peak arousal", f"{peak_arousal:.2f}"),
        ("Peak escape / fear", f"{peak_escape:.2f}"),
        ("Mean motion energy", f"{mean_motion:.4f}"),
    ]
    if hearing_active:
        stats_lines.append(("Mean hearing level", f"{mean_audio:.2f}"))
    y -= 0.055
    for label, value in stats_lines:
        ax_stats.text(
            0.0,
            y,
            label,
            transform=ax_stats.transAxes,
            fontsize=8.5,
            color=INK_SECONDARY,
            va="top",
        )
        ax_stats.text(
            1.0,
            y,
            value,
            transform=ax_stats.transAxes,
            fontsize=8.5,
            color=INK,
            va="top",
            ha="right",
        )
        y -= 0.042

    y -= 0.035
    ax_stats.text(
        0.0,
        y,
        "MOOD KEY",
        transform=ax_stats.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="top",
    )
    y -= 0.045
    for mood in EMOTION_ORDER:
        ax_stats.add_patch(
            Rectangle(
                (0.0, y - 0.018),
                0.075,
                0.026,
                transform=ax_stats.transAxes,
                facecolor=MOOD_COLORS[mood],
                edgecolor="none",
            )
        )
        ax_stats.text(
            0.10,
            y - 0.005,
            f"{mood}  {mood_share[mood] * 100:.0f}%",
            transform=ax_stats.transAxes,
            fontsize=8,
            color=INK_SECONDARY,
            va="center",
        )
        y -= 0.037

    ax_stats.text(
        0.0,
        0.015,
        "Illustrative summary of simulated\n"
        "activity. Mood labels are a playful\n"
        "UI projection - not a claim that the\n"
        "fly experiences anything.",
        transform=ax_stats.transAxes,
        fontsize=7.5,
        color=MUTED,
        va="bottom",
        linespacing=1.35,
    )

    return fig


def write_behavior_summary(
    log: BehaviorLog,
    *,
    video_path: str,
    level: str,
    source: str,
    n_neurons: int,
    output_dir: str | None = None,
) -> str | None:
    """Render and save the summary PNG; returns its path (or None if empty)."""
    if len(log) < 2:
        return None

    from matplotlib import pyplot as plt

    video_name = os.path.basename(str(video_path)) or "video"
    fig = _build_figure(
        log,
        video_name=video_name,
        level=level,
        source=source,
        n_neurons=n_neurons,
    )

    out_dir = Path(output_dir) if output_dir else Path("data") / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_name).stem) or "video"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stem}_behavior_summary_{timestamp}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(path)
