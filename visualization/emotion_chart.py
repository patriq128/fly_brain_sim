"""
visualization/emotion_chart.py
================================

Draws the rolling "emotion mix" as a small colored horizontal bar chart for
the OpenCV video panel.  The mix itself is computed in brain/fly_state.py
(frequency of the illustrative mood labels over recent frames); this module
only renders it, so it has no opinion about what the percentages mean.

Kept dependency-free beyond numpy/OpenCV so it can also be reused by a future
web/logging UI.
"""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

from brain.fly_state import EMOTION_ORDER

# Display colors (BGR, as OpenCV expects).  Chosen for contrast on a dark
# background and to stay distinguishable without relying on hue alone
# (each bar is also labeled with its name and percentage).
EMOTION_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Calm": (165, 194, 102),
    "Settled": (203, 160, 141),
    "Curious": (41, 194, 230),
    "Alert": (43, 142, 242),
    "Wary": (161, 122, 176),
    "Startled": (89, 87, 225),
    "Overwhelmed": (138, 39, 214),
}

ROW_HEIGHT = 18
HEADER_HEIGHT = 20
CHART_HEIGHT = HEADER_HEIGHT + ROW_HEIGHT * len(EMOTION_ORDER) + 8


def render_emotion_chart(
    width: int,
    emotion_mix: Mapping[str, float],
    *,
    frames: int,
    current_mood: str | None = None,
) -> np.ndarray:
    """Return a (CHART_HEIGHT, width, 3) BGR image of the emotion-mix bars."""
    canvas = np.full((CHART_HEIGHT, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"Emotion mix (last {frames} frame{'s' if frames != 1 else ''})",
        (8, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )

    label_width = 86
    percent_width = 52
    bar_x = label_width
    bar_width = max(40, width - label_width - percent_width - 18)

    for i, mood in enumerate(EMOTION_ORDER):
        y = HEADER_HEIGHT + i * ROW_HEIGHT
        fraction = float(min(1.0, max(0.0, emotion_mix.get(mood, 0.0))))
        color = EMOTION_COLORS_BGR.get(mood, (200, 200, 200))
        is_current = mood == current_mood
        text_color = (255, 255, 255) if is_current else (190, 190, 190)

        # Track, then filled portion.
        cv2.rectangle(
            canvas,
            (bar_x, y + 4),
            (bar_x + bar_width, y + ROW_HEIGHT - 5),
            (52, 52, 52),
            -1,
        )
        filled = int(round(fraction * bar_width))
        if filled > 0:
            cv2.rectangle(
                canvas,
                (bar_x, y + 4),
                (bar_x + filled, y + ROW_HEIGHT - 5),
                color,
                -1,
            )
        if is_current:
            cv2.rectangle(
                canvas,
                (bar_x - 1, y + 3),
                (bar_x + bar_width + 1, y + ROW_HEIGHT - 4),
                (255, 255, 255),
                1,
            )

        cv2.putText(
            canvas,
            mood,
            (8, y + ROW_HEIGHT - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{fraction * 100.0:5.1f}%",
            (bar_x + bar_width + 8, y + ROW_HEIGHT - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas
