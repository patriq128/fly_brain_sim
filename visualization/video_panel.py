"""
visualization/video_panel.py
==============================

Displays the currently-playing video frame (plus a small overlay of the
downsampled "retina" luminance grid, so you can see what the model is
actually reacting to) in its own OpenCV window, positioned next to the
PyVista 3D window. Kept intentionally simple/decoupled: this module knows
nothing about the brain simulation beyond the arrays it's handed.
"""

from __future__ import annotations

import cv2
import numpy as np

from visualization.emotion_chart import CHART_HEIGHT, render_emotion_chart


class VideoPanel:
    def __init__(self, window_name: str = "Fly Visual Input", width: int = 480):
        self.window_name = window_name
        self.width = width
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Video + retina grid + footer + emotion chart; extra room so the
        # window does not have to scale the stacked image down.
        cv2.resizeWindow(
            self.window_name, width, int(width * 0.75) + 160 + 70 + CHART_HEIGHT + 20
        )
        cv2.moveWindow(self.window_name, 20, 20)

    def show(
        self,
        frame_bgr: np.ndarray,
        retina_grid: np.ndarray | None,
        overlay_text: str,
        *,
        emotion_mix: dict[str, float] | None = None,
        emotion_frames: int = 0,
        current_mood: str | None = None,
    ):
        h, w = frame_bgr.shape[:2]
        scale = self.width / w
        disp = cv2.resize(frame_bgr, (self.width, int(h * scale)))

        if retina_grid is not None:
            grid_img = (np.clip(retina_grid, 0, 1) * 255).astype(np.uint8)
            grid_img = cv2.resize(
                grid_img, (self.width, self.width // 3), interpolation=cv2.INTER_NEAREST
            )
            grid_img = cv2.applyColorMap(grid_img, cv2.COLORMAP_INFERNO)
            cv2.putText(
                grid_img,
                "Simulated photoreceptor sampling (R1-R6-like channel)",
                (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            disp = np.vstack([disp, grid_img])

        footer = np.zeros((70, self.width, 3), dtype=np.uint8)
        for i, line in enumerate(overlay_text.split("\n")[:4]):
            cv2.putText(
                footer,
                line,
                (6, 16 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 230, 200),
                1,
                cv2.LINE_AA,
            )
        disp = np.vstack([disp, footer])
        if emotion_mix is not None:
            disp = np.vstack(
                [
                    disp,
                    render_emotion_chart(
                        self.width,
                        emotion_mix,
                        frames=emotion_frames,
                        current_mood=current_mood,
                    ),
                ]
            )

        cv2.imshow(self.window_name, disp)

    def poll_key(self) -> int:
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)
