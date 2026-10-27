"""
visualization/video_panel.py
==============================

Displays the currently-playing video frame, a small overlay of the
downsampled "retina" luminance grid, and an audio scope/band-meter strip
showing what the auditory model is hearing, in its own OpenCV window,
positioned next to the PyVista 3D window. Kept intentionally simple/
decoupled: this module knows nothing about the brain simulation beyond the
arrays it's handed.
"""

from __future__ import annotations

import cv2
import numpy as np

from visualization.emotion_chart import CHART_HEIGHT, render_emotion_chart

AUDIO_STRIP_HEIGHT = 58


def _render_audio_strip(
    width: int,
    waveform: np.ndarray | None,
    band_levels: dict[str, float] | None,
    hearing_enabled: bool,
) -> np.ndarray:
    """Small scope for the soundtrack currently entering the auditory model."""
    strip = np.zeros((AUDIO_STRIP_HEIGHT, width, 3), dtype=np.uint8)
    cv2.putText(
        strip,
        "Audio input -> Johnston's organ (hearing)",
        (6, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    plot_left, plot_right = 6, width - 6
    plot_top, plot_bottom = 16, 38
    center = (plot_top + plot_bottom) // 2
    if not hearing_enabled:
        cv2.putText(
            strip,
            "Hearing disabled (--no-hearing)",
            (plot_left, center + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    elif waveform is None:
        cv2.putText(
            strip,
            "No audio decoded (no track, or ffmpeg/ffprobe missing)",
            (plot_left, center + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        n = max(1, plot_right - plot_left)
        if samples.size:
            idx = np.linspace(0, samples.size - 1, n).astype(np.int64)
            amp = np.clip(np.abs(samples[idx]), 0.0, 1.0)
            amp_px = (amp * (plot_bottom - plot_top) * 0.5).astype(np.int32)
            xs = np.arange(plot_left, plot_left + n, dtype=np.int32)
            upper = np.stack([xs, center - amp_px], axis=1)
            lower = np.stack([xs, center + amp_px], axis=1)
            cv2.polylines(strip, [upper], False, (120, 255, 180), 1, cv2.LINE_AA)
            cv2.polylines(strip, [lower], False, (120, 255, 180), 1, cv2.LINE_AA)
            cv2.line(strip, (plot_left, center), (plot_left + n, center), (40, 90, 60), 1)

    # Three band meters, matching the receptor model's bands.
    x = 6
    for label, key in (("LOW", "low"), ("MID", "mid"), ("HIGH", "high")):
        value = 0.0
        if band_levels:
            value = float(min(1.0, max(0.0, band_levels.get(key, 0.0))))
        cv2.putText(
            strip,
            label,
            (x, AUDIO_STRIP_HEIGHT - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (200, 220, 200),
            1,
            cv2.LINE_AA,
        )
        bar_x = x + 30
        cv2.rectangle(
            strip,
            (bar_x, AUDIO_STRIP_HEIGHT - 13),
            (bar_x + 70, AUDIO_STRIP_HEIGHT - 5),
            (45, 45, 45),
            -1,
        )
        filled = int(70 * value)
        if filled > 0:
            cv2.rectangle(
                strip,
                (bar_x, AUDIO_STRIP_HEIGHT - 13),
                (bar_x + filled, AUDIO_STRIP_HEIGHT - 5),
                (120, 200, 255),
                -1,
            )
        x = bar_x + 80

    return strip


class VideoPanel:
    def __init__(self, window_name: str = "Fly Sensory Input", width: int = 480):
        self.window_name = window_name
        self.width = width
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Video + retina grid + audio scope + footer + emotion chart; extra
        # room so the window does not have to scale the stacked image down.
        cv2.resizeWindow(
            self.window_name,
            width,
            int(width * 0.75)
            + 160
            + AUDIO_STRIP_HEIGHT
            + 70
            + CHART_HEIGHT
            + 20,
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
        audio_window: np.ndarray | None = None,
        audio_band_levels: dict[str, float] | None = None,
        hearing_enabled: bool = True,
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

        disp = np.vstack(
            [
                disp,
                _render_audio_strip(
                    self.width, audio_window, audio_band_levels, hearing_enabled
                ),
            ]
        )

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
