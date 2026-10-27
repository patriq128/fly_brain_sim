"""
visualization/summary_panel.py
==============================

Shows the saved end-of-video behavior summary PNG in its own OpenCV window.
Kept separate from `video_panel.py` so the summary can be generated once and
left on screen while the simulation is paused.
"""

from __future__ import annotations

import cv2


class SummaryPanel:
    def __init__(self, window_name: str = "Fly behavior summary"):
        self.window_name = window_name
        self.visible = False
        self._resized = False

    def show(self, image_path: str) -> bool:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if not self._resized:
            h, w = image.shape[:2]
            scale = min(1.0, 1180.0 / max(w, 1), 820.0 / max(h, 1))
            cv2.resizeWindow(
                self.window_name, max(320, int(w * scale)), max(240, int(h * scale))
            )
            self._resized = True
        cv2.imshow(self.window_name, image)
        self.visible = True
        return True

    def poll_key(self) -> int:
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self.visible:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        self.visible = False
