"""
ui/app.py
=========

Wires together the SimulationEngine, the PyVista 3D brain view, and the
OpenCV video panel into one interactive application, and owns the main
loop and keyboard controls.

Controls
--------
    space        play / pause
    r            restart from the beginning
    left / right seek backward / forward ~1 second
    q / Esc      quit (works from either window)

Both windows are driven from the same loop tick so the video timeline and
the neural simulation stay in lockstep (point 7 in the project spec):
video frame N is always the frame whose resulting visual-pathway activity
is currently displayed in the 3D view.
"""

from __future__ import annotations

import time

from simulation.config import SimulationConfig
from simulation.engine import SimulationEngine
from visualization.brain3d import Brain3DView
from visualization.stats_panel import format_stats
from visualization.video_panel import VideoPanel


class FlyBrainApp:
    def __init__(self, config: SimulationConfig, video_path: str, show_video_panel: bool = True):
        self.engine = SimulationEngine(config, video_path)
        self.brain3d = Brain3DView(self.engine.brain.graph)
        self.video_panel = VideoPanel() if show_video_panel else None
        self._bind_keys()
        self._fps_ema = 0.0
        self._last_t = time.perf_counter()
        self._quit = False

    def _bind_keys(self):
        p = self.brain3d.plotter
        p.add_key_event("space", self.engine.toggle_pause)
        p.add_key_event("r", self.engine.restart)
        p.add_key_event("Left", lambda: self.engine.seek(-int(self.engine.video.native_fps or 15)))
        p.add_key_event("Right", lambda: self.engine.seek(int(self.engine.video.native_fps or 15)))
        p.add_key_event("q", self._request_quit)

    def _request_quit(self):
        self._quit = True

    def _update_fps(self):
        now = time.perf_counter()
        dt = max(now - self._last_t, 1e-6)
        self._last_t = now
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst_fps
        return self._fps_ema

    def run(self):
        graph = self.engine.brain.graph
        n_edges = int(graph.adjacency.nnz)
        n_neurons = graph.n_neurons

        self.brain3d.plotter.show(auto_close=False, interactive_update=True)

        try:
            while not self._quit:
                result = self.engine.tick()
                fps = self._update_fps()

                if result.ok:
                    stats_text = format_stats(
                        result.stats,
                        fps=fps,
                        video_time_s=result.video_time_s,
                        video_progress=result.video_progress,
                        paused=self.engine.is_paused,
                        level=self.engine.config.level.value,
                        n_neurons=n_neurons,
                        n_edges=n_edges,
                        source=graph.source,
                    )
                    mood_text = ""
                    if result.stats.fly_state is not None:
                        mood_text = f'Simulated fly mood: {result.stats.fly_state.mood}'
                    self.brain3d.update(
                        self.engine.brain.dynamics.activity,
                        stats_text,
                        mood_text=mood_text,
                    )

                    if self.video_panel is not None:
                        current_mood = (
                            result.stats.fly_state.mood
                            if result.stats.fly_state is not None
                            else None
                        )
                        self.video_panel.show(
                            result.frame_bgr,
                            self.engine.brain._last_luminance_grid,
                            stats_text,
                            emotion_mix=result.stats.emotion_mix,
                            emotion_frames=result.stats.emotion_mix_frames,
                            current_mood=current_mood,
                        )
                        key = self.video_panel.poll_key()
                        if key in (ord("q"), 27):
                            break
                        elif key == ord(" "):
                            self.engine.toggle_pause()
                        elif key == ord("r"):
                            self.engine.restart()
                        elif key == 81:  # left arrow (platform dependent)
                            self.engine.seek(-int(self.engine.video.native_fps or 15))
                        elif key == 83:  # right arrow
                            self.engine.seek(int(self.engine.video.native_fps or 15))

                self.brain3d.render()

                if getattr(self.brain3d.plotter, "_closed", False):
                    break
        finally:
            self.close()

    def close(self):
        try:
            self.brain3d.plotter.close()
        except Exception:
            pass
        if self.video_panel is not None:
            try:
                self.video_panel.close()
            except Exception:
                pass
        self.engine.close()
