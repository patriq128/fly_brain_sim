"""
visualization/brain3d.py
==========================

Interactive 3D visualization of the simulated brain using PyVista.

Shows:
  - every neuron as a point, colored by region (dim) and by current
    activity (bright highlight when active) - toggled by a scalar array
    that is updated every frame.
  - a subsampled set of synaptic connections as translucent lines, colored
    by sign (excitatory / inhibitory), so the picture reads as a wiring
    diagram rather than a point cloud.
  - a curved "signal propagation" pulse effect: recently-activated edges
    are briefly brightened, giving a visual sense of activity flowing from
    retina -> lamina -> medulla -> lobula/lobula plate -> central brain.
  - a text overlay with the live statistics block (see stats_panel.py for
    the text formatting).
  - region outlines/labels so the different brain regions are identifiable
    (a scientific-diagram look, not photorealism - this is a SCHEMATIC
    layout, see brain/regions.py for why the exact coordinates are not a
    biological claim).

Controls (bound via PyVista key events):
    space  - play / pause
    r      - restart
    left   - seek backward
    right  - seek forward
    q      - quit
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from brain.connectome import ConnectomeGraph
from brain.regions import REGION_ORDER, REGIONS

MAX_EDGES_DRAWN = 6000


class Brain3DView:
    def __init__(self, graph: ConnectomeGraph, plotter: pv.Plotter | None = None):
        self.graph = graph
        self.plotter = plotter or pv.Plotter(window_size=(900, 800))
        self.plotter.set_background("#0b0d17")

        self._build_points()
        self._build_edges()
        self._build_region_labels()
        self._set_default_camera()

        self.stats_actor = self.plotter.add_text(
            "", position="upper_left", font_size=10, color="white", name="stats_text"
        )
        self.title_actor = self.plotter.add_text(
            "Simulated Drosophila Brain - Visual Pathway Activity",
            position="upper_right",
            font_size=10,
            color="#aad9ff",
            name="title_text",
        )

    # -- construction ---------------------------------------------------
    def _build_points(self):
        pts = self.graph.positions
        self.point_cloud = pv.PolyData(pts)

        base_colors = np.zeros((self.graph.n_neurons, 3), dtype=np.float32)
        for r in REGION_ORDER:
            sl = self.graph.region_slices.get(r)
            if sl is None:
                continue
            hexcolor = REGIONS[r].color.lstrip("#")
            rgb = np.array([int(hexcolor[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
            base_colors[sl] = rgb
        # neurons without a known region (real-data 'other' bucket)
        unknown = np.array([r not in REGION_ORDER for r in self.graph.region_of])
        base_colors[unknown] = np.array([0.5, 0.5, 0.5])

        self._base_colors = base_colors
        self.point_cloud["activity"] = np.zeros(self.graph.n_neurons, dtype=np.float32)
        self.point_cloud["rgb"] = (base_colors * 255).astype(np.uint8)

        self.points_actor = self.plotter.add_mesh(
            self.point_cloud,
            scalars="rgb",
            rgb=True,
            point_size=6.0,
            render_points_as_spheres=True,
            name="neurons",
        )

    def _build_edges(self):
        adj = self.graph.adjacency.tocoo()
        n_edges = adj.nnz
        if n_edges == 0:
            self.edge_mesh = None
            return
        if n_edges > MAX_EDGES_DRAWN:
            rng = np.random.default_rng(1)
            keep = rng.choice(n_edges, size=MAX_EDGES_DRAWN, replace=False)
        else:
            keep = np.arange(n_edges)

        src = adj.row[keep]
        dst = adj.col[keep]
        weights = adj.data[keep]

        pts = self.graph.positions
        line_points = np.empty((len(keep) * 2, 3), dtype=np.float32)
        line_points[0::2] = pts[src]
        line_points[1::2] = pts[dst]
        lines = np.hstack(
            [np.full((len(keep), 1), 2), np.arange(len(keep) * 2).reshape(-1, 2)]
        ).astype(np.int64)

        self.edge_mesh = pv.PolyData(line_points, lines=lines)
        colors = np.where(weights[:, None] > 0, np.array([0.3, 0.6, 1.0]), np.array([1.0, 0.35, 0.35]))
        edge_colors = np.repeat(colors, 1, axis=0)
        self.edge_mesh["rgb"] = (edge_colors * 255).astype(np.uint8)
        self.plotter.add_mesh(
            self.edge_mesh,
            scalars="rgb",
            rgb=True,
            opacity=0.08,
            line_width=1.0,
            name="synapses",
        )
        self._edge_src = src
        self._edge_dst = dst

    def _build_region_labels(self):
        for r in REGION_ORDER:
            sl = self.graph.region_slices.get(r)
            if sl is None or sl.stop <= sl.start:
                continue
            info = REGIONS[r]
            for anchor in (info.anchor_left, info.anchor_right):
                self.plotter.add_point_labels(
                    [anchor],
                    [info.display_name],
                    font_size=9,
                    text_color="#dddddd",
                    shape=None,
                    always_visible=True,
                    name=f"label_{r}_{anchor[0]}",
                )

    def _set_default_camera(self):
        # Look down the anterior-posterior-ish axis so the two bilateral
        # hemispheres (laid out along x) are visible side by side, with a
        # slight elevation, which reads much more like a "brain diagram"
        # than a face-on view of the thin z-depth we use for the schematic
        # layout.
        self.plotter.camera_position = "xy"
        self.plotter.camera.azimuth = 25
        self.plotter.camera.elevation = 20
        self.plotter.camera.zoom(1.3)

    # -- per-frame update -------------------------------------------------
    def update(self, activity: np.ndarray, stats_text: str):
        highlight = np.clip(activity, 0, 1)[:, None]
        bright = self._base_colors * (0.25 + 0.75 * (1 - highlight)) + np.array(
            [1.0, 0.85, 0.3]
        ) * highlight
        self.point_cloud["rgb"] = (np.clip(bright, 0, 1) * 255).astype(np.uint8)
        self.point_cloud["activity"] = activity

        if self.edge_mesh is not None:
            edge_activity = np.minimum(activity[self._edge_src], activity[self._edge_dst])
            base = self.edge_mesh["rgb"].astype(np.float32) / 255.0
            base_reduced = base.copy()
            flash = np.clip(edge_activity, 0, 1)[:, None]
            flashed = base_reduced * (1 - flash) + np.array([1.0, 1.0, 0.6]) * flash
            self.edge_mesh["rgb"] = (np.clip(flashed, 0, 1) * 255).astype(np.uint8)

        self.plotter.add_text(
            stats_text, position="upper_left", font_size=10, color="white", name="stats_text"
        )

    def render(self):
        self.plotter.update()
