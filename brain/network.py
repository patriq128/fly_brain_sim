"""
brain/network.py
=================

Ties together:
  vision.retina.RetinaModel       -> photoreceptor responses
  brain.motion_pathway.ReichardtMotionDetector -> local motion signals
  brain.connectome.ConnectomeGraph -> wiring
  brain.neuron_model.LeakyRateNetwork -> dynamics

into a single `FlyBrain` object with a simple `step(frame_bgr) -> stats`
interface used by the simulation engine and the visualizers.

Data flow per video frame (this IS the real, documented Drosophila optic
pathway order; the specific numerical treatment at each stage is our
SIMULATION-level simplification - see the module docstrings referenced
below for the detailed biology/assumption breakdown of each stage):

    video frame
      -> RetinaModel (BIOLOGY: photoreceptors, ASSUMPTION: video as world)
      -> injected as external current into 'retina' neurons (SIMULATION)
      -> ReichardtMotionDetector on the lamina-stage grid (BIOLOGY-inspired
         T4/T5 motion detection, SIMULATION: HRC implementation)
      -> injected as extra external current into 'medulla' neurons
      -> full recurrent connectome propagation for N sub-steps (SIMULATION)
      -> downstream regions (lobula, lobula_plate, central_complex,
         mushroom_body) evolve from recurrent input alone, same as any
         other neuron in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.connectome import ConnectomeGraph, build_synthetic_connectome, load_real_connectome
from brain.motion_pathway import ReichardtMotionDetector
from brain.neuron_model import LeakyRateNetwork
from brain.regions import REGION_ORDER
from simulation.config import SimLevel, SimulationConfig
from vision.retina import RetinaModel


@dataclass
class StepStats:
    sim_time_s: float
    active_neuron_count: int
    active_fraction: float
    region_activity: dict[str, float]
    region_active_count: dict[str, int]
    mean_firing_rate_hz: float
    strongest_connections: list[tuple[int, int, float]]
    motion_energy_mean: float
    frame_process_ms: float


class FlyBrain:
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.level == SimLevel.CONNECTOME:
            if not config.connectome_path:
                raise ValueError("CONNECTOME level requires --connectome-path")
            self.graph: ConnectomeGraph = load_real_connectome(config.connectome_path, config)
        else:
            self.graph = build_synthetic_connectome(config)

        self.retina_model = RetinaModel(config.retina_grid_shape)
        self.motion_detector = ReichardtMotionDetector(config.retina_grid_shape)
        self.dynamics = LeakyRateNetwork(self.graph.adjacency, config.dynamics, seed=config.random_seed)

        self._sim_time_s = 0.0
        self._last_motion_energy = 0.0
        self._last_luminance_grid = None
        self._last_direction_maps = None
        self._external_input = np.zeros(self.graph.n_neurons, dtype=np.float32)

    def reset(self):
        self.dynamics.reset()
        self._sim_time_s = 0.0
        self.retina_model._baseline = None

    def _inject_vision(self, frame_bgr: np.ndarray, dt: float):
        vision = self.retina_model.process(frame_bgr)
        r1_6 = vision["r1_6"]
        r7_8 = vision["r7_8"]
        self._last_luminance_grid = vision["raw_luminance"]

        self._external_input[:] = 0.0

        retina_slice = self.graph.region_slices.get("retina")
        lamina_slice = self.graph.region_slices.get("lamina")
        medulla_slice = self.graph.region_slices.get("medulla")

        if retina_slice is not None and self.graph.source == "synthetic":
            n = retina_slice.stop - retina_slice.start
            cols = self.graph.column_of[retina_slice]
            # map each retina neuron to its column's photoreceptor sample,
            # cycling if oversubscribed (n_neurons can exceed n_columns).
            grid_flat = r1_6
            n_grid = grid_flat.shape[0]
            sample = grid_flat[cols % n_grid]
            self._external_input[retina_slice] = sample * 2.0

        rows, cols_ = self.config.retina_grid_shape
        lamina_grid_est = r1_6.reshape(rows, cols_)
        direction_maps = self.motion_detector.step(lamina_grid_est, dt)
        self._last_direction_maps = direction_maps
        self._last_motion_energy = float(direction_maps["motion_energy"].mean())

        if medulla_slice is not None and self.graph.source == "synthetic":
            n = medulla_slice.stop - medulla_slice.start
            cols_m = self.graph.column_of[medulla_slice]
            motion_flat = direction_maps["motion_energy"].flatten()
            n_grid = motion_flat.shape[0]
            sample = motion_flat[cols_m % n_grid]
            self._external_input[medulla_slice] += sample * 1.5

        if lamina_slice is not None and self.graph.source == "synthetic":
            cols_l = self.graph.column_of[lamina_slice]
            n_grid = r7_8.shape[0]
            sample = r7_8[cols_l % n_grid]
            self._external_input[lamina_slice] += sample * 0.5

        if self.graph.source != "synthetic":
            # CONNECTOME mode without column mapping available: inject the
            # mean visual drive uniformly into whatever neurons are tagged
            # 'retina' (or into all neurons if no region info was provided).
            target = retina_slice if retina_slice else slice(0, self.graph.n_neurons)
            self._external_input[target] = float(r1_6.mean()) * 2.0

    def step(self, frame_bgr: np.ndarray, sub_steps: int) -> StepStats:
        import time as _time

        t0 = _time.perf_counter()
        dt = self.dynamics._dt

        self._inject_vision(frame_bgr, dt)

        activity = None
        for _ in range(sub_steps):
            activity = self.dynamics.step(self._external_input)
            self._sim_time_s += dt

        stats = self._compute_stats(activity)
        stats.frame_process_ms = (_time.perf_counter() - t0) * 1000.0
        return stats

    def _compute_stats(self, activity: np.ndarray) -> StepStats:
        thresh = 0.5
        active_mask = activity > thresh
        region_activity = {}
        region_active_count = {}
        for r in REGION_ORDER:
            sl = self.graph.region_slices.get(r)
            if sl is None:
                continue
            region_activity[r] = float(activity[sl].mean()) if sl.stop > sl.start else 0.0
            region_active_count[r] = int(active_mask[sl].sum())

        strongest = self._strongest_active_connections(activity, top_k=5)

        return StepStats(
            sim_time_s=self._sim_time_s,
            active_neuron_count=int(active_mask.sum()),
            active_fraction=float(active_mask.mean()),
            region_activity=region_activity,
            region_active_count=region_active_count,
            mean_firing_rate_hz=float(self.dynamics.firing_rate_hz_estimate().mean()),
            strongest_connections=strongest,
            motion_energy_mean=self._last_motion_energy,
            frame_process_ms=0.0,
        )

    def _strongest_active_connections(self, activity: np.ndarray, top_k: int = 5):
        active_idx = np.where(activity > 0.5)[0]
        if len(active_idx) == 0:
            return []
        sub = self.graph.adjacency[active_idx][:, active_idx]
        if sub.nnz == 0:
            return []
        coo = sub.tocoo()
        weighted = np.abs(coo.data) * activity[active_idx[coo.row]] * activity[active_idx[coo.col]]
        if len(weighted) == 0:
            return []
        top = np.argsort(-weighted)[:top_k]
        results = []
        for t in top:
            src = int(active_idx[coo.row[t]])
            dst = int(active_idx[coo.col[t]])
            results.append((src, dst, float(coo.data[t])))
        return results
