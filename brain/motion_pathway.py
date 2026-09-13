"""
brain/motion_pathway.py
========================

Elementary motion detection feeding the medulla/lobula-plate stage.

BIOLOGY (real, well established): Drosophila detects the direction of
visual motion using T4 (ON-edge selective) and T5 (OFF-edge selective)
cells in the medulla/lobula, each of the four T4 and four T5 subtypes
tuned to one of the four cardinal directions. Their input logic closely
matches a "Hassenstein-Reichardt correlator" (HRC): two neighboring visual
inputs are combined with mismatched temporal filtering (one delayed
relative to the other) and multiplied together, so that motion in the
preferred direction produces a large product while the opposite direction
cancels. This correlator model long predates the connectome and has been
validated at the circuit level using connectomic and physiological data
(see e.g. Borst & Egelhaaf 1989 for the classical HRC theory, and
Fisher et al. 2015 / Haag, Arenz et al. 2016; Shinomiya et al. 2019 for the
identification of T4/T5's actual synaptic inputs in the connectome, which
matches HRC's required "delayed + non-delayed" input structure). The
Lappalainen et al. 2024 connectome-constrained model cited in README.md
reproduces T4/T5 direction selectivity from connectivity + optimization.

SIMULATION (this project's implementation): we implement a standard,
textbook 2-input Hassenstein-Reichardt correlator, vectorized over the
whole retinotopic grid using numpy array shifts (rather than the ~4 input
cell types and nonlinear synapses used by a real T4 dendrite). This is a
deliberate, clearly-labeled simplification: a single delay-line low-pass
filter and elementwise multiply per direction, applied on the lamina
output grid, per the four cardinal directions (up/down/left/right).
"""

from __future__ import annotations

import numpy as np


class ReichardtMotionDetector:
    """Vectorized 4-direction Hassenstein-Reichardt correlator array over a
    (rows, cols) retinotopic grid."""

    def __init__(self, grid_shape: tuple[int, int], tau_fast: float = 0.02, tau_delay: float = 0.08):
        self.rows, self.cols = grid_shape
        # exponential low-pass filter states for the "delayed" input branch
        self._delayed = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.alpha_delay = None  # set on first step() given dt
        self.tau_delay = tau_delay
        self._prev_input = np.zeros((self.rows, self.cols), dtype=np.float32)

    def _shift(self, arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
        return np.roll(np.roll(arr, dr, axis=0), dc, axis=1)

    def step(self, lamina_grid: np.ndarray, dt: float) -> dict[str, np.ndarray]:
        """`lamina_grid` is (rows, cols) contrast/activity input for this
        instant. Returns direction-selective activity maps for the four
        cardinal directions plus a combined motion-energy map."""
        if self.alpha_delay is None:
            self.alpha_delay = dt / (self.tau_delay + dt)

        # low-pass ("delayed") branch
        self._delayed = (
            1 - self.alpha_delay
        ) * self._delayed + self.alpha_delay * lamina_grid
        undelayed = lamina_grid

        directions = {
            "right": (0, 1),
            "left": (0, -1),
            "down": (1, 0),
            "up": (-1, 0),
        }
        out = {}
        motion_energy = np.zeros((self.rows, self.cols), dtype=np.float32)
        for name, (dr, dc) in directions.items():
            neighbor_delayed = self._shift(self._delayed, dr, dc)
            neighbor_undelayed = self._shift(undelayed, dr, dc)
            # classic HRC: (undelayed_A * delayed_B) - (delayed_A * undelayed_B)
            a_branch = undelayed * neighbor_delayed
            b_branch = self._delayed * neighbor_undelayed
            response = a_branch - b_branch
            response = np.clip(response, 0, None)  # ON-type rectification (T4-like)
            out[name] = response
            motion_energy += response

        out["motion_energy"] = motion_energy
        self._prev_input = lamina_grid
        return out
