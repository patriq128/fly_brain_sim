"""
brain/neuron_model.py
======================

SIMULATION: a generic, vectorized leaky-rate neuron model:

    tau * dV/dt = -V + gain * (W^T @ a) + I_ext + noise
    a = sigmoid(slope * (V - threshold))

integrated with simple forward Euler steps. `a` (activation, in [0, 1]) is
what we call "activity" or loosely "firing rate" throughout the project -
NOT a spike train. This is one of the simplest standard choices in
computational neuroscience for getting smoothly varying, interpretable
population activity suitable for real-time visualization; it is NOT fit to
or derived from any Drosophila electrophysiological recording. Swapping in
a biophysically detailed spiking model (e.g. conductance-based / Hodgkin-
Huxley-style, or even a leaky integrate-and-fire with real spikes) is
exactly the kind of extension `docs/SIMULATION_NOTES.md` describes for
"more biologically accurate" future work, and the rest of the codebase
only depends on this class's public interface (step / activity), so it can
be swapped without touching brain/network.py.

Everything here is plain numpy + scipy.sparse and is vectorized across all
neurons at once (no Python-level per-neuron loops in the hot path), which
is what makes DETAILED mode (~10k neurons) run interactively on a laptop.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from simulation.config import NeuronDynamicsParams


class LeakyRateNetwork:
    def __init__(self, adjacency: sparse.csr_matrix, params: NeuronDynamicsParams, seed: int = 0):
        self.adjacency = adjacency.tocsr()
        self.adjacency_T = self.adjacency.T.tocsr()  # for W^T @ a  (pre->post already encoded in adjacency[pre, post])
        self.n = adjacency.shape[0]
        self.params = params
        self.rng = np.random.default_rng(seed)

        self.V = np.zeros(self.n, dtype=np.float32)
        self.activity = np.zeros(self.n, dtype=np.float32)
        self.fatigue = np.zeros(self.n, dtype=np.float32)  # simple adaptation state

        self._dt = params.dt_ms / 1000.0
        self._tau = params.tau_ms / 1000.0

    def reset(self):
        self.V[:] = 0.0
        self.activity[:] = 0.0
        self.fatigue[:] = 0.0

    def step(self, external_input: np.ndarray) -> np.ndarray:
        """Advance the network by one dt. `external_input` is a (n,) array
        added directly to each neuron's driving current (used to inject
        vision-pathway output into the retina/lamina/medulla neurons).
        Returns the new activity array (also stored as self.activity).
        """
        p = self.params
        # recurrent input: for neuron j, sum over i of adjacency[i, j] * activity[i]
        recurrent = self.adjacency_T.dot(self.activity)

        noise = self.rng.normal(0.0, p.noise_std, size=self.n).astype(np.float32)
        drive = p.gain * recurrent + external_input - self.fatigue + noise

        dV = (-self.V + drive) * (self._dt / self._tau)
        self.V = self.V + dV

        z = p.activation_slope * (self.V - p.activation_threshold)
        self.activity = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        self.activity = self.activity.astype(np.float32)

        # simple fatigue/adaptation so sustained input decays a bit over time
        self.fatigue = p.refractory_decay * self.fatigue + (1 - p.refractory_decay) * self.activity

        return self.activity

    def firing_rate_hz_estimate(self) -> np.ndarray:
        """ASSUMPTION: maps the [0,1] activation to a illustrative 0-200 Hz
        range purely for the stats display; this is NOT a measured or
        calibrated firing rate, just a readable rescaling."""
        return self.activity * 200.0
