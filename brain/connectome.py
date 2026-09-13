"""
brain/connectome.py
====================

Builds the weighted directed adjacency matrix ("connectome") the simulation
runs on, as a scipy sparse matrix, plus per-neuron metadata (region label,
retinotopic column index, 3D layout position for visualization).

Two sources are supported:

1. `build_synthetic_connectome(...)`  (used by DEMO and DETAILED levels)
   Procedurally generates a graph. This is CLEARLY LABELED as synthetic.
   Design choices that ARE grounded in the literature (see README/
   docs/SIMULATION_NOTES.md for citations):
     - Region ordering follows the real feedforward optic-lobe pathway
       retina -> lamina -> medulla -> lobula/lobula plate -> central brain.
     - Retinotopy: each retina "ommatidium" connects to a small local
       neighborhood in lamina/medulla (columnar organization is a real,
       well documented feature of the fly optic lobe).
     - A minority of long-range, wide-field-like connections exist between
       lobula plate and central complex/mushroom body, echoing the real
       existence of large tangential neurons (HS/VS cells) that pool over
       many columns - but the specific cells, counts and weights are
       invented for this project, not measured.
     - Mostly local excitatory feedforward connectivity plus a smaller
       fraction of inhibitory lateral connections within a region, which
       mirrors the qualitative textbook picture of lateral inhibition in
       the lamina/medulla but is not derived from real synapse counts.

2. `load_real_connectome(path)`  (used by CONNECTOME level)
   Loads a real, user-supplied adjacency edge list (e.g. exported from
   neuprint-python for the hemibrain, or from the FlyWire "codex" /
   flywire_annotations connectivity export) in a simple CSV schema.
   This repository does not (and cannot license-wise) bundle that data.
   See data/connectome_loader.py for the exact expected schema and
   download instructions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from brain.regions import REGION_ORDER, REGIONS


@dataclass
class ConnectomeGraph:
    n_neurons: int
    adjacency: sparse.csr_matrix          # signed weights, rows=pre, cols=post
    region_of: np.ndarray                 # (n_neurons,) str region key per neuron index
    region_slices: dict[str, slice]       # region_key -> slice into neuron index space
    positions: np.ndarray                 # (n_neurons, 3) float32 layout for 3D viz
    column_of: np.ndarray                 # (n_neurons,) int retinotopic column id (-1 if n/a)
    source: str                           # "synthetic" or path to real data
    level: str

    def region_activity_mask(self, region_key: str) -> slice:
        return self.region_slices[region_key]


def _hex_grid_positions(rows: int, cols: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate an approximate hexagonal lattice of (x, y) sample points in
    [-1, 1] and a matching column-id array. ASSUMPTION/SIMULATION: real
    ommatidial packing is hexagonal (this IS biologically accurate - the
    fly compound eye is a hex lattice) but the exact sampling density here
    is chosen for computational tractability, not measured."""
    xs, ys, col_ids = [], [], []
    cid = 0
    for r in range(rows):
        offset = 0.5 if r % 2 else 0.0
        for c in range(cols):
            x = (c + offset) / cols * 2 - 1
            y = r / rows * 2 - 1
            xs.append(x)
            ys.append(y)
            col_ids.append(cid)
            cid += 1
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(col_ids)


def _normalize_incoming_weights(adjacency: sparse.csr_matrix, target_abs_sum: float) -> sparse.csr_matrix:
    """SIMULATION: rescale each neuron's incoming synaptic weights so the
    total absolute input strength converging on it does not exceed
    `target_abs_sum`, regardless of how many presynaptic partners it has.

    This is a standard numerical-stability technique for recurrent rate
    networks (comparable in spirit to weight normalization used in reservoir
    computing / balanced-network models), not a biological measurement. Real
    fan-in-dependent synaptic scaling/homeostasis does exist in biology
    (e.g. synaptic scaling, Turrigiano 2008) but the specific rule and
    target value used here are simply chosen to keep the demo's dynamics in
    an interesting, non-saturating regime rather than to model that
    phenomenon quantitatively.
    """
    adj = adjacency.tocsc()
    col_abs_sum = np.asarray(np.abs(adj).sum(axis=0)).flatten()
    scale = np.ones_like(col_abs_sum)
    nonzero = col_abs_sum > target_abs_sum
    scale[nonzero] = target_abs_sum / col_abs_sum[nonzero]
    scale_diag = sparse.diags(scale)
    return (adj @ scale_diag).tocsr()


def build_synthetic_connectome(config) -> ConnectomeGraph:
    """Build a procedurally generated, retinotopically-organized graph for
    DEMO / DETAILED simulation levels. Deterministic given config.random_seed.
    """
    rng = np.random.default_rng(config.random_seed)
    sizes = config.region_sizes
    rows, cols = config.retina_grid_shape
    n_columns = rows * cols

    region_counts = {
        "retina": sizes.retina,
        "lamina": sizes.lamina,
        "medulla": sizes.medulla,
        "lobula": sizes.lobula,
        "lobula_plate": sizes.lobula_plate,
        "central_complex": sizes.central_complex,
        "mushroom_body": sizes.mushroom_body,
    }
    n_total = sum(region_counts.values())

    region_of = np.empty(n_total, dtype=object)
    column_of = np.full(n_total, -1, dtype=np.int64)
    positions = np.zeros((n_total, 3), dtype=np.float32)
    region_slices: dict[str, slice] = {}

    idx = 0
    xs, ys, col_ids = _hex_grid_positions(rows, cols, "left")
    # We lay retina/lamina/medulla out retinotopically (each neuron tagged
    # with the nearest sampling column so downstream local wiring and 3D
    # layout both respect columnar organization).
    layered_regions = ["retina", "lamina", "medulla"]
    pooled_regions = ["lobula", "lobula_plate", "central_complex", "mushroom_body"]

    depth_for_region = {"retina": 0.30, "lamina": 0.15, "medulla": 0.0}

    for region in layered_regions:
        n = region_counts[region]
        start = idx
        info = REGIONS[region]
        for i in range(n):
            # assign to nearest retinotopic column (cycled if n != n_columns)
            column = i % n_columns
            column_of[idx] = column
            region_of[idx] = region
            side_sign = -1.0 if (column % cols) < cols / 2 else 1.0
            anchor = info.anchor_left if side_sign < 0 else info.anchor_right
            jitter = rng.normal(0, 0.06, size=3)
            positions[idx] = np.array(
                [
                    anchor[0] + xs[column] * 0.9,
                    anchor[1] + ys[column] * 0.9,
                    anchor[2],
                ],
                dtype=np.float32,
            ) + jitter
            idx += 1
        region_slices[region] = slice(start, idx)

    for region in pooled_regions:
        n = region_counts[region]
        start = idx
        info = REGIONS[region]
        for i in range(n):
            region_of[idx] = region
            side_sign = -1.0 if i % 2 == 0 else 1.0
            anchor = info.anchor_left if side_sign < 0 else info.anchor_right
            spread = 0.5 if region in ("central_complex",) else 0.7
            jitter = rng.normal(0, spread * 0.35, size=3)
            positions[idx] = np.array(anchor, dtype=np.float32) + jitter
            idx += 1
        region_slices[region] = slice(start, idx)

    assert idx == n_total

    # ---- Build sparse weighted adjacency -----------------------------
    rows_i, cols_j, vals = [], [], []

    def add_local_feedforward(src_region, dst_region, fan_out=6, w_mean=0.9, w_std=0.2):
        """Each dst neuron samples from a small local neighborhood of src
        neurons in the SAME retinotopic column region (biologically: local,
        columnar feedforward convergence, e.g. photoreceptors -> LMCs)."""
        src_slice = region_slices[src_region]
        dst_slice = region_slices[dst_region]
        src_cols = column_of[src_slice]
        dst_idx_range = np.arange(dst_slice.start, dst_slice.stop)
        dst_cols = column_of[dst_slice]
        # build a column -> src indices map
        col_to_src: dict[int, list[int]] = {}
        for local_i, c in enumerate(src_cols):
            col_to_src.setdefault(int(c), []).append(src_slice.start + local_i)
        for local_j, dst_i in enumerate(dst_idx_range):
            c = int(dst_cols[local_j])
            candidates = col_to_src.get(c, [])
            if not candidates:
                continue
            n_pick = min(fan_out, len(candidates))
            chosen = rng.choice(candidates, size=n_pick, replace=False)
            w = rng.normal(w_mean, w_std, size=n_pick)
            rows_i.extend(chosen.tolist())
            cols_j.extend([dst_i] * n_pick)
            vals.extend(w.tolist())

    def add_lateral_inhibition(region, fan_out=4, w_mean=-0.4, w_std=0.1):
        """Local lateral inhibitory connections within a region (mirrors the
        qualitative, well-known role of lateral inhibition sharpening
        contrast in early visual processing)."""
        sl = region_slices[region]
        idxs = np.arange(sl.start, sl.stop)
        n = len(idxs)
        if n < 2:
            return
        for i in idxs:
            n_pick = min(fan_out, n - 1)
            choices = rng.choice(idxs, size=n_pick, replace=False)
            choices = choices[choices != i]
            w = rng.normal(w_mean, w_std, size=len(choices))
            rows_i.extend([i] * len(choices))
            cols_j.extend(choices.tolist())
            vals.extend(w.tolist())

    def add_pooled_convergence(src_region, dst_region, fan_in=24, w_mean=0.5, w_std=0.15):
        """Many-to-few convergent connections, e.g. medulla columns -> a
        smaller number of lobula/lobula-plate wide-field-like neurons."""
        src_sl = region_slices[src_region]
        dst_sl = region_slices[dst_region]
        src_idxs = np.arange(src_sl.start, src_sl.stop)
        for dst_i in range(dst_sl.start, dst_sl.stop):
            n_pick = min(fan_in, len(src_idxs))
            chosen = rng.choice(src_idxs, size=n_pick, replace=False)
            w = rng.normal(w_mean, w_std, size=n_pick)
            rows_i.extend(chosen.tolist())
            cols_j.extend([dst_i] * n_pick)
            vals.extend(w.tolist())

    def add_sparse_random(src_region, dst_region, p=0.02, w_mean=0.3, w_std=0.1):
        src_sl = region_slices[src_region]
        dst_sl = region_slices[dst_region]
        n_src = src_sl.stop - src_sl.start
        n_dst = dst_sl.stop - dst_sl.start
        n_edges = max(1, int(p * n_src * n_dst))
        src_choice = rng.integers(src_sl.start, src_sl.stop, size=n_edges)
        dst_choice = rng.integers(dst_sl.start, dst_sl.stop, size=n_edges)
        w = rng.normal(w_mean, w_std, size=n_edges)
        rows_i.extend(src_choice.tolist())
        cols_j.extend(dst_choice.tolist())
        vals.extend(w.tolist())

    # Feedforward visual pathway (BIOLOGY-inspired ordering; ASSUMPTION-level weights)
    add_local_feedforward("retina", "lamina", fan_out=1, w_mean=1.1, w_std=0.1)
    add_lateral_inhibition("lamina", fan_out=6, w_mean=-0.5, w_std=0.1)
    add_local_feedforward("lamina", "medulla", fan_out=8, w_mean=0.7, w_std=0.2)
    add_lateral_inhibition("medulla", fan_out=6, w_mean=-0.3, w_std=0.08)
    add_pooled_convergence("medulla", "lobula", fan_in=16, w_mean=0.6, w_std=0.15)
    add_pooled_convergence("medulla", "lobula_plate", fan_in=20, w_mean=0.5, w_std=0.15)
    add_pooled_convergence("lobula", "central_complex", fan_in=12, w_mean=0.4, w_std=0.1)
    add_pooled_convergence("lobula_plate", "central_complex", fan_in=12, w_mean=0.4, w_std=0.1)
    add_pooled_convergence("lobula", "mushroom_body", fan_in=10, w_mean=0.3, w_std=0.1)
    add_sparse_random("central_complex", "mushroom_body", p=0.05, w_mean=0.25, w_std=0.1)
    add_sparse_random("mushroom_body", "central_complex", p=0.03, w_mean=0.2, w_std=0.1)
    # sparse recurrent feedback within central complex (heading/integration-like loop)
    add_sparse_random("central_complex", "central_complex", p=0.08, w_mean=0.15, w_std=0.1)

    adjacency = sparse.coo_matrix(
        (np.array(vals, dtype=np.float32), (np.array(rows_i), np.array(cols_j))),
        shape=(n_total, n_total),
    ).tocsr()

    adjacency = _normalize_incoming_weights(adjacency, target_abs_sum=2.5)

    return ConnectomeGraph(
        n_neurons=n_total,
        adjacency=adjacency,
        region_of=region_of,
        region_slices=region_slices,
        positions=positions,
        column_of=column_of,
        source="synthetic",
        level=config.level.value,
    )


def load_real_connectome(path: str, config) -> ConnectomeGraph:
    """Load a real connectome edge list for CONNECTOME mode.

    See data/connectome_loader.py for the exact file schema this expects
    and for instructions on producing it from neuprint-python (hemibrain)
    or the FlyWire codex / flywire_annotations connectivity exports.
    """
    from data.connectome_loader import load_edge_list

    return load_edge_list(path, config)


def save_connectome_summary(graph: ConnectomeGraph, path: str) -> None:
    """Write a small JSON summary (not the full matrix) for inspection/tests."""
    summary = {
        "n_neurons": graph.n_neurons,
        "n_edges": int(graph.adjacency.nnz),
        "source": graph.source,
        "level": graph.level,
        "region_counts": {
            r: int(sl.stop - sl.start) for r, sl in graph.region_slices.items()
        },
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
