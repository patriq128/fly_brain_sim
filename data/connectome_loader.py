"""
data/connectome_loader.py
==========================

Loader for REAL connectome data, used only by CONNECTOME simulation level.

Why no real data is bundled
----------------------------
The full FlyWire (FAFB) connectome has ~139,000 neurons and ~16.5 million
weighted edges (Dorkenwald et al. 2024, Schlegel et al. 2024, Nature); the
hemibrain has ~25,000 neurons and ~20 million synapses (Scheffer et al. 2020,
eLife). Both are many hundreds of MB to multiple GB even in compressed,
pre-aggregated form, are governed by their own data-use terms, and require
an account / API token to download. Shipping them in this repository is not
practical or appropriate. Instead, this file documents exactly how to
produce a compatible edge-list CSV yourself and loads whatever you place at
the path you pass to `--connectome-path`.

How to obtain a real edge list
--------------------------------
Option A - hemibrain via neuprint-python (recommended for a first try,
smaller dataset, needs a free neuPrint account/token from
https://neuprint.janelia.org):

    pip install neuprint-python
    python - <<'PY'
    from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC
    c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='<YOUR_TOKEN>')
    # Restrict to the optic-lobe-related ROIs to keep this tractable; see
    # neuprint docs for the full ROI list.
    neuron_df, conn_df = fetch_adjacencies(
        NC(status='Traced', rois=['ME(R)', 'LO(R)', 'LOP(R)', 'CX']),
        NC(status='Traced', rois=['ME(R)', 'LO(R)', 'LOP(R)', 'CX']),
    )
    conn_df = conn_df.rename(columns={'bodyId_pre': 'src', 'bodyId_post': 'dst', 'weight': 'weight'})
    neuron_df[['bodyId', 'type', 'instance']].to_csv('neurons.csv', index=False)
    conn_df[['src', 'dst', 'weight']].to_csv('edges.csv', index=False)
    PY

Option B - FlyWire, via the flyconnectome/flywire_annotations exports or the
fafbseg-py client (requires FlyWire community membership / codex access):
export a `src,dst,weight` (or `pre_root_id,post_root_id,syn_count`) table
for the ROIs/cell types you care about.

Expected input schema
----------------------
Two CSV files, both referenced by `path` (a directory):

    <path>/neurons.csv   columns: id, type (optional), region (optional)
    <path>/edges.csv     columns: src, dst, weight

`region` should, if present, use one of this project's region keys
(retina, lamina, medulla, lobula, lobula_plate, central_complex,
mushroom_body) or "other" - see brain/regions.py. If `region` is absent,
every neuron is placed in a single "other" bucket and the visualization/
simulation still runs, just without the region breakdown.

If real synapse counts are used as weights, they are treated as unsigned
excitatory synapse counts (SIMULATION/ASSUMPTION: true fly synapses are
polyadic and can be excitatory or inhibitory depending on neurotransmitter,
which most simple connectome exports do not disambiguate - if your export
includes a sign/neurotransmitter column, extend `_infer_sign` below rather
than assuming everything is excitatory).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse

KNOWN_REGIONS = {
    "retina",
    "lamina",
    "medulla",
    "lobula",
    "lobula_plate",
    "central_complex",
    "mushroom_body",
}


def _infer_sign(weight: float) -> float:
    # ASSUMPTION: without neurotransmitter annotation, treat all measured
    # synapse-count weights as excitatory. Real inhibitory neurotransmitters
    # (e.g. GABA, glutamate at inhibitory receptors) are common in the fly
    # optic lobe; if your CSV has a `sign` or `neurotransmitter` column,
    # multiply weight by -1 for inhibitory rows before calling this loader,
    # or extend this function.
    return weight


def load_edge_list(path: str, config):
    from brain.connectome import ConnectomeGraph  # local import avoids cycle

    neurons_path = os.path.join(path, "neurons.csv")
    edges_path = os.path.join(path, "edges.csv")
    if not (os.path.exists(neurons_path) and os.path.exists(edges_path)):
        raise FileNotFoundError(
            f"Expected {neurons_path} and {edges_path}. See the docstring in "
            f"data/connectome_loader.py for how to export these from "
            f"neuprint-python (hemibrain) or FlyWire."
        )

    neurons = pd.read_csv(neurons_path)
    edges = pd.read_csv(edges_path)

    if "id" not in neurons.columns:
        raise ValueError("neurons.csv must have an 'id' column")
    ids = neurons["id"].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)

    if "region" in neurons.columns:
        region_of = neurons["region"].fillna("other").astype(str).to_numpy(dtype=object)
        region_of = np.array(
            [r if r in KNOWN_REGIONS else "other" for r in region_of], dtype=object
        )
    else:
        region_of = np.array(["other"] * n, dtype=object)

    rows, cols, vals = [], [], []
    missing = 0
    for src, dst, w in zip(edges["src"], edges["dst"], edges["weight"]):
        si = id_to_idx.get(src)
        di = id_to_idx.get(dst)
        if si is None or di is None:
            missing += 1
            continue
        rows.append(si)
        cols.append(di)
        vals.append(_infer_sign(float(w)))
    if missing:
        print(f"[connectome_loader] skipped {missing} edges referencing unknown neuron ids")

    # normalize weight scale so dynamics are comparable to synthetic modes
    vals = np.array(vals, dtype=np.float32)
    if len(vals) and vals.std() > 0:
        vals = vals / (vals.std() * 4.0)

    adjacency = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    region_slices = {}
    order = sorted(set(region_of.tolist()))
    # build contiguous slices by sorting neuron order by region for fast
    # region-wise indexing/statistics (positions/adjacency are re-indexed too)
    sort_idx = np.argsort([order.index(r) for r in region_of], kind="stable")
    region_of_sorted = region_of[sort_idx]
    adjacency = adjacency[sort_idx][:, sort_idx]
    start = 0
    for r in order:
        count = int((region_of_sorted == r).sum())
        region_slices[r] = slice(start, start + count)
        start += count

    positions = _layout_positions(region_of_sorted, region_slices)
    column_of = np.full(n, -1, dtype=np.int64)

    return ConnectomeGraph(
        n_neurons=n,
        adjacency=adjacency,
        region_of=region_of_sorted,
        region_slices=region_slices,
        positions=positions,
        column_of=column_of,
        source=path,
        level=config.level.value,
    )


def _layout_positions(region_of_sorted, region_slices) -> np.ndarray:
    """Assign schematic 3D positions for real-data neurons, region by region,
    reusing the same anchor points as the synthetic layout (see
    brain/regions.py). This is a VISUALIZATION-ONLY layout, not real
    morphological coordinates (skeleton/soma coordinates are available from
    neuprint/FlyWire but are not required for this project's purposes)."""
    from brain.regions import REGIONS

    n = len(region_of_sorted)
    positions = np.zeros((n, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    for region, sl in region_slices.items():
        count = sl.stop - sl.start
        if region not in REGIONS:
            anchor_l = (-4.0, 0.0, -0.5)
            anchor_r = (4.0, 0.0, -0.5)
        else:
            anchor_l = REGIONS[region].anchor_left
            anchor_r = REGIONS[region].anchor_right
        for local_i in range(count):
            anchor = anchor_l if local_i % 2 == 0 else anchor_r
            jitter = rng.normal(0, 0.4, size=3)
            positions[sl.start + local_i] = np.array(anchor, dtype=np.float32) + jitter
    return positions
