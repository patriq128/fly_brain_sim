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

    <path>/neurons.csv   columns: id, type/cell_type (optional),
                                 region (optional), neurotransmitter/nt (optional)
    <path>/edges.csv     columns: src, dst, weight,
                                 sign or neurotransmitter (optional)

`region` should, if present, use one of this project's region keys
(retina, lamina, medulla, lobula, lobula_plate, central_complex,
mushroom_body) or "other" - see brain/regions.py. If `region` is absent,
every neuron is placed in a single "other" bucket and the visualization/
simulation still runs, just without the region breakdown.

Signs
-----
If `edges.csv` has a numeric `sign` column (+1/-1), it is used. Otherwise, if
it has a `neurotransmitter`/`nt` column, or `neurons.csv` has one, the
transmitter is mapped to an effective sign: acetylcholine +1; GABA,
glutamate and histamine -1; dopamine/serotonin/octopamine +0.25. Without any
of these columns all weights are treated as excitatory (the original
behaviour). See `_infer_sign` and `SIGN_BY_NT` below.

Note that `--connectome-source flywire` (data/flywire_source.py) now provides
a fully automatic, no-token path to the real FAFB v783 connectome and is the
recommended way to run real data. This CSV loader remains useful for
hemibrain exports and custom datasets.
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
    "other",
}

# Predicted/known transmitter -> effective sign for the rate model.  Same
# convention as data/flywire_source.py (kept local to avoid pulling pyarrow
# into the CSV-only path).
SIGN_BY_NT = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "glycine": -1.0,
    "dopamine": 0.25,
    "serotonin": 0.25,
    "octopamine": 0.25,
}


def _sign_for_nt(nt: object) -> float | None:
    if nt is None or (isinstance(nt, float) and np.isnan(nt)):
        return None
    key = str(nt).split(",")[0].strip().lower()
    replacements = {"hist": "histamine", "5-ht": "serotonin", "5ht": "serotonin"}
    key = replacements.get(key, key)
    return SIGN_BY_NT.get(key)


def _infer_sign(weight: float, sign: object = None, nt: object = None) -> float:
    """Apply an optional edge sign / transmitter to a measured weight.

    ASSUMPTION: without any sign/neurotransmitter annotation, treat all
    measured synapse-count weights as excitatory.
    """
    if sign is not None and not (isinstance(sign, float) and np.isnan(sign)):
        return weight * float(sign)
    mapped = _sign_for_nt(nt)
    return weight * (mapped if mapped is not None else 1.0)


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

    # Optional per-neuron metadata from the export.
    cell_type_col = next(
        (c for c in ("cell_type", "type") if c in neurons.columns), None
    )
    cell_type_of = (
        neurons[cell_type_col].fillna("").astype(str).to_numpy(dtype=object)
        if cell_type_col
        else np.array([""] * n, dtype=object)
    )
    neuron_nt_col = next(
        (c for c in ("neurotransmitter", "nt", "top_nt") if c in neurons.columns), None
    )
    nt_of = (
        neurons[neuron_nt_col].fillna("").astype(str).to_numpy(dtype=object)
        if neuron_nt_col
        else np.array([""] * n, dtype=object)
    )
    if neuron_nt_col:
        neuron_sign = np.array(
            [
                (s if (s := _sign_for_nt(x)) is not None else 1.0)
                for x in nt_of
            ],
            dtype=np.float32,
        )
    else:
        neuron_sign = None

    edge_sign_col = "sign" if "sign" in edges.columns else None
    edge_nt_col = next(
        (c for c in ("neurotransmitter", "nt", "top_nt") if c in edges.columns), None
    )
    edge_signs = (
        edges[edge_sign_col].to_numpy(dtype=object)
        if edge_sign_col is not None
        else [None] * len(edges)
    )
    edge_nts = (
        edges[edge_nt_col].to_numpy(dtype=object)
        if edge_nt_col is not None
        else [None] * len(edges)
    )

    rows, cols, vals = [], [], []
    missing = 0
    for src, dst, w, edge_sign, edge_nt in zip(
        edges["src"], edges["dst"], edges["weight"], edge_signs, edge_nts
    ):
        si = id_to_idx.get(src)
        di = id_to_idx.get(dst)
        if si is None or di is None:
            missing += 1
            continue
        rows.append(si)
        cols.append(di)
        if edge_sign is not None:
            vals.append(_infer_sign(float(w), sign=edge_sign))
        elif edge_nt is not None:
            vals.append(_infer_sign(float(w), nt=edge_nt))
        elif neuron_sign is not None:
            vals.append(float(w) * float(neuron_sign[si]))
        else:
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
        neuron_ids=np.array([ids[i] for i in sort_idx], dtype=object),
        cell_type_of=cell_type_of[sort_idx],
        nt_of=nt_of[sort_idx],
        sign_of=neuron_sign[sort_idx] if neuron_sign is not None else None,
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
