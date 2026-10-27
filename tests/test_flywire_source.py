"""Offline tests for the real FlyWire connectome pipeline.

These use a tiny hand-written annotation table and Feather edge list, so they
exercise the real region/NT/sign/selection code without any network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from brain.regions import OTHER_REGION
from data.flywire_source import (
    build_flywire_connectome,
    ensure_flywire_files,
    load_annotations,
    select_neurons,
)
from simulation.config import SimLevel, SimulationConfig


def _write_fixture(tmp_path):
    """16 neurons covering every mapped region plus an 'other' neuron."""
    rows = [
        # root_id, cell_class, cell_type, side, top_nt, known_nt, y, z
        (1, "visual", "R1-6", "left", "acetylcholine", "", 1000, 100),
        (2, "visual", "R7", "left", "acetylcholine", "", 1200, 200),
        (3, "visual", "R8", "right", "acetylcholine", "", 1400, 300),
        (4, "LA>ME", "L1", "left", "glutamate", "glutamate", 1100, 150),
        (5, "LA>ME", "L2", "left", "acetylcholine", "acetylcholine", 1300, 250),
        (6, "ME>LOP", "T4a", "left", "acetylcholine", "acetylcholine", 1500, 400),
        (7, "LO>LOP", "T5b", "right", "acetylcholine", "acetylcholine", 1700, 500),
        (8, "Kenyon_Cell", "", "left", "acetylcholine", "", 2000, 600),
        (9, "CX", "", "left", "gaba", "gaba", 2200, 700),
        (10, "MBON", "", "left", "acetylcholine", "", 2400, 800),
        (11, "olfactory", "", "left", "acetylcholine", "", 2600, 900),
        (12, "LA>ME", "L3", "left", "acetylcholine", "", 1150, 175),
        (13, "ME", "Mi1", "left", "acetylcholine", "", 1600, 450),
        (14, "ME", "Tm3", "right", "acetylcholine", "", 1800, 550),
        (15, "LO", "LC4", "left", "acetylcholine", "", 1900, 650),
        # Johnston's-organ auditory receptor neuron -> 'antenna'.
        (16, "mechanosensory", "JO-A", "left", "acetylcholine", "", 900, 950),
    ]
    ann = pd.DataFrame(
        rows,
        columns=[
            "root_id", "cell_class", "cell_type", "side",
            "top_nt", "known_nt", "soma_y", "soma_z",
        ],
    )
    ann["super_class"] = "optic"
    ann["flow"] = "intrinsic"
    ann["soma_x"] = 1000.0
    ann["pos_x"] = ann["soma_x"]
    ann["pos_y"] = ann["soma_y"]
    ann["pos_z"] = ann["soma_z"]
    ann = ann[
        [
            "root_id", "super_class", "cell_class", "cell_type", "flow", "side",
            "top_nt", "known_nt", "soma_x", "soma_y", "soma_z",
            "pos_x", "pos_y", "pos_z",
        ]
    ]
    ann_path = tmp_path / "annotations.tsv"
    ann.to_csv(ann_path, sep="\t", index=False)

    # (pre, post, syn_count); the first pair appears twice to test summing
    # across neuropil rows.
    edges = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1, 4, 5, 6, 9, 8, 13, 14, 999],
            "post_pt_root_id": [4, 4, 5, 6, 7, 9, 10, 10, 6, 7, 1],
            "neuropil": ["LA_L", "ME_L", "ME_L", "ME_L", "LO_L", "CX",
                          "CX", "MB_L", "ME_L", "ME_R", "ME_L"],
            "syn_count": [5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1],
        }
    )
    con_path = tmp_path / "connections.feather"
    feather.write_feather(pa.Table.from_pandas(edges, preserve_index=False), con_path)
    return ann_path, con_path, ann


def _config():
    config = SimulationConfig.for_level(SimLevel.CONNECTOME)
    config.retina_grid_shape = (4, 4)
    return config


def test_region_and_transmitter_mapping(tmp_path):
    ann_path, _, _ = _write_fixture(tmp_path)
    ann = load_annotations(ann_path)
    by_id = ann.set_index("root_id")
    assert by_id.loc[1, "region"] == "retina"
    assert by_id.loc[4, "region"] == "lamina"
    assert by_id.loc[6, "region"] == "lobula_plate"  # T4 overrides ME>LOP
    assert by_id.loc[7, "region"] == "lobula_plate"  # T5 overrides LO>LOP
    assert by_id.loc[15, "region"] == "lobula"
    assert by_id.loc[16, "region"] == "antenna"  # Johnston's-organ JO neuron
    assert by_id.loc[8, "region"] == "mushroom_body"
    assert by_id.loc[9, "region"] == "central_complex"
    assert by_id.loc[11, "region"] == OTHER_REGION
    # Photoreceptors are known histaminergic even when top_nt is noisy.
    assert by_id.loc[1, "nt"] == "histamine"
    assert by_id.loc[4, "nt"] == "glutamate"
    assert by_id.loc[5, "nt"] == "acetylcholine"
    # Effective sign: photoreceptors are treated as excitatory in the rate
    # model (documented ASSUMPTION), glutamate/GABA are inhibitory.
    assert by_id.loc[1, "sign"] == 1.0
    assert by_id.loc[4, "sign"] == -1.0
    assert by_id.loc[9, "sign"] == -1.0


def test_build_real_graph_from_fixture(tmp_path):
    ann_path, con_path, _ = _write_fixture(tmp_path)
    graph = build_flywire_connectome(
        _config(),
        annotations_path=ann_path,
        connectivity_path=con_path,
        max_neurons=0,
        min_synapses=1,
        use_cache=False,
        verbose=False,
    )
    assert graph.source == "flywire_fafb783"
    assert graph.n_neurons == 16
    assert graph.adjacency.shape == (16, 16)
    assert graph.neuron_ids is not None and len(graph.neuron_ids) == 16
    # All region buckets used by the fixture exist and partition the graph.
    covered = np.zeros(graph.n_neurons, dtype=bool)
    for sl in graph.region_slices.values():
        covered[sl] = True
    assert covered.all()
    assert set(graph.region_slices) == {
        "antenna", "retina", "lamina", "medulla", "lobula", "lobula_plate",
        "central_complex", "mushroom_body", "other",
    }

    index = {int(root): i for i, root in enumerate(graph.neuron_ids)}
    # R1-6 -> L1 is excitatory (effective photoreceptor sign), and the two
    # neuropil rows (5 + 5 synapses) were summed before normalisation.
    assert graph.adjacency[index[1], index[4]] > 0
    # L1 is glutamatergic -> its outgoing edge is inhibitory.
    assert graph.adjacency[index[4], index[6]] < 0
    # L2 is cholinergic -> excitatory.
    assert graph.adjacency[index[5], index[7]] > 0
    # The unknown pre-synaptic id (999) is not part of the graph.
    assert 999 not in index

    assert graph.nt_of[index[1]] == "histamine"
    assert graph.sign_of[index[1]] == pytest.approx(1.0)
    assert graph.sign_of[index[4]] == pytest.approx(-1.0)
    assert graph.cell_type_of[index[1]] == "R1-6"
    assert graph.label_for(index[1]) == "R1-6"

    # Optic-lobe neurons receive a retinotopic column; central ones do not.
    for region in ("retina", "lamina", "medulla", "lobula", "lobula_plate"):
        sl = graph.region_slices[region]
        assert np.all(graph.column_of[sl] >= 0)
    for region in ("antenna", "central_complex", "mushroom_body", "other"):
        sl = graph.region_slices[region]
        assert np.all(graph.column_of[sl] == -1)

    assert graph.positions.shape == (16, 3)
    assert np.isfinite(graph.positions).all()


def test_selection_keeps_named_regions_and_priority_cells(tmp_path):
    ann_path, _, ann = _write_fixture(tmp_path)
    loaded = load_annotations(ann_path)
    selected = select_neurons(loaded, 6, seed=0)
    assert 0 < len(selected) < len(ann)
    assert not np.any(loaded.iloc[selected]["region"].to_numpy() == OTHER_REGION)
    cell_types = set(loaded.iloc[selected]["cell_type"].astype(str))
    # The priority logic ensures at least some visual pathway types survive.
    assert cell_types & {"R1-6", "R7", "R8"}
    assert cell_types & {"L1", "L2", "L3"}
    assert cell_types & {"T4a", "T5b"}
    # Whole-brain request returns every annotated row.
    assert len(select_neurons(loaded, 0, seed=0)) == len(ann)


def test_built_graph_cache_round_trip(tmp_path):
    ann_path, con_path, _ = _write_fixture(tmp_path)
    kwargs = dict(
        annotations_path=ann_path,
        connectivity_path=con_path,
        max_neurons=0,
        min_synapses=1,
        use_cache=True,
        verbose=False,
    )
    first = build_flywire_connectome(_config(), cache_dir=tmp_path, **kwargs)
    caches = list(tmp_path.glob("graph_*.npz"))
    assert len(caches) == 1
    second = build_flywire_connectome(_config(), cache_dir=tmp_path, **kwargs)
    assert second.n_neurons == first.n_neurons
    assert (second.adjacency != first.adjacency).nnz == 0
    assert np.array_equal(second.region_of, first.region_of)
    assert np.array_equal(second.neuron_ids, first.neuron_ids)
    assert np.array_equal(second.column_of, first.column_of)


def test_offline_cache_miss_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_flywire_files(tmp_path / "empty", allow_download=False, verbose=False)


def test_flybrain_injects_vision_into_identified_cells(tmp_path, monkeypatch):
    """The real-connectome path drives R1-6/R7/R8 and T4/T5, not everything."""
    import data.flywire_source as flywire_source
    from brain.network import FlyBrain

    ann_path, con_path, _ = _write_fixture(tmp_path)
    graph = build_flywire_connectome(
        _config(),
        annotations_path=ann_path,
        connectivity_path=con_path,
        max_neurons=0,
        min_synapses=1,
        use_cache=False,
        verbose=False,
    )
    monkeypatch.setattr(
        flywire_source, "build_flywire_connectome", lambda *a, **k: graph
    )

    config = _config()
    config.connectome_source = "flywire"
    config.connectome_cache_dir = str(tmp_path)
    config.allow_connectome_download = False
    brain = FlyBrain(config)

    # Vary all channels spatially so both the luminance-contrast and the
    # chroma-proxy photoreceptor channels get a non-constant image.  Shift the
    # pattern between frames: the HRC's delayed branch needs temporal change
    # before any direction map becomes non-zero.
    ramp = np.linspace(0, 255, 32, dtype=np.uint8)
    frames = []
    for shift in range(3):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = np.roll(ramp, shift)[None, :]          # B
        frame[:, :, 1] = 255 - np.roll(ramp, shift)[None, :]    # G
        frames.append(frame)

    index = {int(root): i for i, root in enumerate(graph.neuron_ids)}

    def external(root_id):
        return float(brain._external_input[index[root_id]])

    t4_t5_drive = 0.0
    for frame in frames:
        brain.step(frame, sub_steps=1)
        t4_t5_drive += external(6) + external(7)

    # CONNECTOME: R1-6 (1), R7 (2) and R8 (3) are identified photoreceptors.
    assert external(1) > 0.0
    assert external(2) > 0.0
    assert external(3) > 0.0
    # T4a (6) / T5b (7) receive the corresponding HRC direction maps.
    assert t4_t5_drive > 0.0
    # A downstream neuron with no direct visual identity (Kenyon cell, 8)
    # must only be driven through synapses, never by direct injection.
    assert external(8) == 0.0

    # Optional: turning motion injection off leaves photoreceptors driven but
    # removes the direct T4/T5 drive.
    config.motion_injection = False
    brain_no_motion = FlyBrain(config)
    for frame in frames:
        brain_no_motion.step(frame, sub_steps=1)
        assert float(brain_no_motion._external_input[index[6]]) == 0.0
        assert float(brain_no_motion._external_input[index[7]]) == 0.0
    assert float(brain_no_motion._external_input[index[1]]) > 0.0

    # Hearing: the JO-A neuron (16) is driven directly by the auditory
    # receptor model, while downstream Kenyon cells are still not.
    t_audio = np.arange(1600) / 16000.0
    tone = (0.5 * np.sin(2.0 * np.pi * 220.0 * t_audio)).astype(np.float32)
    brain.step(frames[0], sub_steps=1, audio_window=tone)
    assert float(brain._external_input[index[16]]) > 0.0
    assert float(brain._external_input[index[8]]) == 0.0
