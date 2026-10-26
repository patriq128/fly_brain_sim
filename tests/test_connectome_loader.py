"""Tests for the local CSV connectome export path (hemibrain/custom data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.connectome_loader import load_edge_list
from simulation.config import SimLevel, SimulationConfig


def _write_csv_fixture(tmp_path):
    neurons = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "cell_type": ["R1-6", "L1", "L2", "T4a"],
            "region": ["retina", "lamina", "lamina", "lobula_plate"],
            "neurotransmitter": ["histamine", "glutamate", "acetylcholine", "acetylcholine"],
        }
    )
    edges = pd.DataFrame(
        {
            "src": [1, 1, 2, 3],
            "dst": [2, 3, 4, 4],
            "weight": [10, 8, 6, 5],
        }
    )
    neurons.to_csv(tmp_path / "neurons.csv", index=False)
    edges.to_csv(tmp_path / "edges.csv", index=False)


def test_csv_loader_signs_and_metadata(tmp_path):
    _write_csv_fixture(tmp_path)
    config = SimulationConfig.for_level(
        SimLevel.CONNECTOME, connectome_path=str(tmp_path)
    )
    graph = load_edge_list(str(tmp_path), config)

    assert graph.n_neurons == 4
    index = {int(nid): i for i, nid in enumerate(graph.neuron_ids)}
    # All three incoming annotations are interpreted from the presynaptic
    # neuron's transmitter: histamine/glutamate inhibitory, ACh excitatory.
    assert graph.adjacency[index[1], index[2]] < 0  # R1-6 (histamine) -> L1
    assert graph.adjacency[index[1], index[3]] < 0  # R1-6 -> L2
    assert graph.adjacency[index[2], index[4]] < 0  # L1 (glutamate) -> T4a
    assert graph.adjacency[index[3], index[4]] > 0  # L2 (ACh) -> T4a

    assert graph.cell_type_of is not None
    assert graph.cell_type_of[index[1]] == "R1-6"
    assert graph.nt_of[index[1]] == "histamine"
    assert graph.sign_of[index[1]] == -1.0
    assert graph.sign_of[index[3]] == 1.0

    covered = np.zeros(graph.n_neurons, dtype=bool)
    for sl in graph.region_slices.values():
        covered[sl] = True
    assert covered.all()
    assert set(graph.region_slices) == {"retina", "lamina", "lobula_plate"}


def test_csv_loader_explicit_edge_sign_overrides_neuron_nt(tmp_path):
    neurons = pd.DataFrame(
        {
            "id": [1, 2],
            "region": ["retina", "lamina"],
            "neurotransmitter": ["acetylcholine", "acetylcholine"],
        }
    )
    edges = pd.DataFrame(
        {"src": [1], "dst": [2], "weight": [4.0], "sign": [-1.0]}
    )
    neurons.to_csv(tmp_path / "neurons.csv", index=False)
    edges.to_csv(tmp_path / "edges.csv", index=False)
    graph = load_edge_list(
        str(tmp_path),
        SimulationConfig.for_level(SimLevel.CONNECTOME, connectome_path=str(tmp_path)),
    )
    index = {int(nid): i for i, nid in enumerate(graph.neuron_ids)}
    assert graph.adjacency[index[1], index[2]] < 0
