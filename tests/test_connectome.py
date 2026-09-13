import numpy as np

from brain.connectome import build_synthetic_connectome, _normalize_incoming_weights
from simulation.config import SimulationConfig, SimLevel


def test_demo_connectome_shapes():
    config = SimulationConfig.for_level(SimLevel.DEMO)
    graph = build_synthetic_connectome(config)
    assert graph.n_neurons == config.region_sizes.total
    assert graph.adjacency.shape == (graph.n_neurons, graph.n_neurons)
    assert graph.positions.shape == (graph.n_neurons, 3)
    assert set(graph.region_slices.keys()) == {
        "retina", "lamina", "medulla", "lobula", "lobula_plate",
        "central_complex", "mushroom_body",
    }


def test_region_slices_partition_all_neurons():
    config = SimulationConfig.for_level(SimLevel.DEMO)
    graph = build_synthetic_connectome(config)
    covered = np.zeros(graph.n_neurons, dtype=bool)
    for sl in graph.region_slices.values():
        covered[sl] = True
    assert covered.all()


def test_deterministic_given_seed():
    config1 = SimulationConfig.for_level(SimLevel.DEMO)
    config1.random_seed = 42
    config2 = SimulationConfig.for_level(SimLevel.DEMO)
    config2.random_seed = 42
    g1 = build_synthetic_connectome(config1)
    g2 = build_synthetic_connectome(config2)
    assert (g1.adjacency != g2.adjacency).nnz == 0


def test_normalize_incoming_weights_caps_column_sum():
    from scipy import sparse

    dense = np.ones((5, 5), dtype=np.float32) * 2.0
    adj = sparse.csr_matrix(dense)
    normed = _normalize_incoming_weights(adj, target_abs_sum=3.0)
    col_sums = np.asarray(np.abs(normed).sum(axis=0)).flatten()
    assert np.all(col_sums <= 3.0 + 1e-4)
