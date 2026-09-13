import numpy as np
from scipy import sparse

from brain.neuron_model import LeakyRateNetwork
from simulation.config import NeuronDynamicsParams


def _small_adjacency(n=10, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.normal(0, 0.3, size=(n, n))
    np.fill_diagonal(dense, 0)
    return sparse.csr_matrix(dense.astype(np.float32))


def test_activity_bounded_in_unit_interval():
    adj = _small_adjacency()
    net = LeakyRateNetwork(adj, NeuronDynamicsParams(), seed=1)
    ext = np.random.default_rng(2).normal(0, 1, size=10).astype(np.float32)
    for _ in range(200):
        a = net.step(ext)
        assert np.all(a >= 0.0) and np.all(a <= 1.0)


def test_zero_input_settles_low():
    adj = _small_adjacency()
    params = NeuronDynamicsParams(noise_std=0.0)
    net = LeakyRateNetwork(adj, params, seed=1)
    ext = np.zeros(10, dtype=np.float32)
    for _ in range(500):
        a = net.step(ext)
    assert a.mean() < 0.5  # no sustained drive -> should not sit near full saturation


def test_reset_clears_state():
    adj = _small_adjacency()
    net = LeakyRateNetwork(adj, NeuronDynamicsParams(), seed=1)
    ext = np.ones(10, dtype=np.float32)
    for _ in range(50):
        net.step(ext)
    assert net.V.sum() != 0
    net.reset()
    assert net.V.sum() == 0
    assert net.activity.sum() == 0


def test_firing_rate_estimate_scaling():
    adj = _small_adjacency()
    net = LeakyRateNetwork(adj, NeuronDynamicsParams(), seed=1)
    net.activity[:] = 0.5
    rate = net.firing_rate_hz_estimate()
    assert np.allclose(rate, 100.0)
