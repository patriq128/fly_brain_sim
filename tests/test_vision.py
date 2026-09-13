import numpy as np

from brain.motion_pathway import ReichardtMotionDetector
from vision.retina import RetinaModel


def test_retina_process_output_shapes():
    model = RetinaModel(grid_shape=(8, 8))
    frame = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    out = model.process(frame)
    assert out["r1_6"].shape == (64,)
    assert out["r7_8"].shape == (64,)
    assert out["raw_luminance"].shape == (8, 8)
    assert np.all(out["r1_6"] >= 0) and np.all(out["r1_6"] <= 1)


def test_motion_detector_responds_to_moving_edge():
    detector = ReichardtMotionDetector(grid_shape=(8, 8))
    dt = 1 / 30.0
    grid = np.zeros((8, 8), dtype=np.float32)
    # sweep a bright column left to right and confirm rightward motion energy > 0 at some point
    max_right_energy = 0.0
    for c in range(8):
        grid[:] = 0.0
        grid[:, c] = 1.0
        out = detector.step(grid, dt)
        max_right_energy = max(max_right_energy, out["right"].mean())
    assert max_right_energy > 0.0


def test_motion_energy_zero_for_static_scene():
    detector = ReichardtMotionDetector(grid_shape=(8, 8))
    dt = 1 / 30.0
    grid = np.full((8, 8), 0.5, dtype=np.float32)
    for _ in range(20):
        out = detector.step(grid, dt)
    # after settling on an unchanging scene, motion energy should be near zero
    assert out["motion_energy"].mean() < 0.05
