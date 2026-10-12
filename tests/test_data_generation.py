"""Tests for the vehicle-trajectory data generator."""

import numpy as np

from MCP.data.data_generation import (
    CALIBRATION_POLICIES,
    PREFIX_SIZE,
    SUFFIX_SIZE,
    generate_calibration_dataset,
    sample_rollouts,
)
from MCP.data.predictor import TrajectoryPredictor


def test_policies_are_drawn_iid_uniformly():
    # Exactly n/3 rollouts per policy would make the samples dependent.
    n = 3000
    counts = np.array([np.bincount(sample_rollouts(n, seed=s)[1], minlength=3) for s in range(5)])
    assert counts.shape == (5, len(CALIBRATION_POLICIES))
    assert np.all(counts.sum(axis=1) == n)
    sd = np.sqrt(n * (1 / 3) * (2 / 3))
    assert np.all(np.abs(counts - n / 3) < 5 * sd)
    assert len({tuple(c) for c in counts}) > 1  # counts vary with the seed
    assert not np.all(counts == n // 3)


def test_rollouts_draw_independent_noise():
    # All policies share the acceleration of the first steps, so equal speed
    # increments would mean shared noise.
    x, y = generate_calibration_dataset(30, seed=5)
    dv = np.diff(np.concatenate([x[:, :, 3], y[:, :, 3]], axis=1), axis=1)
    assert not any(np.allclose(dv[i, :4], dv[j, :4]) for i in range(30) for j in range(i))


def test_residuals_of_consecutive_rollouts_are_uncorrelated():
    x, y = generate_calibration_dataset(3000, seed=101)
    assert x.shape == (3000, PREFIX_SIZE, 4) and y.shape == (3000, SUFFIX_SIZE, 4)
    pred = TrajectoryPredictor(0.1).predict_batched_trajectories(x)
    norms = np.linalg.norm(y[:, 29, :2] - pred[:, 29, :2], axis=1)
    assert abs(np.corrcoef(norms[:-1], norms[1:])[0, 1]) < 0.1


def test_same_seed_reproduces_dataset():
    x1, y1 = generate_calibration_dataset(30, seed=7)
    x2, y2 = generate_calibration_dataset(30, seed=7)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
