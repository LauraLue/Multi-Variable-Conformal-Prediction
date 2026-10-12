"""Toy problems shared by the tests.

Two synthetic problems:

  Correlated regression (make_data): 1-D input, 2-D output with
  heteroscedastic, correlated Gaussian noise around f(x) = [x, 2x].  Used
  with the rectangular and ellipsoidal toy shapes below; the ellipsoid uses
  the symmetric parameterization P(q) = [[q1, q2], [q2, q3]].

  Multi-modal residuals (mixture_residuals): residuals of
  MCP.experiment.make_data_synthetic, a mixture of three Gaussians.
"""

import numpy as np

from MCP.experiment import make_data_synthetic

# ---------------------------------------------------------------------------
# Correlated regression problem
# ---------------------------------------------------------------------------


def make_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """1-D input -> 2-D output with heteroscedastic, correlated noise."""
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    scale = 0.3 + 0.05 * np.abs(X)
    z = rng.normal(size=(n, 2))
    noise = np.column_stack([z[:, 0], 0.7 * z[:, 0] + np.sqrt(0.51) * z[:, 1]]) * scale
    Y = np.column_stack([X[:, 0], 2.0 * X[:, 0]]) + noise
    return X, Y


def point_predictor(x: np.ndarray) -> np.ndarray:
    """f(x) = [x_0, 2*x_0], for a single input (n_x,) or a batch (n, n_x)."""
    x = np.asarray(x)
    return np.stack([x[..., 0], 2.0 * x[..., 0]], axis=-1)


def rectangular_score(X, Y, q):
    """max_j s_j with s_j = |e_j| - q_j, batched over rows.  Shape: (n,)."""
    return np.max(np.abs(Y - point_predictor(X)) - q, axis=1)


def rectangular_cost(q):
    return float(np.sum(q))


def rectangular_cost_grad(q):
    return np.ones_like(q)


def ellipsoidal_score(X, Y, q):
    """s = e^T P(q) e - 1,  linear in q, batched over rows.  Shape: (n,)."""
    E = Y - point_predictor(X)
    q1, q2, q3 = q
    return q1 * E[:, 0] ** 2 + 2.0 * q2 * E[:, 0] * E[:, 1] + q3 * E[:, 1] ** 2 - 1.0


def ellipsoidal_cost(q):
    """J(q) = -log(det P),  log-barrier keeping P positive definite."""
    q1, q2, q3 = q
    det_P = q1 * q3 - q2**2
    return -np.log(det_P) if det_P > 0.0 else 1e10


def ellipsoidal_cost_grad(q):
    q1, q2, q3 = q
    det_P = q1 * q3 - q2**2
    if det_P <= 0.0:
        return np.zeros(3)
    return np.array([-q3 / det_P, 2.0 * q2 / det_P, -q1 / det_P])


def P_from_q(q) -> np.ndarray:
    """Symmetric matrix P(q) = [[q1, q2], [q2, q3]] of the toy ellipsoid."""
    q1, q2, q3 = q
    return np.array([[q1, q2], [q2, q3]])


def coverage_mask(predictor, X: np.ndarray, Y: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Boolean array: True iff point m is in the prediction set.

    The score of every predictor is <= 0 iff y is inside the set; tol > 0
    accommodates numerical error on boundary support points.
    """
    return predictor.scores(X, Y) <= tol


# ---------------------------------------------------------------------------
# Multi-modal residuals
# ---------------------------------------------------------------------------


def mixture_residuals(n: int, seed: int) -> np.ndarray:
    """(n, 2) residuals Y - f(X) of make_data_synthetic: a mixture of three Gaussians."""
    X, Y = make_data_synthetic(n, np.random.default_rng(seed))
    return Y - np.column_stack([X[:, 0] / 2.0, 2.0 * X[:, 0]])
