"""
Unit tests for the ellipsoidal shapes, the shape fits, and the paper methods:
  SCP-1, SCP-2A, SCP-2B, SCP-2C, RemMCP, H-RelMCP-A, H-RelMCP-B

All methods are tested in residual space on synthetic multi-modal residuals
(toy_problems.mixture_residuals).

Run all tests:             pytest tests/test_shapes_and_methods.py
Skip slow (CMA-ES/H-RelMCP): pytest tests/test_shapes_and_methods.py -m "not slow"
Run as a script:           python tests/test_shapes_and_methods.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

from MCP.experiment import min_ncal_for_rho
from MCP.fitting import fit_ellipsoid_kmeans, fit_ellipsoid_tumu, init_q0_from_fit_ellipsoid
from MCP.HRelMCP import HRelMCP
from MCP.RemMCP import RemMCP
from MCP.SCP import SCP
from MCP.shapes import (
    Ellipsoid,
    FittedEllipsoids,
    UnionOfEllipsoids,
    ellipsoid_param_count,
    score_2norm,
)

from toy_problems import mixture_residuals

_N_Q_PER_ELLIPSOID = ellipsoid_param_count(2)


def min_ncal_for_rho0(epsilon: float) -> int:
    """Smallest calibration-set size with a valid SCP outlier budget (rho = 0, n_q = 1)."""
    return min_ncal_for_rho(epsilon, rho=0, nq=1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
N_CAL = 150
EPSILON = 0.1
N_SHAPES = 2
ELLIPSOID = Ellipsoid(2)
UNION = UnionOfEllipsoids(2, N_SHAPES)


def _zeros(n: int) -> np.ndarray:
    """Dummy input array for residual-space calibration."""
    return np.zeros((n, 1), dtype=float)


def _cholesky_q_to_matrix(q: np.ndarray) -> np.ndarray:
    """Q = L L^T for the 2-D Cholesky parameterization q = [L00, L10, L11]."""
    L = np.array([[q[0], 0.0], [q[1], q[2]]], dtype=float)
    return L @ L.T


def _is_pd(Q: np.ndarray) -> bool:
    """True iff the 2×2 matrix Q is positive definite."""
    return bool(np.all(np.linalg.eigvalsh(np.asarray(Q, dtype=float)) > 0))


# ===========================================================================
# 1 — score_2norm
# ===========================================================================


def test_score_2norm():
    """Scalar for a single sample, (n,) array for a batch; known values."""
    assert score_2norm(None, np.array([3.0, 4.0])) == pytest.approx(5.0)

    Y = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    out = score_2norm(None, Y)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [5.0, 1.0, 0.0])


# ===========================================================================
# 2 — ELLIPSOID.score / ELLIPSOID.cost / ELLIPSOID.cost_grad
# ===========================================================================


def test_ellipsoidal_score():
    """Shape (1,) for single, (n,) for batch; known values for Q = I."""
    q_id = np.array([1.0, 0.0, 1.0])  # Q = I  →  score = ‖e‖² − 1

    # e = [1, 0] lies on the unit-circle boundary: score = 0
    score = ELLIPSOID.score(None, np.array([[1.0, 0.0]]), q_id)
    assert score.shape == (1,)
    assert float(score[0]) == pytest.approx(0.0)

    # five points at origin: score = −1 each
    scores = ELLIPSOID.score(None, np.zeros((5, 2)), q_id)
    assert scores.shape == (5,)
    np.testing.assert_allclose(scores, -1.0)


def test_ellipsoidal_cost():
    """Finite and < 1e9 for a valid Cholesky factor; sentinel 1e10 otherwise."""
    q_pd = np.array([2.0, 0.5, 2.0])  # L = [[2, 0], [0.5, 2]], positive diagonal
    q_bad = np.array([1.0, 0.5, -1.0])  # L[1, 1] < 0: not a valid Cholesky factor

    assert np.isfinite(ELLIPSOID.cost(q_pd)) and ELLIPSOID.cost(q_pd) < 1e9
    assert ELLIPSOID.cost(q_bad) == pytest.approx(1e10)


def test_ellipsoidal_cost_grad():
    """Finite (3,) gradient for a valid Cholesky factor; zero gradient otherwise."""
    q_pd = np.array([2.0, 0.5, 2.0])
    q_bad = np.array([1.0, 0.5, -1.0])

    g_pd = ELLIPSOID.cost_grad(q_pd)
    assert g_pd.shape == (3,) and np.all(np.isfinite(g_pd))

    g_bad = ELLIPSOID.cost_grad(q_bad)
    assert g_bad.shape == (3,) and np.all(g_bad == 0.0)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_ellipsoidal_score_grad_matches_finite_differences(d):
    """Analytic score gradient agrees with central differences, shape (n, d(d+1)/2)."""
    rng = np.random.default_rng(d)
    n_tri = d * (d + 1) // 2
    q = rng.standard_normal(n_tri)
    E = rng.standard_normal((40, d))

    shape = Ellipsoid(d)
    grad = shape.score_grad(None, E, q)
    assert grad.shape == (40, n_tri)

    h = 1e-6
    fd = np.zeros_like(grad)
    for k in range(n_tri):
        step = np.zeros(n_tri)
        step[k] = h
        fd[:, k] = (shape.score(None, E, q + step) - shape.score(None, E, q - step)) / (2 * h)
    np.testing.assert_allclose(grad, fd, atol=1e-6)


# ===========================================================================
# 3 — FittedEllipsoids.component_scores
# ===========================================================================


def test_fitted_component_scores():
    """Output shape (K, n); three identical shapes give identical rows."""
    K, n = 3, 20
    rng = np.random.default_rng(0)
    E = rng.standard_normal((n, 2))
    fitted = FittedEllipsoids([np.zeros(2)] * K, [np.eye(2)] * K)
    assert fitted.n_shapes == K
    out = fitted.component_scores(E)
    assert out.shape == (K, n)

    # All three shapes are identical → rows must agree
    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], out[2])

    # At the origin with Q=I: score = 0² + 0² − 1 = −1
    np.testing.assert_allclose(fitted.component_scores(np.zeros((1, 2))), -1.0)


# ===========================================================================
# 4 — FittedEllipsoids.score / ellipsoids
# ===========================================================================


def test_fitted_union_score_and_geometry():
    """Batched union score; a single sample is a batch of one; centroid is inside."""
    rng = np.random.default_rng(1)
    E_fit = rng.standard_normal((60, 2))
    fitted = fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, n_shapes=1, normalization="common")

    assert fitted.score(None, np.zeros((10, 2))).shape == (10,)
    assert fitted.score(None, np.zeros(2)).shape == (1,)

    # Cluster centroid must be well inside the fitted ellipsoid (score < 0)
    assert float(fitted.score(None, fitted.centers[0])[0]) < 0.0

    # Threshold q scales the precision matrices by 1 / (1 + q); 1 + q <= 0 gives no set
    ((Q, c),) = fitted.ellipsoids(threshold=1.0)
    np.testing.assert_allclose(Q, fitted.Q[0] / 2.0)
    assert fitted.ellipsoids(threshold=-1.0) == []


def test_fitted_weights_score_and_geometry():
    """Weights scale the component scores; the threshold inflates each shape by q / w_k."""
    fitted = FittedEllipsoids([np.zeros(2), np.full(2, 5.0)], [np.eye(2), np.eye(2)], [2.0, 0.5])
    E = np.array([[0.0, 0.0], [5.0, 5.0], [2.0, 0.0]])
    # Point 0: min(2 * -1, 0.5 * 49) = -2; point 1: min(2 * 49, 0.5 * -1) = -0.5
    np.testing.assert_allclose(fitted.score(None, E)[:2], [-2.0, -0.5])

    threshold = 1.0
    geometry = fitted.ellipsoids(threshold)
    np.testing.assert_allclose(geometry[0][0], np.eye(2) / 1.5)
    np.testing.assert_allclose(geometry[1][0], np.eye(2) / 3.0)
    # Geometry agrees with the score: e is inside some ellipsoid iff score <= threshold
    rng = np.random.default_rng(3)
    pts = rng.uniform(-3.0, 8.0, size=(2000, 2))
    in_geometry = np.zeros(len(pts), dtype=bool)
    for Q, c in geometry:
        in_geometry |= np.einsum("ni,ij,nj->n", pts - c, Q, pts - c) <= 1.0
    np.testing.assert_array_equal(in_geometry, fitted.score(None, pts) <= threshold)

    # A negative threshold empties the shape with the smaller weight first
    assert len(fitted.ellipsoids(-0.6)) == 1
    assert fitted.scaled(2.0).weights == fitted.weights
    with pytest.raises(ValueError, match="positive"):
        FittedEllipsoids([np.zeros(2)], [np.eye(2)], [0.0])


def test_tumu_normalization_matches_eq9():
    """w_k = 1 / (Quantile(R_k, 1 - alpha) - min R_k) over all fitting residuals (Tumu Eq. 9)."""
    E_fit = mixture_residuals(600, seed=4)
    common = fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, n_shapes=3, normalization="common")
    tumu = fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, n_shapes=3, normalization="tumu")
    raw = FittedEllipsoids(tumu.centers, tumu.Q).component_scores(E_fit)
    expected = 1.0 / (np.quantile(raw, 1 - EPSILON, axis=1) - raw.min(axis=1) + 1e-8)
    np.testing.assert_allclose(tumu.weights, expected)
    # Same clusters; the common normalization rescales Q and keeps unit weights
    for Q_common, Q_tumu in zip(common.Q, tumu.Q, strict=True):
        np.testing.assert_allclose(Q_common / Q_common[0, 0], Q_tumu / Q_tumu[0, 0])
    assert common.weights == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="Unknown normalization"):
        fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, normalization="other")


def test_kmeans_outlier_cluster_is_regularized():
    """A single far outlier forms its own cluster; its covariance stays invertible."""
    rng = np.random.default_rng(5)
    E_fit = np.vstack([rng.standard_normal((400, 2)), rng.standard_normal((400, 2)) + 6.0])
    E_fit = np.vstack([E_fit, [[-60.0, -80.0]]])
    for normalization in ("common", "tumu"):
        fitted = fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, n_shapes=3, normalization=normalization)
        for Q in fitted.Q:
            assert np.all(np.isfinite(Q))
            assert np.linalg.cond(Q) < 1e6


# ===========================================================================
# 5 — fit_ellipsoid_kmeans
# ===========================================================================


def test_fit_ellipsoid_kmeans():
    """FittedEllipsoids with n_shapes PD ellipsoids; error on too-small data."""
    rng = np.random.default_rng(2)
    E = rng.standard_normal((80, 2))

    for n_shapes in (1, N_SHAPES):
        params = fit_ellipsoid_kmeans(E, alpha=EPSILON, n_shapes=n_shapes)
        assert isinstance(params, FittedEllipsoids)
        assert params.n_shapes == n_shapes
        assert len(params.centers) == len(params.Q) == n_shapes
        for Q in params.Q:
            assert _is_pd(Q), f"Q not PD: eigenvalues = {np.linalg.eigvalsh(Q)}"
        for c in params.centers:
            assert np.asarray(c).shape == (2,)

    with pytest.raises(ValueError, match="Not enough"):
        fit_ellipsoid_kmeans(E[:1], alpha=EPSILON, n_shapes=3)


# ===========================================================================
# 6 — init_q0_from_fit_ellipsoid
# ===========================================================================


def test_init_q0_from_fit_ellipsoid():
    """Length = n_shapes × 5; diagonal Cholesky entries L[0,0], L[1,1] > 0."""
    rng = np.random.default_rng(3)
    E = rng.standard_normal((60, 2))

    for n_shapes in (1, N_SHAPES):
        q0 = init_q0_from_fit_ellipsoid(E, alpha=EPSILON, n_shapes=n_shapes)
        assert q0.shape == (n_shapes * _N_Q_PER_ELLIPSOID,)
        for i in range(n_shapes):
            base = i * _N_Q_PER_ELLIPSOID
            assert q0[base + 0] > 0.0, f"L[0,0] not positive for shape {i}"
            assert q0[base + 2] > 0.0, f"L[1,1] not positive for shape {i}"


# ===========================================================================
# 7 — UnionOfEllipsoids
# ===========================================================================


def test_union_of_ellipsoids_score_and_cost():
    """Correct output shapes for scores, cost, and their analytic gradients."""
    rng = np.random.default_rng(4)
    E = rng.standard_normal((30, 2))
    q = init_q0_from_fit_ellipsoid(E, alpha=EPSILON, n_shapes=N_SHAPES)
    scores = UNION.score(None, E, q)
    assert scores.shape == (E.shape[0],)

    cost = UNION.cost(q)
    assert np.isfinite(cost) and cost > 0.0

    g_cost = UNION.cost_grad(q)
    assert g_cost.shape == q.shape and np.all(np.isfinite(g_cost))

    g_score = UNION.score_grad(None, E, q)
    assert g_score.shape == (E.shape[0], q.size)


# ===========================================================================
# 8 — SCP-1: isotropic 2-norm prediction set
# ===========================================================================


def test_scp1():
    """rho formula, outlier count, active coverage = 1, cal coverage ≥ 1 − ε."""
    E = mixture_residuals(N_CAL, seed=42)
    x_cal, y_cal = _zeros(N_CAL), E

    predictor = SCP(score_fn=score_2norm, alpha=EPSILON)
    predictor.calibrate(x_cal, y_cal)

    expected_rho = int(np.floor(EPSILON * (N_CAL + 1))) - 1

    assert predictor.q_opt is not None
    assert float(predictor.q_opt) > 0.0, "q_opt must be positive"
    assert predictor.rho == expected_rho, f"rho={predictor.rho}, expected {expected_rho}"
    assert int(predictor.outlier_mask.sum()) == expected_rho, "outlier count != rho"

    cal_cov = predictor.empirical_coverage(x_cal, y_cal)
    assert cal_cov >= 1.0 - EPSILON, f"SCP-1 cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"

    # All non-outlier calibration points must be inside the set
    active = ~predictor.outlier_mask
    active_cov = predictor.empirical_coverage(x_cal[active], y_cal[active])
    assert active_cov == pytest.approx(1.0), "SCP-1: active calibration points not fully covered"

    # Scores of outliers must strictly exceed the threshold
    scores = score_2norm(x_cal, y_cal)
    assert np.all(scores[predictor.outlier_mask] > predictor.q_opt)

    # coverage_confidence must be a valid probability
    conf = predictor.coverage_confidence(epsilon=EPSILON)
    assert 0.0 <= conf <= 1.0


# ===========================================================================
# 9 — SCP-2A: single fitted ellipsoid (kmeans, 1 shape)
# ===========================================================================


def test_scp_2a():
    """One PD ellipsoid fitted; calibration coverage ≥ 1 − ε on the SCP stage."""
    E = mixture_residuals(N_CAL, seed=42)
    n_cal2 = min_ncal_for_rho0(EPSILON)
    E_fit, E_cal2 = E[: N_CAL - n_cal2], E[N_CAL - n_cal2 :]

    params = fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, n_shapes=1)
    assert params.n_shapes == 1
    assert _is_pd(params.Q[0])

    x_cal2, y_cal2 = _zeros(len(E_cal2)), E_cal2
    predictor = SCP(score_fn=params.score, alpha=EPSILON)
    predictor.calibrate(x_cal2, y_cal2)

    assert predictor.q_opt is not None
    cal_cov = predictor.empirical_coverage(x_cal2, y_cal2)
    assert cal_cov >= 1.0 - EPSILON, f"SCP-2A cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"


# ===========================================================================
# 10 — SCP-2B: union of N_SHAPES ellipsoids (kmeans)
# ===========================================================================


def test_scp_2b():
    """N_SHAPES PD ellipsoids fitted; calibration coverage ≥ 1 − ε."""
    E = mixture_residuals(N_CAL, seed=42)
    n_cal2 = min_ncal_for_rho0(EPSILON)
    E_fit, E_cal2 = E[: N_CAL - n_cal2], E[N_CAL - n_cal2 :]

    params = fit_ellipsoid_kmeans(E_fit, alpha=EPSILON, n_shapes=N_SHAPES)
    assert params.n_shapes == N_SHAPES
    assert len(params.Q) == N_SHAPES
    for Q in params.Q:
        assert _is_pd(Q), f"SCP-2B Q not PD: eigenvalues = {np.linalg.eigvalsh(Q)}"

    x_cal2, y_cal2 = _zeros(len(E_cal2)), E_cal2
    predictor = SCP(score_fn=params.score, alpha=EPSILON)
    predictor.calibrate(x_cal2, y_cal2)

    assert predictor.q_opt is not None
    cal_cov = predictor.empirical_coverage(x_cal2, y_cal2)
    assert cal_cov >= 1.0 - EPSILON, f"SCP-2B cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"


# ===========================================================================
# 11 — SCP-2C: union of ellipsoids via CMA-ES (Tumu)          [slow]
# ===========================================================================


@pytest.mark.slow
def test_scp_2c():
    """N_SHAPES PD ellipsoids from CMA-ES; calibration coverage ≥ 1 − ε."""
    E = mixture_residuals(N_CAL, seed=42)
    n_cal2 = min_ncal_for_rho0(EPSILON)
    E_fit, E_cal2 = E[: N_CAL - n_cal2], E[N_CAL - n_cal2 :]

    params = fit_ellipsoid_tumu(E_fit, alpha=EPSILON, n_shapes=N_SHAPES, seed=0, max_iter=200)
    assert params.n_shapes == N_SHAPES
    for Q in params.Q:
        assert _is_pd(Q), f"SCP-2C Q not PD: eigenvalues = {np.linalg.eigvalsh(Q)}"

    x_cal2, y_cal2 = _zeros(len(E_cal2)), E_cal2
    predictor = SCP(score_fn=params.score, alpha=EPSILON)
    predictor.calibrate(x_cal2, y_cal2)

    assert predictor.q_opt is not None
    cal_cov = predictor.empirical_coverage(x_cal2, y_cal2)
    assert cal_cov >= 1.0 - EPSILON, f"SCP-2C cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"


# ===========================================================================
# 12 — RemMCP: convex ellipsoidal predictor
# ===========================================================================


def test_remmcp():
    """rho formula; PD Q; active coverage = 1; cal coverage ≥ 1 − ε."""
    E = mixture_residuals(N_CAL, seed=42)
    x_cal, y_cal = _zeros(N_CAL), E
    n_q = 3

    predictor = RemMCP(
        score_fn=ELLIPSOID.score,
        cost_fn=ELLIPSOID.cost,
        n_q=n_q,
        alpha=EPSILON,
        q0=np.array([1.0, 0.0, 1.0]),
        cost_grad_fn=ELLIPSOID.cost_grad,
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
    )
    predictor.calibrate(x_cal, y_cal)

    expected_rho = int(np.floor(EPSILON * (N_CAL + 1) / n_q)) - 1

    assert predictor.q_opt is not None and predictor.q_opt.shape == (n_q,)
    assert predictor.rho == expected_rho, f"RemMCP rho={predictor.rho}, expected {expected_rho}"

    Q = _cholesky_q_to_matrix(predictor.q_opt)
    assert _is_pd(Q), f"RemMCP Q not PD: eigenvalues = {np.linalg.eigvalsh(Q)}"

    # All active (non-outlier) calibration points must be covered; support
    # points lie on the boundary, so allow a small numerical tolerance.
    active = ~predictor.outlier_mask
    active_scores = ELLIPSOID.score(None, y_cal[active], predictor.q_opt)
    assert np.all(active_scores <= 1e-6), (
        f"RemMCP: max active score {active_scores.max():.2e} > 1e-6"
    )

    cal_cov = predictor.empirical_coverage(x_cal, y_cal)
    assert cal_cov >= 1.0 - EPSILON, f"RemMCP cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"

    conf = predictor.coverage_confidence(epsilon=EPSILON)
    assert 0.0 <= conf <= 1.0


# ===========================================================================
# 13 — H-RelMCP-A: non-convex, single ellipsoid
# ===========================================================================


def test_hrelmcp_a():
    """PD Q; epsilon_i_opt ≤ ε; confidence_final valid; cal coverage ≥ 1 − ε."""
    E = mixture_residuals(N_CAL, seed=42)
    x_cal, y_cal = _zeros(N_CAL), E

    # beta derived from SCP-1 confidence, matching run_single
    scp1 = SCP(score_fn=score_2norm, alpha=EPSILON)
    scp1.calibrate(x_cal, y_cal)
    beta = max(1e-10, 1.0 - scp1.coverage_confidence(epsilon=EPSILON))

    predictor = HRelMCP(
        score_fn=ELLIPSOID.score,
        cost_fn=ELLIPSOID.cost,
        n_q=3,
        epsilon=EPSILON,
        beta=beta,
        q0=np.array([1.0, 0.0, 1.0]),
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
        score_grad_fn=ELLIPSOID.score_grad,
        cost_grad_fn=ELLIPSOID.cost_grad,
        n_restarts=2,
        max_outer_iter=10,
        max_iter=300,
        random_state=0,
    )
    predictor.calibrate(x_cal, y_cal)

    assert predictor.q_opt is not None and predictor.q_opt.shape == (3,)
    assert np.all(np.isfinite(predictor.q_opt)), "H-RelMCP-A q_opt contains non-finite values"

    Q = _cholesky_q_to_matrix(predictor.q_opt)
    assert _is_pd(Q), f"H-RelMCP-A Q not PD: eigenvalues = {np.linalg.eigvalsh(Q)}"

    assert predictor.confidence_final is not None
    assert 0.0 <= predictor.confidence_final <= 1.0

    # The accepted iteration must satisfy the epsilon target
    assert predictor.epsilon_i_opt is not None
    assert predictor.epsilon_i_opt <= EPSILON + 1e-10, (
        f"H-RelMCP-A epsilon_i_opt {predictor.epsilon_i_opt:.6f} > epsilon {EPSILON}"
    )

    cal_cov = predictor.empirical_coverage(x_cal, y_cal)
    assert cal_cov >= 1.0 - EPSILON, f"H-RelMCP-A cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"


# ===========================================================================
# 14 — H-RelMCP-B: non-convex, union of N_SHAPES ellipsoids          [slow]
# ===========================================================================


@pytest.mark.slow
def test_hrelmcp_b():
    """All N_SHAPES ellipsoids PD; epsilon_i_opt ≤ ε; cal coverage ≥ 1 − ε."""
    # n_q = 10 needs more data than N_CAL: with 150 points even a solution
    # without violations only certifies epsilon_i ≈ 0.109 > EPSILON.
    n_cal = 400
    E = mixture_residuals(n_cal, seed=42)
    x_cal, y_cal = _zeros(n_cal), E
    n_q = N_SHAPES * _N_Q_PER_ELLIPSOID

    scp1 = SCP(score_fn=score_2norm, alpha=EPSILON)
    scp1.calibrate(x_cal, y_cal)
    beta = max(1e-10, 1.0 - scp1.coverage_confidence(epsilon=EPSILON))

    predictor = HRelMCP(
        score_fn=UNION.score,
        cost_fn=UNION.cost,
        n_q=n_q,
        epsilon=EPSILON,
        beta=beta,
        q0=init_q0_from_fit_ellipsoid(E, alpha=EPSILON, n_shapes=N_SHAPES),
        bounds=([(1e-6, None), (None, None), (1e-6, None), (None, None), (None, None)] * N_SHAPES),
        score_grad_fn=UNION.score_grad,
        cost_grad_fn=UNION.cost_grad,
        n_restarts=2,
        max_outer_iter=8,
        max_iter=300,
        random_state=0,
    )
    predictor.calibrate(x_cal, y_cal)

    assert predictor.q_opt is not None and predictor.q_opt.shape == (n_q,)
    assert np.all(np.isfinite(predictor.q_opt)), "H-RelMCP-B q_opt contains non-finite values"

    for Q, _ in UNION.ellipsoids(predictor.q_opt):
        assert _is_pd(Q), f"H-RelMCP-B ellipsoid not PD: eigenvalues = {np.linalg.eigvalsh(Q)}"

    assert predictor.confidence_final is not None
    assert 0.0 <= predictor.confidence_final <= 1.0

    assert predictor.epsilon_i_opt is not None
    assert predictor.epsilon_i_opt <= EPSILON + 1e-10, (
        f"H-RelMCP-B epsilon_i_opt {predictor.epsilon_i_opt:.6f} > epsilon {EPSILON}"
    )

    cal_cov = predictor.empirical_coverage(x_cal, y_cal)
    assert cal_cov >= 1.0 - EPSILON, f"H-RelMCP-B cal coverage {cal_cov:.4f} < {1 - EPSILON:.4f}"


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    slow_tests = []
    fast_tests = [
        test_score_2norm,
        test_ellipsoidal_score,
        test_ellipsoidal_cost,
        test_ellipsoidal_cost_grad,
        test_fitted_component_scores,
        test_fitted_union_score_and_geometry,
        test_fitted_weights_score_and_geometry,
        test_tumu_normalization_matches_eq9,
        test_kmeans_outlier_cluster_is_regularized,
        test_fit_ellipsoid_kmeans,
        test_init_q0_from_fit_ellipsoid,
        test_union_of_ellipsoids_score_and_cost,
        test_scp1,
        test_scp_2a,
        test_scp_2b,
        test_remmcp,
        test_hrelmcp_a,
    ]
    slow_tests = [
        test_scp_2c,
        test_hrelmcp_b,
    ]

    run_slow = "--slow" in sys.argv
    all_tests = fast_tests + (slow_tests if run_slow else [])

    passed = failed = 0
    for t in all_tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    skipped = len(slow_tests) if not run_slow else 0
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (pass --slow to include).")
