"""
Tests for the public predictor API shared by SCP, RemMCP, and H-RelMCP:
batched scores()/covers()/predict(), input validation, CalibrationError, and
the H-RelMCP gradient and tradeoff helpers.
"""

import numpy as np
import pytest

from MCP.CP import CalibrationError
from MCP.HRelMCP import HRelMCP, tradeoff_beta, tradeoff_epsilon
from MCP.RemMCP import RemMCP
from MCP.SCP import SCP
from MCP.shapes import Ellipsoid


def _norm_score(X, Y):
    return np.linalg.norm(Y, axis=1)


def _box_score(X, Y, q):
    """max_j |y_j| - q_j for each row."""
    return np.max(np.abs(Y) - q, axis=1)


def _box_score_grad(X, Y, q):
    """d/dq of max_j (|y_j| - q_j): -1 at the active coordinate."""
    active = np.argmax(np.abs(Y) - q, axis=1)
    grads = np.zeros((Y.shape[0], q.size))
    grads[np.arange(Y.shape[0]), active] = -1.0
    return grads


def _box_cost(q):
    return float(np.sum(q**2))


def _box_cost_grad(q):
    return 2.0 * q


@pytest.fixture
def residuals():
    return np.random.default_rng(0).standard_normal((300, 2))


@pytest.fixture
def calibrated(residuals):
    scp = SCP(_norm_score, alpha=0.1).calibrate(None, residuals)
    remmcp = RemMCP(
        _box_score,
        _box_cost,
        n_q=2,
        alpha=0.1,
        q0=np.ones(2),
        cost_grad_fn=_box_cost_grad,
        bounds=[(0.0, None)] * 2,
    ).calibrate(None, residuals)
    return {"SCP": scp, "RemMCP": remmcp}


# ===========================================================================
# scores / covers / predict
# ===========================================================================


@pytest.mark.parametrize("name", ["SCP", "RemMCP"])
def test_scores_and_covers_agree(calibrated, residuals, name):
    predictor = calibrated[name]
    scores = predictor.scores(None, residuals)
    assert scores.shape == (residuals.shape[0],)
    tol = predictor.cover_tol
    assert tol == (0.0 if name == "SCP" else 1e-6)
    np.testing.assert_array_equal(predictor.covers(None, residuals), scores <= tol)
    assert predictor.empirical_coverage(None, residuals) == pytest.approx(np.mean(scores <= tol))


def test_remmcp_covers_all_kept_calibration_points(residuals):
    # The support points of the final stage lie on the boundary up to solver
    # round-off; they must count as covered.
    shape = Ellipsoid(2)
    predictor = RemMCP(
        shape.score,
        shape.cost,
        n_q=shape.n_q,
        alpha=0.1,
        q0=shape.initial_q(),
        cost_grad_fn=shape.cost_grad,
        bounds=shape.bounds,
    ).calibrate(None, residuals)
    kept = ~predictor.outlier_mask
    scores = predictor.scores(None, residuals)
    assert np.any(np.abs(scores[kept]) < 1e-8)  # binding support points
    assert predictor.covers(None, residuals)[kept].all()
    assert predictor.empirical_coverage(None, residuals) >= kept.mean()


def test_scp_scores_are_threshold_margins(calibrated, residuals):
    scp = calibrated["SCP"]
    np.testing.assert_allclose(
        scp.scores(None, residuals), _norm_score(None, residuals) - scp.q_opt
    )


@pytest.mark.parametrize("name", ["SCP", "RemMCP"])
def test_single_sample_is_a_batch_of_one(calibrated, residuals, name):
    predictor = calibrated[name]
    assert predictor.scores(None, residuals[0]).shape == (1,)
    assert predictor.covers(None, residuals[0])[0] == predictor.covers(None, residuals)[0]


def test_predict_repeats_input_for_candidates():
    def shifted_score(X, Y):  # residual relative to the input x
        return np.linalg.norm(Y - X, axis=1)

    rng = np.random.default_rng(1)
    X_cal = rng.standard_normal((200, 2))
    Y_cal = X_cal + rng.standard_normal((200, 2))
    scp = SCP(shifted_score, alpha=0.1).calibrate(X_cal, Y_cal)

    x = np.array([5.0, -5.0])
    candidates = np.array([[5.0, -5.0], [5.1, -4.9], [50.0, 50.0]])
    np.testing.assert_array_equal(
        scp.predict(x, candidates), scp.covers(np.tile(x, (3, 1)), candidates)
    )
    assert scp.predict(x, candidates).tolist() == [True, True, False]


def test_scores_require_calibration():
    with pytest.raises(RuntimeError, match="calibrate"):
        SCP(_norm_score, alpha=0.1).scores(None, np.zeros((3, 2)))


# ===========================================================================
# Validation and calibration failures
# ===========================================================================


def test_score_fn_with_wrong_shape_raises(residuals):
    def per_component_score(X, Y, q):  # returns (n, 2) instead of (n,)
        return np.abs(Y) - q

    predictor = RemMCP(per_component_score, _box_cost, n_q=2, alpha=0.1, q0=np.ones(2))
    with pytest.raises(ValueError, match="one score per sample"):
        predictor.calibrate(None, residuals)


def test_score_grad_fn_with_wrong_shape_raises(residuals):
    predictor = HRelMCP(
        _box_score,
        _box_cost,
        n_q=2,
        epsilon=0.1,
        beta=0.1,
        q0=np.ones(2),
        score_grad_fn=lambda X, Y, q: np.zeros(q.size),
    )
    with pytest.raises(ValueError, match="score_grad_fn"):
        predictor.calibrate(None, residuals)


def test_calibration_error_is_a_value_error():
    assert issubclass(CalibrationError, ValueError)


def test_scp_too_few_samples_raises_calibration_error():
    with pytest.raises(CalibrationError):
        SCP(_norm_score, alpha=0.05).calibrate(None, np.zeros((10, 2)))


def test_hrelmcp_unreachable_target_raises_calibration_error():
    # 30 points cannot certify 95 % coverage with 90 % confidence for n_q = 2,
    # even when all calibration constraints are satisfied.
    residuals = np.random.default_rng(2).standard_normal((30, 2))
    predictor = HRelMCP(
        _box_score,
        _box_cost,
        n_q=2,
        epsilon=0.05,
        beta=0.1,
        q0=np.ones(2),
        bounds=[(0.0, None)] * 2,
        n_restarts=1,
        random_state=0,
    )
    with pytest.raises(CalibrationError, match="certifies"):
        predictor.calibrate(None, residuals)
    assert predictor.q_opt is None


def test_hrelmcp_stops_after_max_total_iter_without_a_valid_solution(residuals):
    # Points with y_1 > 0 are violated for every q, so no iteration is valid.
    def score(X, Y, q):
        return np.where(Y[:, 0] > 0, 1.0, _box_score(X, Y, q))

    predictor = HRelMCP(
        score,
        _box_cost,
        n_q=2,
        epsilon=0.05,
        beta=0.1,
        q0=np.ones(2),
        bounds=[(0.0, None)] * 2,
        max_outer_iter=3,
        max_total_iter=5,
        n_restarts=1,
        random_state=0,
    )
    with pytest.raises(CalibrationError, match="max_total_iter=5"):
        predictor.calibrate(None, residuals)
    assert predictor.n_iterations == 5
    assert predictor.q_opt is None


def test_hrelmcp_rejects_total_limit_below_i_soft():
    with pytest.raises(ValueError, match="max_total_iter"):
        HRelMCP(
            _box_score, _box_cost, n_q=2, epsilon=0.1, beta=0.1, max_outer_iter=10, max_total_iter=5
        )


# ===========================================================================
# H-RelMCP helpers
# ===========================================================================


def test_analytic_and_finite_difference_gradients_agree(residuals):
    q = np.array([0.8, 1.1])  # generic point: no ties in the max, no slack at zero
    kwargs = dict(n_q=2, epsilon=0.1, beta=0.1)
    analytic = HRelMCP(
        _box_score, _box_cost, score_grad_fn=_box_score_grad, cost_grad_fn=_box_cost_grad, **kwargs
    )
    numeric = HRelMCP(_box_score, _box_cost, **kwargs)

    g_analytic = analytic._penalized_gradient(q, 0.7, None, residuals)
    g_numeric = numeric._penalized_gradient(q, 0.7, None, residuals)
    np.testing.assert_allclose(g_analytic, g_numeric, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "i, d_i, n_cal, beta", [(1, 3, 500, 0.1), (4, 25, 1000, 0.05), (2, 10, 2000, 0.5)]
)
def test_tradeoff_beta_inverts_tradeoff_epsilon(i, d_i, n_cal, beta):
    epsilon_i = tradeoff_epsilon(beta, i, d_i, n_cal)
    assert 0.0 < epsilon_i < 1.0
    assert tradeoff_beta(i, d_i, n_cal, epsilon_i) == pytest.approx(beta, rel=1e-6)


def test_tradeoff_epsilon_grows_with_complexity_and_iterations():
    eps = [tradeoff_epsilon(0.1, 1, d, 1000) for d in (2, 10, 50)]
    assert eps[0] < eps[1] < eps[2]
    assert tradeoff_epsilon(0.1, 1, 10, 1000) < tradeoff_epsilon(0.1, 5, 10, 1000)


def test_hrelmcp_reports_violations_instead_of_an_outlier_budget(residuals):
    predictor = HRelMCP(
        _box_score,
        _box_cost,
        n_q=2,
        epsilon=0.1,
        beta=0.1,
        q0=np.ones(2),
        score_grad_fn=_box_score_grad,
        cost_grad_fn=_box_cost_grad,
        bounds=[(0.0, None)] * 2,
        n_restarts=1,
        random_state=0,
    ).calibrate(None, residuals)

    assert predictor.rho is None
    violated = ~predictor.covers(None, residuals)
    np.testing.assert_array_equal(predictor.outlier_mask, violated)
    assert predictor.n_violations == int(violated.sum())
    assert predictor.d_opt == predictor.n_violations + min(
        2, residuals.shape[0] - predictor.n_violations
    )

    summary = predictor.summary()
    assert "Violated calibration pts  : " in summary and "rho" not in summary


@pytest.mark.parametrize("name", ["SCP", "RemMCP"])
def test_scp_and_remmcp_summaries_report_the_outlier_budget(calibrated, name):
    predictor = calibrated[name]
    assert f"Outlier budget       rho  : {predictor.rho}" in predictor.summary()
    assert int(predictor.outlier_mask.sum()) == predictor.rho * predictor.n_q
