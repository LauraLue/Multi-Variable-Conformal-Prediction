"""
Tests for the two coverage guarantees of SCP and RemMCP (Sec. 3 of the paper):

  expected coverage        E[eta] >= 1 - alpha,                 rho from the alpha rule
  high-confidence coverage P(eta >= 1 - epsilon) >= 1 - beta,
                           rho = max{rho' : beta_bar(rho') <= beta}

Besides checking the outlier-budget rules directly, the Monte Carlo tests
calibrate on many independent Gaussian calibration sets and evaluate the exact
coverage eta of each calibrated set in closed form, so that both guarantees
are verified empirically.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from MCP.CP import (
    confidence_bound,
    outlier_budget_expected,
    outlier_budget_high_confidence,
)
from MCP.HRelMCP import HRelMCP
from MCP.RemMCP import RemMCP
from MCP.SCP import SCP


def _abs_score(x, y):
    """s̃(x, y) = |y| for 1-D residuals y of shape (n, 1)."""
    return np.abs(np.asarray(y)[..., 0])


def _box_score(x, y, q):
    """s_j(x, y, q) = |y_j| - q_j, batched as the max over j (rectangular set)."""
    return np.max(np.abs(np.atleast_2d(y)) - q, axis=1)


def _box_cost(q):
    return float(np.sum(q))


def _box_cost_grad(q):
    return np.ones_like(q)


def _make_box_remmcp(**targets) -> RemMCP:
    return RemMCP(
        score_fn=_box_score,
        cost_fn=_box_cost,
        n_q=2,
        **targets,
        q0=np.ones(2),
        cost_grad_fn=_box_cost_grad,
        bounds=[(0.0, None), (0.0, None)],
    )


def _gaussian_interval_coverage(q: float) -> float:
    """Exact coverage of {y : |y| <= q} for y ~ N(0, 1)."""
    return 2.0 * norm.cdf(q) - 1.0


def _gaussian_box_coverage(q: np.ndarray) -> float:
    """Exact coverage of {y : |y_j| <= q_j} for y ~ N(0, I)."""
    return float(np.prod(2.0 * norm.cdf(q) - 1.0))


# ===========================================================================
# Constructor targets
# ===========================================================================


@pytest.mark.parametrize(
    "targets",
    [
        {},  # nothing given
        {"epsilon": 0.1},  # beta missing
        {"beta": 0.1},  # epsilon missing
        {"alpha": 0.1, "epsilon": 0.1, "beta": 0.1},  # both guarantees
        {"alpha": 1.5},  # out of range
        {"epsilon": 0.1, "beta": 0.0},  # out of range
    ],
)
def test_invalid_targets_raise(targets):
    with pytest.raises(ValueError):
        SCP(_abs_score, **targets)
    with pytest.raises(ValueError):
        _make_box_remmcp(**targets)


def test_guarantee_selected_by_targets():
    expected = SCP(_abs_score, alpha=0.1)
    assert expected.guarantee == "expected"
    assert expected.coverage_bound == pytest.approx(0.9)

    high_conf = SCP(_abs_score, epsilon=0.05, beta=0.2)
    assert high_conf.guarantee == "high_confidence"
    assert high_conf.coverage_bound == pytest.approx(0.95)


def test_hrelmcp_requires_high_confidence_targets():
    with pytest.raises(TypeError):
        HRelMCP(_box_score, _box_cost, n_q=2, alpha=0.1)  # no alpha argument


# ===========================================================================
# Outlier-budget rules
# ===========================================================================


@pytest.mark.parametrize(
    "n_cal, n_q, alpha", [(100, 1, 0.1), (2000, 1, 0.05), (2000, 3, 0.05), (19, 1, 0.05)]
)
def test_outlier_budget_expected_formula(n_cal, n_q, alpha):
    rho = outlier_budget_expected(n_cal, n_q, alpha)
    assert rho == int(np.floor(alpha * (n_cal + 1) / n_q)) - 1
    # rho is the largest budget whose expected-coverage bound still meets alpha
    assert n_q * (rho + 1) / (n_cal + 1) <= alpha + 1e-12
    assert n_q * (rho + 2) / (n_cal + 1) > alpha


@pytest.mark.parametrize(
    "n_cal, n_q, epsilon, beta",
    [(100, 1, 0.1, 0.1), (500, 1, 0.05, 0.5), (2000, 3, 0.05, 0.1), (2000, 6, 0.02, 0.01)],
)
def test_outlier_budget_high_confidence_is_largest_feasible(n_cal, n_q, epsilon, beta):
    rho = outlier_budget_high_confidence(n_cal, n_q, epsilon, beta)
    assert rho >= 0
    assert confidence_bound(rho, n_cal, n_q, epsilon) <= beta
    assert confidence_bound(rho + 1, n_cal, n_q, epsilon) > beta

    # Brute force over all admissible budgets
    feasible = [r for r in range(n_cal // n_q) if confidence_bound(r, n_cal, n_q, epsilon) <= beta]
    assert rho == max(feasible)


def test_outlier_budget_high_confidence_infeasible():
    # With 20 samples, even rho = 0 cannot certify 95 % coverage with 99 % confidence.
    assert outlier_budget_high_confidence(20, 1, 0.05, 0.01) == -1
    with pytest.raises(ValueError, match="No valid outlier budget"):
        SCP(_abs_score, epsilon=0.05, beta=0.01).calibrate(None, np.zeros((20, 1)))


def test_confidence_bound_increasing_in_rho():
    values = [confidence_bound(r, 1000, 2, 0.05) for r in range(40)]
    assert np.all(np.diff(values) >= 0)


# ===========================================================================
# Calibrated predictors
# ===========================================================================


def test_scp_high_confidence_calibration():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((500, 1))
    predictor = SCP(_abs_score, epsilon=0.05, beta=0.1).calibrate(None, Y)

    assert predictor.rho == outlier_budget_high_confidence(500, 1, 0.05, 0.1)
    assert int(predictor.outlier_mask.sum()) == predictor.rho
    assert predictor.coverage_confidence() >= 0.9
    # the expected-coverage bound holds for any rho
    assert predictor.expected_coverage_lower_bound() == pytest.approx(1 - (predictor.rho + 1) / 501)


def test_remmcp_high_confidence_calibration():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((400, 2))
    predictor = _make_box_remmcp(epsilon=0.1, beta=0.1).calibrate(np.zeros((400, 1)), Y)

    assert predictor.rho == outlier_budget_high_confidence(400, 2, 0.1, 0.1)
    assert predictor.coverage_confidence() >= 0.9
    active = ~predictor.outlier_mask
    assert np.all(_box_score(None, Y[active], predictor.q_opt) <= 1e-6)


def test_coverage_confidence_needs_epsilon_for_expected_predictor():
    rng = np.random.default_rng(2)
    predictor = SCP(_abs_score, alpha=0.1).calibrate(None, rng.standard_normal((100, 1)))
    with pytest.raises(ValueError, match="epsilon"):
        predictor.coverage_confidence()
    assert 0.0 <= predictor.coverage_confidence(epsilon=0.1) <= 1.0


# ===========================================================================
# Monte Carlo verification of the guarantees
# ===========================================================================

N_RUNS = 2000


def _scp_coverages(n_cal: int, **targets) -> np.ndarray:
    rng = np.random.default_rng(123)
    coverages = np.empty(N_RUNS)
    for run in range(N_RUNS):
        Y = rng.standard_normal((n_cal, 1))
        predictor = SCP(_abs_score, **targets).calibrate(None, Y)
        coverages[run] = _gaussian_interval_coverage(predictor.q_opt)
    return coverages


def test_scp_expected_guarantee_monte_carlo():
    alpha, n_cal = 0.1, 99
    coverages = _scp_coverages(n_cal, alpha=alpha)
    standard_error = coverages.std(ddof=1) / np.sqrt(N_RUNS)
    assert coverages.mean() >= 1 - alpha - 3 * standard_error


def test_scp_high_confidence_guarantee_monte_carlo():
    epsilon, beta, n_cal = 0.1, 0.1, 200
    coverages = _scp_coverages(n_cal, epsilon=epsilon, beta=beta)
    hit_rate = np.mean(coverages >= 1 - epsilon)
    standard_error = np.sqrt(beta * (1 - beta) / N_RUNS)
    assert hit_rate >= 1 - beta - 3 * standard_error


def test_remmcp_high_confidence_guarantee_monte_carlo():
    epsilon, beta, n_cal, n_runs = 0.1, 0.2, 150, 300
    rng = np.random.default_rng(7)
    hits = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_runs):
            Y = rng.standard_normal((n_cal, 2))
            predictor = _make_box_remmcp(epsilon=epsilon, beta=beta).calibrate(
                np.zeros((n_cal, 1)), Y
            )
            hits += _gaussian_box_coverage(predictor.q_opt) >= 1 - epsilon
    standard_error = np.sqrt(beta * (1 - beta) / n_runs)
    assert hits / n_runs >= 1 - beta - 3 * standard_error


def test_scp_coverage_follows_the_beta_distribution_at_small_beta():
    """With rho from (14), the coverage of SCP is exactly Beta(n - rho, rho + 1) (Fig. 4)."""
    from scipy.stats import beta as beta_dist
    from scipy.stats import kstest

    epsilon, beta, n_cal = 0.05, 1e-6, 500  # the threshold split of SCP-2 in Sec. 4
    rho = outlier_budget_high_confidence(n_cal, 1, epsilon, beta)
    assert rho == 5
    coverages = _scp_coverages(n_cal, epsilon=epsilon, beta=beta)
    assert kstest(coverages, beta_dist(n_cal - rho, rho + 1).cdf).pvalue > 1e-3
    assert coverages.min() >= 1 - epsilon  # P(miss) = 9.2e-7 per run
