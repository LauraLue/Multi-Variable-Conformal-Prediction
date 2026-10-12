"""Base classes for all conformal predictors.

CP is the abstract base class.  Concrete subclasses must implement:
  calibrate(X_cal, Y_cal)        -- calibration logic
  _scores(X, Y) -> Array         -- batched signed scores, <= 0 iff covered
  _extra_summary_lines()         -- subclass-specific summary rows

The public methods scores(), covers(), predict(), empirical_coverage(), and
summary() are implemented here via the template-method pattern.
MultiVarCP adds the parts shared by RemMCP and H-RelMCP: a parameter vector q,
a batched score function s(X, Y, q), and a cost function J(q).

Two coverage guarantees are supported (Sec. 3 of the paper):
  expected coverage        E[eta(q*)] >= 1 - alpha            (exchangeable data)
  high-confidence coverage P(eta(q*) >= 1 - epsilon) >= 1 - beta   (i.i.d. data)
A predictor targets the expected guarantee if constructed with ``alpha`` and
the high-confidence guarantee if constructed with ``epsilon`` and ``beta``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from scipy.stats import binom

Array = np.ndarray

ScoreFn = Callable[[Array | None, Array, Array], Array]  # (X, Y, q) -> R^n
ScoreGradFn = Callable[[Array | None, Array, Array], Array]  # (X, Y, q) -> R^{n x n_q}
CostFn = Callable[[Array], float]  # q -> R
CostGradFn = Callable[[Array], Array]  # q -> R^{n_q}

EXPECTED = "expected"
HIGH_CONFIDENCE = "high_confidence"


class CalibrationError(ValueError):
    """Raised when no calibration achieves the requested coverage target."""


T = TypeVar("T")


def _require(value: T | None, name: str) -> T:
    """Return a value that is only set by calibrate(), or raise if it is missing."""
    if value is None:
        raise RuntimeError(f"{name} is not available; call calibrate() first.")
    return value


def _check_unit_interval(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1), got {value}.")


def confidence_bound(rho: int, n_cal: int, n_q: int, epsilon: float) -> float:
    """Return beta_bar(rho), the risk of the high-confidence guarantee.

    Evaluates

        beta_bar(rho) = sum_{j=0}^{n_q(rho+1)-1} C(n_cal, j) eps^j (1-eps)^{n_cal-j},

    so that P(eta(q*) >= 1 - epsilon) >= 1 - beta_bar(rho) for RemMCP with
    outlier budget rho (Thm. 1) and for SCP with n_q = 1.  beta_bar is
    monotonically increasing in rho.

    Args:
        rho: Outlier budget.
        n_cal: Number of calibration samples.
        n_q: Dimension of the parameter vector.
        epsilon: Miscoverage level.

    Returns:
        beta_bar(rho) in [0, 1].
    """
    return float(binom.cdf(n_q * (rho + 1) - 1, n_cal, epsilon))


def outlier_budget_expected(n_cal: int, n_q: int, alpha: float) -> int:
    """Outlier budget for the expected coverage guarantee (Sec. 3.1).

    rho = floor(alpha (n_cal + 1) / n_q) - 1.  A negative value means that no
    valid budget exists.
    """
    return int(np.floor(alpha * (n_cal + 1) / n_q)) - 1


def outlier_budget_high_confidence(n_cal: int, n_q: int, epsilon: float, beta: float) -> int:
    """Outlier budget for the high-confidence guarantee (Sec. 3.1).

    rho = max{rho' in N_0 : beta_bar(rho') <= beta}.  Returns -1 if even
    rho = 0 violates the target, i.e. no valid budget exists.
    """
    rho_max = n_cal // n_q - 1  # largest rho with n_q (rho + 1) <= n_cal
    if rho_max < 0:
        return -1
    rhos = np.arange(rho_max + 1)
    beta_bar = binom.cdf(n_q * (rhos + 1) - 1, n_cal, epsilon)
    return int(np.count_nonzero(beta_bar <= beta)) - 1


class CP(ABC):
    """Abstract base class for conformal predictors.

    Exactly one of the following must be given:
      alpha            -- target miscoverage of the expected guarantee, or
      epsilon and beta -- target miscoverage and misconfidence of the
                          high-confidence guarantee.

    Args:
        alpha: Target miscoverage level in (0, 1) for E[eta] >= 1 - alpha.
        epsilon: Target miscoverage level in (0, 1) for
            P(eta >= 1 - epsilon) >= 1 - beta.
        beta: Target misconfidence level in (0, 1).

    Attributes:
        guarantee: Either "expected" or "high_confidence".
        alpha, epsilon, beta: Target levels; unused ones are None.
        n_q: Dimension of the calibrated parameter (1 for SCP).
        q_opt: Calibrated parameter (Array for RemMCP/H-RelMCP, float for SCP).
        rho: Outlier budget (None for H-RelMCP, which removes no samples).
        outlier_mask: Boolean array of shape (n_cal,), True for samples that
            are not covered by construction (removed or violated).
        coverage_bound: Target coverage level, 1 - alpha or 1 - epsilon.
    """

    alpha: float | None
    epsilon: float | None
    beta: float | None
    cover_tol: float = 0.0
    q_opt: Array | float | None
    rho: int | None
    outlier_mask: Array | None
    _X_cal: Array | None
    _Y_cal: Array | None

    def __init__(
        self,
        alpha: float | None = None,
        epsilon: float | None = None,
        beta: float | None = None,
    ) -> None:
        if alpha is not None:
            if epsilon is not None or beta is not None:
                raise ValueError(
                    "Specify either alpha (expected coverage) or epsilon and beta "
                    "(high-confidence coverage), not both."
                )
            _check_unit_interval("alpha", alpha)
            self.guarantee = EXPECTED
            self.coverage_bound = 1.0 - alpha
        elif epsilon is not None and beta is not None:
            _check_unit_interval("epsilon", epsilon)
            _check_unit_interval("beta", beta)
            self.guarantee = HIGH_CONFIDENCE
            self.coverage_bound = 1.0 - epsilon
        else:
            raise ValueError(
                "Specify alpha for the expected coverage guarantee, or epsilon and "
                "beta for the high-confidence guarantee."
            )
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.n_q = 1

        # Populated by calibrate()
        self.q_opt = None
        self.rho = None
        self.outlier_mask = None
        self._X_cal = None
        self._Y_cal = None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def calibrate(self, X_cal: Array | None, Y_cal: Array) -> CP:
        """Calibrate the predictor from a held-out calibration set."""

    @abstractmethod
    def _scores(self, X: Array | None, Y: Array) -> Array:
        """Return signed scores of shape (n,); a sample is covered iff its score <= 0."""

    @abstractmethod
    def _extra_summary_lines(self) -> list[str]:
        """Subclass-specific lines appended after the common summary block."""

    # ------------------------------------------------------------------
    # Helpers for subclass calibrate() methods
    # ------------------------------------------------------------------

    def _validate_cal(self, X_cal: Array | None, Y_cal: Array) -> tuple[Array | None, Array, int]:
        """Validate, coerce, and store calibration data.

        Args:
            X_cal: Feature array of shape (n_cal, n_x), or None for
                residual-space problems.
            Y_cal: Target array of shape (n_cal, n_y).

        Returns:
            Tuple of (X_cal, Y_cal, n_cal).

        Raises:
            ValueError: If X_cal and Y_cal have different numbers of rows.
        """
        Y_cal = np.atleast_2d(Y_cal)
        if X_cal is not None and X_cal.shape[0] != Y_cal.shape[0]:
            raise ValueError("X_cal and Y_cal must have the same number of rows.")
        self._X_cal = X_cal
        self._Y_cal = Y_cal
        return X_cal, Y_cal, Y_cal.shape[0]

    def _outlier_budget(self, n_cal: int) -> int:
        """Outlier budget rho for the targeted guarantee (Sec. 3.1).

        Raises:
            ValueError: If no rho >= 0 achieves the target.
        """
        if self.guarantee == EXPECTED:
            assert self.alpha is not None
            rho = outlier_budget_expected(n_cal, self.n_q, self.alpha)
            target = f"alpha={self.alpha}"
        else:
            assert self.epsilon is not None and self.beta is not None
            rho = outlier_budget_high_confidence(n_cal, self.n_q, self.epsilon, self.beta)
            target = f"epsilon={self.epsilon}, beta={self.beta}"
        if rho < 0:
            raise CalibrationError(
                f"No valid outlier budget for n_cal={n_cal}, n_q={self.n_q}, {target}. "
                "Increase n_cal or reduce n_q."
            )
        return rho

    def _check_calibrated(self, method: str) -> None:
        if self.q_opt is None:
            raise RuntimeError(f"Call calibrate() before {method}().")

    @property
    def n_cal(self) -> int:
        """Number of calibration samples (available after calibrate())."""
        return _require(self._Y_cal, "n_cal").shape[0]

    # ------------------------------------------------------------------
    # Public API — fully implemented using the abstract methods above
    # ------------------------------------------------------------------

    def scores(self, X: Array | None, Y: Array) -> Array:
        """Signed scores of the calibrated prediction set, batched over rows.

        For SCP the score is s̃(x, y) - q*; for RemMCP/H-RelMCP it is
        max_j s_j(x, y, q*).  In both cases y is in the prediction set for x
        iff the score is <= 0.

        Args:
            X: Inputs of shape (n, n_x), or None for residual-space problems.
            Y: Outputs of shape (n, n_y); a single output (n_y,) is treated
                as a batch of one.

        Returns:
            Score array of shape (n,).

        Raises:
            RuntimeError: If calibrate() has not been called.
        """
        self._check_calibrated("scores")
        X = None if X is None else np.atleast_2d(X)
        return self._scores(X, np.atleast_2d(Y))

    def covers(self, X: Array | None, Y: Array) -> Array:
        """Boolean array of shape (n,): True iff Y[m] is in the set for X[m].

        A sample counts as covered if its score is <= cover_tol (0 for SCP,
        see MultiVarCP for RemMCP/H-RelMCP).
        """
        return self.scores(X, Y) <= self.cover_tol

    def predict(self, x: Array | None, Y_candidates: Array) -> Array:
        """Check set membership of several candidate outputs for one input.

        Args:
            x: Input array of shape (n_x,), or None for residual-space problems.
            Y_candidates: Candidate outputs of shape (n_candidates, n_y).

        Returns:
            Boolean array of shape (n_candidates,).
        """
        Y_candidates = np.atleast_2d(Y_candidates)
        X = None if x is None else np.repeat(np.atleast_2d(x), len(Y_candidates), axis=0)
        return self.covers(X, Y_candidates)

    def empirical_coverage(self, X_test: Array | None, Y_test: Array) -> float:
        """Fraction of test points covered by their prediction set."""
        return float(self.covers(X_test, Y_test).mean())

    def summary(self) -> str:
        """Human-readable calibration summary."""
        if self.q_opt is None:
            return f"{type(self).__name__}: not yet calibrated (call calibrate() first)."
        lines = [
            f"{type(self).__name__} calibration summary",
            "=" * 40,
            f"  Calibration samples n_cal : {self.n_cal}",
            f"  Guarantee                 : {self._target_description()}",
            *self._budget_summary_lines(),
            f"  Target coverage           : {self.coverage_bound:.4f}",
        ] + self._extra_summary_lines()
        return "\n".join(lines)

    def _budget_summary_lines(self) -> list[str]:
        n_removed = int(_require(self.outlier_mask, "outlier_mask").sum())
        return [
            f"  Outlier budget       rho  : {self.rho}",
            f"  Total outliers removed    : {n_removed}",
        ]

    def _target_description(self) -> str:
        if self.guarantee == EXPECTED:
            return f"E[coverage] >= 1 - alpha, alpha = {self.alpha}"
        return (
            f"P(coverage >= 1 - epsilon) >= 1 - beta, epsilon = {self.epsilon}, beta = {self.beta}"
        )

    def expected_coverage_lower_bound(self) -> float:
        """Return the guaranteed lower bound on expected coverage.

        Evaluates 1 - n_q (rho + 1) / (n_cal + 1) from the bound
        E[V(q*)] <= n_q (rho + 1) / (n_cal + 1) on the expected violation
        probability (Sec. 2.1; Thm. 1(a)).  Holds for any rho, i.e. also when
        the predictor targets the high-confidence guarantee.

        Returns:
            Lower bound on expected coverage.

        Raises:
            RuntimeError: If calibrate() has not been called.
        """
        self._check_calibrated("expected_coverage_lower_bound")
        rho = _require(self.rho, "rho")
        return float(1 - self.n_q * (rho + 1) / (self.n_cal + 1))

    def coverage_confidence(self, epsilon: float | None = None) -> float:
        """Return the guaranteed confidence 1 - beta_bar(rho) of coverage >= 1 - epsilon.

        This is the high-confidence part of Thm. 1 (Thm. 1(b)); for SCP
        (n_q = 1) it follows from the Beta coverage distribution (Sec. 2.2).

        Args:
            epsilon: Miscoverage level.  Defaults to the target epsilon of a
                high-confidence predictor; must be given for predictors that
                target the expected guarantee.

        Returns:
            Confidence as a float in [0, 1].

        Raises:
            RuntimeError: If calibrate() has not been called.
            ValueError: If epsilon is missing for an expected-coverage predictor.
        """
        self._check_calibrated("coverage_confidence")
        epsilon = self._resolve_epsilon(epsilon)
        return 1.0 - confidence_bound(_require(self.rho, "rho"), self.n_cal, self.n_q, epsilon)

    def _resolve_epsilon(self, epsilon: float | None) -> float:
        if epsilon is None:
            if self.epsilon is None:
                raise ValueError(
                    "This predictor targets the expected guarantee (alpha); pass "
                    "the miscoverage level epsilon explicitly."
                )
            return self.epsilon
        _check_unit_interval("epsilon", epsilon)
        return epsilon


class MultiVarCP(CP):
    """Shared base class of RemMCP and H-RelMCP.

    Args:
        score_fn: Batched score function (X, Y, q) -> Array of shape (n,)
            returning max_j s_j(x_m, y_m, q) for every sample m, so that y_m
            is in the prediction set Y(x_m, q) iff the score is <= 0.  X may
            be None for residual-space problems.
        cost_fn: Cost J(q) encoding preferences over prediction sets.
        n_q: Dimension of the parameter vector q.
        alpha, epsilon, beta: Coverage targets, see CP.
        q0: Initial value for q. Defaults to zeros.
        bounds: Box constraints on q passed to the scipy optimiser.
        tol: Solver convergence tolerance.
        max_iter: Maximum solver iterations.
        cover_tol: A sample counts as covered if its score is <= cover_tol.
            The numerical solvers return the binding calibration constraints
            only up to round-off (scores of about 1e-12 to 1e-9), so with a
            zero tolerance these points, which are covered by construction,
            would count as uncovered.
    """

    q_opt: Array | None

    def __init__(
        self,
        score_fn: ScoreFn,
        cost_fn: CostFn,
        n_q: int,
        *,
        alpha: float | None = None,
        epsilon: float | None = None,
        beta: float | None = None,
        q0: Array | None = None,
        bounds=None,
        tol: float = 1e-8,
        max_iter: int = 2000,
        cover_tol: float = 1e-6,
    ) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, beta=beta)
        if n_q < 1:
            raise ValueError(f"n_q must be >= 1, got {n_q}.")
        if cover_tol < 0:
            raise ValueError(f"cover_tol must be >= 0, got {cover_tol}.")
        self.cover_tol = float(cover_tol)
        self.score_fn = score_fn
        self.cost_fn = cost_fn
        self.n_q = n_q
        self._q0 = q0
        self.bounds = bounds
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _initial_q(self) -> Array:
        if self._q0 is None:
            return np.zeros(self.n_q)
        return np.array(self._q0, dtype=float)

    def _scores_at(self, q: Array, X: Array | None, Y: Array) -> Array:
        """Evaluate score_fn at parameter q and check its output shape."""
        scores = np.asarray(self.score_fn(X, Y, q), dtype=float)
        if scores.shape != (Y.shape[0],):
            raise ValueError(
                f"score_fn must return one score per sample, i.e. shape ({Y.shape[0]},); "
                f"got {scores.shape}."
            )
        return scores

    def _scores(self, X: Array | None, Y: Array) -> Array:
        return self._scores_at(_require(self.q_opt, "q_opt"), X, Y)

    def _parameter_summary_lines(self) -> list[str]:
        return [
            f"  Parameter dimension  n_q  : {self.n_q}",
            f"  Optimal q*                : "
            f"{np.array2string(_require(self.q_opt, 'q_opt'), precision=4, suppress_small=True)}",
        ]
