"""Heuristic Multi-Variable Conformal Prediction with Constraint Relaxation (H-RelMCP).

Implementation of H-RelMCP, the adaptive, heuristic version of RelMCP
(Sec. 3.2), with the calibration of Alg. 1 (App. B) of:
  "Multi-Variable Conformal Prediction"

Given target coverage 1 - epsilon and confidence level 1 - beta, seeks q
such that P(eta(q) >= 1 - epsilon) >= 1 - beta (high-confidence guarantee).
Unlike RelMCP, H-RelMCP selects the penalties adaptively and approximates the
solution complexity d_i heuristically, so Thm. 2 does not cover it: its
miscoverage epsilon_i and misconfidence are nominal values (App. B).

Unlike RemMCP, score and cost functions are not required to be convex.
Calibration uses an outer loop over a penalty parameter phi_i (Alg. 1) and
solves penalized non-convex programs at each iteration, evaluating the
achievable miscoverage epsilon_i a posteriori via the confidence-coverage
tradeoff (20) of Garatti & Campi (2024).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, logsumexp

from MCP.CP import (
    Array,
    CalibrationError,
    CostFn,
    CostGradFn,
    MultiVarCP,
    ScoreFn,
    ScoreGradFn,
    _require,
)

_FD_STEP = 1e-6


def _forward_difference(f: Callable[[Array], float], q: Array) -> Array:
    """Forward-difference gradient of a scalar function, relative step per entry."""
    q = np.asarray(q, dtype=float)
    f0 = float(f(q))
    g = np.zeros_like(q, dtype=float)
    for j in range(q.size):
        h = _FD_STEP * max(1.0, abs(q[j]))
        q_plus = q.copy()
        q_plus[j] += h
        g[j] = (float(f(q_plus)) - f0) / h
    return g


def _log_binom(n, k):
    """log C(n, k) via log-gamma; works elementwise on arrays."""
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def tradeoff_epsilon(beta: float, i: int, d_i: int, n_cal: int) -> float:
    """Solve the confidence-coverage tradeoff (Sec. 3.2) for epsilon_i.

    Finds epsilon_i such that

        beta / (i n_cal) * sum_{j=d_i}^{n_cal-1} C(j, d_i) / C(n_cal, d_i)
            * (1 - epsilon_i)^{-(n_cal - j)} = 1,

    in log space (brentq) to avoid overflow in the binomial sum.  The factor
    1/i is the union bound over the i evaluated penalty parameters.

    Args:
        beta: Total misconfidence budget.
        i: Number of evaluated penalty parameters.
        d_i: Solution complexity.
        n_cal: Number of calibration samples.

    Returns:
        Miscoverage level epsilon_i in [0, 1].
    """
    if d_i < 0 or d_i >= n_cal:
        return 1.0

    log_comb_n_d = _log_binom(n_cal, d_i)
    if not np.isfinite(log_comb_n_d):
        return 1.0

    j_vals = np.arange(d_i, n_cal, dtype=float)
    log_ratios = _log_binom(j_vals, d_i) - log_comb_n_d
    powers = n_cal - j_vals
    log_prefactor = np.log(beta) - np.log(float(i)) - np.log(float(n_cal))

    def g_t(t: float) -> float:
        # g(t) = 0  <=>  tradeoff equation holds for t = 1 - epsilon_i
        if t <= 0.0 or t > 1.0:
            return np.nan
        return float(log_prefactor + logsumexp(log_ratios - powers * np.log(t)))

    g_at_1 = g_t(1.0)
    if g_at_1 > 0.0:
        # Root would require t > 1 (epsilon_i < 0): not feasible.
        return 0.0
    if np.isnan(g_at_1):
        return 1.0

    try:
        t_star = brentq(g_t, 1e-300, 1.0)
        return float(np.clip(1.0 - t_star, 0.0, 1.0))
    except ValueError:
        warnings.warn(
            "Could not solve H-RelMCP tradeoff equation robustly; returning epsilon_i=1.",
            RuntimeWarning,
            stacklevel=3,
        )
        return 1.0


def tradeoff_beta(i: int, d_i: int, n_cal: int, epsilon: float) -> float:
    """Solve the confidence-coverage tradeoff for beta at a given epsilon.

    Inverse of tradeoff_epsilon: returns the total misconfidence beta for
    which epsilon is certified after i evaluated penalty parameters.

    Args:
        i: Number of evaluated penalty parameters.
        d_i: Solution complexity.
        n_cal: Number of calibration samples.
        epsilon: Miscoverage level.

    Returns:
        beta in [0, 1].
    """
    if d_i < 0 or d_i >= n_cal:
        return 1.0
    t = 1.0 - epsilon
    if t <= 0.0:
        return 1.0
    log_t = np.log(t)

    log_part1 = _log_binom(n_cal, d_i) + (n_cal - d_i) * log_t
    j_vals = np.arange(d_i, n_cal, dtype=float)
    log_part2 = logsumexp(_log_binom(j_vals, d_i) + (j_vals - d_i) * log_t)

    log_beta = np.log(float(i)) + np.log(float(n_cal)) + log_part1 - log_part2
    return float(np.clip(np.exp(log_beta), 0.0, 1.0))


@dataclass
class _Iterate:
    """Result of one outer iteration of Alg. 1."""

    i: int
    phi: float
    q: Array
    violation_mask: Array
    d_i: int
    epsilon_i: float


class HRelMCP(MultiVarCP):
    """Non-Convex Multi-Variable Conformal Predictor.

    Args:
        score_fn: Batched score function (X, Y, q) -> Array of shape (n,)
            returning max_j s_j(x_m, y_m, q). No convexity is required in q.
        cost_fn: Cost function J(q). No convexity is required.
        n_q: Dimension of the parameter vector q.
        epsilon: Target miscoverage level in (0, 1).
        beta: Target misconfidence level in (0, 1).
        q0: Initial value for q. Defaults to zeros.
        bounds: Box constraints on q passed to the scipy optimizer.
        score_grad_fn: Gradient of score_fn with respect to q, (X, Y, q) ->
            Array of shape (n, n_q). If None, finite differences are used.
        cost_grad_fn: Gradient of cost_fn. If None, finite differences are used.
        phi0: Initial penalty parameter phi_1 > 0.
        phi_increase: Multiplicative factor (> 1) when target coverage is
            not met.
        phi_decrease: Multiplicative factor in (0, 1) when seeking a less
            conservative solution after strongly exceeding target coverage.
        max_outer_iter: Soft iteration limit i_soft of Alg. 1.  Once a valid
            solution exists, the algorithm returns at the latest at this
            iteration; usually it stops earlier, when the next iteration could
            no longer certify the best valid solution.  Without a valid
            solution, the penalty increase accelerates beyond i_soft.
        max_total_iter: Maximum number of iterations i_max >= i_soft of
            Alg. 1.  If no valid solution is found within i_max iterations,
            calibration fails with CalibrationError.
        n_restarts: Number of starting points of the inner optimization:
            the best previous solution, q0, and perturbations of both (see
            _solve_inner_problem for the selection among their solutions).
        restart_scale: Standard deviation of Gaussian perturbations for
            restarts.
        tol: Inner optimizer tolerance.
        max_iter: Inner optimizer maximum iterations.
        random_state: RNG seed for restart perturbations.

    Attributes:
        q_opt: Calibrated parameter vector.
        i_opt: Accepted outer iteration index.
        n_iterations: Number of outer iterations, i.e. evaluated penalties.
        phi_opt: Penalty parameter at the accepted iteration.
        epsilon_i_opt: Certified miscoverage of the accepted solution: the
            tradeoff (20) with n_eval = n_iterations, since the union
            bound covers all evaluated penalties, also those after i_opt.
        d_opt: Solution complexity estimate d_i at the accepted iteration.
        n_violations: Number of calibration points violated by q_opt.
        outlier_mask: Boolean array of shape (n_cal,), True for violated points.
        beta_i_opt: Per-evaluation confidence budget beta / n_iterations.
        confidence_final: Certified confidence of coverage >= 1 - epsilon
            after all n_iterations evaluations.
        rho: Always None; H-RelMCP has no outlier budget.  Its certificate
            depends on d_opt and n_iterations instead.
    """

    epsilon: float
    beta: float

    def __init__(
        self,
        score_fn: ScoreFn,
        cost_fn: CostFn,
        n_q: int,
        epsilon: float,
        beta: float,
        *,
        q0: Array | None = None,
        bounds=None,
        score_grad_fn: ScoreGradFn | None = None,
        cost_grad_fn: CostGradFn | None = None,
        phi0: float = 0.1,
        phi_increase: float = 2.0,
        phi_decrease: float = 0.3,
        max_outer_iter: int = 100,
        max_total_iter: int = 200,
        n_restarts: int = 5,
        restart_scale: float = 0.25,
        tol: float = 1e-8,
        max_iter: int = 2000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            score_fn,
            cost_fn,
            n_q,
            epsilon=epsilon,
            beta=beta,
            q0=q0,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
        )
        if phi0 <= 0.0:
            raise ValueError(f"phi0 must be > 0, got {phi0}.")
        if phi_increase <= 1.0:
            raise ValueError(f"phi_increase must be > 1, got {phi_increase}.")
        if not 0.0 < phi_decrease < 1.0:
            raise ValueError(f"phi_decrease must be in (0, 1), got {phi_decrease}.")
        if max_outer_iter < 1:
            raise ValueError(f"max_outer_iter must be >= 1, got {max_outer_iter}.")
        if max_total_iter < max_outer_iter:
            raise ValueError(
                f"max_total_iter must be >= max_outer_iter={max_outer_iter}, got {max_total_iter}."
            )
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}.")

        self.score_grad_fn = score_grad_fn
        self.cost_grad_fn = cost_grad_fn
        self.phi0 = float(phi0)
        self.phi_increase = float(phi_increase)
        self.phi_decrease = float(phi_decrease)
        self.max_outer_iter = int(max_outer_iter)
        self.max_total_iter = int(max_total_iter)
        self.n_restarts = int(n_restarts)
        self.restart_scale = float(restart_scale)
        self._rng = np.random.default_rng(random_state)

        # Populated after calibrate()
        self.i_opt: int | None = None
        self.n_iterations: int | None = None
        self.phi_opt: float | None = None
        self.epsilon_i_opt: float | None = None
        self.d_opt: int | None = None
        self.n_violations: int | None = None
        self.beta_i_opt: float | None = None
        self.confidence_final: float | None = None

    # ------------------------------------------------------------------
    # Calibration (Alg. 1)
    # ------------------------------------------------------------------

    def calibrate(self, X_cal: Array | None, Y_cal: Array) -> HRelMCP:
        """Calibrate with an iterative non-convex penalty approach (Alg. 1).

        Each outer iteration solves the penalized program (Sec. 3.2):

          min_q J(q) + phi_i * sum_m xi_m(q),
          xi_m(q) := max(0, max_j s_j(x_m, y_m, q)).

        Then computes d_i (solution complexity, Sec. 3.2) and epsilon_i from
        the confidence-coverage tradeoff, and updates phi_i following the
        bisection-like rule in Alg. 1 (line 16).

        Args:
            X_cal: Feature array of shape (n_cal, n_x), or None for
                residual-space problems.
            Y_cal: Target array of shape (n_cal, n_y).

        Returns:
            self

        Raises:
            CalibrationError: If all calibration constraints are satisfied but
                the tradeoff still cannot certify the target coverage, or if
                no valid solution is found within max_total_iter iterations.
        """
        X_cal, Y_cal, n_cal = self._validate_cal(X_cal, Y_cal)
        q0 = self._initial_q()
        q_current = q0
        best_valid: _Iterate | None = None
        best_invalid: _Iterate | None = None

        i, phi = 1, self.phi0
        while True:
            iterate = self._run_iteration(i, phi, q0, q_current, X_cal, Y_cal)
            self.n_iterations = i

            if iterate.epsilon_i <= self.epsilon:
                best_valid = iterate
            elif not iterate.violation_mask.any():
                raise CalibrationError(
                    f"H-RelMCP satisfies all {n_cal} calibration constraints but only "
                    f"certifies epsilon_i={iterate.epsilon_i:.4f} > epsilon={self.epsilon}. "
                    "Increase n_cal, epsilon, or beta, or reduce n_q."
                )
            else:
                best_invalid = iterate

            if best_valid is not None:
                epsilon_next = tradeoff_epsilon(self.beta, i + 1, best_valid.d_i, n_cal)
                if epsilon_next > self.epsilon or i >= self.max_outer_iter:
                    self._accept(best_valid, n_cal)
                    return self
            elif i >= self.max_total_iter:
                raise CalibrationError(
                    f"H-RelMCP found no certified solution within max_total_iter={i} "
                    f"iterations; the last one certifies only epsilon_i="
                    f"{iterate.epsilon_i:.4f} > epsilon={self.epsilon}."
                )

            phi = self._next_penalty(i, best_valid, best_invalid)
            best = best_valid if best_valid is not None else best_invalid
            assert best is not None  # each iteration is valid, invalid, or raises
            q_current = np.array(best.q, dtype=float)
            i += 1

    def _run_iteration(
        self, i: int, phi: float, q0: Array, q_start: Array, X_cal: Array | None, Y_cal: Array
    ) -> _Iterate:
        """Solve the penalized program for phi and certify its solution."""
        q_i = self._solve_inner_problem(q0, q_start, phi, X_cal, Y_cal)
        # Violated means not covered by the prediction set (see covers()).
        violation_mask = self._scores_at(q_i, X_cal, Y_cal) > self.cover_tol
        d_i = self._estimate_d_i(violation_mask)
        epsilon_i = tradeoff_epsilon(self.beta, i, d_i, Y_cal.shape[0])
        return _Iterate(
            i=i,
            phi=phi,
            q=np.array(q_i, dtype=float),
            violation_mask=violation_mask,
            d_i=d_i,
            epsilon_i=epsilon_i,
        )

    def _next_penalty(
        self, i: int, best_valid: _Iterate | None, best_invalid: _Iterate | None
    ) -> float:
        """Penalty update rule of Alg. 1 (line 16)."""
        if best_valid is None:
            assert best_invalid is not None
            exponent = max(1, i - self.max_outer_iter)
            phi_next = best_invalid.phi * self.phi_increase**exponent
        elif best_invalid is None:
            phi_next = best_valid.phi * self.phi_decrease
        else:
            # Bisection between best valid and best invalid penalties.
            phi_next = 0.5 * (best_valid.phi + best_invalid.phi)
        return float(max(phi_next, 1e-12))

    def _accept(self, iterate: _Iterate, n_cal: int) -> None:
        """Populate calibrated attributes from the accepted iteration.

        The certificate uses n_eval = n_iterations evaluated penalties, as in Thm. 2.
        Alg. 1 only continues past the accepted iteration if the solution
        stays certified at the next iteration count, so epsilon_i_opt <=
        epsilon holds.
        """
        n_eval = _require(self.n_iterations, "n_iterations")
        self.q_opt = iterate.q
        self.i_opt = iterate.i
        self.phi_opt = iterate.phi
        self.epsilon_i_opt = tradeoff_epsilon(self.beta, n_eval, iterate.d_i, n_cal)
        self.beta_i_opt = self.beta / n_eval
        self.d_opt = iterate.d_i
        self.outlier_mask = iterate.violation_mask
        self.n_violations = int(iterate.violation_mask.sum())
        self.confidence_final = 1 - tradeoff_beta(n_eval, iterate.d_i, n_cal, self.epsilon)

    def _estimate_d_i(self, violation_mask: Array) -> int:
        """Estimate the solution complexity d_i.

        Uses the conservative heuristic
          d_i = (# violated points) + min(n_q, # non-violated points).
        """
        n_cal = violation_mask.size
        n_viol = int(violation_mask.sum())
        d_i = n_viol + min(self.n_q, n_cal - n_viol)
        return int(min(max(d_i, 0), n_cal - 1))

    # ------------------------------------------------------------------
    # Inner penalized problem
    # ------------------------------------------------------------------

    def _restart_points(self, q0: Array, q_start: Array) -> list[Array]:
        """Exactly n_restarts initial points: q_start, q0, then Gaussian
        perturbations alternating between q_start and q0."""
        starts = [q_start, q0][: self.n_restarts]
        for k in range(self.n_restarts - len(starts)):
            base = q_start if k % 2 == 0 else q0
            starts.append(base + self.restart_scale * self._rng.standard_normal(self.n_q))
        return starts

    def _solve_inner_problem(
        self, q0: Array, q_start: Array, phi: float, X_cal: Array | None, Y_cal: Array
    ) -> Array:
        """Solve one penalized calibration subproblem with restarts.

        Among the solutions of all restarts, returns the one with the
        smallest cost J(q) among those that violate at most as many
        calibration points as the solution with the smallest penalized
        objective.  The penalty sums the slacks, whereas the certificate
        (20) only counts the violated points, so the smallest objective
        can belong to a larger set: e.g. a union of ellipsoids in which one
        ellipsoid covers several residual clusters, because this keeps the
        points between the clusters inside, while the other ellipsoids
        shrink to sets that contain hardly any point.  Since d_i only grows
        with the number of violations, the selection never weakens the
        certificate of the subproblem.

        Returns:
            Selected parameter vector.
        """
        q_start = np.array(q_start, dtype=float)
        q0 = np.array(q0, dtype=float)

        def objective(q: Array) -> float:
            return self._penalized_objective(q, phi, X_cal, Y_cal)

        def gradient(q: Array) -> Array:
            return self._penalized_gradient(q, phi, X_cal, Y_cal)

        candidates: list[tuple[float, int, float, Array]] = []  # (objective, violations, cost, q)
        for q_init in self._restart_points(q0, q_start):
            result = minimize(
                fun=objective,
                x0=q_init,
                method="L-BFGS-B",
                jac=gradient,
                bounds=self.bounds,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            q = np.array(result.x, dtype=float)
            n_violations = int(np.count_nonzero(self._scores_at(q, X_cal, Y_cal) > self.cover_tol))
            candidates.append((objective(q), n_violations, float(self.cost_fn(q)), q))
        max_violations = min(candidates, key=lambda c: c[0])[1]
        eligible = [c for c in candidates if c[1] <= max_violations]
        return min(eligible, key=lambda c: c[2])[3]

    def _slack_sum(self, q: Array, X: Array | None, Y: Array) -> float:
        """sum_m xi_m(q) with xi_m(q) = max(0, score_m(q))."""
        return float(np.sum(np.maximum(self._scores_at(q, X, Y), 0.0)))

    def _penalized_objective(self, q: Array, phi: float, X: Array | None, Y: Array) -> float:
        slacks = np.maximum(self._scores_at(q, X, Y), 0.0)
        return float(self.cost_fn(q) + phi * np.sum(slacks))

    def _penalized_gradient(self, q: Array, phi: float, X: Array | None, Y: Array) -> Array:
        q = np.asarray(q, dtype=float)
        if self.cost_grad_fn is not None:
            g_cost = np.asarray(self.cost_grad_fn(q), dtype=float)
        else:
            g_cost = _forward_difference(self.cost_fn, q)

        if self.score_grad_fn is not None:
            violated = self._scores_at(q, X, Y) > 0.0
            grads = np.asarray(self.score_grad_fn(X, Y, q), dtype=float)
            if grads.shape != (Y.shape[0], q.size):
                raise ValueError(
                    f"score_grad_fn must return shape ({Y.shape[0]}, {q.size}); got {grads.shape}."
                )
            g_slacks = np.sum(grads[violated], axis=0) if violated.any() else np.zeros(q.size)
        else:
            g_slacks = _forward_difference(lambda q_: self._slack_sum(q_, X, Y), q)
        return g_cost + float(phi) * g_slacks

    # ------------------------------------------------------------------
    # Guarantees and reporting
    # ------------------------------------------------------------------

    def coverage_confidence(self, epsilon: float | None = None) -> float:
        """Return the certified confidence of coverage >= 1 - epsilon (nominal, App. B).

        Evaluates the confidence-coverage tradeoff for the accepted solution,
        including the union bound over all n_iterations evaluated penalties.

        Args:
            epsilon: Miscoverage level.  Defaults to the target epsilon.

        Returns:
            Confidence as a float in [0, 1].
        """
        self._check_calibrated("coverage_confidence")
        epsilon = self._resolve_epsilon(epsilon)
        return 1.0 - tradeoff_beta(
            _require(self.n_iterations, "n_iterations"),
            _require(self.d_opt, "d_opt"),
            self.n_cal,
            epsilon,
        )

    def expected_coverage_lower_bound(self) -> float:
        """Not available: H-RelMCP only certifies the high-confidence guarantee."""
        raise NotImplementedError(
            "H-RelMCP certifies the high-confidence guarantee only; use coverage_confidence()."
        )

    def _budget_summary_lines(self) -> list[str]:
        return [f"  Violated calibration pts  : {self.n_violations}"]

    def _extra_summary_lines(self) -> list[str]:
        beta_i = _require(self.beta_i_opt, "beta_i_opt")
        epsilon_i = _require(self.epsilon_i_opt, "epsilon_i_opt")
        q_opt = _require(self.q_opt, "q_opt")
        return [
            f"  Parameter dimension  n_q  : {self.n_q}",
            f"  Confidence level 1-beta   : {1.0 - self.beta:.6f}",
            f"  Outer iteration i*        : {self.i_opt} of {self.n_iterations}",
            f"  Evaluation confidence     : {1.0 - beta_i:.6f}",
            f"  Certified epsilon         : {epsilon_i:.6f}",
            f"  Certified coverage        : {1.0 - epsilon_i:.6f}",
            f"  Tradeoff dimension d_i*   : {self.d_opt}",
            f"  Penalty phi*              : {self.phi_opt:.6f}",
            f"  Optimal q*                : "
            f"{np.array2string(q_opt, precision=4, suppress_small=True)}",
        ]

    def __repr__(self) -> str:
        return (
            f"HRelMCP(n_q={self.n_q}, epsilon={self.epsilon}, beta={self.beta}, "
            f"calibrated={self.q_opt is not None})"
        )
