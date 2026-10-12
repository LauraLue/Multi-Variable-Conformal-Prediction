"""Multi-Variable Conformal Prediction with Constraint Removal (RemMCP).

Implementation of the RemMCP framework from Sec. 3.1 of:
  "Multi-Variable Conformal Prediction"

The calibration stage solves a cascading scenario program (Sec. 3.1) to find
an optimal parameter vector q* whose prediction sets satisfy either the
expected coverage guarantee E[eta] >= 1 - alpha (exchangeable data) or the
high-confidence guarantee P(eta >= 1 - epsilon) >= 1 - beta (i.i.d. data),
depending on how the outlier budget rho is chosen (Thm. 1).

Notation follows the paper:
  n_cal   ... number of calibration samples
  n_q     ... dimension of parameter vector q
  rho     ... outlier budget  (Sec. 3.1)
  alpha   ... target miscoverage of the expected guarantee
  epsilon ... target miscoverage of the high-confidence guarantee
  beta    ... target misconfidence of the high-confidence guarantee
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from MCP.CP import Array, CostFn, CostGradFn, MultiVarCP, ScoreFn


def _minimise_convex(
    cost_fn: CostFn,
    grad_fn: CostGradFn | None,
    constraints: list[dict],
    q0: Array,
    bounds=None,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> Array:
    """Minimise J(q) subject to a list of scipy-style inequality constraints.

    Args:
        cost_fn: Convex objective J(q).
        grad_fn: Gradient of J, or None to use finite differences.
        constraints: List of scipy SLSQP constraint dicts.
        q0: Initial parameter vector.
        bounds: Optional box constraints passed to scipy.
        tol: Solver convergence tolerance.
        max_iter: Maximum solver iterations.

    Returns:
        Optimal parameter vector.
    """
    jac = grad_fn if grad_fn is not None else "2-point"
    result = minimize(
        fun=cost_fn,
        x0=q0,
        jac=jac,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol},
    )
    if not result.success:
        warnings.warn(
            f"RemMCP optimiser did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=3,
        )
    return result.x


class RemMCP(MultiVarCP):
    """Convex Multi-Variable Conformal Predictor.

    Implements the RemMCP procedure from Sec. 3.1 of the paper.  After calling
    ``calibrate()``, the object stores the calibrated parameter ``q_opt``.

    Args:
        score_fn: Batched score function (X, Y, q) -> Array of shape (n,)
            returning max_j s_j(x_m, y_m, q), **convex in q**; y_m is in the
            prediction set Y(x_m, q) iff the score is <= 0.
        cost_fn: Convex cost J(q) encoding preferences over prediction sets
            (e.g., set volume or width).
        n_q: Dimension of the parameter vector q.
        alpha: Target miscoverage level in (0, 1) of the expected guarantee
            E[eta] >= 1 - alpha.  Mutually exclusive with epsilon/beta.
        epsilon: Target miscoverage level in (0, 1) of the high-confidence
            guarantee P(eta >= 1 - epsilon) >= 1 - beta.
        beta: Target misconfidence level in (0, 1).
        q0: Initial value for q. Defaults to zeros.
        cost_grad_fn: Gradient of J. If None, finite differences are used.
        bounds: Box constraints on q passed to the scipy optimiser.
        tol: Solver convergence tolerance.
        max_iter: Maximum solver iterations per cascading stage.
        support_tol: A constraint counts as binding (support candidate) if
            its score at the stage solution is >= -support_tol.

    Attributes:
        q_opt: Calibrated parameter vector q_rho* of shape (n_q,).
        rho: Outlier budget.
        outlier_mask: Boolean array of shape (n_cal,), True for removed samples.
        coverage_bound: Target coverage level, 1 - alpha or 1 - epsilon.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        cost_fn: CostFn,
        n_q: int,
        alpha: float | None = None,
        *,
        epsilon: float | None = None,
        beta: float | None = None,
        q0: Array | None = None,
        cost_grad_fn: CostGradFn | None = None,
        bounds=None,
        tol: float = 1e-8,
        max_iter: int = 2000,
        support_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            score_fn,
            cost_fn,
            n_q,
            alpha=alpha,
            epsilon=epsilon,
            beta=beta,
            q0=q0,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
        )
        self.cost_grad_fn = cost_grad_fn
        self.support_tol = float(support_tol)

    def calibrate(self, X_cal: Array | None, Y_cal: Array) -> RemMCP:
        """Calibrate via the cascading scenario program (Sec. 3.1).

        At each stage k = 0, ..., rho the constraint set is M_cal \\ R_k.
        The support of each solution, filled up to n_q samples by sample
        index if needed (see _removal_set), is added to R_{k+1} following
        the removal scheme of Sec. 2.1.  The support of
        the final stage is not removed — the total number of removed points
        is n_q * rho (not n_q * (rho+1)), consistent with |R_k| = k * n_q.

        Args:
            X_cal: Feature array of shape (n_cal, n_x), or None for
                residual-space problems.
            Y_cal: Target array of shape (n_cal, n_y).

        Returns:
            self

        Raises:
            CalibrationError: If n_cal is too small for the targets and n_q.
        """
        X_cal, Y_cal, n_cal = self._validate_cal(X_cal, Y_cal)
        self.rho = rho = self._outlier_budget(n_cal)

        q_current = self._initial_q()
        active = np.ones(n_cal, dtype=bool)

        for k in range(rho + 1):
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                warnings.warn(
                    f"No active calibration points at stage k={k}. Stopping early.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            X_act = None if X_cal is None else X_cal[active_idx]
            Y_act = Y_cal[active_idx]

            # scipy 'ineq' convention: fun(q) >= 0, i.e. -score >= 0.  A single
            # vectorised constraint is far cheaper than one per sample.
            constraint = {
                "type": "ineq",
                "fun": lambda q, X=X_act, Y=Y_act: -self._scores_at(q, X, Y),
            }
            q_current = _minimise_convex(
                cost_fn=self.cost_fn,
                grad_fn=self.cost_grad_fn,
                constraints=[constraint],
                q0=q_current,
                bounds=self.bounds,
                tol=self.tol,
                max_iter=self.max_iter,
            )

            # supp(q*_k) is only needed to build R_{k+1}: skip at final stage
            if k < rho:
                scores_active = self._scores_at(q_current, X_act, Y_act)
                active[self._removal_set(active_idx, scores_active, k)] = False

        self.q_opt = q_current
        self.outlier_mask = ~active
        return self

    def _removal_set(self, active_idx: Array, scores_active: Array, k: int) -> Array:
        """Indices of the n_q samples removed after stage k (Romao et al. 2023, Eq. 20-21).

        The support set is taken as the binding constraints (score >=
        -support_tol); a nondegenerate convex program has at most n_q
        support constraints, so only the n_q largest scores are kept if more
        are within the tolerance.  If there are fewer than n_q, the removal
        set is filled with the non-support samples of smallest index, the
        label order of Romao et al. (any fixed order is admissible; the
        calibration data is in random order).
        """
        n_remove = min(self.n_q, active_idx.size)
        order = np.argsort(scores_active)[::-1]  # most binding first
        n_binding = int(np.count_nonzero(scores_active >= -self.support_tol))
        support = order[: min(n_binding, n_remove)]
        if support.size == 0:
            warnings.warn(
                f"No binding constraint within support_tol={self.support_tol:g} at stage "
                f"k={k}; the solver may not have converged.",
                RuntimeWarning,
                stacklevel=3,
            )
        fill = np.setdiff1d(np.arange(active_idx.size), support)[: n_remove - support.size]
        return active_idx[np.concatenate([support, fill])]

    def _extra_summary_lines(self) -> list[str]:
        return self._parameter_summary_lines()

    def __repr__(self) -> str:
        return (
            f"RemMCP(n_q={self.n_q}, {self._target_description()}, "
            f"calibrated={self.q_opt is not None})"
        )
