"""Split Conformal Prediction (SCP).

Implementation of the SCP framework described in Section 2.2 of:
  "Multi-Variable Conformal Prediction"

SCP is a special case of RemMCP (Corollary 1) obtained by setting
    s(x, y, q) = s̃(x, y) - q,    J(q) = q,
where s̃ is a scalar nonconformity score.  The calibration stage then
reduces to a single closed-form quantile computation — no optimisation needed.

Notation follows the paper:
  n_cal   ... number of calibration samples
  rho     ... outlier budget  (Sec. 2.2)
  alpha   ... target miscoverage of the expected guarantee
  epsilon ... target miscoverage of the high-confidence guarantee
  beta    ... target misconfidence of the high-confidence guarantee
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from MCP.CP import CP, Array, _require

ScoreFn = Callable[[Array | None, Array], Array | float]  # s̃(X, Y) -> R^n (batched over rows)


class SCP(CP):
    q_opt: float | None
    """Split Conformal Predictor.

    Implements the calibration procedure in Section 2.2 of the paper.
    After calling ``calibrate()``, the object stores the calibrated threshold
    ``q_opt`` = q̃_rho.

    SCP is recovered from RemMCP (Corollary 1) by setting
        s(x, y, q) = s̃(x, y) - q   and   J(q) = q.

    Args:
        score_fn: Scalar nonconformity score s̃(x, y), evaluated row-wise:
            for X of shape (n, n_x) and Y of shape (n, n_y) it must return
            an array of shape (n,).  Larger values indicate worse
            conformity.  Common choice: s̃(x, y) = ||f(x) - y||_p.
        alpha: Target miscoverage level in (0, 1) of the expected guarantee
            E[eta] >= 1 - alpha.  Mutually exclusive with epsilon/beta.
        epsilon: Target miscoverage level in (0, 1) of the high-confidence
            guarantee P(eta >= 1 - epsilon) >= 1 - beta.
        beta: Target misconfidence level in (0, 1).

    Attributes:
        q_opt: Calibrated threshold q̃_rho.
        rho: Outlier budget.
        outlier_mask: Boolean array of shape (n_cal,), True for the rho
            largest scores.
        coverage_bound: Target coverage level, 1 - alpha or 1 - epsilon.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        alpha: float | None = None,
        *,
        epsilon: float | None = None,
        beta: float | None = None,
    ) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, beta=beta)
        self.score_fn = score_fn
        self._scores_cal: Array | None = None

    def calibrate(self, X_cal: Array | None, Y_cal: Array) -> SCP:
        """Calibrate via empirical quantile of nonconformity scores.

        Args:
            X_cal: Feature array of shape (n_cal, n_x), or None for
                residual-space problems.
            Y_cal: Target array of shape (n_cal, n_y).

        Returns:
            self

        Raises:
            CalibrationError: If n_cal is too small to achieve any valid rho >= 0.
        """
        X_cal, Y_cal, n_cal = self._validate_cal(X_cal, Y_cal)

        rho = self._outlier_budget(n_cal)  # RemMCP budget with n_q = 1 (Cor. 1)
        self.rho = rho

        scores = np.asarray(self.score_fn(X_cal, Y_cal), dtype=float)
        self._scores_cal = scores

        sorted_idx = np.argsort(scores)
        quantile_pos = n_cal - rho - 1  # 0-indexed
        self.q_opt = float(scores[sorted_idx[quantile_pos]])

        outlier_mask = np.zeros(n_cal, dtype=bool)
        outlier_mask[sorted_idx[quantile_pos + 1 :]] = True
        self.outlier_mask = outlier_mask

        return self

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _scores(self, X: Array | None, Y: Array) -> Array:
        """s̃(x, y) - q* — in the set iff <= 0."""
        return np.asarray(self.score_fn(X, Y), dtype=float) - _require(self.q_opt, "q_opt")

    def _extra_summary_lines(self) -> list[str]:
        rho = _require(self.rho, "rho")
        beta_a = self.n_cal - rho
        beta_b = rho + 1
        return [
            f"  Optimal threshold  q̃*    : {self.q_opt:.6f}",
            f"  Coverage distribution     : Beta({beta_a}, {beta_b})  [Sec. 2.2]",
        ]

    def __repr__(self) -> str:
        return f"SCP({self._target_description()}, calibrated={self.q_opt is not None})"
