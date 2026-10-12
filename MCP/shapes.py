"""Ellipsoidal prediction-set shapes: parameterization, scores, and costs.

Building blocks for the ellipsoidal prediction sets of the experiments
(App. C of the paper):

  - score_2norm: spherical sets of SCP-1.
  - FittedEllipsoids: fixed union of ellipsoids fitted on held-out data
    (split-SCP baselines); SCP calibrates a threshold on its score.
  - Ellipsoid: centered ellipsoid parameterized by a Cholesky factor
    (RemMCP, H-RelMCP-A); score and cost are convex in q.
  - UnionOfEllipsoids: union with jointly optimized factors and centers
    (H-RelMCP-B).

Ellipsoid and UnionOfEllipsoids provide the batched score s(X, Y, q), the
cost J(q), their analytic gradients, box bounds, an initial q, and the set
geometry, i.e. everything RemMCP/H-RelMCP need.  Their scores are norms minus
one, ||L^T (e - c)|| - 1, rather than the squared norm: both define the same
sets, but the H-RelMCP penalty on a violated point then grows linearly, not
quadratically, with its distance from the set, so that a few far outliers do
not dominate the penalty (RemMCP only uses the sign of the score).  The
split-SCP baselines keep the squared scores of Tumu et al. (FittedEllipsoids).

Parameter layout: q stores the lower-triangular entries of L (row-major),
Q = L L^T, and for unions [L_1, c_1, L_2, c_2, ...].

All scores operate on residuals E = Y - f(X); callers holding residuals pass
X=None and Y=E.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Ellipsoids = list[
    tuple[np.ndarray, np.ndarray]
]  # [(Q, center), ...] with {e : (e-c)^T Q (e-c) <= 1}

# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------


def lower_triangular_param_count(d: int) -> int:
    return d * (d + 1) // 2


def ellipsoid_param_count(d: int) -> int:
    """Parameters of one ellipsoid with free center: Cholesky factor + center."""
    return lower_triangular_param_count(d) + d


def lower_triangular_indices(d: int):
    return np.tril_indices(d)


def _diag_param_indices(d: int) -> np.ndarray:
    return np.cumsum(np.arange(1, d + 1)) - 1


def lower_triangular_bounds(d: int) -> list[tuple[float | None, float | None]]:
    """Box bounds for the lower-triangular Cholesky factor.

    Diagonal entries are constrained to (1e-6, None) to ensure positive
    definiteness; off-diagonal entries are unconstrained.
    """
    bounds: list[tuple[float | None, float | None]] = []
    for row in range(d):
        for col in range(row + 1):
            bounds.append((1e-6, None) if row == col else (None, None))
    return bounds


def q_to_lower_triangular_matrix(q: np.ndarray, d: int) -> np.ndarray:
    """Unpack a flat parameter vector of length d(d+1)/2 into a lower-triangular matrix.

    Raises:
        ValueError: If len(q) does not match d.
    """
    q = np.asarray(q, dtype=float).ravel()
    expected = lower_triangular_param_count(d)
    if q.size != expected:
        raise ValueError(f"Expected {expected} lower-triangular params for d={d}, got {q.size}.")
    L = np.zeros((d, d), dtype=float)
    L[lower_triangular_indices(d)] = q
    return L


def identity_lower_triangular_q(d: int) -> np.ndarray:
    """Parameter vector of the identity matrix (unit ball)."""
    q = np.zeros(lower_triangular_param_count(d), dtype=float)
    q[_diag_param_indices(d)] = 1.0
    return q


# Lower bound on ||L^T e|| in the score gradients (the score is not differentiable at e = 0)
_NORM_FLOOR = 1e-12


def _unit_ball_volume(d: int) -> float:
    return float(np.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0))


# ---------------------------------------------------------------------------
# Residual-space convention
# ---------------------------------------------------------------------------


def point_predictor(x: np.ndarray) -> np.ndarray:
    """Identity predictor used for residual-space problems.

    All score functions in this module operate on residuals E = Y - f(X).
    Callers either pass X=None and Y=E, or X as the prediction and Y as the
    ground truth, in which case f is this identity.
    """
    return x


def _residuals(X: np.ndarray | None, Y: np.ndarray) -> np.ndarray:
    """Residuals E = Y - f(X) as a 2-D array (a single sample becomes one row)."""
    return np.atleast_2d(Y if X is None else Y - point_predictor(X))


# ---------------------------------------------------------------------------
# SCP-1: spherical (2-norm) prediction set
# ---------------------------------------------------------------------------


def score_2norm(x: np.ndarray | None, y: np.ndarray) -> np.ndarray | float:
    """Euclidean distance score for SCP-1: s̃(x,y) = ||y - f(x)||_2 (Sec. 4).

    Args:
        x: Input array, or None for residual-space calls.
        y: Target array of shape (n_y,) for a single sample, or (n, n_y)
            for a batch.

    Returns:
        Scalar norm for a single sample, or array of shape (n,) for a batch.
    """
    E = y if x is None else y - point_predictor(x)
    if np.ndim(E) == 1:
        return float(np.linalg.norm(E))
    return np.linalg.norm(E, axis=1)


# ---------------------------------------------------------------------------
# Split-SCP baselines: fixed fitted union of ellipsoids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedEllipsoids:
    """Fixed union of ellipsoids {e : min_k w_k ((e - c_k)^T Q_k (e - c_k) - 1) <= 0}.

    Produced by the shape fits in MCP.fitting and used as the score of the
    split-SCP baselines: SCP calibrates a threshold q on score(), so the
    calibrated set is the union of {e : (e - c_k)^T Q_k (e - c_k) <= 1 + q / w_k}.
    The weights w_k are the normalizing constants of Tumu et al. (2024,
    Eq. 9); with unit weights all ellipsoids are inflated alike.

    Unlike Ellipsoid and UnionOfEllipsoids, the component scores keep the
    squared form of Tumu et al.: SCP only ranks the scores, so for a single
    ellipsoid the form does not matter, but with weights, w_k (d_k - 1) would
    inflate the ellipsoids of a union differently than w_k (d_k^2 - 1).

    Attributes:
        centers: Ellipsoid centers c_k of shape (d,).
        Q: Precision matrices Q_k of shape (d, d).
        weights: Positive score weights w_k; defaults to ones.
    """

    centers: Sequence[np.ndarray]
    Q: Sequence[np.ndarray]
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "centers", tuple(self.centers))
        object.__setattr__(self, "Q", tuple(self.Q))
        if len(self.centers) != len(self.Q):
            raise ValueError("centers and Q must have the same length.")
        weights = (1.0,) * len(self.Q) if self.weights is None else tuple(map(float, self.weights))
        if len(weights) != len(self.Q):
            raise ValueError("weights and Q must have the same length.")
        if not all(w > 0.0 for w in weights):
            raise ValueError("weights must be positive.")
        object.__setattr__(self, "weights", weights)

    @property
    def n_shapes(self) -> int:
        return len(self.centers)

    def component_scores(self, E: np.ndarray) -> np.ndarray:
        """Unweighted scores (e - c_k)^T Q_k (e - c_k) - 1 per ellipsoid, shape (n_shapes, n)."""
        out = np.zeros((self.n_shapes, E.shape[0]))
        for k in range(self.n_shapes):
            diff = E - self.centers[k]
            out[k] = np.einsum("ni,ij,nj->n", diff, self.Q[k], diff) - 1.0
        return out

    def score(self, X: np.ndarray | None, Y: np.ndarray) -> np.ndarray:
        """Union score min_k w_k (component score k), batched over rows; shape (n,)."""
        weights = np.asarray(self.weights)[:, None]
        return np.min(weights * self.component_scores(_residuals(X, Y)), axis=0)

    def scaled(self, factor: float) -> FittedEllipsoids:
        """Same centers and weights with all Q_k multiplied by factor."""
        return FittedEllipsoids(self.centers, [factor * Q for Q in self.Q], self.weights)

    def weighted(self, weights: Sequence[float]) -> FittedEllipsoids:
        """Same ellipsoids with the score weights replaced."""
        return FittedEllipsoids(self.centers, self.Q, weights)

    def ellipsoids(self, threshold: float = 0.0) -> Ellipsoids:
        """Geometry of {e : score(e) <= threshold} as [(Q_k / (1 + threshold / w_k), c_k)].

        Ellipsoids whose margin 1 + threshold / w_k is not positive are empty
        and omitted.
        """
        out: Ellipsoids = []
        for Q, c, w in zip(self.Q, self.centers, np.asarray(self.weights), strict=True):
            margin = 1.0 + float(threshold) / w
            if np.isfinite(margin) and margin > 0.0:
                out.append((np.asarray(Q, dtype=float) / margin, np.asarray(c, dtype=float)))
        return out


# ---------------------------------------------------------------------------
# RemMCP / H-RelMCP-A: centered Cholesky ellipsoid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ellipsoid:
    """Centered ellipsoid {e : ||L^T e|| <= 1}, q = lower-triangular entries of L (App. C.1).

    Score and cost are convex in q, so this family can be used with RemMCP.

    Attributes:
        dim: Output dimension d.
    """

    dim: int

    @property
    def n_q(self) -> int:
        return lower_triangular_param_count(self.dim)

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        return lower_triangular_bounds(self.dim)

    def initial_q(self) -> np.ndarray:
        """Parameters of the unit ball (L = I)."""
        return identity_lower_triangular_q(self.dim)

    def score(self, X, Y, q) -> np.ndarray:
        """s(x, y, q) = ||L^T e|| - 1 = sqrt(e^T (L L^T) e) - 1, batched; shape (n,)."""
        E = _residuals(X, Y)
        L = q_to_lower_triangular_matrix(q, self.dim)
        return np.linalg.norm(E @ L, axis=1) - 1.0

    def score_grad(self, X, Y, q) -> np.ndarray:
        """Gradient of score w.r.t. q, shape (n, n_q).

        With z = L^T e, ds/dL_ab = e_a z_b / ||z|| for every lower-triangular
        entry (a >= b) of L (zero for e = 0, where the score is not
        differentiable).
        """
        E = _residuals(X, Y)
        L = q_to_lower_triangular_matrix(q, self.dim)
        Z = E @ L
        norms = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), _NORM_FLOOR)
        rows, cols = lower_triangular_indices(self.dim)
        return E[:, rows] * Z[:, cols] / norms

    def cost(self, q) -> float:
        """Volume proxy J(q) = -2 sum_i log L_ii = log det (L L^T)^{-1}.

        Returns 1e10 if a diagonal entry of L is not positive.
        """
        diag = np.diag(q_to_lower_triangular_matrix(q, self.dim))
        if np.any(~np.isfinite(diag)) or np.any(diag <= 0.0):
            return 1e10
        return float(-2.0 * np.sum(np.log(diag)))

    def cost_grad(self, q) -> np.ndarray:
        """Gradient of cost w.r.t. q (zeros if a diagonal entry of L is not positive)."""
        q = np.asarray(q, dtype=float).ravel()
        g = np.zeros_like(q, dtype=float)
        diag_idx = _diag_param_indices(self.dim)
        diag = q[diag_idx]
        if np.any(~np.isfinite(diag)) or np.any(diag <= 0.0):
            return g
        g[diag_idx] = -2.0 / diag
        return g

    def ellipsoids(self, q) -> Ellipsoids:
        """Geometry as [(L L^T, 0)]."""
        L = q_to_lower_triangular_matrix(q, self.dim)
        return [(L @ L.T, np.zeros(self.dim))]


# ---------------------------------------------------------------------------
# H-RelMCP-B: union of ellipsoids with free centers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnionOfEllipsoids:
    """Union of n_shapes ellipsoids with free factors and centers (App. C.3).

    q concatenates [lower-triangular entries of L_k, center c_k] per
    ellipsoid.  The score is the minimum over the ellipsoids, which is not
    convex in q, so this family requires H-RelMCP.

    Attributes:
        dim: Output dimension d.
        n_shapes: Number of ellipsoids.
    """

    dim: int
    n_shapes: int

    @property
    def n_q(self) -> int:
        return self.n_shapes * ellipsoid_param_count(self.dim)

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        bounds: list[tuple[float | None, float | None]] = []
        for _ in range(self.n_shapes):
            bounds.extend(lower_triangular_bounds(self.dim))
            bounds.extend([(None, None)] * self.dim)
        return bounds

    def _components(self, q) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(L_k, c_k)] for every ellipsoid."""
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.n_q:
            raise ValueError(f"Expected {self.n_q} parameters, got {q.size}.")
        p, m = ellipsoid_param_count(self.dim), lower_triangular_param_count(self.dim)
        return [
            (
                q_to_lower_triangular_matrix(q[k * p : k * p + m], self.dim),
                q[k * p + m : (k + 1) * p],
            )
            for k in range(self.n_shapes)
        ]

    def score(self, X, Y, q) -> np.ndarray:
        """s(x, y, q) = min_k ||L_k^T (e - c_k)|| - 1, batched; shape (n,)."""
        E = _residuals(X, Y)
        scores = np.full(E.shape[0], np.inf, dtype=float)
        for L, c in self._components(q):
            scores = np.minimum(scores, np.linalg.norm((E - c) @ L, axis=1) - 1.0)
        return scores

    def score_grad(self, X, Y, q) -> np.ndarray:
        """Gradient of score w.r.t. q, shape (n, n_q): that of the active (minimal) ellipsoid.

        With u = e - c_k and z = L_k^T u, ds/d(L_k)_ab = u_a z_b / ||z|| and
        ds/dc_k = -L_k z / ||z||.
        """
        E = _residuals(X, Y)
        components = self._components(q)
        p, m = ellipsoid_param_count(self.dim), lower_triangular_param_count(self.dim)
        tri = lower_triangular_indices(self.dim)

        vals = np.empty((self.n_shapes, E.shape[0]), dtype=float)
        for k, (L, c) in enumerate(components):
            vals[k] = np.linalg.norm((E - c) @ L, axis=1)
        active = np.argmin(vals, axis=0)

        grads = np.zeros((E.shape[0], self.n_q), dtype=float)
        for k, (L, c) in enumerate(components):
            mask = active == k
            if not np.any(mask):
                continue
            u = E[mask] - c
            z = u @ L
            norms = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), _NORM_FLOOR)
            g = np.zeros((u.shape[0], self.n_q), dtype=float)
            g_full = u[:, :, None] * z[:, None, :] / norms[:, :, None]
            g[:, k * p : k * p + m] = g_full[:, tri[0], tri[1]]
            g[:, k * p + m : (k + 1) * p] = -(z @ L.T) / norms
            grads[mask] = g
        return grads

    def cost(self, q) -> float:
        """Total volume J(q) = sum_k vol(unit ball) / prod_i (L_k)_ii.

        A degenerate ellipsoid (non-positive diagonal of L_k) contributes 1e10.
        """
        unit_volume = _unit_ball_volume(self.dim)
        total = 0.0
        for L, _ in self._components(q):
            prod = float(np.prod(np.diag(L)))
            if np.isfinite(prod) and prod > 0:
                total += unit_volume / max(prod, 1e-30)
            else:
                total += 1e10
        return total

    def cost_grad(self, q) -> np.ndarray:
        """Gradient of cost w.r.t. q; only the diagonal entries of each L_k contribute."""
        p = ellipsoid_param_count(self.dim)
        diag_idx = _diag_param_indices(self.dim)
        unit_volume = _unit_ball_volume(self.dim)
        g = np.zeros(self.n_q, dtype=float)
        for k, (L, _) in enumerate(self._components(q)):
            diag = np.diag(L)
            prod = float(np.prod(diag))
            if np.isfinite(prod) and prod > 0:
                g[k * p + diag_idx] = -(unit_volume / max(prod, 1e-30)) / diag
        return g

    def ellipsoids(self, q) -> Ellipsoids:
        """Geometry as [(L_k L_k^T, c_k)] for every ellipsoid in the union."""
        return [(L @ L.T, np.asarray(c, dtype=float)) for L, c in self._components(q)]
