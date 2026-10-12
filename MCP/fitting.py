"""Fit ellipsoidal set shapes on held-out data for the SCP baselines.

  - fit_ellipsoid_kmeans: k-means clusters + sample covariances (suffix A/B).
  - fit_ellipsoid_tumu: minimum-volume enclosing ellipsoids via CMA-ES
    (suffix C, Tumu et al. 2024).
  - init_q0_from_fit_ellipsoid: initial H-RelMCP-B parameters from a fit.

All fits return a FittedEllipsoids normalized on the fitting data with one
of two normalizations (alpha is the miscoverage level of the expected
coverage guarantee targeted by the split-SCP baselines):

  - "tumu" (default): each shape score is weighted by the normalizing
    constant of Tumu et al. (2024, Eq. 9), see apply_tumu_normalization.
  - "common": all Q_k are scaled by one factor so that a fraction 1 - alpha
    of the fitting data lies inside the union (apply_normalizing_constants).
"""

from __future__ import annotations

from typing import Literal

import cma
import numpy as np
from sklearn.cluster import KMeans

from MCP.shapes import (
    FittedEllipsoids,
    ellipsoid_param_count,
    lower_triangular_indices,
    lower_triangular_param_count,
)

Normalization = Literal["common", "tumu"]
NORMALIZATIONS: tuple[Normalization, ...] = ("common", "tumu")
COV_REG = 1e-2  # ridge of the k-means covariances, relative to the residual variance

# ---------------------------------------------------------------------------
# k-means + sample covariance (SCP-2A/B and H-RelMCP-B initialization)
# ---------------------------------------------------------------------------


def fit_ellipsoid_kmeans(
    E_fit: np.ndarray,
    alpha: float,
    n_shapes: int = 3,
    min_points_per_cluster: int = 2,
    normalization: Normalization = "tumu",
    cov_reg: float = COV_REG,
) -> FittedEllipsoids:
    """Fit union-of-ellipsoids via KMeans clustering and sample covariance.

    Clusters the observed fit residuals directly.  The resulting ellipsoids
    are anchored to the actual sample geometry and then normalized on E_fit
    (see normalize).  For n_shapes > 1, each cluster covariance is regularized
    as Sigma_k + lambda I with lambda = cov_reg * tr(Sigma) / d, where Sigma is
    the covariance of all of E_fit; otherwise a cluster of one outlier,
    backfilled to min_points_per_cluster points, has a (nearly) singular
    covariance.  For n_shapes = 1, Sigma itself is used.

    Args:
        E_fit: Residual array of shape (n_fit, d) used for fitting.
        alpha: Target miscoverage level of the normalization.
        n_shapes: Number of ellipsoids (clusters).
        min_points_per_cluster: Minimum cluster size; tiny clusters are
            backfilled with the nearest points.
        normalization: "common" or "tumu", see the module docstring.
        cov_reg: Ridge of the cluster covariances, relative to the average
            variance of E_fit.

    Returns:
        FittedEllipsoids.

    Raises:
        ValueError: If E_fit has too few rows.
    """
    n, d = E_fit.shape
    if n < max(n_shapes, min_points_per_cluster):
        raise ValueError(
            f"Not enough fitting samples for kmeans-cov fit: n_fit={n}, "
            f"n_shapes={n_shapes}, min_points_per_cluster={min_points_per_cluster}."
        )

    cov_all = np.atleast_2d(np.cov(E_fit.T))
    ridge = cov_reg * np.trace(cov_all) / d * np.eye(d)
    centers: list[np.ndarray] = []
    q_matrices: list[np.ndarray] = []
    if n_shapes == 1:
        centers.append(np.zeros(d, dtype=float))
        q_matrices.append(np.linalg.inv(cov_all))
    else:
        km = KMeans(n_clusters=n_shapes, n_init=10, random_state=0)
        labels = km.fit_predict(E_fit)
        for k in range(n_shapes):
            pts = E_fit[labels == k]
            if pts.shape[0] < min_points_per_cluster:
                # Backfill tiny clusters with the nearest points so that the
                # sample covariance is defined.
                center_k = km.cluster_centers_[k]
                nearest_idx = np.argsort(np.sum((E_fit - center_k) ** 2, axis=1))[
                    :min_points_per_cluster
                ]
                pts = E_fit[nearest_idx]
            c = pts.mean(axis=0)
            cov = np.atleast_2d(np.cov(pts.T))
            centers.append(c)
            q_matrices.append(np.linalg.inv(cov + ridge))

    return normalize(FittedEllipsoids(centers, q_matrices), E_fit, alpha, normalization)


def init_q0_from_fit_ellipsoid(
    E_fit: np.ndarray,
    alpha: float,
    n_shapes: int = 1,
) -> np.ndarray:
    """Initialise H-RelMCP q0 from fitted SCP-2A ellipsoids.

    Fits KMeans ellipsoids and converts the result into the flattened H-RelMCP
    parameter layout [L_lower_tri, center] per shape.

    Args:
        E_fit: Residual array of shape (n_fit, d).
        alpha: Target miscoverage level for ellipsoid fitting.
        n_shapes: Number of ellipsoids.

    Returns:
        Initial parameter vector of length n_shapes * ellipsoid_param_count(d).
    """
    # H-RelMCP's parameters cannot represent score weights, so scale the Q_k instead.
    fitted = fit_ellipsoid_kmeans(E_fit, alpha=alpha, n_shapes=n_shapes, normalization="common")
    d = E_fit.shape[1]
    p = ellipsoid_param_count(d)
    m = lower_triangular_param_count(d)
    q0 = np.zeros(n_shapes * p, dtype=float)
    for i in range(n_shapes):
        Q = np.asarray(fitted.Q[i], dtype=float)
        c = np.asarray(fitted.centers[i], dtype=float)
        # Factor Q = L @ L.T with lower-triangular L (Cholesky). If Q is
        # numerically singular, add a tiny jitter to ensure PD.
        jitter = 1e-12
        try:
            L = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Q + jitter * np.eye(Q.shape[0]))
        tri = lower_triangular_indices(d)
        base = i * p
        q0[base : base + m] = L[tri]
        q0[base + m : base + p] = c
    return q0


# ---------------------------------------------------------------------------
# CMA-ES minimum-volume ellipsoids (SCP-2C, Tumu et al.)
# ---------------------------------------------------------------------------


def cluster_kmeans(
    E_fit: np.ndarray,
    n_shapes: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run KMeans on E_fit.

    Args:
        E_fit: Residual array of shape (n, d).
        n_shapes: Number of clusters.
        seed: Random seed.

    Returns:
        Tuple of (labels, cluster_centers).
    """
    km = KMeans(n_clusters=n_shapes, n_init=10, random_state=seed)
    labels = km.fit_predict(E_fit)
    return labels, km.cluster_centers_


def fit_cma_raw(
    E_fit: np.ndarray,
    labels: np.ndarray,
    n_shapes: int,
    sigma0: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
) -> FittedEllipsoids:
    """Fit minimum-volume enclosing ellipsoids per cluster using CMA-ES.

    Implements the Tumu et al. objective: minimise D_cp^(d/2) / sqrt(det Q)
    where D_cp = max_i (e_i - c)^T Q (e_i - c) is the enclosing scale.
    After optimisation each Q is divided by D_cp so all cluster points have
    Mahalanobis score <= 1.

    Args:
        E_fit: Residual array of shape (n, d).
        labels: Cluster assignment array of shape (n,).
        n_shapes: Number of clusters.
        sigma0: CMA-ES initial step size.
        max_iter: Maximum CMA-ES iterations per cluster.
        seed: Random seed.

    Returns:
        FittedEllipsoids whose Q matrices
        satisfy (e - c)^T Q (e - c) <= 1 for all cluster points.
    """
    _, d = E_fit.shape
    n_L = d * d
    n_params = n_L + d

    centers: list[np.ndarray] = []
    q_matrices: list[np.ndarray] = []

    for k in range(n_shapes):
        pts_k = E_fit[labels == k]
        if pts_k.shape[0] == 0:
            pts_k = E_fit

        x0 = np.zeros(n_params)
        x0[:n_L] = np.eye(d).reshape(n_L)
        x0[n_L:] = pts_k.mean(axis=0)

        def _obj(x: np.ndarray, pts: np.ndarray = pts_k) -> float:
            L = x[:n_L].reshape(d, d)
            c = x[n_L:]
            Q = L.T @ L
            if np.linalg.matrix_rank(Q) < d:
                return np.nan
            diffs = pts - c
            scores = np.einsum("ni,ij,nj->n", diffs, Q, diffs)
            D_cp = float(np.max(scores))
            if D_cp <= 0.0:
                return np.nan
            det_Q = np.linalg.det(Q)
            if det_Q <= 0.0:
                return np.nan
            return float((D_cp ** (d / 2)) / np.sqrt(det_Q))

        result = cma.fmin2(
            _obj,
            x0,
            sigma0,
            options={"maxiter": max_iter, "verbose": -9, "seed": seed * 10 + k},
        )
        x_best = np.asarray(result[0], dtype=float)

        L = x_best[:n_L].reshape(d, d)
        Q = L.T @ L
        c = x_best[n_L:]
        diffs = pts_k - c
        scores = np.einsum("ni,ij,nj->n", diffs, Q, diffs)
        D_cp = max(float(np.max(scores)), 1e-12)

        centers.append(c)
        q_matrices.append(Q / D_cp)

    return FittedEllipsoids(centers, q_matrices)


def apply_normalizing_constants(
    fitted: FittedEllipsoids, E_fit: np.ndarray, alpha: float
) -> FittedEllipsoids:
    """Scale all Q_k by a common factor so that a fraction 1 - alpha of E_fit lies inside.

    Scaling Q -> scale * Q changes the union score of point i to
    scale * (union_raw[i] + 1) - 1.  Setting the (1 - alpha) quantile of the
    scaled scores to 0 gives scale = 1 / (q_union + 1), where q_union is the
    (1 - alpha) quantile of the unscaled union scores.

    Args:
        fitted: Unscaled ellipsoids.
        E_fit: Residual array used for the coverage quantile.
        alpha: Target miscoverage level.

    Returns:
        Rescaled ellipsoids.
    """
    coverage_target = 1.0 - alpha
    union_raw = np.min(fitted.component_scores(E_fit), axis=0)  # (n_fit,)
    q_union = float(np.quantile(union_raw, coverage_target))
    scale = 1.0 / (q_union + 1.0 + 1e-8)
    return fitted.scaled(scale)


def apply_tumu_normalization(
    fitted: FittedEllipsoids, E_fit: np.ndarray, alpha: float
) -> FittedEllipsoids:
    """Weight each shape score by the normalizing constant of Tumu et al. (2024, Eq. 9).

    With R_k the unweighted scores of shape k on all of E_fit (not only on
    its cluster), the weight is

        w_k = 1 / (Quantile(R_k, 1 - alpha) - min(R_k)),

    as in ConformalRegion.fit of conformal_region_designer, including its
    1e-8 safeguard.  The ellipsoids themselves are not rescaled.

    Args:
        fitted: Unweighted ellipsoids.
        E_fit: Residual array used for the quantiles.
        alpha: Target miscoverage level.

    Returns:
        Weighted ellipsoids.
    """
    scores = fitted.component_scores(E_fit)  # (n_shapes, n_fit)
    spread = np.quantile(scores, 1.0 - alpha, axis=1) - np.min(scores, axis=1)
    return fitted.weighted(1.0 / (1e-8 + spread))


def normalize(
    fitted: FittedEllipsoids, E_fit: np.ndarray, alpha: float, normalization: Normalization
) -> FittedEllipsoids:
    """Normalize fitted ellipsoids on E_fit ("common" or "tumu", see the module docstring)."""
    if normalization == "common":
        return apply_normalizing_constants(fitted, E_fit, alpha)
    if normalization == "tumu":
        return apply_tumu_normalization(fitted, E_fit, alpha)
    raise ValueError(f"Unknown normalization {normalization!r}; choose from {NORMALIZATIONS}.")


def fit_ellipsoid_tumu(
    E_fit: np.ndarray,
    alpha: float,
    n_shapes: int = 3,
    labels: np.ndarray | None = None,
    sigma0: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
    normalization: Normalization = "tumu",
) -> FittedEllipsoids:
    """Fit minimum-volume enclosing ellipsoids per cluster (Tumu et al. / CMA-ES).

    Pipeline: cluster_kmeans → fit_cma_raw → normalize.

    For each cluster k, CMA-ES minimises the volume of the smallest ellipsoid
    enclosing all cluster points.  Parameterisation Q = L^T L (L unconstrained
    d×d) ensures Q is always positive semi-definite.

    Objective (per cluster, see callCMAESMatrixManyEllipse in Tumu et al.):
        min  D_cp^(d/2) / sqrt(det Q),   D_cp = max_i (e_i - c)^T Q (e_i - c)

    Args:
        E_fit: Residual array of shape (n_fit, d).
        alpha: Target miscoverage level for normalisation.
        n_shapes: Number of ellipsoids (clusters).
        labels: Pre-computed cluster labels; computed via KMeans if None.
        sigma0: CMA-ES initial step size.
        max_iter: CMA-ES maximum iterations per cluster.
        seed: Random seed.
        normalization: "common" or "tumu", see the module docstring.

    Returns:
        FittedEllipsoids.
    """
    if labels is None:
        labels, _ = cluster_kmeans(E_fit, n_shapes, seed=0)
    fitted_raw = fit_cma_raw(E_fit, labels, n_shapes, sigma0=sigma0, max_iter=max_iter, seed=seed)
    return normalize(fitted_raw, E_fit, alpha, normalization)
