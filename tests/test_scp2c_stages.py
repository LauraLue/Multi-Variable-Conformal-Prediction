"""
Visualization test for the four SCP-2C pipeline stages.

  Stage 1 – KMeans clustering of E_fit
  Stage 2 – CMA-ES minimum-volume ellipsoids per cluster  (Q/D_cp, before normalizing)
  Stage 3 – Normalized ellipsoids  (coverage quantile baked in, before SCP calibration)
  Stage 4 – Calibrated ellipsoids + E_cal2 residuals  (after SCP quantile step)

Run:
    python test_scp2c_stages.py
Output is saved to tests/plots/fig_scp2c_stages.png (pytest: a temporary directory).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from MCP.fitting import apply_normalizing_constants, cluster_kmeans, fit_cma_raw
from MCP.SCP import SCP

from toy_problems import mixture_residuals

_CLUSTER_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _draw_ellipse(ax, Q: np.ndarray, c: np.ndarray, **kw) -> bool:
    """Draw { e : (e-c)^T Q (e-c) = 1 }. Returns False if Q is not PD."""
    evals, evecs = np.linalg.eigh(Q)
    if np.any(evals <= 0):
        return False
    semi = 1.0 / np.sqrt(evals)
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    ax.add_patch(mpatches.Ellipse(xy=c, width=2 * semi[0], height=2 * semi[1], angle=angle, **kw))
    return True


def _ellipse_max_radius(Q: np.ndarray) -> float:
    """Largest semi-axis length (conservative bounding radius)."""
    evals, _ = np.linalg.eigh(Q)
    evals = evals[evals > 0]
    return float(1.0 / np.sqrt(evals.min())) if evals.size else 1.0


def _axis_limits(points_list, params_list, pad: float = 0.25):
    """Compute (xlim, ylim) that fits all scatter points and all ellipses."""
    all_pts = np.vstack([p for p in points_list if p.size > 0])
    xs = list(all_pts[:, 0])
    ys = list(all_pts[:, 1])
    for params in params_list:
        for Q, c in zip(params.Q, params.centers, strict=True):
            r = _ellipse_max_radius(Q)
            xs += [c[0] - r, c[0] + r]
            ys += [c[1] - r, c[1] + r]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xpad = (xhi - xlo) * pad
    ypad = (yhi - ylo) * pad
    return (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------


def test_scp2c_stages(
    plot_dir,
    epsilon: float = 0.05,
    n_fit: int = 400,
    n_shapes: int = 3,
    seed: int = 42,
    cma_max_iter: int = 500,
) -> None:
    """Run and visualise the four SCP-2C pipeline stages.

    Generates synthetic residuals, runs the full SCP-2C pipeline (KMeans
    clustering, CMA-ES ellipsoid fitting, normalisation, SCP calibration),
    and saves a 2×2 diagnostic figure.

    Args:
        plot_dir: Directory of the diagnostic figure fig_scp2c_stages.png.
        epsilon: Target mis-coverage level.
        n_fit: Number of residuals used for fitting (Stages 1–3).
        n_shapes: Number of ellipsoids (clusters) K.
        seed: Random seed for data generation and CMA-ES.
        cma_max_iter: Maximum CMA-ES iterations per cluster.
    """
    # Minimum n_cal2 for rho >= 0, i.e. MCP.experiment.min_ncal_for_rho(epsilon, 0, 1)
    n_cal2 = int(np.ceil(1.0 / epsilon - 1.0))

    E_fit = mixture_residuals(n_fit, seed)
    E_cal2 = mixture_residuals(n_cal2, seed + 1)

    print(f"SCP-2C stages test:  K={n_shapes}, ε={epsilon}, n_fit={n_fit}, n_cal2={n_cal2}")

    # ------------------------------------------------------------------
    # Stage 1: KMeans clustering
    # ------------------------------------------------------------------
    labels, km_centers = cluster_kmeans(E_fit, n_shapes, seed=seed)
    print(f"  Stage 1 – cluster sizes: {[int((labels == k).sum()) for k in range(n_shapes)]}")

    # ------------------------------------------------------------------
    # Stage 2: CMA-ES raw ellipsoids (Q/D_cp, enclosing each cluster)
    # ------------------------------------------------------------------
    params_cma = fit_cma_raw(E_fit, labels, n_shapes, max_iter=cma_max_iter, seed=seed)
    raw_cma = params_cma.component_scores(E_fit)  # (K, n_fit); ≤ 0 inside
    inside_cma = [(raw_cma[k] <= 0).sum() for k in range(n_shapes)]
    print(f"  Stage 2 – cluster pts inside CMA ellipsoid: {inside_cma}")

    # ------------------------------------------------------------------
    # Stage 3: Normalized ellipsoids
    # ------------------------------------------------------------------
    params_norm = apply_normalizing_constants(params_cma, E_fit, epsilon)
    union_scores_fit = params_norm.score(None, E_fit)
    n_outside_fit = int((union_scores_fit > 0).sum())
    frac_covered_fit = 1.0 - n_outside_fit / n_fit
    print(
        f"  Stage 3 – E_fit coverage by normalized union: "
        f"{frac_covered_fit:.3f}  ({n_outside_fit}/{n_fit} outside,  target ≤{epsilon:.2f})"
    )

    # ------------------------------------------------------------------
    # Stage 4: SCP calibration on E_cal2
    # ------------------------------------------------------------------
    score_fn = params_norm.score
    x_dummy = np.zeros((n_cal2, 1))
    predictor = SCP(score_fn=score_fn, alpha=epsilon)
    predictor.calibrate(x_dummy, E_cal2)

    q_opt = float(predictor.q_opt)
    margin = 1.0 + q_opt
    params_cal = params_norm.scaled(1.0 / margin)
    cal_scores = score_fn(None, E_cal2)
    n_outliers = int((cal_scores > q_opt).sum())
    print(
        f"  Stage 4 – q*={q_opt:.4f},  margin={margin:.4f},  "
        f"cal outliers: {n_outliers}/{n_cal2}  (rho={predictor.rho})"
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    colors = _CLUSTER_COLORS[:n_shapes]

    # Compute shared axis limits
    xlim, ylim = _axis_limits(
        [E_fit, E_cal2],
        [params_cma, params_norm, params_cal],
    )

    # ---- Stage 1: KMeans clustering ----
    ax = axes[0]
    for k in range(n_shapes):
        mask = labels == k
        ax.scatter(
            E_fit[mask, 0],
            E_fit[mask, 1],
            color=colors[k],
            s=12,
            alpha=0.55,
            label=f"Cluster {k}  (n={mask.sum()})",
        )
    ax.scatter(
        km_centers[:, 0],
        km_centers[:, 1],
        c="black",
        marker="X",
        s=90,
        zorder=5,
        label="KMeans centers",
    )
    ax.set_title("Stage 1 – KMeans clustering  (E_fit)", fontweight="bold")
    _style_ax(ax, xlim, ylim)

    # ---- Stage 2: CMA-ES ellipsoids ----
    ax = axes[1]
    for k in range(n_shapes):
        mask = labels == k
        ax.scatter(E_fit[mask, 0], E_fit[mask, 1], color=colors[k], s=12, alpha=0.45)
        drawn = _draw_ellipse(
            ax,
            params_cma.Q[k],
            params_cma.centers[k],
            facecolor=colors[k],
            alpha=0.12,
            zorder=1,
        )
        if drawn:
            _draw_ellipse(
                ax,
                params_cma.Q[k],
                params_cma.centers[k],
                facecolor="none",
                edgecolor=colors[k],
                linewidth=2.0,
                zorder=3,
                label=f"CMA ellipsoid {k}",
            )
    ax.set_title("Stage 2 – CMA-ES ellipsoids  (Q/D_cp, before normalizing)", fontweight="bold")
    _style_ax(ax, xlim, ylim)

    # ---- Stage 3: Normalized ellipsoids ----
    ax = axes[2]
    # Color each E_fit point by the ellipsoid it falls closest to (min score)
    raw_norm = params_norm.component_scores(E_fit)  # (K, n_fit)
    best_k_fit = np.argmin(raw_norm, axis=0)
    union_fit = np.min(raw_norm, axis=0)
    inside_mask = union_fit <= 0
    outside_mask = ~inside_mask

    for k in range(n_shapes):
        m = (best_k_fit == k) & inside_mask
        ax.scatter(E_fit[m, 0], E_fit[m, 1], color=colors[k], s=12, alpha=0.45)
        _draw_ellipse(
            ax, params_norm.Q[k], params_norm.centers[k], facecolor=colors[k], alpha=0.12, zorder=1
        )
        _draw_ellipse(
            ax,
            params_norm.Q[k],
            params_norm.centers[k],
            facecolor="none",
            edgecolor=colors[k],
            linewidth=2.0,
            zorder=3,
            label=f"Norm. ellipsoid {k}",
        )
    ax.scatter(
        E_fit[outside_mask, 0],
        E_fit[outside_mask, 1],
        color="red",
        s=22,
        marker="x",
        zorder=6,
        label=f"Outside union  ({outside_mask.sum()}/{n_fit})",
    )
    ax.set_title(
        f"Stage 3 – Normalized ellipsoids  "
        f"(fit coverage {frac_covered_fit:.1%},  target {1 - epsilon:.0%})",
        fontweight="bold",
    )
    _style_ax(ax, xlim, ylim)

    # ---- Stage 4: Calibrated ellipsoids + E_cal2 ----
    ax = axes[3]
    outlier_cal2 = cal_scores > q_opt
    ax.scatter(
        E_cal2[~outlier_cal2, 0],
        E_cal2[~outlier_cal2, 1],
        color="black",
        s=30,
        alpha=0.65,
        label=f"E_cal2 inliers  ({(~outlier_cal2).sum()})",
    )
    ax.scatter(
        E_cal2[outlier_cal2, 0],
        E_cal2[outlier_cal2, 1],
        color="red",
        s=50,
        marker="x",
        zorder=5,
        label=f"E_cal2 outliers  (ρ={outlier_cal2.sum()})",
    )
    for k in range(n_shapes):
        _draw_ellipse(
            ax, params_cal.Q[k], params_cal.centers[k], facecolor=colors[k], alpha=0.12, zorder=1
        )
        _draw_ellipse(
            ax,
            params_cal.Q[k],
            params_cal.centers[k],
            facecolor="none",
            edgecolor=colors[k],
            linewidth=2.0,
            zorder=3,
            label=f"Calibrated ellipsoid {k}",
        )
    ax.set_title(
        f"Stage 4 – Calibrated ellipsoids  (q*={q_opt:.3f},  margin={margin:.3f})",
        fontweight="bold",
    )
    _style_ax(ax, xlim, ylim)

    fig.suptitle(
        f"SCP-2C Pipeline  –  K={n_shapes},  ε={epsilon},  "
        f"n_fit={n_fit},  n_cal2={n_cal2},  seed={seed}",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = plot_dir / "fig_scp2c_stages.png"
    fig.savefig(save_path, dpi=130)
    print(f"\nFigure saved to {save_path}")
    plt.close(fig)

    print("\n" + predictor.summary())


def _style_ax(ax, xlim, ylim):
    ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.6, ls="--", zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$e_1$")
    ax.set_ylabel("$e_2$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)
    test_scp2c_stages(
        out_dir,
        epsilon=0.05,
        n_fit=400,
        n_shapes=3,
        seed=42,
        cma_max_iter=500,
    )
