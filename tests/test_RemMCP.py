"""
Unit test: RemMCP with rectangular and ellipsoidal prediction sets
================================================================
Both set types are calibrated on the same dataset.  The single plot shows
prediction errors e^(m) = y^(m) - f(x^(m)) in the (e1, e2) plane with both
set boundaries overlaid.  Points are coloured by joint coverage status:

  green  dot      -- covered by both sets
  blue   square   -- covered by rectangular set only
  orange triangle -- covered by ellipsoidal set only
  red    cross    -- covered by neither set

Rectangular set:   s_j(x, y, q) = |e_j| - q_j,       J(q) = sum(q)
Ellipsoidal set:   s(x, y, q)   = e^T P(q) e - 1,    J(q) = -log(det P)
                   P(q) = [[q1, q2], [q2, q3]],  n_q = 3
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from MCP.RemMCP import RemMCP

from toy_problems import (
    P_from_q,
    coverage_mask,
    ellipsoidal_cost,
    ellipsoidal_cost_grad,
    ellipsoidal_score,
    make_data,
    point_predictor,
    rectangular_cost,
    rectangular_cost_grad,
    rectangular_score,
)


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------
def plot_results(
    remmcp_rect: RemMCP,
    remmcp_ellip: RemMCP,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    cov_rect: np.ndarray,  # bool mask for rectangular set
    cov_ellip: np.ndarray,  # bool mask for ellipsoidal set
    save_path: str | None = None,
):
    """Prediction errors in the (e1, e2) plane with both set boundaries.

    Points are coloured by joint coverage status (see module docstring).
    """
    errors = np.array([Y_cal[m] - point_predictor(X_cal[m]) for m in range(X_cal.shape[0])])

    # Joint-coverage categories
    both = cov_rect & cov_ellip
    rect_only = cov_rect & ~cov_ellip
    ellip_only = ~cov_rect & cov_ellip
    neither = ~cov_rect & ~cov_ellip

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- Rectangular set boundary ---
    q1, q2 = remmcp_rect.q_opt
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-q1, -q2),
            2 * q1,
            2 * q2,
            boxstyle="square,pad=0",
            facecolor="steelblue",
            alpha=0.12,
            zorder=2,
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-q1, -q2),
            2 * q1,
            2 * q2,
            boxstyle="square,pad=0",
            facecolor="none",
            edgecolor="steelblue",
            linewidth=1.8,
            zorder=4,
            label=f"Rectangular set  ($q_1^*={q1:.3f},\\ q_2^*={q2:.3f}$)",
        )
    )

    # --- Ellipsoidal set boundary ---
    P = P_from_q(remmcp_ellip.q_opt)
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    semi_axes = 1.0 / np.sqrt(eigenvalues)
    angle_deg = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(
        mpatches.Ellipse(
            (0.0, 0.0),
            2.0 * semi_axes[0],
            2.0 * semi_axes[1],
            angle=angle_deg,
            facecolor="coral",
            alpha=0.12,
            zorder=2,
        )
    )
    ax.add_patch(
        mpatches.Ellipse(
            (0.0, 0.0),
            2.0 * semi_axes[0],
            2.0 * semi_axes[1],
            angle=angle_deg,
            facecolor="none",
            edgecolor="coral",
            linewidth=1.8,
            zorder=4,
            label=(
                f"Ellipsoidal set  ($a_1={semi_axes[0]:.3f},\\ a_2={semi_axes[1]:.3f},"
                f"\\ \\theta={angle_deg:.1f}°$)"
            ),
        )
    )

    # --- Scatter by joint coverage ---
    scatter_kw = [
        (both, "green", "o", 20, 0.65, f"Both covered        ({both.sum()})"),
        (rect_only, "steelblue", "s", 35, 0.85, f"Rectangular only    ({rect_only.sum()})"),
        (ellip_only, "coral", "^", 40, 0.85, f"Ellipsoidal only    ({ellip_only.sum()})"),
        (neither, "red", "x", 55, 1.00, f"Neither             ({neither.sum()})"),
    ]
    for mask, color, marker, size, alpha, label in scatter_kw:
        if mask.any():
            ax.scatter(
                errors[mask, 0],
                errors[mask, 1],
                color=color,
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=1.0,
                zorder=3,
                label=label,
            )

    ax.axhline(0.0, color="gray", lw=0.7, ls="--", zorder=1)
    ax.axvline(0.0, color="gray", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("$e_1 = y_1 - f_1(x)$", fontsize=12)
    ax.set_ylabel("$e_2 = y_2 - f_2(x)$", fontsize=12)
    ax.set_title(
        f"RemMCP prediction sets in error space  "
        f"($\\alpha={remmcp_rect.alpha}$, coverage bound $= {remmcp_rect.coverage_bound:.2f}$)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"Figure saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------
def test_RemMCP(plot_dir):
    rng = np.random.default_rng(42)

    n_total = 2000
    n_cal = 200
    epsilon = 0.15

    X_all, Y_all = make_data(n_total, rng)
    X_cal, Y_cal = X_all[:n_cal], Y_all[:n_cal]
    X_test, Y_test = X_all[n_cal:], Y_all[n_cal:]

    # ------------------------------------------------------------------ #
    # Rectangular RemMCP  (n_q = 2)                                         #
    # ------------------------------------------------------------------ #
    remmcp_rect = RemMCP(
        score_fn=rectangular_score,
        cost_fn=rectangular_cost,
        n_q=2,
        alpha=epsilon,
        q0=np.ones(2) * 0.5,
        cost_grad_fn=rectangular_cost_grad,
        bounds=[(0.0, None), (0.0, None)],
    )
    remmcp_rect.calibrate(X_cal, Y_cal)
    print(remmcp_rect.summary())

    rect_cal_cov = remmcp_rect.empirical_coverage(X_cal, Y_cal)
    rect_test_cov = remmcp_rect.empirical_coverage(X_test, Y_test)
    print(f"\nRect  cal coverage  : {rect_cal_cov:.4f}  (bound: {1 - epsilon:.4f})")
    print(f"Rect  test coverage : {rect_test_cov:.4f}")

    # Active points all covered (boundary tol for support points)
    active_rect = ~remmcp_rect.outlier_mask
    assert (
        np.mean(coverage_mask(remmcp_rect, X_cal[active_rect], Y_cal[active_rect], tol=1e-6)) == 1.0
    ), "Rectangular: active calibration points not fully covered"
    assert rect_cal_cov >= 1.0 - epsilon, (
        f"Rectangular: calibration coverage {rect_cal_cov:.4f} < bound {1 - epsilon:.4f}"
    )
    assert np.all(remmcp_rect.q_opt >= -1e-6), "Rectangular: q* has negative entries"

    # ------------------------------------------------------------------ #
    # Ellipsoidal RemMCP  (n_q = 3)                                         #
    # ------------------------------------------------------------------ #
    remmcp_ellip = RemMCP(
        score_fn=ellipsoidal_score,
        cost_fn=ellipsoidal_cost,
        n_q=3,
        alpha=epsilon,
        q0=np.array([1.0, 0.0, 1.0]),
        cost_grad_fn=ellipsoidal_cost_grad,
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
    )
    remmcp_ellip.calibrate(X_cal, Y_cal)
    print(f"\n{remmcp_ellip.summary()}")

    P = P_from_q(remmcp_ellip.q_opt)
    print(f"\n  P* =\n{P}")

    ellip_cal_cov = remmcp_ellip.empirical_coverage(X_cal, Y_cal)
    ellip_test_cov = remmcp_ellip.empirical_coverage(X_test, Y_test)
    print(f"\nEllip cal coverage  : {ellip_cal_cov:.4f}  (bound: {1 - epsilon:.4f})")
    print(f"Ellip test coverage : {ellip_test_cov:.4f}")

    # P* must be positive definite
    assert np.all(np.linalg.eigvalsh(P) > 0), "Ellipsoidal: P* is not positive definite"
    active_ellip = ~remmcp_ellip.outlier_mask
    assert (
        np.mean(coverage_mask(remmcp_ellip, X_cal[active_ellip], Y_cal[active_ellip], tol=1e-6))
        == 1.0
    ), "Ellipsoidal: active calibration points not fully covered"
    assert ellip_cal_cov >= 1.0 - epsilon, (
        f"Ellipsoidal: calibration coverage {ellip_cal_cov:.4f} < bound {1 - epsilon:.4f}"
    )

    # ------------------------------------------------------------------ #
    # Plot — both sets on the same error scatter                           #
    # ------------------------------------------------------------------ #
    # Use tol=1e-6 for the plot masks so boundary support points render as covered
    cov_rect = coverage_mask(remmcp_rect, X_cal, Y_cal, tol=1e-6)
    cov_ellip = coverage_mask(remmcp_ellip, X_cal, Y_cal, tol=1e-6)

    save_path = plot_dir / "test_RemMCP.png"
    plot_results(remmcp_rect, remmcp_ellip, X_cal, Y_cal, cov_rect, cov_ellip, save_path=save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)
    test_RemMCP(out_dir)
    print("\nAll assertions passed.")


def test_removal_set_fills_by_sample_index():
    """Support = binding constraints; missing ones are the lowest-index non-support samples."""
    remmcp = RemMCP(rectangular_score, rectangular_cost, n_q=3, alpha=0.1)
    active_idx = np.array([2, 5, 7, 9, 11])
    # Two binding constraints (positions 3 and 1); the most violated non-binding one
    # (position 4) must not be preferred over the lowest-index one (position 0).
    scores = np.array([-0.5, 0.0, -0.9, -1e-9, -1e-3])
    np.testing.assert_array_equal(remmcp._removal_set(active_idx, scores, k=0), [5, 9, 2])
    # More binding constraints than n_q: the n_q largest scores
    scores = np.array([0.0, -1e-8, -2e-7, 1e-9, -0.4])
    np.testing.assert_array_equal(remmcp._removal_set(active_idx, scores, k=0), [9, 2, 5])
