"""
Unit test: SCP with 2-norm and infinity-norm score functions
============================================================
Both score functions are calibrated on the same dataset.  The single plot
shows prediction errors e^(m) = y^(m) - f(x^(m)) in the (e1, e2) plane
with both set boundaries overlaid.  Points are coloured by joint coverage:

  green  dot      -- covered by both sets
  blue   square   -- covered by 2-norm set only
  orange triangle -- covered by inf-norm set only
  red    cross    -- covered by neither set

2-norm score:    s̃(x, y) = ||y - f(x)||_2    ->  disk  of radius q*_2
Inf-norm score:  s̃(x, y) = ||y - f(x)||_inf  ->  square of half-side q*_inf

Relationship between norms in R^2:
    ||e||_inf <= ||e||_2 <= sqrt(2) * ||e||_inf
so q*_inf <= q*_2, and the disk extends beyond the square corners.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from MCP.SCP import SCP

from toy_problems import coverage_mask, make_data, point_predictor


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------
def score_2norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s̃(x, y) = ||y - f(x)||_2, row-wise for batches."""
    return np.linalg.norm(y - point_predictor(x), axis=-1)


def score_infnorm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s̃(x, y) = ||y - f(x)||_inf = max_j |e_j|, row-wise for batches."""
    return np.max(np.abs(y - point_predictor(x)), axis=-1)


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------
def plot_results(
    scp_2: SCP,
    scp_inf: SCP,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    cov_2: np.ndarray,  # bool mask for 2-norm set
    cov_inf: np.ndarray,  # bool mask for inf-norm set
    save_path: str | None = None,
):
    """Prediction errors in the (e1, e2) plane with both set boundaries.

    Points are coloured by joint coverage status (see module docstring).
    """
    errors = np.array([Y_cal[m] - point_predictor(X_cal[m]) for m in range(X_cal.shape[0])])

    both = cov_2 & cov_inf
    norm2_only = cov_2 & ~cov_inf
    inf_only = ~cov_2 & cov_inf
    neither = ~cov_2 & ~cov_inf

    q2 = scp_2.q_opt
    qi = scp_inf.q_opt

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- 2-norm set boundary: disk of radius q*_2 ---
    ax.add_patch(
        mpatches.Circle(
            (0.0, 0.0),
            q2,
            facecolor="steelblue",
            alpha=0.12,
            zorder=2,
        )
    )
    ax.add_patch(
        mpatches.Circle(
            (0.0, 0.0),
            q2,
            facecolor="none",
            edgecolor="steelblue",
            linewidth=1.8,
            zorder=4,
            label=f"2-norm set  ($q^*={q2:.3f}$)",
        )
    )

    # --- Inf-norm set boundary: square of half-side q*_inf ---
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-qi, -qi),
            2 * qi,
            2 * qi,
            boxstyle="square,pad=0",
            facecolor="coral",
            alpha=0.12,
            zorder=2,
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-qi, -qi),
            2 * qi,
            2 * qi,
            boxstyle="square,pad=0",
            facecolor="none",
            edgecolor="coral",
            linewidth=1.8,
            zorder=4,
            label=f"$\\infty$-norm set  ($q^*={qi:.3f}$)",
        )
    )

    # --- Scatter by joint coverage ---
    scatter_kw = [
        (both, "green", "o", 20, 0.65, f"Both covered        ({both.sum()})"),
        (norm2_only, "steelblue", "s", 35, 0.85, f"2-norm only         ({norm2_only.sum()})"),
        (inf_only, "coral", "^", 40, 0.85, f"$\\infty$-norm only ({inf_only.sum()})"),
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
        f"SCP prediction sets in error space  "
        f"($\\alpha={scp_2.alpha}$, coverage bound $= {scp_2.coverage_bound:.2f}$)",
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
def test_SCP(plot_dir):
    rng = np.random.default_rng(42)

    n_total = 2000
    n_cal = 200
    epsilon = 0.15

    X_all, Y_all = make_data(n_total, rng)
    X_cal, Y_cal = X_all[:n_cal], Y_all[:n_cal]
    X_test, Y_test = X_all[n_cal:], Y_all[n_cal:]

    # rho = floor(epsilon * (n_cal + 1)) - 1 = floor(0.15 * 201) - 1 = 29
    expected_rho = int(np.floor(epsilon * (n_cal + 1))) - 1

    # ------------------------------------------------------------------ #
    # 2-norm SCP                                                           #
    # ------------------------------------------------------------------ #
    scp_2 = SCP(score_fn=score_2norm, alpha=epsilon)
    scp_2.calibrate(X_cal, Y_cal)
    print(scp_2.summary())

    cal_cov_2 = scp_2.empirical_coverage(X_cal, Y_cal)
    test_cov_2 = scp_2.empirical_coverage(X_test, Y_test)
    print(f"\n2-norm  cal coverage  : {cal_cov_2:.4f}  (bound: {1 - epsilon:.4f})")
    print(f"2-norm  test coverage : {test_cov_2:.4f}")

    assert scp_2.rho == expected_rho, f"2-norm: rho={scp_2.rho}, expected {expected_rho}"
    assert int(scp_2.outlier_mask.sum()) == scp_2.rho, "2-norm: outlier count != rho"
    assert cal_cov_2 >= 1.0 - epsilon, (
        f"2-norm: calibration coverage {cal_cov_2:.4f} < bound {1 - epsilon:.4f}"
    )
    active_2 = ~scp_2.outlier_mask
    assert coverage_mask(scp_2, X_cal[active_2], Y_cal[active_2]).mean() == 1.0, (
        "2-norm: active calibration points not fully covered"
    )

    # ------------------------------------------------------------------ #
    # Inf-norm SCP                                                         #
    # ------------------------------------------------------------------ #
    scp_inf = SCP(score_fn=score_infnorm, alpha=epsilon)
    scp_inf.calibrate(X_cal, Y_cal)
    print(f"\n{scp_inf.summary()}")

    cal_cov_inf = scp_inf.empirical_coverage(X_cal, Y_cal)
    test_cov_inf = scp_inf.empirical_coverage(X_test, Y_test)
    print(f"\nInf-norm cal coverage  : {cal_cov_inf:.4f}  (bound: {1 - epsilon:.4f})")
    print(f"Inf-norm test coverage : {test_cov_inf:.4f}")

    assert scp_inf.rho == expected_rho, f"Inf-norm: rho={scp_inf.rho}, expected {expected_rho}"
    assert int(scp_inf.outlier_mask.sum()) == scp_inf.rho, "Inf-norm: outlier count != rho"
    assert cal_cov_inf >= 1.0 - epsilon, (
        f"Inf-norm: calibration coverage {cal_cov_inf:.4f} < bound {1 - epsilon:.4f}"
    )
    active_inf = ~scp_inf.outlier_mask
    assert coverage_mask(scp_inf, X_cal[active_inf], Y_cal[active_inf]).mean() == 1.0, (
        "Inf-norm: active calibration points not fully covered"
    )

    # Norm inequality: ||e||_inf <= ||e||_2  =>  q*_inf <= q*_2
    assert scp_inf.q_opt <= scp_2.q_opt + 1e-10, (
        f"Expected q*_inf ({scp_inf.q_opt:.4f}) <= q*_2 ({scp_2.q_opt:.4f})"
    )
    print(f"\nq*_2   = {scp_2.q_opt:.6f}")
    print(f"q*_inf = {scp_inf.q_opt:.6f}  (expected <= q*_2)")

    # ------------------------------------------------------------------ #
    # Plot — both sets on the same error scatter                           #
    # ------------------------------------------------------------------ #
    cov_2 = coverage_mask(scp_2, X_cal, Y_cal)
    cov_inf = coverage_mask(scp_inf, X_cal, Y_cal)

    save_path = plot_dir / "test_SCP.png"
    plot_results(scp_2, scp_inf, X_cal, Y_cal, cov_2, cov_inf, save_path=save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)
    test_SCP(out_dir)
    print("\nAll assertions passed.")
