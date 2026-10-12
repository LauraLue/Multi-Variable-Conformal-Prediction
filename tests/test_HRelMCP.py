"""
Unit test: H-RelMCP with rectangular and ellipsoidal prediction sets
================================================================
Both set types are calibrated on the same dataset. The single plot shows
prediction errors e^(m) = y^(m) - f(x^(m)) in the (e1, e2) plane with both
set boundaries overlaid. Points are colored by joint coverage status.

Rectangular set:   s_j(x, y, q) = |e_j| - q_j,       J(q) = sum(q)
Ellipsoidal set:   s(x, y, q)   = e^T P(q) e - 1,    J(q) = -log(det P)
                   P(q) = [[q1, q2], [q2, q3]],  n_q = 3
"""

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from MCP.HRelMCP import HRelMCP

from toy_problems import (
    P_from_q,
    coverage_mask,
    ellipsoidal_cost,
    ellipsoidal_score,
    make_data,
    point_predictor,
    rectangular_cost,
    rectangular_score,
)


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------
def plot_results(
    hrelmcp_rect: HRelMCP,
    hrelmcp_ellip: HRelMCP,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    cov_rect: np.ndarray,
    cov_ellip: np.ndarray,
    save_path: str | None = None,
):
    """Prediction errors in the (e1, e2) plane with both set boundaries."""
    errors = np.array([Y_cal[m] - point_predictor(X_cal[m]) for m in range(X_cal.shape[0])])

    both = cov_rect & cov_ellip
    rect_only = cov_rect & ~cov_ellip
    ellip_only = ~cov_rect & cov_ellip
    neither = ~cov_rect & ~cov_ellip

    fig, ax = plt.subplots(figsize=(7, 6))

    # Rectangular boundary
    q1, q2 = hrelmcp_rect.q_opt
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

    # Ellipsoidal boundary
    p = P_from_q(hrelmcp_ellip.q_opt)
    eigenvalues, eigenvectors = np.linalg.eigh(p)
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
        f"H-RelMCP prediction sets in error space  "
        f"($\\varepsilon={hrelmcp_rect.epsilon}$, "
        f"target confidence $= {1 - hrelmcp_rect.beta:.2f}$)",
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
def test_HRelMCP(plot_dir):
    rng = np.random.default_rng(42)

    n_total = 2000
    n_cal = 200
    epsilon = 0.15
    beta = 0.6

    X_all, Y_all = make_data(n_total, rng)
    X_cal, Y_cal = X_all[:n_cal], Y_all[:n_cal]
    X_test, Y_test = X_all[n_cal:], Y_all[n_cal:]

    # ------------------------------------------------------------------ #
    # Rectangular H-RelMCP                                                   #
    # ------------------------------------------------------------------ #
    hrelmcp_rect = HRelMCP(
        score_fn=rectangular_score,
        cost_fn=rectangular_cost,
        n_q=2,
        epsilon=epsilon,
        beta=beta,
        q0=np.ones(2) * 0.5,
        bounds=[(0.0, None), (0.0, None)],
        n_restarts=3,
        max_outer_iter=50,
        random_state=42,
    )
    hrelmcp_rect.calibrate(X_cal, Y_cal)
    print(hrelmcp_rect.summary())

    rect_cal_cov = hrelmcp_rect.empirical_coverage(X_cal, Y_cal)
    rect_test_cov = hrelmcp_rect.empirical_coverage(X_test, Y_test)
    print(f"\nRect  cal coverage  : {rect_cal_cov:.4f}  (target: {1 - epsilon:.4f})")
    print(f"Rect  test coverage : {rect_test_cov:.4f}")

    assert hrelmcp_rect.q_opt.shape == (2,)
    assert np.all(np.isfinite(hrelmcp_rect.q_opt)), "Rectangular: q* contains non-finite values"
    assert np.all(hrelmcp_rect.q_opt >= -1e-6), "Rectangular: q* has negative entries"
    assert hrelmcp_rect.i_opt is not None and hrelmcp_rect.i_opt >= 1
    assert hrelmcp_rect.beta_i_opt is not None and hrelmcp_rect.beta_i_opt <= beta
    assert hrelmcp_rect.epsilon_i_opt is not None and hrelmcp_rect.epsilon_i_opt <= epsilon + 1e-10

    active_rect = ~hrelmcp_rect.outlier_mask
    assert (
        np.mean(coverage_mask(hrelmcp_rect, X_cal[active_rect], Y_cal[active_rect], tol=1e-6))
        == 1.0
    ), "Rectangular: active calibration points not fully covered"
    assert rect_cal_cov >= 1.0 - epsilon, (
        f"Rectangular: calibration coverage {rect_cal_cov:.4f} < target {1 - epsilon:.4f}"
    )

    # ------------------------------------------------------------------ #
    # Ellipsoidal H-RelMCP                                                   #
    # ------------------------------------------------------------------ #
    hrelmcp_ellip = HRelMCP(
        score_fn=ellipsoidal_score,
        cost_fn=ellipsoidal_cost,
        n_q=3,
        epsilon=epsilon,
        beta=beta,
        q0=np.array([1.0, 0.0, 1.0]),
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
        n_restarts=3,
        max_outer_iter=50,
        random_state=43,
    )
    hrelmcp_ellip.calibrate(X_cal, Y_cal)
    print(f"\n{hrelmcp_ellip.summary()}")

    p = P_from_q(hrelmcp_ellip.q_opt)
    print(f"\n  P* =\n{p}")

    ellip_cal_cov = hrelmcp_ellip.empirical_coverage(X_cal, Y_cal)
    ellip_test_cov = hrelmcp_ellip.empirical_coverage(X_test, Y_test)
    print(f"\nEllip cal coverage  : {ellip_cal_cov:.4f}  (target: {1 - epsilon:.4f})")
    print(f"Ellip test coverage : {ellip_test_cov:.4f}")

    assert hrelmcp_ellip.q_opt.shape == (3,)
    assert np.all(np.isfinite(hrelmcp_ellip.q_opt)), "Ellipsoidal: q* contains non-finite values"
    assert np.all(np.linalg.eigvalsh(p) > 0), "Ellipsoidal: P* is not positive definite"
    assert hrelmcp_ellip.i_opt is not None and hrelmcp_ellip.i_opt >= 1
    assert hrelmcp_ellip.beta_i_opt is not None and hrelmcp_ellip.beta_i_opt <= beta
    assert (
        hrelmcp_ellip.epsilon_i_opt is not None and hrelmcp_ellip.epsilon_i_opt <= epsilon + 1e-10
    )

    active_ellip = ~hrelmcp_ellip.outlier_mask
    assert (
        np.mean(coverage_mask(hrelmcp_ellip, X_cal[active_ellip], Y_cal[active_ellip], tol=1e-6))
        == 1.0
    ), "Ellipsoidal: active calibration points not fully covered"
    assert ellip_cal_cov >= 1.0 - epsilon, (
        f"Ellipsoidal: calibration coverage {ellip_cal_cov:.4f} < target {1 - epsilon:.4f}"
    )

    # ------------------------------------------------------------------ #
    # Plot                                                               #
    # ------------------------------------------------------------------ #
    cov_rect = coverage_mask(hrelmcp_rect, X_cal, Y_cal, tol=1e-6)
    cov_ellip = coverage_mask(hrelmcp_ellip, X_cal, Y_cal, tol=1e-6)

    save_path = plot_dir / "test_HRelMCP.png"
    plot_results(
        hrelmcp_rect, hrelmcp_ellip, X_cal, Y_cal, cov_rect, cov_ellip, save_path=save_path
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)
    test_HRelMCP(out_dir)
    print("\nAll assertions passed.")


def test_certificate_covers_all_evaluated_penalties():
    """Alg. 1 may evaluate penalties after the accepted one; the union bound counts them all."""
    import warnings

    from MCP.HRelMCP import tradeoff_beta, tradeoff_epsilon
    from MCP.methods import METHODS, CalibrationTask, Target

    from toy_problems import mixture_residuals

    target = Target(0.05, beta=1e-6)
    E = mixture_residuals(2000, seed=5)
    setup = METHODS["H-RelMCP-A"].build(CalibrationTask(E_cal=E, target=target, n_shapes=3, seed=5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        predictor = setup.predictor.calibrate(None, setup.Y_cal)

    n_eval, d = predictor.n_iterations, predictor.d_opt
    assert predictor.i_opt < n_eval  # the case this test is about
    beta_reported = 1.0 - predictor.coverage_confidence()
    assert beta_reported == pytest.approx(tradeoff_beta(n_eval, d, 2000, 0.05))
    assert beta_reported > tradeoff_beta(predictor.i_opt, d, 2000, 0.05)  # not i_opt
    assert beta_reported <= target.beta
    assert predictor.epsilon_i_opt == pytest.approx(tradeoff_epsilon(1e-6, n_eval, d, 2000))
    assert predictor.epsilon_i_opt <= target.epsilon
    assert predictor.beta_i_opt == pytest.approx(1e-6 / n_eval)


@pytest.mark.parametrize("n_restarts", [1, 2, 3, 4, 10])
def test_restart_points_match_n_restarts(n_restarts):
    hrelmcp = HRelMCP(
        rectangular_score,
        rectangular_cost,
        n_q=2,
        epsilon=0.1,
        beta=0.1,
        n_restarts=n_restarts,
        random_state=0,
    )
    q0, q_start = np.zeros(2), np.ones(2)
    starts = hrelmcp._restart_points(q0, q_start)
    assert len(starts) == n_restarts
    np.testing.assert_array_equal(starts[0], q_start)
    if n_restarts >= 2:
        np.testing.assert_array_equal(starts[1], q0)


def test_inner_problem_selects_smallest_cost_without_more_violations(monkeypatch):
    """Among the restart solutions, the smallest cost with at most as many violations as the
    smallest-objective solution is selected, even if its objective is larger."""
    # candidate id -> scores of the 4 calibration points; cost = q[1]
    scores = {
        1: np.array([-1.0, -1.0, -1.0, 0.5]),  # cost 3.0, 1 violation, objective 3.5
        2: np.array([-1.0, -1.0, -1.0, 2.0]),  # cost 2.5, 1 violation, objective 4.5
        3: np.array([5.0, 0.5, 0.5, -1.0]),  # cost 1.0, 3 violations, objective 7.0
    }
    candidates = iter([np.array([1.0, 3.0]), np.array([2.0, 2.5]), np.array([3.0, 1.0])])

    class _Result:
        def __init__(self, x):
            self.x = x

    module = importlib.import_module("MCP.HRelMCP")
    monkeypatch.setattr(module, "minimize", lambda **kwargs: _Result(next(candidates)))
    hrelmcp = HRelMCP(
        lambda X, Y, q: scores[int(q[0])],
        lambda q: float(q[1]),
        n_q=2,
        epsilon=0.1,
        beta=0.1,
        n_restarts=3,
        random_state=0,
    )
    q = hrelmcp._solve_inner_problem(np.zeros(2), np.zeros(2), 1.0, None, np.zeros((4, 2)))
    np.testing.assert_array_equal(q, [2.0, 2.5])
