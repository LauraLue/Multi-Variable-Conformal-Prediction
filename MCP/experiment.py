"""Experiments of Sec. 4 and App. E and F: compare SCP baselines, RemMCP, and H-RelMCP.

Each run draws calibration and test data, calibrates all requested methods on
the calibration residuals, and evaluates test coverage and set volume.  All
methods of a run share one coverage Target (expected or high-confidence
guarantee); methods that cannot certify the target with the given number of
calibration points are skipped.  Runs are repeated over seeds and targets;
results are reported as
console tables, a compact pickle, TikZ figures/tables, and PNG figures (see
MCP.reporting).

Run from the repository root, e.g.:

    python -m MCP.experiment --data Vehicle --timesteps 2 30 50 --n-runs 500
    python -m MCP.experiment --data Vehicle --timesteps 30 --betas 1e-6 --n-runs 500
    python -m MCP.experiment --help
"""

import os

# Avoid known Windows+MKL KMeans warning/leak behavior by capping OpenMP threads.
# Must be set before numpy/scikit-learn are imported to take effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import random
import time
import traceback
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np

from MCP.CP import CalibrationError
from MCP.data.data_generation import generate_calibration_dataset
from MCP.data.predictor import TrajectoryPredictor
from MCP.data.regression import train_casp_nn, train_scm20d_nn
from MCP.methods import (
    METHODS,
    PAPER_METHODS,
    UNIMODAL_METHODS,
    CalibrationTask,
    Target,
    feasibility_issue,
    hrelmcp_target_confidence,
)
from MCP.reporting import Tee, print_run, write_report
from MCP.results import MethodResult, RunResult
from MCP.shapes import Ellipsoids, point_predictor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ("CASP", "Vehicle", "SCM20D")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_data_synthetic(n: int, rng: np.random.Generator, modes=None):
    """1-D input -> 2-D output; residuals are x-independent Gaussian mixture."""
    if modes is None:
        modes = [
            (1 / 3, np.array([-1.5, 0.5]), np.array([[0.01, 0.04], [0.04, 0.18]])),
            (1 / 3, np.array([0.6, -0.8]), np.array([[0.15, -0.06], [-0.06, 0.10]])),
            (1 / 3, np.array([2.0, 0.7]), np.array([[0.10, 0.05], [0.05, 0.08]])),
        ]

    weights = np.array([w for w, _, _ in modes])
    weights /= weights.sum()
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    k_idx = rng.choice(len(modes), size=n, p=weights)
    noise = np.zeros((n, 2))
    for k, (_, mean, cov) in enumerate(modes):
        mask = k_idx == k
        if mask.sum() > 0:
            noise[mask] = rng.multivariate_normal(mean, cov, size=int(mask.sum()))
    Y = np.column_stack([X[:, 0] / 2.0, 2.0 * X[:, 0]]) + noise
    return X, Y


def make_data_vehicle(n: int, timestep: int = 10, seed: int = 0) -> tuple:
    """Generate n vehicle-trajectory samples and return (Y_pred, Y_true).

    Args:
        n: Number of samples to generate.
        timestep: Prediction horizon (1-indexed); selects position at
            ``Y[:, timestep-1, :2]``.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (Y_pred, Y_true), each of shape (n, 2).
    """
    # Seed both Python's RNG and NumPy's legacy RNG and create a Generator
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env_X, env_Y = generate_calibration_dataset(n, seed=seed)
    idx_shuffled = rng.permutation(env_X.shape[0])
    env_X = env_X[idx_shuffled]
    env_Y = env_Y[idx_shuffled]
    Y_true = env_Y[:, timestep - 1, :2]  # convert to residuals from initial position

    predictor = TrajectoryPredictor(0.1)
    x_arr = np.asarray(env_X)
    # If a single sequence (T, features) is provided, call the single-sequence
    # predictor and return a (2,) residual for the requested time step.
    if x_arr.ndim == 2:
        Y_pred = predictor.predict_trajectory(x_arr)
    else:  # If a batch of sequences (N, T, features) is provided, use the batched API
        Y_pred = predictor.predict_batched_trajectories(x_arr)
    Y_pred = Y_pred[:, timestep - 1, :2]
    if timestep == 2:
        # scale by 1000 to avoid numerical issues with small residuals at short horizons
        Y_pred = Y_pred * 1000
        Y_true = Y_true * 1000
    return Y_pred, Y_true


def make_data_NN(
    n: int,
    seed: int,
    nn_model,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
):
    """Sample n points from the holdout pool and return (Y_pred, Y_true).

    Y_pred is the NNpredictor output; Y_true is the ground-truth target.
    Residuals E = Y_true - Y_pred are d-D where d is determined by nn_model.
    """
    rng = np.random.default_rng(seed)
    n = np.minimum(n, len(X_pool))
    idx = rng.choice(len(X_pool), size=n, replace=False)
    X_sel = X_pool[idx]
    y_sel = y_pool[idx]
    Y_pred = nn_model.predict(X_sel)
    return Y_pred, y_sel


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def ellipsoid_bounding_box(ellipsoids: Ellipsoids) -> tuple[np.ndarray, np.ndarray]:
    """Smallest axis-aligned box containing a union of ellipsoids {e : (e-c)^T Q (e-c) <= 1}.

    The half-widths of one ellipsoid are sqrt(diag(Q^{-1})).
    """
    lows, highs = [], []
    for Q, c in ellipsoids:
        half_width = np.sqrt(np.diag(np.linalg.inv(np.asarray(Q, dtype=float))))
        lows.append(np.asarray(c, dtype=float) - half_width)
        highs.append(np.asarray(c, dtype=float) + half_width)
    return np.min(lows, axis=0), np.max(highs, axis=0)


def estimate_volume(
    predictor,
    reference_points: np.ndarray,
    n_samples: int = 100_000,
    rng: np.random.Generator | None = None,
    expansion: float = 1.6,
    boundary_samples: int = 256,
    max_expand_steps: int = 8,
    ellipsoids: Ellipsoids | None = None,
) -> float:
    """Monte Carlo volume estimate using only a score-based inside indicator.

    Points are sampled uniformly in a box containing the prediction set.  If
    the set geometry is given as ellipsoids, the box is their exact bounding
    box (an empty list means an empty set).  Otherwise the box starts at the
    bounding box of reference_points and is enlarged while the set reaches
    its faces, at most max_expand_steps times, so that larger sets may be
    truncated.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if ellipsoids is not None:
        if not ellipsoids:
            return 0.0
        mins, maxs = ellipsoid_bounding_box(ellipsoids)
    elif reference_points.size == 0:
        return np.nan
    else:
        mins, maxs = _expanded_box(
            predictor, reference_points, rng, expansion, boundary_samples, max_expand_steps
        )

    box_volume = float(np.prod(maxs - mins))
    if not np.isfinite(box_volume) or box_volume <= 0.0:
        return np.nan

    samples = rng.uniform(mins, maxs, size=(n_samples, mins.size))
    inside = predictor.covers(None, samples)
    return float(box_volume * np.mean(inside))


def _expanded_box(
    predictor,
    reference_points: np.ndarray,
    rng: np.random.Generator,
    expansion: float,
    boundary_samples: int,
    max_expand_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of reference_points, enlarged while the set reaches its faces."""
    mins = np.min(reference_points, axis=0).astype(float)
    maxs = np.max(reference_points, axis=0).astype(float)

    for axis in range(reference_points.shape[1]):
        if maxs[axis] <= mins[axis]:
            mins[axis] -= 1.0
            maxs[axis] += 1.0

    center = 0.5 * (mins + maxs)
    half_width = 0.5 * (maxs - mins)

    for _ in range(max_expand_steps):
        faces = []
        for axis in range(reference_points.shape[1]):
            lower_face = rng.uniform(
                center - half_width,
                center + half_width,
                size=(boundary_samples, reference_points.shape[1]),
            )
            upper_face = rng.uniform(
                center - half_width,
                center + half_width,
                size=(boundary_samples, reference_points.shape[1]),
            )
            lower_face[:, axis] = center[axis] - half_width[axis]
            upper_face[:, axis] = center[axis] + half_width[axis]
            faces.extend([lower_face, upper_face])
        boundary = np.vstack(faces)
        if np.any(predictor.covers(None, boundary)):
            half_width *= expansion
        else:
            break
    return center - half_width, center + half_width


def min_ncal_for_rho(alpha: float, rho: int, nq: int) -> int:
    """Minimum n_cal for which the expected-coverage outlier budget reaches rho."""
    ncal = int(np.ceil((rho + 1) * nq / alpha - 1))
    return ncal


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _calibrate_method(name: str, task: CalibrationTask, E_test: np.ndarray) -> MethodResult:
    """Build, calibrate, and evaluate one method; record failures instead of raising."""
    result = MethodResult(name)
    t0 = time.perf_counter()
    try:
        setup = METHODS[name].build(task)
        predictor = setup.predictor
        predictor.calibrate(None, setup.Y_cal)
        result.cal_time = time.perf_counter() - t0
        result.predictor = predictor
        result.n_cal = setup.Y_cal.shape[0]
        result.fitted_shape = setup.fitted_shape
        result.cal_cov = predictor.empirical_coverage(None, task.E_cal)
        result.test_cov = predictor.empirical_coverage(None, E_test)
        result.ellipsoids = setup.geometry()
        result.volume = estimate_volume(predictor, task.E_cal, ellipsoids=result.ellipsoids)
    except CalibrationError as exc:
        # Expected: the coverage target cannot be certified with this data.
        result.predictor = None
        result.error = str(exc)
        result.uncertified = True
    except Exception as exc:
        result.predictor = None
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"  {name} failed unexpectedly:\n{traceback.format_exc()}")
    return result


def _check_run(run: RunResult) -> None:
    """Sanity checks that must hold for every successful run."""
    eps, beta = run.epsilon, run.beta
    where = f"{run.target.label}, seed {run.seed}"
    scp1 = run.methods.get("SCP-1")
    if scp1 is not None and scp1.ok:
        assert scp1.cal_cov >= 1.0 - eps, (
            f"{where}: SCP-1 cal coverage {scp1.cal_cov:.4f} < {1 - eps:.4f}"
        )
    if beta is not None:
        for name, method_result in run.methods.items():
            if method_result.predictor is not None:
                certified = 1.0 - method_result.predictor.coverage_confidence(epsilon=eps)
                assert certified <= beta * (1.0 + 1e-6), (
                    f"{where}: {name} certifies beta={certified:.3e} > target {beta:.3e}"
                )
    for name in ("H-RelMCP-A", "H-RelMCP-B"):
        result = run.methods.get(name)
        if result is not None and result.ok:
            for Q, _ in result.ellipsoids:
                assert np.all(np.linalg.eigvalsh(np.asarray(Q)[:2, :2]) > 0), (
                    f"{where}: {name} ellipsoid not positive definite"
                )


def run_single(
    seed: int,
    n_total: int,
    n_cal: int,
    target: Target,
    n_shapes: int = 1,
    timestep: int = 10,
    data: str = "Vehicle",
    nn_model=None,
    X_pool=None,
    y_pool=None,
    methods: Sequence[str] = (),
) -> RunResult:
    """Calibrate all requested methods on one data draw and collect metrics.

    Args:
        seed: Random seed for data generation.
        n_total: Total number of data points (calibration + test).
        n_cal: Number of calibration samples.
        target: Coverage target of all methods (see MCP.methods.Target).
            Methods that cannot certify it with n_cal points are skipped and
            recorded with infeasible=True.
        n_shapes: Number of ellipsoids in unions (SCP-kB/C, H-RelMCP-B).
        timestep: Vehicle prediction horizon (ignored for CASP/SCM20D).
        data: "Vehicle", "CASP", or "SCM20D".
        nn_model: Pre-trained model (required for CASP/SCM20D).
        X_pool: Feature holdout pool (required for CASP/SCM20D).
        y_pool: Target holdout pool (required for CASP/SCM20D).
        methods: Method names from MCP.methods.METHODS.

    Returns:
        RunResult with one MethodResult per method.
    """
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {list(METHODS)}.")

    if data in ("CASP", "SCM20D"):
        X_all, Y_all = make_data_NN(
            n_total, seed=seed, nn_model=nn_model, X_pool=X_pool, y_pool=y_pool
        )
    else:
        X_all, Y_all = make_data_vehicle(n_total, timestep=timestep, seed=seed)
    E_all = np.asarray(Y_all - point_predictor(X_all))
    E_cal, E_test = E_all[:n_cal], E_all[n_cal:]

    task = CalibrationTask(E_cal=E_cal, target=target, n_shapes=n_shapes, seed=seed)
    run = RunResult(
        seed=seed,
        target=target,
        E_cal=E_cal,
        E_test=E_test,
        hrelmcp_target_conf=(
            hrelmcp_target_confidence(n_cal, target.epsilon) if target.beta is None else np.nan
        ),
        hrelmcp_beta=task.hrelmcp_beta,
    )
    for name in methods:
        issue = feasibility_issue(name, target, n_cal, E_cal.shape[1], n_shapes)
        if issue is None:
            run.methods[name] = _calibrate_method(name, task, E_test)
        else:
            run.methods[name] = MethodResult(name, error=issue, infeasible=True)
    return run


# ---------------------------------------------------------------------------
# Repeated runs
# ---------------------------------------------------------------------------


def main_compare(
    n_runs: int = 10,
    targets: Sequence[Target] = (Target(0.05),),
    n_total: int = 11000,
    n_cal: int = 1000,
    exp_name: str = "ellips",
    n_shapes: int = 3,
    timestep: int = 10,
    data: str = "Vehicle",
    methods: Sequence[str] = PAPER_METHODS,
    output_dim: int = 2,
    output_dir: str = REPO_ROOT,
) -> dict[Target, list[RunResult]]:
    """Run the multi-run comparison and write the report.

    Calibrates all methods on n_runs independent data draws (seeds 101, 102,
    ...) for each target, prints a summary, and writes results, TikZ files,
    and figures below output_dir (see MCP.reporting).

    Args:
        n_runs: Number of independent calibration-evaluation trials per target.
        targets: Coverage targets to sweep over, in display order (see run_single).
        n_total: Total samples per run (calibration + test).
        n_cal: Number of calibration samples per run.
        exp_name: Experiment identifier used as a filename prefix.
        n_shapes: Number of ellipsoids in unions (SCP-kB/C, H-RelMCP-B).
        timestep: Vehicle prediction horizon (ignored for CASP/SCM20D).
        data: "Vehicle", "CASP", or "SCM20D".
        methods: Method names to include.
        output_dim: Number of output dimensions for SCM20D (2-16).
        output_dir: Root directory of results/, figures/, and tikz_figures/.

    Returns:
        Mapping from target to the list of RunResults.
    """
    nn_model, X_pool, y_pool = None, None, None
    if data == "CASP":
        nn_model, X_pool, y_pool = train_casp_nn(
            n_train=5000, layers=[64, 64], epochs=200, lr=1e-3, random_state=42
        )
    elif data == "SCM20D":
        nn_model, X_pool, y_pool = train_scm20d_nn(
            n_train=3000,
            layers=[64, 64],
            epochs=500,
            lr=1e-3,
            random_state=42,
            output_dim=output_dim,
        )
    seeds = list(range(101, 101 + n_runs))

    print(f"Running {n_runs} independent trial(s) per target ({exp_name}, n_cal={n_cal})...")
    print(f"Targets: {[target.label for target in targets]}\n")

    dim = output_dim if data == "SCM20D" else 2
    runs_by_target: dict[Target, list[RunResult]] = {target: [] for target in targets}
    for target in runs_by_target:
        print(target.label)
        for name in methods:
            issue = feasibility_issue(name, target, n_cal, dim, n_shapes)
            if issue is not None:
                print(f"  Skipping {issue}")
        for i, seed in enumerate(seeds):
            run = run_single(
                seed,
                n_total,
                n_cal,
                target,
                n_shapes=n_shapes,
                timestep=timestep,
                data=data,
                nn_model=nn_model,
                X_pool=X_pool,
                y_pool=y_pool,
                methods=methods,
            )
            runs_by_target[target].append(run)
            print_run(run, i, methods)
            _check_run(run)
        print()

    write_report(runs_by_target, exp_name, methods, output_dir)
    return runs_by_target


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _epsilon_tag(epsilon: float) -> str:
    """Epsilon in percent without a dot (LaTeX \\input file names), e.g. eps5 for 0.05."""
    return "eps" + f"{epsilon * 100:g}".replace(".", "p")


def _experiment_name(
    prefix: str,
    data: str,
    timestep: int,
    output_dim: int,
    epsilons: Sequence[float],
    betas: Sequence[float] | None = None,
) -> str:
    """Name such as mainHc_Vehicle30_eps5_beta1e-06; sweeps end in _epssweep/_betasweep."""
    if data == "Vehicle":
        name = f"{prefix}{data}{timestep}"
    elif data == "SCM20D":
        name = f"{prefix}{data}dim{output_dim}"
    else:
        name = f"{prefix}{data}"
    name += f"_{_epsilon_tag(epsilons[0])}" if len(epsilons) == 1 else "_epssweep"
    if betas:
        name += f"_beta{betas[0]:.0e}" if len(betas) == 1 else "_betasweep"
    return name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m MCP.experiment",
        description="Compare SCP baselines, RemMCP, and H-RelMCP (Sec. 4, App. E, F of the paper).",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="datasets to run (default: all)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=1, help="calibration runs per target (paper: 500)"
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.05],
        help="common miscoverage levels epsilon (= alpha of the expected guarantee)",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=None,
        help="misconfidence levels: all methods target P(eta >= 1 - epsilon) >= 1 - beta "
        "for every combination of epsilon and beta (default: SCP and RemMCP target the "
        "expected guarantee with alpha = epsilon, H-RelMCP the confidence of SCP-1)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=None,
        help="methods to compare (default: all paper methods for Vehicle, "
        "unimodal ones for CASP/SCM20D)",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=None,
        help="calibration samples (default: 2000; 1000 for SCM20D)",
    )
    parser.add_argument("--n-test", type=int, default=10000, help="test samples")
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=[30], help="vehicle prediction horizons T"
    )
    parser.add_argument(
        "--output-dims", type=int, nargs="+", default=[2], help="SCM20D output dimensions"
    )
    parser.add_argument("--n-shapes", type=int, default=3, help="ellipsoids in unions")
    parser.add_argument("--name-prefix", default="test_", help="prefix of experiment names")
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT,
        help="root of results/, figures/, tikz_figures/ (default: repository root)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    targets = [Target(eps, beta) for eps in args.epsilons for beta in (args.betas or [None])]
    for data in args.data:
        methods = args.methods or (PAPER_METHODS if data == "Vehicle" else UNIMODAL_METHODS)
        n_cal = args.n_cal or (1000 if data == "SCM20D" else 2000)
        timesteps = args.timesteps if data == "Vehicle" else [0]
        output_dims = args.output_dims if data == "SCM20D" else [2]
        for timestep in timesteps:
            for output_dim in output_dims:
                exp_name = _experiment_name(
                    args.name_prefix, data, timestep, output_dim, args.epsilons, args.betas
                )
                log_path = os.path.join(args.output_dir, "results", f"{exp_name}_log.txt")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with Tee(log_path):
                    main_compare(
                        n_runs=args.n_runs,
                        targets=targets,
                        n_total=args.n_test + n_cal,
                        n_cal=n_cal,
                        exp_name=exp_name,
                        n_shapes=args.n_shapes,
                        timestep=timestep,
                        data=data,
                        methods=methods,
                        output_dim=output_dim,
                        output_dir=args.output_dir,
                    )
                    print("\nDone.")
                print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
