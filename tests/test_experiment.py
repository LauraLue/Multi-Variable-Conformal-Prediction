"""
Tests for the experiment pipeline: method registry, shape families, run_single,
main_compare output, and the command-line interface.
"""

import os
import pickle

import numpy as np
import pytest

import MCP.experiment as experiment
from MCP.CP import EXPECTED, HIGH_CONFIDENCE, CalibrationError
from MCP.experiment import _check_run, _experiment_name, main, main_compare, parse_args, run_single
from MCP.methods import (
    METHODS,
    PAPER_METHODS,
    UNIMODAL_METHODS,
    CalibrationTask,
    Target,
    feasibility_issue,
    hrelmcp_target_confidence,
    min_ncal,
)
from MCP.SCP import SCP
from MCP.shapes import Ellipsoid, UnionOfEllipsoids, score_2norm

from toy_problems import mixture_residuals

SLOW_METHODS = {name for name in METHODS if name.endswith("C")} | {"H-RelMCP-B"}


TARGETS = [Target(0.1), Target(0.1, beta=1e-3)]


def _task(E: np.ndarray, target: Target = TARGETS[0]) -> CalibrationTask:
    return CalibrationTask(E_cal=E, target=target, n_shapes=2, seed=0)


# ===========================================================================
# Registry
# ===========================================================================


def test_registry_is_consistent():
    assert set(PAPER_METHODS) <= set(METHODS) and set(UNIMODAL_METHODS) <= set(METHODS)
    keys = [spec.key for spec in METHODS.values()]
    assert len(keys) == len(set(keys)), "method keys must be unique"
    for name, spec in METHODS.items():
        assert spec.name == name
        assert spec.tikz_style in ("", "dashed", "dotted")
        assert (spec.split_ratio is not None) == (name.startswith("SCP-") and name != "SCP-1")


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW_METHODS else n for n in METHODS],
)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.slug)
def test_every_method_calibrates_and_draws_its_set(name, target):
    """Each registered method calibrates and yields a positive-definite set geometry."""
    E = mixture_residuals(400, seed=0)
    setup = METHODS[name].build(_task(E, target))
    setup.predictor.calibrate(None, setup.Y_cal)

    predictor = setup.predictor
    if name.startswith("H-RelMCP"):
        assert predictor.guarantee == HIGH_CONFIDENCE
        assert predictor.beta == target.hrelmcp_beta(E.shape[0])
        assert (predictor.max_outer_iter, predictor.max_total_iter) == (100, 200)
        assert 1 <= predictor.i_opt <= predictor.n_iterations <= predictor.max_outer_iter
    else:
        assert predictor.guarantee == target.guarantee
        assert (predictor.alpha, predictor.epsilon, predictor.beta) == (
            (target.epsilon, None, None) if target.beta is None else (None, 0.1, 1e-3)
        )
    if target.beta is not None:
        assert 1.0 - predictor.coverage_confidence() <= target.beta

    expected_n = (
        E.shape[0] - int(np.floor(METHODS[name].split_ratio * E.shape[0]))
        if METHODS[name].split_ratio
        else 0
    )
    assert setup.Y_cal.shape[0] == E.shape[0] - expected_n

    ellipsoids = setup.geometry()
    assert ellipsoids, f"{name} has no plottable geometry"
    for Q, c in ellipsoids:
        assert np.all(np.linalg.eigvalsh(Q) > 0)
        assert np.asarray(c).shape == (2,)


def test_target_selects_the_guarantee():
    expected, high = Target(0.05), Target(0.05, beta=1e-6)
    assert expected.guarantee == EXPECTED and high.guarantee == HIGH_CONFIDENCE
    assert expected.predictor_kwargs() == {"alpha": 0.05}
    assert high.predictor_kwargs() == {"epsilon": 0.05, "beta": 1e-6}
    assert high.hrelmcp_beta(2000) == 1e-6
    assert expected.hrelmcp_beta(2000) == pytest.approx(1.0 - hrelmcp_target_confidence(2000, 0.05))
    assert (expected.label, expected.slug) == ("eps=0.05", "eps0.05")
    assert (high.label, high.slug) == ("eps=0.05, beta=1e-06", "eps0.05_beta1e-06")
    task = CalibrationTask(E_cal=np.zeros((2000, 2)), target=high, n_shapes=3, seed=0)
    assert task.epsilon == 0.05 and task.hrelmcp_beta == 1e-6


@pytest.mark.parametrize("epsilon, beta", [(0.0, None), (1.0, None), (0.05, 0.0), (0.05, 1.5)])
def test_target_rejects_levels_outside_the_unit_interval(epsilon, beta):
    with pytest.raises(ValueError):
        Target(epsilon, beta)


def test_hrelmcp_target_matches_scp1_confidence():
    E = mixture_residuals(500, seed=0)
    scp1 = SCP(score_2norm, alpha=0.05).calibrate(None, E)
    assert hrelmcp_target_confidence(500, 0.05) == pytest.approx(
        scp1.coverage_confidence(epsilon=0.05)
    )


# ===========================================================================
# Shape families
# ===========================================================================


def test_ellipsoid_shape_family():
    shape = Ellipsoid(3)
    assert shape.n_q == 6 and len(shape.bounds) == 6
    q = shape.initial_q()
    E = np.random.default_rng(0).standard_normal((10, 3))
    # unit ball
    np.testing.assert_allclose(shape.score(None, E, q), np.linalg.norm(E, axis=1) - 1.0)
    ((Q, c),) = shape.ellipsoids(q)
    np.testing.assert_allclose(Q, np.eye(3))
    np.testing.assert_array_equal(c, np.zeros(3))


def test_union_of_ellipsoids_shape_family():
    shape = UnionOfEllipsoids(dim=2, n_shapes=3)
    assert shape.n_q == 15 and len(shape.bounds) == 15
    q = np.tile([1.0, 0.0, 1.0, 0.5, -0.5], 3)
    E = np.random.default_rng(1).standard_normal((10, 2))
    # three identical unit balls centered at (0.5, -0.5): union score = single score
    np.testing.assert_allclose(
        shape.score(None, E, q), np.linalg.norm(E - [0.5, -0.5], axis=1) - 1.0
    )
    assert shape.score_grad(None, E, q).shape == (10, 15)
    assert len(shape.ellipsoids(q)) == 3


@pytest.mark.parametrize("shape", [Ellipsoid(3), UnionOfEllipsoids(dim=2, n_shapes=3)])
def test_norm_score_gradients_match_finite_differences(shape):
    rng = np.random.default_rng(2)
    if isinstance(shape, Ellipsoid):
        q = shape.initial_q()
    else:
        q = np.tile([1.5, 0.3, 0.8, 0.2, -0.1], 3)
    q = q + 0.1 * rng.standard_normal(q.size)
    E = rng.standard_normal((20, shape.dim))
    h = 1e-6
    steps = h * np.eye(q.size)
    numeric = np.stack(
        [(shape.score(None, E, q + dq) - shape.score(None, E, q - dq)) / (2 * h) for dq in steps],
        axis=1,
    )
    np.testing.assert_allclose(shape.score_grad(None, E, q), numeric, atol=1e-6)


def test_shape_classes_validate_inputs():
    from MCP.shapes import FittedEllipsoids

    with pytest.raises(ValueError, match="same length"):
        FittedEllipsoids([np.zeros(2)], [np.eye(2), np.eye(2)])
    with pytest.raises(ValueError, match="Expected 10 parameters"):
        UnionOfEllipsoids(dim=2, n_shapes=2).score(None, np.zeros((3, 2)), np.ones(9))
    with pytest.raises(ValueError, match="lower-triangular"):
        Ellipsoid(2).score(None, np.zeros((3, 2)), np.ones(4))


# ===========================================================================
# run_single and main_compare
# ===========================================================================


def test_run_single_skips_infeasible_methods():
    # 20 calibration points: SCP-1 needs n_cal >= 9, RemMCP (n_q = 3) needs n_cal >= 29.
    run = run_single(
        seed=101,
        n_total=80,
        n_cal=20,
        target=Target(0.1),
        timestep=30,
        data="Vehicle",
        methods=("SCP-1", "RemMCP"),
    )
    assert run.methods["SCP-1"].ok and not run.methods["SCP-1"].infeasible
    remmcp = run.methods["RemMCP"]
    assert not remmcp.ok and remmcp.infeasible
    assert "no valid outlier budget" in remmcp.error and "needs n_cal >= 29" in remmcp.error
    assert np.isnan(remmcp.test_cov) and remmcp.ellipsoids == []


def test_run_single_records_calibration_errors(monkeypatch):
    """Without the up-front check, the same method fails in calibration instead."""
    monkeypatch.setattr(experiment, "feasibility_issue", lambda *args: None)
    run = run_single(
        seed=101,
        n_total=80,
        n_cal=20,
        target=Target(0.1),
        timestep=30,
        data="Vehicle",
        methods=("RemMCP",),
    )
    remmcp = run.methods["RemMCP"]
    assert not remmcp.ok and not remmcp.infeasible
    assert "No valid outlier budget" in remmcp.error


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.slow) if n == "H-RelMCP-B" else n
        for n in ("SCP-1", "SCP-2A", "SCP-3B", "RemMCP", "H-RelMCP-A", "H-RelMCP-B")
    ],
)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.slug)
def test_feasibility_check_matches_calibration(name, target):
    """Below min_ncal calibration must fail; from min_ncal on, SCP and RemMCP succeed.

    For H-RelMCP the check is only necessary: a solution without violations may
    first be found at a later iteration, whose confidence budget beta / i is
    tighter.
    """
    n_min = min_ncal(name, target, dim=2, n_shapes=2)
    assert feasibility_issue(name, target, n_min, dim=2, n_shapes=2) is None
    issue = feasibility_issue(name, target, n_min - 1, dim=2, n_shapes=2)
    assert issue is not None and f"needs n_cal >= {n_min}" in issue

    if target.beta is not None:  # bisection in min_ncal assumes monotonicity in n_cal
        assert all(
            feasibility_issue(name, target, n, dim=2, n_shapes=2) is None
            for n in range(n_min, n_min + 30)
        )

    E = mixture_residuals(n_min, seed=0)
    setup = METHODS[name].build(_task(E[: n_min - 1], target))
    with pytest.raises(CalibrationError):
        setup.predictor.calibrate(None, setup.Y_cal)
    if not METHODS[name].relaxation:
        setup = METHODS[name].build(_task(E, target))
        setup.predictor.calibrate(None, setup.Y_cal)  # certifiable: must not raise


def test_hrelmcp_admissible_violations_shrink_logarithmically():
    """Plan values at n_cal = 2,000, epsilon = 0.05, beta = 1e-6 with d_i = violations + n_q."""
    from MCP.HRelMCP import tradeoff_epsilon

    def d_max(i: int) -> int:
        d = 0
        while tradeoff_epsilon(1e-6, i, d + 1, 2000) <= 0.05:
            d += 1
        return d

    max_violations = {i: d_max(i) - 3 for i in (1, 15, 100, 1000)}  # H-RelMCP-A, n_q = 3
    assert max_violations == {1: 47, 15: 43, 100: 41, 1000: 38}


def test_min_ncal_matches_known_budgets():
    # Plan values for n_cal = 2,000 at epsilon = 0.05, beta = 1e-6 (all feasible), and the
    # infeasible SCP-2 and H-RelMCP-B (n_q = 15) at epsilon = 0.02.
    target = Target(0.05, beta=1e-6)
    assert all(feasibility_issue(m, target, 2000, dim=2) is None for m in PAPER_METHODS)
    target = Target(0.02, beta=1e-6)
    infeasible = [m for m in PAPER_METHODS if feasibility_issue(m, target, 2000, dim=2)]
    assert infeasible == ["SCP-2A", "SCP-2B", "H-RelMCP-B"]
    assert min_ncal("SCP-1", Target(0.1), dim=2) == 9  # floor(0.1 (n + 1)) >= 1
    assert min_ncal("RemMCP", Target(0.1), dim=2) == 29  # floor(0.1 (n + 1) / 3) >= 1


def test_run_single_reports_unexpected_errors(monkeypatch, capsys):
    def broken_build(task):
        raise RuntimeError("boom")

    spec = METHODS["SCP-1"]
    monkeypatch.setitem(METHODS, "SCP-1", type(spec)(**{**spec.__dict__, "build": broken_build}))
    run = run_single(
        seed=101,
        n_total=200,
        n_cal=100,
        target=Target(0.1),
        timestep=30,
        data="Vehicle",
        methods=("SCP-1",),
    )
    assert run.methods["SCP-1"].error == "RuntimeError: boom"
    assert "Traceback" in capsys.readouterr().out


def test_run_single_rejects_unknown_methods():
    with pytest.raises(ValueError, match="Unknown methods"):
        run_single(seed=101, n_total=200, n_cal=100, target=Target(0.1), methods=("SCP-9Z",))


def test_main_compare_writes_report_to_output_dir(tmp_path):
    runs = main_compare(
        n_runs=2,
        targets=[Target(0.1)],
        n_total=600,
        n_cal=300,
        exp_name="unit",
        timestep=30,
        data="Vehicle",
        methods=("SCP-1", "SCP-2A", "RemMCP"),
        output_dir=str(tmp_path),
    )
    assert [run.seed for run in runs[Target(0.1)]] == [101, 102]

    expected = [
        "results/unit_results.pkl",
        "figures/fig_unit_sets.png",
        "figures/fig_unit_boxplot.png",
        "tikz_figures/table_unit.tex",
        "tikz_figures/fig_unit_boxplot.tex",
        "tikz_figures/fig_unit_sets0.tex",
    ]
    for rel in expected:
        assert (tmp_path / rel).is_file(), rel

    saved = pickle.load(open(tmp_path / "results/unit_results.pkl", "rb"))
    assert saved[(0.1, None)][0]["guarantee"] == EXPECTED
    first = saved[(0.1, None)][0]["methods"]
    assert first["SCP-2A"]["n_cal"] == 75 and len(first["SCP-2A"]["fitted_shape"]["Q"]) == 1
    assert first["RemMCP"]["ok"] and len(first["RemMCP"]["q_opt"]) == 3


def test_main_compare_logs_hrelmcp_iterations(tmp_path, capsys):
    runs = main_compare(
        n_runs=1,
        targets=[Target(0.1, beta=1e-3)],
        n_total=500,
        n_cal=200,
        exp_name="it",
        timestep=30,
        data="Vehicle",
        methods=("SCP-1", "H-RelMCP-A"),
        output_dir=str(tmp_path),
    )
    hrelmcp = runs[Target(0.1, beta=1e-3)][0].methods["H-RelMCP-A"].predictor
    row = next(line for line in capsys.readouterr().out.splitlines() if "Iterations" in line)
    assert row.split()[-2:] == ["--", f"{hrelmcp.n_iterations:.1f}/{hrelmcp.n_iterations}"]
    saved = pickle.load(open(tmp_path / "results/it_results.pkl", "rb"))
    entry = saved[(0.1, 1e-3)][0]["methods"]["H-RelMCP-A"]
    assert (entry["i_opt"], entry["n_iterations"]) == (hrelmcp.i_opt, hrelmcp.n_iterations)


def test_main_compare_high_confidence_target(tmp_path, capsys):
    target = Target(0.1, beta=1e-6)  # SCP-2A keeps 75 threshold points: no valid budget
    runs = main_compare(
        n_runs=2,
        targets=[target],
        n_total=600,
        n_cal=300,
        exp_name="hc",
        timestep=30,
        data="Vehicle",
        methods=("SCP-1", "SCP-2A", "RemMCP"),
        output_dir=str(tmp_path),
    )
    assert "Skipping SCP-2A infeasible" in capsys.readouterr().out
    for run in runs[target]:
        assert run.beta == 1e-6 and run.hrelmcp_beta == 1e-6 and np.isnan(run.hrelmcp_target_conf)
        assert run.methods["SCP-2A"].infeasible
        for name in ("SCP-1", "RemMCP"):
            predictor = run.methods[name].predictor
            assert predictor.guarantee == HIGH_CONFIDENCE
            assert 1.0 - predictor.coverage_confidence() <= 1e-6

    saved = pickle.load(open(tmp_path / "results/hc_results.pkl", "rb"))
    run = saved[(0.1, 1e-6)][0]
    assert run["guarantee"] == HIGH_CONFIDENCE and run["beta"] == 1e-6
    assert run["methods"]["SCP-2A"]["infeasible"] and not run["methods"]["SCP-1"]["infeasible"]
    table = (tmp_path / "tikz_figures/table_hc.tex").read_text()
    assert "$\\varepsilon = 0.10$, $\\beta = 10^{-6}$" in table and "infeasible" in table
    boxplot = (tmp_path / "tikz_figures/fig_hc_boxplot.tex").read_text()
    assert "nan" not in boxplot and "% SCP-2A: no results (infeasible)" in boxplot


def test_check_run_rejects_uncertified_high_confidence_runs():
    run = run_single(
        seed=101,
        n_total=600,
        n_cal=300,
        target=Target(0.1, beta=1e-6),
        timestep=30,
        data="Vehicle",
        methods=("SCP-1", "RemMCP"),
    )
    _check_run(run)
    run.methods["RemMCP"].predictor.coverage_confidence = lambda epsilon=None: 0.9
    with pytest.raises(AssertionError, match="RemMCP certifies beta=1.000e-01"):
        _check_run(run)


# ===========================================================================
# Command line
# ===========================================================================


def test_parse_args_defaults():
    args = parse_args([])
    assert args.data == ["CASP", "Vehicle", "SCM20D"]
    assert args.n_runs == 1 and args.epsilons == [0.05] and args.methods is None
    assert args.betas is None


def test_experiment_names_mark_the_guarantee():
    assert _experiment_name("p_", "Vehicle", 30, 2, [0.05]) == "p_Vehicle30_eps5"
    assert _experiment_name("p_", "Vehicle", 30, 2, [0.05], [1e-6]) == "p_Vehicle30_eps5_beta1e-06"
    assert _experiment_name("p_", "CASP", 0, 2, [0.15], [1e-6]) == "p_CASP_eps15_beta1e-06"
    assert _experiment_name("p_", "SCM20D", 0, 4, [0.05], [1e-1, 1e-6]) == (
        "p_SCM20Ddim4_eps5_betasweep"
    )
    assert _experiment_name("p_", "Vehicle", 30, 2, [0.02, 0.15]) == "p_Vehicle30_epssweep"
    assert _experiment_name("p_", "Vehicle", 30, 2, [0.025]) == "p_Vehicle30_eps2p5"  # no dot


def test_cli_runs_selected_experiment(tmp_path):
    main(
        [
            "--data",
            "Vehicle",
            "--timesteps",
            "30",
            "--n-cal",
            "200",
            "--n-test",
            "300",
            "--methods",
            "SCP-1",
            "RemMCP",
            "--epsilons",
            "0.1",
            "--name-prefix",
            "cli_",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "results" / "cli_Vehicle30_eps10_log.txt").is_file()
    log = (tmp_path / "results" / "cli_Vehicle30_eps10_log.txt").read_text()
    assert "Running 1 independent trial(s) per target (cli_Vehicle30_eps10, n_cal=200)" in log
    assert os.path.isfile(tmp_path / "tikz_figures" / "table_cli_Vehicle30_eps10.tex")


def test_cli_single_beta_skips_infeasible_methods(tmp_path):
    args = ["--data", "Vehicle", "--timesteps", "30", "--n-cal", "300", "--n-test", "300"]
    args += ["--methods", "SCP-1", "SCP-2A", "--epsilons", "0.1", "--betas", "1e-6"]
    main([*args, "--name-prefix", "cli_", "--output-dir", str(tmp_path)])
    log = (tmp_path / "results" / "cli_Vehicle30_eps10_beta1e-06_log.txt").read_text()
    assert "Targets: ['eps=0.1, beta=1e-06']" in log
    assert "Skipping SCP-2A infeasible" in log and "needs n_cal >= " in log
    saved = pickle.load(
        open(tmp_path / "results" / "cli_Vehicle30_eps10_beta1e-06_results.pkl", "rb")
    )
    methods = saved[(0.1, 1e-6)][0]["methods"]
    assert methods["SCP-2A"]["infeasible"] and methods["SCP-1"]["confidence"] >= 1 - 1e-6


def test_cli_beta_sweep(tmp_path):
    main(
        [
            "--data",
            "Vehicle",
            "--timesteps",
            "30",
            "--n-cal",
            "300",
            "--n-test",
            "300",
            "--methods",
            "SCP-1",
            "RemMCP",
            "--epsilons",
            "0.1",
            "--betas",
            "1e-2",
            "1e-6",
            "--name-prefix",
            "cli_",
            "--output-dir",
            str(tmp_path),
        ]
    )
    saved = pickle.load(
        open(tmp_path / "results" / "cli_Vehicle30_eps10_betasweep_results.pkl", "rb")
    )
    assert list(saved) == [(0.1, 1e-2), (0.1, 1e-6)]
    rho = [saved[key][0]["methods"]["SCP-1"]["rho"] for key in saved]
    assert rho[0] > rho[1]  # smaller beta, smaller outlier budget


def test_experiment_module_has_no_private_cross_module_imports():
    source = open(experiment.__file__).read()
    assert "import _" not in source and "._" not in source.replace("__", "")


# ===========================================================================
# Vehicle predictor
# ===========================================================================


def test_trajectory_predictor_constant_turn_rate():
    from MCP.data.predictor import TrajectoryPredictor

    # Straight line at v = 2 along theta = 0, then a constant turn rate of 0.1 per step.
    straight = np.array([[0.0, 0.0, 0.0, 2.0], [0.2, 0.0, 0.0, 2.0]])
    turning = np.array([[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.1, 1.0], [0.2, 0.0, 0.2, 1.0]])
    predictor = TrajectoryPredictor(dt=0.1, predict_steps=3)

    pred = predictor.predict_trajectory(straight)
    assert pred.dtype == np.float64 and pred.shape == (3, 4)
    np.testing.assert_allclose(pred[:, 0], [0.4, 0.6, 0.8])
    np.testing.assert_allclose(pred[:, 1:3], 0.0)

    pred = predictor.predict_trajectory(turning)
    np.testing.assert_allclose(pred[:, 2], [0.3, 0.4, 0.5])  # theta grows by the average turn rate
    np.testing.assert_allclose(pred[0, :2], [0.2 + 0.1 * np.cos(0.2), 0.1 * np.sin(0.2)])

    batch = predictor.predict_batched_trajectories(np.stack([turning, turning[::-1]]))
    np.testing.assert_array_equal(batch[0], pred)


# ===========================================================================
# Figure settings
# ===========================================================================


def test_tex_boxplot_axes_follow_data_and_explicit_settings(tmp_path):
    import contextlib
    import io
    import re

    from MCP.reporting import AxisRange, BoxplotAxes
    from MCP.reporting.tikz import write_tex_boxplot

    methods = ("SCP-1", "SCP-2A", "RemMCP")
    with contextlib.redirect_stdout(io.StringIO()):
        runs = main_compare(
            n_runs=3,
            targets=[Target(0.1)],
            n_total=600,
            n_cal=300,
            exp_name="axes",
            timestep=30,
            data="Vehicle",
            methods=methods,
            output_dir=str(tmp_path),
        )

    tex = (tmp_path / "tikz_figures" / "fig_axes_boxplot.tex").read_text()
    assert tex.count("xmin=0.5, xmax=3.5,") == 2  # both panels span all methods
    assert "(axis cs:0.5,0.900000)--(axis cs:3.5,0.900000)" in tex  # target line 1 - epsilon
    (cov_lo, cov_hi), (vol_lo, vol_hi) = [
        tuple(map(float, m)) for m in re.findall(r"ymin=([-\d.e]+), ymax=([-\d.e]+),", tex)
    ]
    coverages = [run.methods[m].test_cov for run in runs[Target(0.1)] for m in methods]
    assert cov_lo < min(coverages) and cov_hi > max(coverages)  # violins span all points
    assert tex.count("-- cycle;") == 3  # one coverage violin per method
    whiskers = [float(v) for v in re.findall(r"(?:lower|upper) whisker=([-\d.]+)", tex)]
    assert len(whiskers) == 6  # volume boxplots only
    assert vol_lo < min(whiskers) and vol_hi > max(whiskers)

    axes = BoxplotAxes(
        style="myboxVeh", coverage={0.1: AxisRange(0.8, 1.0, (0.85, 0.95))}, volume=AxisRange(1, 7)
    )
    write_tex_boxplot(runs, "axes_custom", methods, str(tmp_path), axes=axes)
    tex = (tmp_path / "fig_axes_custom_boxplot.tex").read_text()
    assert tex.count("myboxVeh,") == 2 and "mybox," not in tex
    assert "ymin=0.8, ymax=1, ytick={0.85,0.95}," in tex and "ymin=1, ymax=7," in tex


def test_png_axis_limits_cover_sets_but_cap_degenerate_ones():
    from MCP.reporting.plots import _axis_limits

    E = np.array([[0.0, 0.0], [1.0, 2.0]])  # residual span (1, 2)
    small = (np.eye(2) / 0.25, np.array([1.2, 1.0]))  # radius 0.5 around (1.2, 1)
    huge = (np.eye(2) * 1e-8, np.zeros(2))  # radius 1e4

    (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(E, [small], pad=0.0)
    assert (x_lo, x_hi, y_lo, y_hi) == pytest.approx((0.0, 1.7, 0.0, 2.0))

    (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(E, [huge], pad=0.0)
    assert (x_lo, x_hi, y_lo, y_hi) == pytest.approx(
        (-1.0, 2.0, -2.0, 4.0)
    )  # one span beyond the data


def test_ellipsoid_bounding_box_is_tight():
    """Half-widths sqrt(diag(Q^-1)) bound a rotated ellipse and are attained."""
    R = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    Q = R @ np.diag([1 / 4.0, 1 / 0.25]) @ R.T  # semi-axes 2 and 0.5
    c = np.array([1.0, -1.0])
    mins, maxs = experiment.ellipsoid_bounding_box([(Q, c)])
    t = np.linspace(0, 2 * np.pi, 20001)
    boundary = c + (R @ np.vstack([2 * np.cos(t), 0.5 * np.sin(t)])).T
    np.testing.assert_allclose(mins, boundary.min(axis=0), atol=1e-6)
    np.testing.assert_allclose(maxs, boundary.max(axis=0), atol=1e-6)


def test_estimate_volume_from_ellipsoid_geometry():
    """Volume of a far-reaching union matches the analytic area; an empty set has volume 0."""
    Q1, c1 = np.eye(2), np.zeros(2)  # unit disk
    Q2, c2 = np.diag([1 / 400.0, 1 / 25.0]), np.array([300.0, 0.0])  # far, large ellipse
    ellipsoids = [(Q1, c1), (Q2, c2)]

    class Union:
        def covers(self, X, Y):
            return np.any(
                [np.einsum("ni,ij,nj->n", Y - c, Q, Y - c) <= 1.0 for Q, c in ellipsoids], axis=0
            )

    exact = np.pi * (1.0 + 20.0 * 5.0)
    reference = np.random.default_rng(0).standard_normal((100, 2))  # data far from the big ellipse
    volume = experiment.estimate_volume(Union(), reference, ellipsoids=ellipsoids)
    assert volume == pytest.approx(exact, rel=0.05)
    assert experiment.estimate_volume(Union(), reference, ellipsoids=[]) == 0.0
