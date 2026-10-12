"""
Tests for the reporting of coverage targets: sweep-aware labels, certified
misconfidence in tables and console summaries, and figures with infeasible methods.
"""

import contextlib
import io

import numpy as np
import pytest

from MCP.methods import Target
from MCP.reporting.io import print_summary
from MCP.reporting.labels import plain_label, tex_fixed_levels, tex_label, tex_number
from MCP.reporting.tikz import write_tex_boxplot, write_tex_table
from MCP.results import MethodResult, RunResult


class _Certified:
    """Stand-in for a calibrated predictor with a given certified beta."""

    def __init__(self, beta: float):
        self.beta = beta

    def coverage_confidence(self, epsilon=None):
        return 1.0 - self.beta


def _run(target: Target, seed: int, betas: dict[str, float | None]) -> RunResult:
    """One run; a beta of None marks a failed method, a missing beta an infeasible one."""
    run = RunResult(seed=seed, target=target, E_cal=np.zeros((4, 2)), E_test=np.zeros((4, 2)))
    for name in ("SCP-1", "SCP-2A", "H-RelMCP-A"):
        if name not in betas:
            run.methods[name] = MethodResult(name, error="infeasible", infeasible=True)
        elif betas[name] is None:
            run.methods[name] = MethodResult(name, error="no certified solution", uncertified=True)
        else:
            run.methods[name] = MethodResult(
                name,
                predictor=_Certified(betas[name]),
                n_cal=4,
                cal_cov=0.97,
                test_cov=0.96 + 0.01 * seed,
                cal_time=0.1,
                volume=2.0 + seed,
            )
    return run


@pytest.fixture
def runs_by_target():
    """beta sweep at epsilon = 0.05: SCP-2A infeasible at beta = 1e-6, H-RelMCP-A fails once."""
    loose, tight = Target(0.05, beta=1e-2), Target(0.05, beta=1e-6)
    return {
        loose: [
            _run(loose, 0, {"SCP-1": 8e-3, "SCP-2A": 9e-3, "H-RelMCP-A": 5e-3}),
            _run(loose, 1, {"SCP-1": 8e-3, "SCP-2A": 9e-3, "H-RelMCP-A": 7e-3}),
        ],
        tight: [
            _run(tight, 0, {"SCP-1": 6.9e-7, "H-RelMCP-A": 9.4e-7}),
            _run(tight, 1, {"SCP-1": 6.9e-7, "H-RelMCP-A": None}),
        ],
    }


METHODS = ("SCP-1", "SCP-2A", "H-RelMCP-A")


def test_labels_show_the_levels_that_vary():
    beta_sweep = [Target(0.05, beta=1e-1), Target(0.05, beta=1e-6)]
    assert tex_label(beta_sweep[1], beta_sweep) == "$\\beta = 10^{-6}$"
    assert tex_fixed_levels(beta_sweep) == "$\\varepsilon = 0.05$"
    assert plain_label(beta_sweep[0], beta_sweep) == "beta=1e-01"

    eps_sweep = [Target(0.02, beta=1e-6), Target(0.15, beta=1e-6)]
    assert tex_label(eps_sweep[0], eps_sweep) == "$\\varepsilon = 0.02$"
    assert tex_fixed_levels(eps_sweep) == "$\\beta = 10^{-6}$"

    expected = [Target(0.05), Target(0.15)]
    assert tex_label(expected[0], expected) == "$\\varepsilon = 0.05$"
    assert tex_fixed_levels(expected) == "expected guarantee"

    single = [Target(0.05, beta=5e-7)]
    assert tex_label(single[0], single) == "$\\varepsilon = 0.05$, $\\beta = 5{\\cdot}10^{-7}$"
    assert (
        tex_fixed_levels(single) == "" and plain_label(single[0], single) == "eps=0.05, beta=5e-07"
    )


@pytest.mark.parametrize(
    "value, text",
    [(0.4857, "0.4857"), (0.001, "0.0010"), (9.37e-8, "9.4{\\cdot}10^{-8}"), (1e-6, "10^{-6}")],
)
def test_tex_number(value, text):
    assert tex_number(value) == text


def test_table_reports_certified_beta_infeasible_and_failed_runs(tmp_path, runs_by_target):
    write_tex_table(runs_by_target, "rep", METHODS, str(tmp_path))
    tex = (tmp_path / "table_rep.tex").read_text()
    assert "& $\\beta = 10^{-2}$ & $\\beta = 10^{-6}$ \\\\" in tex
    assert "differ). Target: $\\varepsilon = 0.05$.}" in tex  # fixed level in the caption
    rows = {line.split("&")[1].strip(): line for line in tex.splitlines() if line.startswith("  &")}
    beta_rows = [line for line in tex.splitlines() if "Certified $\\bar{\\beta}$" in line]
    # SCP-1: equal in all runs -> one value; SCP-2A: infeasible at 1e-6;
    # H-RelMCP-A: mean and max at 1e-2, the single successful run at 1e-6.
    assert beta_rows[0].endswith("& $0.0080$ & $6.9{\\cdot}10^{-7}$ \\\\")
    assert beta_rows[1].endswith("& $0.0090$ & infeasible \\\\")
    assert beta_rows[2].endswith("& $0.0060$ (max $0.0070$) & $9.4{\\cdot}10^{-7}$ \\\\")
    assert rows["No certified solution"].endswith("& 0/2 & 1/2 \\\\")
    assert tex.count("No certified solution") == 1  # only for the method that failed
    assert "Errors" not in tex  # no unexpected failures
    assert "\\multirow{3}{*}{H-RelMCP-A}" in tex and "\\multirow{2}{*}{SCP-1}" in tex


def test_boxplot_leaves_infeasible_methods_empty(tmp_path, runs_by_target):
    write_tex_boxplot(runs_by_target, "rep", METHODS, str(tmp_path))
    tex = (tmp_path / "fig_rep_boxplot.tex").read_text()
    assert "nan" not in tex
    assert tex.count("% SCP-2A: no results (infeasible)") == 2  # coverage and volume panels
    assert "\\caption{$\\beta = 10^{-6}$}" in tex
    assert "\\label{fig:rep_box_eps050_beta1e-06}" in tex


def test_summary_prints_certified_beta_and_failures(runs_by_target):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_summary(runs_by_target, METHODS)
    text = out.getvalue()
    tight = text[text.index("Target: eps=0.05, beta=1e-06") :]
    assert "Target beta                        1.00e-06" in tight
    max_row = next(line for line in tight.splitlines() if line.startswith("Certified beta (max)"))
    assert max_row.split()[3:] == ["6.90e-07", "infeasible", "9.40e-07"]
    uncertified_row = next(
        line for line in tight.splitlines() if line.startswith("No certified solution")
    )
    assert uncertified_row.split()[3:] == ["0/2", "infeasible", "1/2"]
    errors_row = next(line for line in tight.splitlines() if line.startswith("Errors"))
    assert errors_row.split()[1:] == ["0/2", "infeasible", "0/2"]
    min_row = next(line for line in tight.splitlines() if line.startswith("Min. test coverage"))
    assert min_row.split()[3:] == ["0.9600", "--", "0.9600"]


# ===========================================================================
# Colors and axis ranges
# ===========================================================================


def test_methods_use_the_paper_colors():
    from MCP.methods import METHODS, PAPER_COLORS

    legend = {  # legend of the paper figures
        "SCP-1": ("scp1color", ""),
        "SCP-2A": ("scp3color", ""),
        "SCP-3A": ("scp4color", ""),
        "RemMCP": ("cmcp1color", ""),
        "H-RelMCP-A": ("nmcp1color", ""),
        "SCP-2B": ("scp3color", "dashed"),
        "SCP-3B": ("scp4color", "dashed"),
        "H-RelMCP-B": ("nmcp1color", "dashed"),
    }
    for name, (color, style) in legend.items():
        assert (METHODS[name].tikz_color, METHODS[name].tikz_style) == (color, style)
    assert all(spec.tikz_color in PAPER_COLORS for spec in METHODS.values())
    assert METHODS["RemMCP"].color == "#4472C4"  # RGB 68, 114, 196


def test_coverage_axis_starts_below_target_or_lowest_point():
    from MCP.reporting.axes import coverage_range

    above = [np.array([0.97, 0.98]), np.array([0.975, 0.99])]
    # 5% of the shown span below the target and above the highest point
    assert coverage_range(above, Target(0.05, beta=1e-6)) == pytest.approx((0.948, 0.992))
    below = [np.array([0.94, 0.98]), np.array([np.nan, 0.99])]
    lo, _ = coverage_range(below, Target(0.05, beta=1e-6))
    assert lo == pytest.approx(0.9375)  # just below the lowest point, which is below 0.95
    # expected guarantee: just below the lowest point, whether or not it is below the target
    assert coverage_range(above, Target(0.05)) == pytest.approx((0.969, 0.991))


def test_volume_axis_spans_the_whiskers_and_clips_extremes():
    from MCP.reporting.axes import outliers, volume_range
    from MCP.reporting.stats import boxplot_stats

    regular = np.linspace(50.0, 60.0, 21)
    degenerate = np.append(np.linspace(40.0, 45.0, 20), 50000.0)
    stats = [boxplot_stats(regular), boxplot_stats(degenerate)]
    assert volume_range(stats) == pytest.approx((39.0, 61.0))  # 50,000 lies outside
    np.testing.assert_array_equal(outliers(degenerate, stats[1]), [50000.0])
    assert volume_range([boxplot_stats(np.array([0.1, 0.2, 0.3]))])[0] >= 0.0


def test_boxplot_draws_outliers_and_marks_means_above_the_axis(tmp_path):
    target = Target(0.05, beta=1e-6)
    runs = []
    volumes = [*np.linspace(40.0, 45.0, 19), 50000.0]
    for seed, volume in enumerate(volumes):
        run = _run(target, seed % 2, {"SCP-1": 1e-7, "SCP-2A": 1e-7, "H-RelMCP-A": 1e-7})
        run.methods["SCP-2A"].volume = volume
        run.methods["SCP-1"].test_cov = 0.94 if seed == 0 else 0.97  # one point below target
        runs.append(run)
    write_tex_boxplot({target: runs}, "out", METHODS, str(tmp_path))
    tex = (tmp_path / "fig_out_boxplot.tex").read_text()
    coverage = tex.split("bottom plot")[0]
    assert ",0.940000)" in coverage  # the violin reaches the point below the target
    assert float(coverage.split("ymin=")[1].split(",")[0]) < 0.94  # and the axis shows it
    assert "% SCP-2A: mean 2540.38 lies above the axis" in tex
    assert "mark=triangle*" in tex and "(0,50000" not in tex  # clipped outlier left out


# ===========================================================================
# Violins and regenerating figures from saved results
# ===========================================================================


def test_violin_outline_spans_the_observed_range():
    from MCP.reporting.stats import violin_outline

    values = np.random.default_rng(0).beta(495, 6, 100)
    y, half_width = violin_outline(values)
    assert (y[0], y[-1]) == (values.min(), values.max())
    assert half_width.max() == 1.0 and np.all(half_width > 0)
    assert violin_outline(np.full(10, 0.97)) is None  # no density for constant values


def test_coverage_violins_in_tikz(tmp_path, runs_by_target):
    write_tex_boxplot(runs_by_target, "vio", METHODS, str(tmp_path))
    tex = (tmp_path / "fig_vio_boxplot.tex").read_text()
    coverage = tex.split("bottom plot")[0]
    # two coverage values per method (seeds 0, 1): violins for SCP-1 and SCP-2A at 1e-2,
    # H-RelMCP-A has values in both targets; SCP-2A is infeasible at 1e-6
    assert coverage.count("-- cycle;") == 3
    assert "pattern=" not in coverage  # none of these methods is hatched
    assert "rectangle (axis cs:" in coverage and "mark=diamond*" in coverage


def test_regenerate_reproduces_figures_and_tables(tmp_path, capsys):
    import os

    from MCP.experiment import main_compare
    from MCP.reporting.regenerate import main as regenerate_main

    methods = ("SCP-1", "SCP-2A", "RemMCP", "H-RelMCP-A")
    targets = [Target(0.1, beta=1e-3), Target(0.1)]
    main_compare(
        n_runs=2,
        targets=targets,
        n_total=500,
        n_cal=200,
        exp_name="regen",
        timestep=30,
        data="Vehicle",
        methods=methods,
        output_dir=str(tmp_path / "run"),
    )
    capsys.readouterr()
    pickle_path = tmp_path / "run" / "results" / "regen_results.pkl"
    regenerate_main([str(pickle_path), "--output-dir", str(tmp_path / "again")])
    summary = capsys.readouterr().out
    assert "Target: eps=0.1, beta=1e-03" in summary and "Iterations Alg. 1" in summary

    for rel in ("tikz_figures/table_regen.tex", "tikz_figures/fig_regen_boxplot.tex"):
        original = (tmp_path / "run" / rel).read_text()
        assert (tmp_path / "again" / rel).read_text() == original, rel
    assert os.path.isfile(tmp_path / "again" / "figures" / "fig_regen_boxplot.png")

    regenerate_main([str(pickle_path), "--methods", "RemMCP", "SCP-1"])  # next to results/
    table = (tmp_path / "run" / "tikz_figures" / "table_regen.tex").read_text()
    assert "SCP-2A" not in table and table.index("RemMCP") < table.index("SCP-1")
    with pytest.raises(ValueError, match="are not in the results of regen"):
        regenerate_main([str(pickle_path), "--methods", "H-RelMCP-B"])


def test_combine_sweep_values_run_as_separate_experiments(tmp_path, capsys):
    import pickle

    from MCP.experiment import main_compare
    from MCP.reporting.regenerate import main as regenerate_main

    methods = ("SCP-1", "RemMCP")
    paths = []
    for beta in (1e-1, 1e-3):
        main_compare(
            n_runs=2,
            targets=[Target(0.1, beta=beta)],
            n_total=500,
            n_cal=200,
            exp_name=f"part_beta{beta:.0e}",
            timestep=30,
            data="Vehicle",
            methods=methods,
            output_dir=str(tmp_path),
        )
        paths.append(str(tmp_path / "results" / f"part_beta{beta:.0e}_results.pkl"))
    capsys.readouterr()

    regenerate_main([*paths, "--combine", "sweep"])
    assert "Target: eps=0.1, beta=1e-01" in capsys.readouterr().out
    table = (tmp_path / "tikz_figures" / "table_sweep.tex").read_text()
    assert "& $\\beta = 10^{-1}$ & $\\beta = 10^{-3}$ \\\\" in table  # panels in given order
    boxplot = (tmp_path / "tikz_figures" / "fig_sweep_boxplot.tex").read_text()
    assert boxplot.count("% SUBFIGURE:") == 2
    merged = pickle.load(open(tmp_path / "results" / "sweep_results.pkl", "rb"))
    assert list(merged) == [(0.1, 1e-1), (0.1, 1e-3)]
    regenerate_main([str(tmp_path / "results" / "sweep_results.pkl")])  # like any other

    with pytest.raises(ValueError, match="appears in more than one file"):
        regenerate_main([paths[0], paths[0], "--combine", "twice"])


def test_combine_experiments_with_the_same_target_by_label(tmp_path, capsys):
    import pickle

    from MCP.experiment import main_compare
    from MCP.reporting.regenerate import main as regenerate_main

    paths = []
    for T in (2, 30):  # stand-ins for the SCM20D output dimensions: same target, other data
        main_compare(
            n_runs=2,
            targets=[Target(0.1, beta=1e-3)],
            n_total=500,
            n_cal=200,
            exp_name=f"part_T{T}",
            timestep=T,
            data="Vehicle",
            methods=("SCP-1", "RemMCP"),
            output_dir=str(tmp_path),
        )
        paths.append(str(tmp_path / "results" / f"part_T{T}_results.pkl"))
    capsys.readouterr()

    with pytest.raises(ValueError, match="give one label per file"):
        regenerate_main([*paths, "--combine", "nolabels"])
    with pytest.raises(ValueError, match="Got 1 labels for 2 files"):
        regenerate_main([*paths, "--combine", "few", "--labels", "$T = 2$"])
    with pytest.raises(SystemExit):
        regenerate_main([*paths, "--labels", "$T = 2$", "$T = 30$"])  # needs --combine
    capsys.readouterr()

    regenerate_main([*paths, "--combine", "horizons", "--labels", "$T = 2$", "$T = 30$"])
    assert "Target: eps=0.1, beta=1e-03 [T = 30]" in capsys.readouterr().out
    table = (tmp_path / "tikz_figures" / "table_horizons.tex").read_text()
    assert "& $T = 2$ & $T = 30$ \\\\" in table
    assert "Target: $\\varepsilon = 0.10$, $\\beta = 10^{-3}$." in table  # shared levels
    boxplot = (tmp_path / "tikz_figures" / "fig_horizons_boxplot.tex").read_text()
    assert "\\caption{$T = 30$}" in boxplot
    assert "\\label{fig:horizons_box_eps100_beta1e-03_T30}" in boxplot  # unique labels
    merged = pickle.load(open(tmp_path / "results" / "horizons_results.pkl", "rb"))
    assert list(merged) == [(0.1, 1e-3, "$T = 2$"), (0.1, 1e-3, "$T = 30$")]
    regenerate_main([str(tmp_path / "results" / "horizons_results.pkl")])  # keeps the labels
    assert "[T = 2]" in capsys.readouterr().out


def test_target_variants_only_label_panels():
    variant = Target(0.05, beta=1e-6, variant="$n_y = 4$")
    assert variant != Target(0.05, beta=1e-6) and variant.plain_variant == "n_y = 4"
    assert variant.predictor_kwargs() == {"epsilon": 0.05, "beta": 1e-6}
    assert variant.label == "eps=0.05, beta=1e-06 [n_y = 4]"
    targets = [Target(0.05, 1e-6, "$n_y = 2$"), variant]
    assert [plain_label(t, targets) for t in targets] == ["n_y = 2", "n_y = 4"]
    assert tex_fixed_levels(targets) == "$\\varepsilon = 0.05$, $\\beta = 10^{-6}$"


def test_tee_log_shows_each_line_immediately(tmp_path):
    from MCP.reporting.io import Tee

    path = tmp_path / "log.txt"
    with Tee(str(path)) as tee:
        tee.write("  Run  1 (seed=101): ...\n")
        assert path.read_text() == "  Run  1 (seed=101): ...\n"  # before closing the log


def test_coverage_and_volume_axes_line_up(tmp_path, runs_by_target):
    """Both panels of a subfigure start at the same position and reserve the same tick width."""
    write_tex_boxplot(runs_by_target, "align", METHODS, str(tmp_path))
    tex = (tmp_path / "fig_align_boxplot.tex").read_text()
    assert "\\hfill\\begin{tikzpicture}" not in tex
    assert tex.count("yticklabel style={text width=2.2em, align=right, inner xsep=1pt}") == (
        2 * len(runs_by_target)
    )
    assert "ylabel={Coverage\\strut}" in tex and "ylabel={Volume\\strut}" in tex


def test_boxplot_panels_share_one_picture_with_fixed_axis_sizes(tmp_path, runs_by_target):
    """The volume axis hangs below the coverage axis; sizes and subfigure widths are fixed."""
    from MCP.reporting.tikz import BoxplotAxes

    axes = BoxplotAxes(
        style="mybox", axis_width="3.5cm", axis_height="2cm", subfigure_widths=("0.4", "0.3")
    )
    write_tex_boxplot(runs_by_target, "fixed", METHODS, str(tmp_path), axes=axes)
    tex = (tmp_path / "fig_fixed_boxplot.tex").read_text()
    n = len(runs_by_target)
    assert tex.count("\\begin{tikzpicture}[baseline]") == n  # one picture per subfigure
    assert tex.count("at={(coverage.south west)}, anchor=north west") == n
    assert tex.count("scale only axis, width=3.5cm, height=2cm,") == 2 * n
    assert tex.count("enlarge x limits=false,") == 2 * n
    assert "\\begin{subfigure}[t]{0.4\\textwidth}" in tex and "{0.3\\textwidth}" in tex
