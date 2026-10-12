"""TikZ/PGFPlots figures and LaTeX tables for the paper.

The generated files use axis styles (default: myplotSet, mybox) and colors
(scp1color, ...) defined in the paper preamble.  Prediction sets are always
rendered in the first two output dimensions.  Axis ranges of the boxplots are
derived from the data unless given explicitly via BoxplotAxes.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from MCP.methods import METHODS, MethodSpec, Target
from MCP.reporting.axes import coverage_range, outliers, volume_range
from MCP.reporting.labels import (
    all_infeasible,
    certified_betas,
    n_errors,
    n_uncertified,
    tex_fixed_levels,
    tex_label,
    tex_number,
)
from MCP.reporting.stats import boxplot_stats, method_mean_std, method_values, violin_outline
from MCP.results import RunResult


def _beta_suffix(target: Target) -> str:
    suffix = "" if target.beta is None else f"_beta{target.beta:.0e}"
    if target.plain_variant is not None:
        suffix += "_" + re.sub(r"[^A-Za-z0-9]+", "", target.plain_variant)
    return suffix


def _ellipse_geometry(Q_eff: np.ndarray, c: np.ndarray):
    evals, evecs = np.linalg.eigh(Q_eff)
    if np.any(evals <= 0):
        raise ValueError("ellipse precision matrix must be positive definite")
    semi_axes = 1.0 / np.sqrt(evals)
    angle = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0]))
    return semi_axes, angle, np.asarray(c, dtype=float)


def _parametric_ellipse_addplot(
    style_name: str, color_name: str, Q_eff: np.ndarray, c: np.ndarray
) -> list[str]:
    semi_axes, angle, center = _ellipse_geometry(Q_eff, c)
    cx, cy = float(center[0]), float(center[1])
    a, b = float(semi_axes[0]), float(semi_axes[1])
    return [
        f"\\addplot[{style_name},{color_name}, line width=1.5pt, fill={color_name}, fill opacity=0.,",
        "  domain=0:360, samples=120,",
        "  smooth, tension=0.5,",
        "  variable=\\t,",
        f"] ({{{cx:.6f} + {a:.6f}*cos(\\t)*cos({angle:.6f}) - {b:.6f}*sin(\\t)*sin({angle:.6f})}},",
        f"   {{{cy:.6f} + {a:.6f}*cos(\\t)*sin({angle:.6f}) + {b:.6f}*sin(\\t)*cos({angle:.6f})}});",
    ]


def _sets_figure_lines(run: RunResult, methods: Sequence[str], axis_style: str) -> list[str]:
    E_cal = np.asarray(run.E_cal, dtype=float)
    if E_cal.ndim == 2 and E_cal.shape[1] >= 2:
        E_cal = E_cal[:, :2]

    lines = ["  \\begin{tikzpicture}", "    \\begin{axis}[", f"      {axis_style},", "    ]"]
    for m in methods:
        spec = METHODS[m]
        for Q_full, c_full in run.methods[m].ellipsoids:
            Q_eff = np.asarray(Q_full, dtype=float)[:2, :2]
            c = np.asarray(c_full, dtype=float)[:2]
            lines.append(f"      % --- {m} ---")
            lines.extend(
                "      " + s
                for s in _parametric_ellipse_addplot(spec.tikz_style, spec.tikz_color, Q_eff, c)
            )

    lines += [
        "      % --- Calibration error scatter ---",
        "      \\addplot[",
        "        only marks,",
        "        mark=asterisk,",
        "        mark size=1.7pt,",
        "        color=black,",
        "        mark options={fill=black, draw=black,draw opacity=0.2},",
        "      ] coordinates {",
    ]
    lines += [f"        ({float(x):.6f},{float(y):.6f})" for x, y in E_cal]
    lines += ["      };", "    \\end{axis}", "  \\end{tikzpicture}"]
    return lines


def write_tex_sets(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    methods: Sequence[str],
    out_dir: str,
    axis_style: str = "myplotSet",
) -> None:
    """One TikZ figure per target with the prediction sets of the first run."""
    os.makedirs(out_dir, exist_ok=True)
    targets = list(runs_by_target)
    for idx, target in enumerate(targets):
        filename = os.path.join(out_dir, f"fig_{exp_name}_sets{idx}.tex")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("% Auto-generated TikZ/PGFPlots figure\n")
            runs = runs_by_target[target]
            if not runs:
                continue
            for line in _sets_figure_lines(runs[0], methods, axis_style):
                f.write(line + "\n")
            if idx < len(targets) - 1 and idx % 2 == 1:
                f.write("\\par\\medskip\n")


def _tex_beta_cell(runs: list[RunResult], m: str) -> str:
    """Certified beta of one method: its value if equal in all runs, else mean and maximum."""
    if all_infeasible(runs, m):
        return "infeasible"
    betas = certified_betas(runs, m)
    if betas.size == 0:
        return "--"
    lo, hi = float(betas.min()), float(betas.max())
    if hi - lo <= 1e-9 * hi:
        return f"${tex_number(hi)}$"
    return f"${tex_number(float(betas.mean()))}$ (max ${tex_number(hi)}$)"


def write_tex_table(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    methods: Sequence[str],
    out_dir: str,
) -> None:
    """LaTeX table with computation time and certified misconfidence per target.

    The certified misconfidence beta_bar of coverage >= 1 - epsilon is shown
    once if it is equal in all runs (SCP, RemMCP), else as mean and maximum
    over the runs (H-RelMCP).  Methods that cannot certify the target are
    marked "infeasible"; rows with the number of runs without a certified
    solution and with errors are added for methods where these occurred.
    """
    targets = list(runs_by_target)
    if not targets:
        raise ValueError("runs_by_target is empty")

    def _fmt_time(runs: list[RunResult], m: str) -> str:
        mu, sd = method_mean_std(runs, m, "cal_time")
        if not np.isfinite(mu) or not np.isfinite(sd):
            return "--"
        return f"${mu:.4f} \\pm {sd:.4f}$"

    def _fmt_count(runs: list[RunResult], m: str, count) -> str:
        return "--" if all_infeasible(runs, m) else f"{count(runs, m)}/{len(runs)}"

    headers = " ".join(f"& {tex_label(target, targets)}" for target in targets)
    fixed = tex_fixed_levels(targets)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Computation time [s] (mean$\\pm$standard deviation) and certified misconfidence"
        " $\\bar{\\beta}$ of achieving coverage $1-\\varepsilon$ (mean and maximum over the runs"
        f" if they differ).{' Target: ' + fixed + '.' if fixed else ''}}}",
        "\\label{tab:resultsComplex}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{ll{'c' * len(targets)}}}",
        "\\toprule",
        f"\\textbf{{Method}} & \\textbf{{Metric}} {headers} \\\\",
        "\\midrule",
    ]
    runs_per_target = [runs_by_target[target] for target in targets]
    for m in methods:
        rows = [
            ("Computation time", [_fmt_time(runs, m) for runs in runs_per_target]),
            ("Certified $\\bar{\\beta}$", [_tex_beta_cell(runs, m) for runs in runs_per_target]),
        ]
        for label, count in (("No certified solution", n_uncertified), ("Errors", n_errors)):
            if any(count(runs, m) for runs in runs_per_target):
                rows.append((label, [_fmt_count(runs, m, count) for runs in runs_per_target]))
        lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{m}}}")
        for label, cells in rows:
            lines.append(f"  & {label} & {' & '.join(cells)} \\\\")
        if m != methods[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"table_{exp_name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@dataclass(frozen=True)
class AxisRange:
    """y-axis range of a boxplot panel; ticks=None lets PGFPlots choose."""

    ymin: float
    ymax: float
    ticks: tuple[float, ...] | None = None

    def option_line(self) -> str:
        line = f"      ymin={self.ymin:.4g}, ymax={self.ymax:.4g},"
        if self.ticks:
            line += " ytick={" + ",".join(f"{t:g}" for t in self.ticks) + "},"
        return line


@dataclass(frozen=True)
class BoxplotAxes:
    """Axis settings of the TikZ boxplots.

    Attributes:
        style: PGFPlots style applied to every panel.
        coverage: Coverage-axis range per epsilon; missing epsilons follow
            MCP.reporting.axes.coverage_range.
        volume: Volume-axis range; None follows MCP.reporting.axes.volume_range.
        axis_width: Width of each axis box without labels (e.g. "3.6cm"); None
            keeps the sizes of style, which include the labels.
        axis_height: Height of each axis box without labels (with axis_width).
        gap: Vertical gap between the coverage and the volume axis box.
        subfigure_widths: Subfigure widths as fractions of the text width, one
            per target; None uses 0.27 for the first and 0.225 for the others.
    """

    style: str = "mybox"
    coverage: dict[float, AxisRange] = field(default_factory=dict)
    volume: AxisRange | None = None
    axis_width: str | None = None
    axis_height: str = "2.4cm"
    gap: str = "0.15cm"
    subfigure_widths: tuple[str, ...] | None = None

    def size_line(self) -> str:
        if self.axis_width is None:
            return ""
        return f"      scale only axis, width={self.axis_width}, height={self.axis_height},"


def _boxplot_lines(
    method_idx: int,
    color_name: str,
    style_name: str,
    stats,
    mean_value: float,
    outlier_values: Sequence[float] = (),
) -> list[str]:
    _, _, _, q1, med, q3, wl, wh = stats
    if style_name == "dashed":
        line_opts = [
            "pattern={Lines[angle=45, distance=2pt, line width=0.8pt]},",
            f"pattern color={color_name},",
        ]
    else:
        line_opts = [f"fill={color_name},"]
    return [
        "        \\addplot[",
        "          boxplot prepared={",
        f"            lower whisker={wl:.6f},",
        f"            lower quartile={q1:.6f},",
        f"            median={med:.6f},",
        f"            upper quartile={q3:.6f},",
        f"            upper whisker={wh:.6f},",
        f"            draw position={method_idx},",
        "            box extend=0.6,",
        "          },",
        *[f"          {opt}" for opt in line_opts],
        "          draw=black!70, line width=0.55pt,",
        "          mark=o, mark size=1pt, mark options={draw=black!60, fill=none, solid},",
        f"        ] coordinates {{{' '.join(f'(0,{v:.6f})' for v in outlier_values)}}};",
        "        \\addplot[only marks, mark=diamond*, mark size=2.0pt,",
        "          mark options={fill=black, draw=black},",
        f"        ] coordinates {{({method_idx},{mean_value:.6f})}};",
    ]


def _method_boxes(
    specs: list[MethodSpec],
    values: list[np.ndarray],
    stats: list[tuple],
    runs: list[RunResult],
    y_range: AxisRange,
) -> list[str]:
    """Boxplot commands per method; methods without results only get a comment.

    Outliers outside y_range are left out (they would be clipped), and means
    above it are marked by a triangle at the upper edge of the axis.
    """
    lines = []
    for method_idx, (spec, method_values_, method_stats) in enumerate(
        zip(specs, values, stats, strict=True), start=1
    ):
        if not np.isfinite(method_stats[0]):
            reason = "infeasible" if all_infeasible(runs, spec.name) else "all runs failed"
            lines.append(f"      % {spec.name}: no results ({reason})")
            continue
        shown = [
            float(v)
            for v in outliers(method_values_, method_stats)
            if y_range.ymin <= v <= y_range.ymax
        ]
        mean = float(method_stats[1])
        lines.append(f"      % {spec.name}")
        lines.extend(
            _boxplot_lines(method_idx, spec.tikz_color, spec.tikz_style, method_stats, mean, shown)
        )
        if mean > y_range.ymax:  # marked by a triangle at the upper edge
            lines += [
                f"      % {spec.name}: mean {mean:.6g} lies above the axis",
                "        \\addplot[only marks, mark=triangle*, mark size=2.2pt, clip=false,",
                "          mark options={fill=black, draw=black},",
                f"        ] coordinates {{({method_idx},{y_range.ymax:.6g})}};",
            ]
    return lines


VIOLIN_HALF_WIDTH = 0.32  # maximum half-width of a violin in x units (as in the PNGs)
INNER_HALF_WIDTH = 0.08  # half-width of the box inside a violin


def _fill_options(color_name: str, style_name: str) -> str:
    if style_name == "dashed":
        return f"pattern={{Lines[angle=45, distance=2pt, line width=0.8pt]}}, pattern color={color_name}"
    return f"fill={color_name}"


def _violin_lines(x: int, color_name: str, style_name: str, values: np.ndarray) -> list[str]:
    """Violin (stats.violin_outline) with an inner box and the mean as a diamond."""
    _, mean, _, q1, med, q3, wl, wh = boxplot_stats(values)
    lines = []
    outline = violin_outline(values)
    if outline is not None:
        y, half = outline[0], VIOLIN_HALF_WIDTH * outline[1]
        left = [f"({x - h:.4f},{v:.6f})" for v, h in zip(y, half, strict=True)]
        right = [f"({x + h:.4f},{v:.6f})" for v, h in zip(y[::-1], half[::-1], strict=True)]
        lines += [
            f"        \\addplot[{_fill_options(color_name, style_name)}, draw=black!70,"
            " line width=0.45pt]",
            f"          coordinates {{{' '.join(left + right)}}} -- cycle;",
        ]
    w = INNER_HALF_WIDTH
    lines += [
        f"        \\draw[black!80, line width=0.5pt] (axis cs:{x},{wl:.6f}) -- (axis cs:{x},{wh:.6f});",
        f"        \\draw[fill=white, draw=black!80, line width=0.5pt] (axis cs:{x - w},{q1:.6f})"
        f" rectangle (axis cs:{x + w},{q3:.6f});",
        f"        \\draw[black, line width=0.8pt] (axis cs:{x - w},{med:.6f})"
        f" -- (axis cs:{x + w},{med:.6f});",
        "        \\addplot[only marks, mark=diamond*, mark size=1.8pt,",
        "          mark options={fill=black, draw=black},",
        f"        ] coordinates {{({x},{mean:.6f})}};",
    ]
    return lines


def _method_violins(
    specs: list[MethodSpec], values: list[np.ndarray], runs: list[RunResult]
) -> list[str]:
    """Violin commands per method; methods without results only get a comment."""
    lines = []
    for x, (spec, method_values_) in enumerate(zip(specs, values, strict=True), start=1):
        if method_values_.size == 0:
            reason = "infeasible" if all_infeasible(runs, spec.name) else "all runs failed"
            lines.append(f"      % {spec.name}: no results ({reason})")
            continue
        lines.append(f"      % {spec.name}")
        lines.extend(_violin_lines(x, spec.tikz_color, spec.tikz_style, method_values_))
    return lines


# Equal tick-label widths (and struts in the y labels, "Coverage" has a descender), so that
# the coverage and volume axes of a subfigure line up.
TICK_LABEL_STYLE = (
    "      yticklabel style={text width=2.2em, align=right, inner xsep=1pt},"
    " major tick length=0pt,"  # the ticks are not drawn: no space for them
)

# The picture ends at the bottom of the volume axis, also where the lowest tick label reaches
# below it, so that all subfigures have the same depth and their captions line up.
BOTTOM_AT_VOLUME_AXIS = [
    "  \\coordinate (bbnw) at (current bounding box.north west);",
    "  \\coordinate (bbse) at (current bounding box.south east);",
    "  \\pgfresetboundingbox",
    "  \\useasboundingbox (bbnw) rectangle (volume.south -| bbse);",
]


def write_tex_boxplot(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    methods: Sequence[str],
    out_dir: str,
    axes: BoxplotAxes | None = None,
) -> None:
    """TikZ violins of test coverage and boxplots of set volume, one subfigure per target.

    Subfigures are captioned with the levels that vary across the targets.
    Methods without results (infeasible or failed in all runs) leave an empty
    slot.  Axis ranges follow MCP.reporting.axes unless given in axes; points
    beyond the whiskers are drawn as outliers.
    """
    targets = list(runs_by_target)
    if not targets:
        raise ValueError("runs_by_target is empty")
    axes = axes or BoxplotAxes()
    specs = [METHODS[m] for m in methods]
    x_range = f"      xmin=0.5, xmax={len(methods) + 0.5:g},"

    lines = ["% Auto-generated TikZ/PGFPlots boxplot figure"]
    for eps_idx, target in enumerate(targets):
        runs = runs_by_target[target]
        eps = target.epsilon
        cov_values = [method_values(runs, m, "test_cov") for m in methods]
        vol_values = [method_values(runs, m, "volume") for m in methods]
        vol_stats = [boxplot_stats(v) for v in vol_values]
        cov_range = axes.coverage.get(eps) or AxisRange(*coverage_range(cov_values, target))
        vol_range = axes.volume or AxisRange(*volume_range(vol_stats))

        width = (
            axes.subfigure_widths[eps_idx]
            if axes.subfigure_widths is not None
            else ("0.225" if eps_idx > 0 else "0.27")
        )
        size = [axes.size_line()] if axes.size_line() else []
        lines += [
            "% =========================================================",
            f"% SUBFIGURE: epsilon = {eps:.2f}",
            f"% TARGET: {target.label}",
            "% =========================================================",
            f"\\begin{{subfigure}}[t]{{{width}\\textwidth}}",
            "  \\centering",
            "  % One picture per subfigure: the volume axis is anchored below the coverage axis,",
            "  % and the baseline is the bottom of the coverage axis, so all panels line up.",
            "  \\begin{tikzpicture}[baseline]",
            "",
            "  % --- top plot: test coverage (violins) ---",
            "    \\begin{axis}[",
            f"      {axes.style},",
            *size,
            "      name=coverage,",
            x_range,
            "      enlarge x limits=false,",
        ]
        cov_target = 1.0 - eps
        lines.append(cov_range.option_line())
        if eps_idx == 0:
            lines.append("      ylabel={Coverage\\strut},")
        lines.append(
            "      yticklabel={\\pgfmathprintnumber[fixed zerofill, precision=2]{\\tick}},"
        )
        lines.append(TICK_LABEL_STYLE)
        lines.append("    ]")
        lines.append(
            f"      \\draw[dashed,thick,black!40!red](axis cs:0.5,{cov_target:.6f})"
            f"--(axis cs:{len(methods) + 0.5:g},{cov_target:.6f});"
        )
        lines += _method_violins(specs, cov_values, runs)
        lines += [
            "    \\end{axis}",
            "",
            "  % --- bottom plot: set volume ---",
            "    \\begin{axis}[",
            f"      {axes.style},",
            *size,
            f"      at={{(coverage.south west)}}, anchor=north west, yshift=-{axes.gap},",
            "      name=volume,",
            x_range,
            "      enlarge x limits=false,",
            vol_range.option_line(),
            TICK_LABEL_STYLE,
        ]
        if eps_idx == 0:
            lines.append("      ylabel={Volume\\strut},")
        lines.append("    ]")
        lines += _method_boxes(specs, vol_values, vol_stats, runs, vol_range)
        lines += [
            "    \\end{axis}",
            *BOTTOM_AT_VOLUME_AXIS,
            "  \\end{tikzpicture}",
            "",
            f"  \\caption{{{tex_label(target, targets)}}}",
            f"  \\label{{fig:{exp_name}_box_eps{int(round(eps * 1000)):03d}{_beta_suffix(target)}}}",
            "\\end{subfigure}%",
        ]
        if eps_idx < len(targets) - 1:
            lines.append("\\hfill")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"fig_{exp_name}_boxplot.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
