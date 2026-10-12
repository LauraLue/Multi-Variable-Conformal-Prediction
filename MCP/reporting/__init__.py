"""Reporting of experiment results: console tables, pickles, TikZ, and PNG figures.

Output layout below ``output_dir``:
  results/{exp_name}_results.pkl     compact results (see io.save_results)
  tikz_figures/fig_{exp_name}_*.tex  TikZ figures, table_{exp_name}.tex table
  figures/fig_{exp_name}_*.png       matplotlib overview figures
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from MCP.methods import Target
from MCP.reporting.io import Tee, print_run, print_summary, save_results
from MCP.reporting.plots import plot_boxplot, plot_sets
from MCP.reporting.tikz import (
    AxisRange,
    BoxplotAxes,
    write_tex_boxplot,
    write_tex_sets,
    write_tex_table,
)
from MCP.results import RunResult

__all__ = [
    "AxisRange",
    "BoxplotAxes",
    "Tee",
    "print_run",
    "print_summary",
    "save_results",
    "write_report",
]


def write_report(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    methods: Sequence[str],
    output_dir: str,
    boxplot_axes: BoxplotAxes | None = None,
    sets_axis_style: str = "myplotSet",
) -> None:
    """Save results first, then print the summary and write all figures and tables.

    Args:
        runs_by_target: Runs per coverage target, in display order.
        exp_name: Experiment identifier used as a filename prefix.
        methods: Method names in display order.
        output_dir: Root directory of results/, figures/, and tikz_figures/.
        boxplot_axes: Axis settings of the TikZ boxplots (default: data ranges).
        sets_axis_style: PGFPlots style of the TikZ prediction-set figures.
    """
    save_results(runs_by_target, os.path.join(output_dir, "results", f"{exp_name}_results.pkl"))
    print_summary(runs_by_target, methods)

    figures_dir = os.path.join(output_dir, "figures")
    plot_sets(runs_by_target, exp_name, methods, figures_dir)
    plot_boxplot(runs_by_target, exp_name, methods, figures_dir)

    tikz_dir = os.path.join(output_dir, "tikz_figures")
    write_tex_table(runs_by_target, exp_name, methods, tikz_dir)
    write_tex_boxplot(runs_by_target, exp_name, methods, tikz_dir, axes=boxplot_axes)
    write_tex_sets(runs_by_target, exp_name, methods, tikz_dir, axis_style=sets_axis_style)
