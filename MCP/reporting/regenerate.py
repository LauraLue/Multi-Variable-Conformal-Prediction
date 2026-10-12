"""Regenerate figures and tables from saved results, without re-running experiments.

Reads the compact pickles written by MCP.experiment (results/{name}_results.pkl,
see io.save_results), prints the summary table, and writes the coverage/volume
figures (PNG and TikZ) and the LaTeX table, as at the end of an experiment.
The prediction-set figures need the calibration residuals, which are not saved,
and are therefore not regenerated.

Run from the repository root, e.g.:

    python -m MCP.reporting.regenerate results/mainHc_Vehicle30_eps5_beta1e-06_results.pkl
    python -m MCP.reporting.regenerate path/to/results/*_results.pkl
    python -m MCP.reporting.regenerate results/a_results.pkl results/b_results.pkl --combine ab
    python -m MCP.reporting.regenerate results/dim2_results.pkl results/dim4_results.pkl \\
        --combine dims --labels '$n_y = 2$' '$n_y = 4$'
    python -m MCP.reporting.regenerate --help

By default, outputs go next to the pickle's results/ directory (figures/ and
tikz_figures/ of the same experiment folder).
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import numpy as np

from MCP.CP import CP
from MCP.methods import METHODS, Target
from MCP.reporting.io import print_summary
from MCP.reporting.plots import plot_boxplot
from MCP.reporting.tikz import write_tex_boxplot, write_tex_table
from MCP.results import MethodResult, RunResult

SUFFIX = "_results.pkl"
# Method names in results saved before the renaming of RelMCP to H-RelMCP.
LEGACY_NAMES = {"RelMCP-A": "H-RelMCP-A", "RelMCP-B": "H-RelMCP-B"}


class SavedPredictor:
    """Stand-in for a calibrated predictor, restored from a results pickle.

    Provides what the reporting uses: the certified confidence at the run's
    epsilon and, for H-RelMCP, the iteration counts.
    """

    def __init__(self, entry: dict):
        self.confidence = float(entry["confidence"])
        self.q_opt = entry.get("q_opt")
        self.rho = entry.get("rho")
        self.coverage_bound = entry.get("coverage_bound")
        self.i_opt = entry.get("i_opt")
        self.n_iterations = entry.get("n_iterations")

    def coverage_confidence(self, epsilon: float | None = None) -> float:
        """Saved confidence of coverage >= 1 - epsilon at the run's epsilon."""
        return self.confidence


def _method_result(name: str, entry: dict) -> MethodResult:
    return MethodResult(
        name,
        # duck-typed: the reporting only uses coverage_confidence and n_iterations
        predictor=cast(CP, SavedPredictor(entry)) if entry["ok"] else None,
        error=entry.get("error"),
        infeasible=entry.get("infeasible", False),
        uncertified=entry.get("uncertified", False),
        n_cal=entry.get("n_cal"),
        cal_cov=entry["cal_cov"],
        test_cov=entry["test_cov"],
        cal_time=entry["cal_time"],
        volume=entry["volume"],
        ellipsoids=[(np.asarray(Q), np.asarray(c)) for Q, c in entry.get("ellipsoids", [])],
    )


def load_results(path: str) -> dict[Target, list[RunResult]]:
    """Runs per target from a results pickle.

    Keys are (epsilon, beta), (epsilon, beta, variant) for combined experiments,
    or epsilon only (results from before the coverage targets).
    """
    with open(path, "rb") as f:
        compact = pickle.load(f)
    runs_by_target: dict[Target, list[RunResult]] = {}
    for key, runs in compact.items():
        target = Target(*key) if isinstance(key, tuple) else Target(float(key))
        runs_by_target[target] = []
        for saved in runs:
            run = RunResult(
                seed=saved["seed"],
                target=target,
                E_cal=np.empty((0, 2)),  # not saved
                E_test=np.empty((0, 2)),
                hrelmcp_target_conf=saved.get(
                    "hrelmcp_target_conf", saved.get("relmcp_target_conf", np.nan)
                ),
                hrelmcp_beta=saved.get("hrelmcp_beta", saved.get("relmcp_beta", np.nan)),
            )
            for name, entry in saved["methods"].items():
                name = LEGACY_NAMES.get(name, name)
                run.methods[name] = _method_result(name, entry)
            runs_by_target[target].append(run)
    return runs_by_target


def _experiment_name(path: str) -> str:
    name = os.path.basename(path)
    if not name.endswith(SUFFIX):
        raise ValueError(f"Expected a file named <experiment>{SUFFIX}, got {name!r}.")
    return name[: -len(SUFFIX)]


def _default_output_dir(path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(path)))


def _report(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    output_dir: str,
    methods: Sequence[str] | None,
) -> None:
    """Print the summary and write the figures and table of one experiment."""
    runs = [run for runs in runs_by_target.values() for run in runs]
    methods = list(methods or runs[0].methods)
    unknown = [m for m in methods if m not in METHODS or any(m not in r.methods for r in runs)]
    if unknown:
        raise ValueError(f"Methods {unknown} are not in the results of {exp_name}.")

    print(f"\n{exp_name}")
    print_summary(runs_by_target, methods)
    plot_boxplot(runs_by_target, exp_name, methods, os.path.join(output_dir, "figures"))
    tikz_dir = os.path.join(output_dir, "tikz_figures")
    write_tex_table(runs_by_target, exp_name, methods, tikz_dir)
    write_tex_boxplot(runs_by_target, exp_name, methods, tikz_dir)


def regenerate(
    path: str, output_dir: str | None = None, methods: Sequence[str] | None = None
) -> dict[Target, list[RunResult]]:
    """Print the summary and write the figures and table of one results pickle.

    Args:
        path: results/{name}_results.pkl.
        output_dir: Root of figures/ and tikz_figures/ (default: the folder
            containing the pickle's results/ directory).
        methods: Methods to show, in display order (default: all saved ones).
    """
    exp_name = _experiment_name(path)
    runs_by_target = load_results(path)
    _report(runs_by_target, exp_name, output_dir or _default_output_dir(path), methods)
    return runs_by_target


def _with_variant(key, variant: str | None):
    """Pickle key (epsilon, beta[, variant]) with the variant replaced, if given."""
    if variant is None:
        return key
    epsilon, beta = (key[0], key[1]) if isinstance(key, tuple) else (float(key), None)
    return (epsilon, beta, variant)


def combine(
    paths: Sequence[str],
    name: str,
    output_dir: str | None = None,
    methods: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[Target, list[RunResult]]:
    """Combine experiments into one with a panel per experiment target, in the order of paths.

    Runs each value of a sweep as its own experiment (in parallel), then merges
    them here.  Experiments with different targets (epsilon or beta sweeps) are
    labeled by the varying level; experiments with the same target, e.g. the
    output dimensions n_y of SCM20D, need one LaTeX label per file, e.g.
    "$n_y = 4$", which becomes the panel caption (Target.variant).  Writes the
    summary, figures, and table under the new name, and the merged pickle to
    results/{name}_results.pkl, so that it can be regenerated like any other.

    Raises:
        ValueError: If labels do not match the files, or two panels would have
            the same target and label.
    """
    if not paths:
        raise ValueError("No results to combine.")
    if labels is not None and len(labels) != len(paths):
        raise ValueError(f"Got {len(labels)} labels for {len(paths)} files.")
    output_dir = output_dir or _default_output_dir(paths[0])
    merged: dict[Target, list[RunResult]] = {}
    compact: dict = {}
    for i, path in enumerate(paths):
        variant = None if labels is None else labels[i]
        for target, runs in load_results(path).items():
            if variant is not None:
                target = replace(target, variant=variant)
                for run in runs:
                    run.target = target
            if target in merged:
                raise ValueError(
                    f"{target.label} appears in more than one file; give one label per file "
                    "(--labels) to combine experiments with the same target."
                )
            merged[target] = runs
        with open(path, "rb") as f:
            for key, saved in pickle.load(f).items():
                compact[_with_variant(key, variant)] = saved
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{name}{SUFFIX}"), "wb") as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
    _report(merged, name, output_dir, methods)
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m MCP.reporting.regenerate",
        description="Regenerate figures and tables from saved experiment results.",
    )
    parser.add_argument("paths", nargs="+", help="results/<experiment>_results.pkl files")
    parser.add_argument(
        "--combine",
        metavar="NAME",
        default=None,
        help="merge all given experiments into one named NAME, one panel per experiment "
        "target in the given order",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="with --combine: LaTeX panel label per file, e.g. '$n_y = 2$' '$n_y = 4$' "
        "(needed if the experiments have the same target)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="root of results/, figures/, and tikz_figures/ (default: next to the results/ "
        "directory of each file)",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="methods to show, in order")
    args = parser.parse_args(argv)
    if args.labels and not args.combine:
        parser.error("--labels requires --combine")
    if args.combine:
        combine(args.paths, args.combine, args.output_dir, args.methods, args.labels)
        return
    for path in args.paths:
        regenerate(path, args.output_dir, args.methods)


if __name__ == "__main__":
    main()
