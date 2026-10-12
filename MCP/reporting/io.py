"""Console output, log files, and result pickles."""

from __future__ import annotations

import os
import pickle
import sys
from collections.abc import Sequence

import numpy as np

from MCP.HRelMCP import HRelMCP
from MCP.methods import METHODS, Target
from MCP.reporting.labels import all_infeasible, certified_betas, n_errors, n_uncertified
from MCP.reporting.stats import method_mean_std, method_values
from MCP.results import MethodResult, RunResult


class Tee:
    """Mirror stdout writes to both the terminal and a log file."""

    def __init__(self, path: str):
        # Line-buffered, so that the log shows the progress of long runs immediately.
        self._file = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._stdout
        self._file.close()


def print_run(run: RunResult, index: int, methods: Sequence[str]) -> None:
    """One summary line with test coverage and volume per method."""

    def _fmt(v):
        return f"{v:.3f}" if np.isfinite(v) else "fail"

    method_summary = " | ".join(
        f"{m} test={_fmt(run.methods[m].test_cov)} vol={_fmt(run.methods[m].volume)}"
        for m in methods
    )
    print(f"  Run {index + 1:2d} (seed={run.seed}): {method_summary}")


def _column_label(method: str, runs: list[RunResult]) -> str:
    """Method name, with the threshold-calibration size for split-SCP methods."""
    n_cal = runs[0].methods[method].n_cal
    if METHODS[method].split_ratio is not None and n_cal is not None:
        return f"{method}(n={n_cal})"
    return method


def print_summary(runs_by_target: dict[Target, list[RunResult]], methods: Sequence[str]) -> None:
    """Table of coverage, certified beta, volume, time, and failures per target."""

    def _r4(mu, sd):
        return f"{mu:7.4f}±{sd:.4f}"

    def _beta(runs: list[RunResult], m: str, reduce) -> str:
        if all_infeasible(runs, m):
            return "infeasible"
        betas = certified_betas(runs, m)
        return f"{reduce(betas):.2e}" if betas.size else "--"

    def _min(values) -> str:
        return f"{values.min():.4f}" if values.size else "--"

    def _iterations(runs: list[RunResult], m: str) -> str:
        values = [
            n
            for run in runs
            if (n := getattr(run.methods[m].predictor, "n_iterations", None)) is not None
        ]
        return f"{np.mean(values):.1f}/{max(values)}" if values else "--"

    def _failures(runs: list[RunResult], m: str, count) -> str:
        if all_infeasible(runs, m):
            return "infeasible"
        return f"{count(runs, m)}/{len(runs)}"

    label_w, col_w = 34, 16
    for target, runs in runs_by_target.items():
        eps = target.epsilon

        def _row(label: str, cells: list[str]) -> str:
            return f"{label:{label_w}s} " + "  ".join(f"{c:>{col_w}s}" for c in cells)

        sep = "=" * (label_w + 1 + (col_w + 2) * len(methods))
        print(f"\nTarget: {target.label}")
        print(sep)
        print(_row("Metric", [_column_label(m, runs) for m in methods]))
        print(sep)
        print(
            _row(
                "Calibration coverage", [_r4(*method_mean_std(runs, m, "cal_cov")) for m in methods]
            )
        )
        print(_row("Test coverage", [_r4(*method_mean_std(runs, m, "test_cov")) for m in methods]))
        print(
            _row("Min. test coverage", [_min(method_values(runs, m, "test_cov")) for m in methods])
        )
        print(f"{'Coverage bound (1-eps)':{label_w}s} {1 - eps:.4f}")
        print(_row("Certified beta (mean)", [_beta(runs, m, np.mean) for m in methods]))
        print(_row("Certified beta (max)", [_beta(runs, m, np.max) for m in methods]))
        if target.beta is not None:
            print(f"{'Target beta':{label_w}s} {target.beta:.2e}")
        print(_row("Set volume", [_r4(*method_mean_std(runs, m, "volume")) for m in methods]))
        print(
            _row(
                "Calibration time [s]",
                [_r4(*method_mean_std(runs, m, "cal_time")) for m in methods],
            )
        )
        print(_row("Iterations Alg. 1 (mean/max)", [_iterations(runs, m) for m in methods]))
        print(
            _row(
                "No certified solution",
                [_failures(runs, m, n_uncertified) for m in methods],
            )
        )
        print(_row("Errors", [_failures(runs, m, n_errors) for m in methods]))
        print(sep)


def _to_list(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return float(arr) if arr.ndim == 0 else arr.tolist()


def _method_summary(result: MethodResult, epsilon: float) -> dict:
    predictor = result.predictor
    summary = {
        "ok": result.ok,
        "error": result.error,
        "infeasible": result.infeasible,
        "uncertified": result.uncertified,
        "n_cal": result.n_cal,
        "cal_cov": result.cal_cov,
        "test_cov": result.test_cov,
        "cal_time": result.cal_time,
        "volume": result.volume,
        "ellipsoids": [(_to_list(Q), _to_list(c)) for Q, c in result.ellipsoids],
        "fitted_shape": None
        if result.fitted_shape is None
        else {
            "centers": [_to_list(c) for c in result.fitted_shape.centers],
            "Q": [_to_list(Q) for Q in result.fitted_shape.Q],
        },
    }
    if predictor is not None:
        summary.update(
            q_opt=_to_list(predictor.q_opt),
            rho=predictor.rho,
            coverage_bound=predictor.coverage_bound,
            confidence=predictor.coverage_confidence(epsilon=epsilon),
        )
        if isinstance(predictor, HRelMCP):
            summary.update(
                d_opt=predictor.d_opt,
                n_violations=predictor.n_violations,
                i_opt=predictor.i_opt,
                n_iterations=predictor.n_iterations,
            )
    return summary


def save_results(runs_by_target: dict[Target, list[RunResult]], path: str) -> None:
    """Pickle a compact summary of all runs (no residual arrays or predictor objects).

    Structure: {(epsilon, beta): [run, ...]} with beta = None for the expected
    guarantee and run = {"seed", "epsilon", "beta", "guarantee",
    "hrelmcp_target_conf", "hrelmcp_beta", "methods": {name: {...}}}; each
    method entry holds the metrics, calibrated parameters, confidence, and
    set geometry; rho is None for H-RelMCP, whose entries also hold d_opt,
    n_violations, i_opt, and n_iterations (outer iterations of Alg. 1).
    """
    compact = {
        (target.epsilon, target.beta): [
            {
                "seed": run.seed,
                "epsilon": run.epsilon,
                "beta": run.beta,
                "guarantee": run.target.guarantee,
                "hrelmcp_target_conf": run.hrelmcp_target_conf,
                "hrelmcp_beta": run.hrelmcp_beta,
                "methods": {m: _method_summary(r, run.epsilon) for m, r in run.methods.items()},
            }
            for run in runs
        ]
        for target, runs in runs_by_target.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {path}")
