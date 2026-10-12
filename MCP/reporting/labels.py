"""Labels of coverage targets and certified misconfidence levels.

A sweep over several targets labels each target only by the levels that vary
across the sweep (epsilon for an epsilon sweep, beta for a beta sweep, the
variant, e.g. n_y, for combined experiments with the same levels); the fixed
levels are stated once, e.g. in a caption.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from MCP.methods import Target
from MCP.results import RunResult


def _shown_levels(targets: Sequence[Target]) -> tuple[bool, bool, bool]:
    """Whether to show (epsilon, beta, variant) in per-target labels.

    The varying ones; if none varies, epsilon and beta (and a variant, if any).
    """
    eps_varies = len({t.epsilon for t in targets}) > 1
    beta_varies = len({t.beta for t in targets}) > 1
    variant_varies = len({t.variant for t in targets}) > 1
    if not (eps_varies or beta_varies or variant_varies):
        return True, True, any(t.variant is not None for t in targets)
    return eps_varies, beta_varies, variant_varies


def tex_number(value: float) -> str:
    """LaTeX number: fixed for values >= 0.001, otherwise m{\\cdot}10^{k} with one decimal."""
    if value == 0.0 or abs(value) >= 1e-3:
        return f"{value:.4f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return power if mantissa == "1.0" else f"{mantissa}{{\\cdot}}{power}"


def _tex_beta(beta: float) -> str:
    mantissa, exponent = f"{beta:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return power if mantissa == "1" else f"{mantissa}{{\\cdot}}{power}"


def _tex_parts(
    target: Target, show_eps: bool, show_beta: bool, show_variant: bool = False
) -> list[str]:
    parts = []
    if show_variant and target.variant is not None:
        parts.append(target.variant)
    if show_eps:
        parts.append(f"$\\varepsilon = {target.epsilon:.2f}$")
    if show_beta:
        parts.append(
            "expected guarantee" if target.beta is None else f"$\\beta = {_tex_beta(target.beta)}$"
        )
    return parts


def tex_label(target: Target, targets: Sequence[Target]) -> str:
    """LaTeX label of a target within a sweep, e.g. "$\\beta = 10^{-6}$"."""
    return ", ".join(_tex_parts(target, *_shown_levels(targets)))


def tex_fixed_levels(targets: Sequence[Target]) -> str:
    """LaTeX description of the levels shared by all targets, or "" if none."""
    show_eps, show_beta, _ = _shown_levels(targets)
    if show_eps and show_beta:
        return ""
    return ", ".join(_tex_parts(targets[0], not show_eps, not show_beta))


def plain_label(target: Target, targets: Sequence[Target]) -> str:
    """Plain-text label of a target within a sweep, e.g. "beta=1e-06"."""
    show_eps, show_beta, show_variant = _shown_levels(targets)
    parts = []
    if show_variant and target.plain_variant is not None:
        parts.append(target.plain_variant)
    if show_eps:
        parts.append(f"eps={target.epsilon:g}")
    if show_beta:
        parts.append("expected" if target.beta is None else f"beta={target.beta:.0e}")
    return ", ".join(parts)


def certified_betas(runs: Sequence[RunResult], method: str) -> np.ndarray:
    """Certified misconfidence of coverage >= 1 - epsilon in each successful run.

    beta_bar(rho) for SCP and RemMCP (Thm. 1), the a-posteriori beta of the
    tradeoff (20) for H-RelMCP (nominal, App. B).
    """
    return np.array(
        [
            1.0 - predictor.coverage_confidence(epsilon=run.epsilon)
            for run in runs
            if (predictor := run.methods[method].predictor) is not None
        ],
        dtype=float,
    )


def all_infeasible(runs: Sequence[RunResult], method: str) -> bool:
    """Whether the method was skipped as infeasible in every run."""
    return bool(runs) and all(run.methods[method].infeasible for run in runs)


def n_uncertified(runs: Sequence[RunResult], method: str) -> int:
    """Number of runs in which a feasible method found no certified solution."""
    return sum(run.methods[method].uncertified for run in runs)


def n_errors(runs: Sequence[RunResult], method: str) -> int:
    """Number of runs in which a feasible method failed for another reason (an exception)."""
    return sum(
        not r.ok and not r.infeasible and not r.uncertified
        for run in runs
        if (r := run.methods[method])
    )
