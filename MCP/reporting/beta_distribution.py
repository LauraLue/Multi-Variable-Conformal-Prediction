"""Coverage distributions of SCP and RemMCP as PGFPlots tables (Fig. 1, 4, and 14).

With outlier budget rho, the coverage of RemMCP satisfies

    eta(q*) ~ Beta(n - n_q (rho + 1) + 1, n_q (rho + 1))

under the assumptions of Thm. 1(c) and is stochastically larger otherwise,
so the curve is a lower bound in general.  For SCP (n_q = 1, continuous
scores) it is exact.  Its CDF at 1 - epsilon is the certified misconfidence
beta_bar(rho) of (14).

Run from the repository root, e.g.:

    python -m MCP.reporting.beta_distribution --epsilon 0.05 --beta 1e-6
    python -m MCP.reporting.beta_distribution --help
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

from MCP.CP import outlier_budget_expected, outlier_budget_high_confidence
from MCP.methods import METHODS, Target, feasibility_issue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURE_METHODS = ("SCP-1", "SCP-2A", "SCP-3A", "RemMCP")


@dataclass(frozen=True)
class CoverageDistribution:
    """Coverage distribution of a method calibrated with outlier budget rho.

    Attributes:
        method: Method name.
        n: Number of points the guarantee is certified on (the split for split SCP).
        n_q: Parameter dimension.
        rho: Outlier budget.
        epsilon: Miscoverage level, used in the file name.
    """

    method: str
    n: int
    n_q: int
    rho: int
    epsilon: float

    @property
    def params(self) -> tuple[int, int]:
        """Parameters (a, b) of the Beta distribution."""
        k = self.n_q * (self.rho + 1)
        return self.n - k + 1, k

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return beta_dist.pdf(x, *self.params)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return beta_dist.cdf(x, *self.params)

    @property
    def filename(self) -> str:
        eps = f"{self.epsilon * 100:g}"
        return f"beta_coords_ncal{self.n}_eps{eps}_nq{self.n_q}_rho{self.rho}.tex"


def coverage_distribution(
    method: str, target: Target, n_cal: int, dim: int = 2, n_shapes: int = 3
) -> CoverageDistribution:
    """Coverage distribution of an SCP or RemMCP method for a target.

    Raises:
        ValueError: For H-RelMCP (no a-priori outlier budget) or infeasible targets.
    """
    spec = METHODS[method]
    if spec.relaxation:
        raise ValueError(f"{method} has no a-priori coverage distribution.")
    issue = feasibility_issue(method, target, n_cal, dim, n_shapes)
    if issue is not None:
        raise ValueError(issue)
    n, n_q = spec.n_threshold(n_cal), spec.n_q(dim, n_shapes)
    if target.beta is None:
        rho = outlier_budget_expected(n, n_q, target.epsilon)
    else:
        rho = outlier_budget_high_confidence(n, n_q, target.epsilon, target.beta)
    return CoverageDistribution(method, n, n_q, rho, target.epsilon)


def write_table(path: str, x: np.ndarray, y: np.ndarray, fmt: str = "{:.5f} {:.5f}") -> None:
    """Write "x y" rows as a PGFPlots table (no trailing newline, as in the paper files)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(fmt.format(xi, yi) for xi, yi in zip(x, y, strict=True)))


def write_density(
    dist: CoverageDistribution,
    out_dir: str,
    x_min: float = 0.65,
    x_max: float = 1.0,
    n_points: int = 200,
) -> str:
    """Write the density on an equidistant grid; returns the file path.

    The defaults reproduce the grid of the paper's coordinate files.
    """
    x = np.linspace(x_min, x_max, n_points)
    path = os.path.join(out_dir, dist.filename)
    write_table(path, x, dist.pdf(x))
    return path


def write_lower_tail(dist: CoverageDistribution, out_dir: str, n_points: int = 200) -> str:
    """Write log10 P(eta <= x) for x up to the mean, for a log-scale tail panel.

    At x = 1 - epsilon the value is log10 beta_bar(rho).  Returns the file path.
    """
    a, b = dist.params
    x = np.linspace(max(0.0, 1.0 - 3.0 * dist.epsilon), a / (a + b), n_points)
    log_tail = beta_dist.logcdf(x, a, b) / np.log(10.0)
    path = os.path.join(out_dir, dist.filename.replace("beta_coords", "beta_logcdf"))
    write_table(path, x, log_tail, fmt="{:.5f} {:.4f}")
    return path


def write_figure_tables(
    target: Target,
    n_cal: int = 2000,
    methods: Sequence[str] = FIGURE_METHODS,
    out_dir: str = os.path.join(REPO_ROOT, "tikz_figures", "fig_helpers"),
    dim: int = 2,
    lower_tail: bool = False,
    **grid,
) -> list[CoverageDistribution]:
    """Write the density tables (and optionally the lower tails) of all methods.

    Prints one line per method with its distribution, file name, and
    beta_bar(rho) = P(eta <= 1 - epsilon).
    """
    dists = [coverage_distribution(m, target, n_cal, dim) for m in methods]
    for dist in dists:
        path = write_density(dist, out_dir, **grid)
        if lower_tail:
            write_lower_tail(dist, out_dir)
        a, b = dist.params
        tail = float(dist.cdf(np.array(1.0 - target.epsilon)))
        print(
            f"{dist.method:8s} n={dist.n:5d} n_q={dist.n_q} rho={dist.rho:4d}  "
            f"Beta({a}, {b})  P(eta <= {1 - target.epsilon:g}) = {tail:.3e}  -> {path}"
        )
    return dists


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m MCP.reporting.beta_distribution",
        description="Write the coverage distributions of SCP and RemMCP as PGFPlots tables.",
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="miscoverage level")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="misconfidence of the high-confidence guarantee (default: expected guarantee "
        "with alpha = epsilon)",
    )
    parser.add_argument("--n-cal", type=int, default=2000, help="calibration samples")
    parser.add_argument("--methods", nargs="+", default=list(FIGURE_METHODS), help="methods")
    parser.add_argument("--x-min", type=float, default=0.65, help="left end of the grid")
    parser.add_argument("--n-points", type=int, default=200, help="grid points")
    parser.add_argument("--lower-tail", action="store_true", help="also write log10 CDF tables")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "tikz_figures", "fig_helpers"),
        help="output directory (default: tikz_figures/fig_helpers)",
    )
    args = parser.parse_args(argv)
    write_figure_tables(
        Target(args.epsilon, args.beta),
        n_cal=args.n_cal,
        methods=args.methods,
        out_dir=args.out_dir,
        lower_tail=args.lower_tail,
        x_min=args.x_min,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()
