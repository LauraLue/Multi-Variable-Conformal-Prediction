"""
Tests for the coverage distributions of SCP and RemMCP (MCP.reporting.beta_distribution).
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from MCP.CP import confidence_bound
from MCP.methods import Target
from MCP.reporting.beta_distribution import (
    FIGURE_METHODS,
    coverage_distribution,
    main,
    write_density,
    write_lower_tail,
)

TARGETS = [Target(0.05), Target(0.05, beta=1e-6)]


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.slug)
@pytest.mark.parametrize("method", FIGURE_METHODS)
def test_cdf_at_target_coverage_is_the_certified_beta(method, target):
    """P(eta <= 1 - epsilon) under Beta(n - k + 1, k) equals beta_bar(rho) of (14)."""
    dist = coverage_distribution(method, target, n_cal=2000)
    beta_bar = confidence_bound(dist.rho, dist.n, dist.n_q, target.epsilon)
    assert dist.cdf(np.array(1.0 - target.epsilon)) == pytest.approx(beta_bar, rel=1e-6)
    if target.beta is not None:
        assert beta_bar <= target.beta


def test_distributions_of_the_paper_figure():
    expected = {"SCP-1": (2000, 1, 56), "SCP-2A": (500, 1, 5), "SCP-3A": (1000, 1, 20)}
    expected["RemMCP"] = (2000, 3, 18)
    for method, (n, n_q, rho) in expected.items():
        dist = coverage_distribution(method, Target(0.05, beta=1e-6), n_cal=2000)
        assert (dist.n, dist.n_q, dist.rho) == (n, n_q, rho)
    scp1 = coverage_distribution("SCP-1", Target(0.05), n_cal=2000)
    assert scp1.params == (1901, 100)  # Beta(n - rho, rho + 1) for SCP
    assert scp1.filename == "beta_coords_ncal2000_eps5_nq1_rho99.tex"


def test_density_integrates_to_one():
    dist = coverage_distribution("SCP-2A", Target(0.05, beta=1e-6), n_cal=2000)
    x = np.linspace(0.0, 1.0, 200001)
    assert trapezoid(dist.pdf(x), x) == pytest.approx(1.0, abs=1e-6)


def test_methods_without_a_priori_distribution_are_rejected():
    with pytest.raises(ValueError, match="no a-priori coverage distribution"):
        coverage_distribution("H-RelMCP-A", Target(0.05), n_cal=2000)
    with pytest.raises(ValueError, match="SCP-2A infeasible"):
        coverage_distribution("SCP-2A", Target(0.02, beta=1e-6), n_cal=2000)


def test_tables_use_the_paper_format(tmp_path):
    dist = coverage_distribution("SCP-1", Target(0.05), n_cal=2000)
    text = open(write_density(dist, str(tmp_path))).read()
    lines = text.split("\n")
    assert len(lines) == 200 and not text.endswith("\n")
    assert lines[0] == "0.65000 0.00000" and lines[-1] == "1.00000 0.00000"

    tail = np.loadtxt(write_lower_tail(dist, str(tmp_path)))
    log_beta_bar = np.log10(confidence_bound(dist.rho, dist.n, dist.n_q, 0.05))
    assert np.interp(0.95, tail[:, 0], tail[:, 1]) == pytest.approx(log_beta_bar, abs=0.02)
    assert np.all(np.diff(tail[:, 1]) > 0)  # the CDF increases


def test_cli_writes_one_table_per_method(tmp_path, capsys):
    main(["--beta", "1e-6", "--x-min", "0.9", "--n-points", "400", "--out-dir", str(tmp_path)])
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "beta_coords_ncal1000_eps5_nq1_rho20.tex",
        "beta_coords_ncal2000_eps5_nq1_rho56.tex",
        "beta_coords_ncal2000_eps5_nq3_rho18.tex",
        "beta_coords_ncal500_eps5_nq1_rho5.tex",
    ]
    assert len((tmp_path / files[0]).read_text().split("\n")) == 400
    assert "P(eta <= 0.95) = 6.897e-07" in capsys.readouterr().out
