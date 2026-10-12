"""Axis ranges of the coverage and volume boxplots, shared by the PNG and TikZ figures.

Coverage: all points are shown.  Under the high-confidence guarantee the axis
starts just below the target 1 - epsilon, or just below the lowest point if a
point lies below the target; under the expected guarantee it starts just below
the lowest point.

Volume: the axis spans the whiskers of all methods, so that every box is
clearly visible; more extreme points (e.g. degenerate split-SCP shapes) may lie
outside and are clipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from MCP.methods import Target

PAD = 0.05  # margin as a fraction of the shown data span


def _padded(lo: float, hi: float, pad: float) -> tuple[float, float]:
    span = hi - lo
    margin = pad * span if span > 0 else 0.01 * max(abs(hi), 1.0)
    return lo - margin, hi + margin


def coverage_range(
    values: Sequence[np.ndarray], target: Target, pad: float = PAD
) -> tuple[float, float]:
    """Coverage-axis range from the coverage of all methods and runs."""
    finite = [v[np.isfinite(v)] for v in values]
    points = np.concatenate(finite) if finite else np.array([])
    goal = 1.0 - target.epsilon
    if points.size == 0:
        return _padded(goal, 1.0, pad)
    lo = float(points.min())
    if target.beta is not None:
        lo = min(lo, goal)
    return _padded(lo, float(points.max()), pad)


def volume_range(stats: Sequence[tuple], pad: float = PAD) -> tuple[float, float]:
    """Volume-axis range spanning all whiskers (see boxplot_stats), clipped at 0."""
    whiskers = [v for s in stats for v in (s[6], s[7]) if np.isfinite(v)]
    if not whiskers:
        return 0.0, 1.0
    lo, hi = _padded(min(whiskers), max(whiskers), pad)
    return max(lo, 0.0), hi


def outliers(values: np.ndarray, stats: tuple) -> np.ndarray:
    """Points beyond the whiskers of one method (see boxplot_stats)."""
    values = values[np.isfinite(values)]
    return values[(values < stats[6]) | (values > stats[7])]
