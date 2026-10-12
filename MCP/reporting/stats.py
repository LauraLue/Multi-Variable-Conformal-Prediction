"""Summary statistics over experiment runs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import gaussian_kde

from MCP.results import RunResult


def finite_array(values) -> np.ndarray:
    """Return finite values as a float array."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_std(values) -> tuple[float, float]:
    """Mean and (population) standard deviation of the finite values."""
    arr = finite_array(values)
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std())


def method_values(runs: Sequence[RunResult], method: str, attr: str) -> np.ndarray:
    """Finite values of MethodResult.<attr> for one method across runs."""
    return finite_array([getattr(run.methods[method], attr) for run in runs])


def method_mean_std(runs: Sequence[RunResult], method: str, attr: str) -> tuple[float, float]:
    """Mean and std of MethodResult.<attr> for one method across runs."""
    return mean_std([getattr(run.methods[method], attr) for run in runs])


def boxplot_stats(arr: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    """Return sample count, mean, std, quartiles, and whisker endpoints."""
    if arr.size == 0:
        return (np.nan,) * 8
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    low_mask = arr >= q1 - 1.5 * iqr
    high_mask = arr <= q3 + 1.5 * iqr
    whisker_low = float(np.min(arr[low_mask])) if np.any(low_mask) else float(np.min(arr))
    whisker_high = float(np.max(arr[high_mask])) if np.any(high_mask) else float(np.max(arr))
    return (
        float(arr.size),
        float(arr.mean()),
        float(arr.std()),
        float(q1),
        float(med),
        float(q3),
        whisker_low,
        whisker_high,
    )


VIOLIN_POINTS = 100  # evaluation points of the density along the value axis


def violin_outline(values: np.ndarray, n_points: int = VIOLIN_POINTS):
    """Half-width profile of a violin, scaled to a maximum of 1, or None.

    Gaussian kernel density estimate with Scott's bandwidth (scipy's
    default), evaluated from the lowest to the highest value, so that the
    violin spans exactly the observed range.  Returns None if the density is
    undefined (fewer than two distinct values).

    Returns:
        Tuple (y, half_width) of arrays of length n_points.
    """
    values = finite_array(values)
    if np.unique(values).size < 2:
        return None
    y = np.linspace(values.min(), values.max(), n_points)
    density = gaussian_kde(values)(y)
    return y, density / density.max()
