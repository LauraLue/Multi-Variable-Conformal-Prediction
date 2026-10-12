"""Matplotlib overview figures (PNG) of prediction sets and coverage/volume."""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from MCP.methods import METHODS, Target
from MCP.reporting.axes import coverage_range, volume_range
from MCP.reporting.labels import plain_label
from MCP.reporting.stats import boxplot_stats, method_values, violin_outline
from MCP.results import RunResult


def _axis_limits(
    E: np.ndarray, ellipses: list[tuple[np.ndarray, np.ndarray]], pad: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits covering the residuals and the drawn sets in the first two dimensions.

    Sets are included up to one residual span beyond the residual range on
    each side, so that a degenerate (huge) set does not hide the data.
    """
    lo, hi = E[:, :2].min(axis=0), E[:, :2].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    set_lo, set_hi = lo.copy(), hi.copy()
    for Q, c in ellipses:
        half_widths = np.sqrt(
            np.diag(np.linalg.inv(Q))
        )  # axis-aligned extent of {(e-c)^T Q (e-c) <= 1}
        set_lo, set_hi = np.minimum(set_lo, c - half_widths), np.maximum(set_hi, c + half_widths)
    lo, hi = np.maximum(set_lo, lo - span), np.minimum(set_hi, hi + span)
    margin = pad * (hi - lo)
    return (lo[0] - margin[0], hi[0] + margin[0]), (lo[1] - margin[1], hi[1] + margin[1])


_LINESTYLES = {"": "-", "dashed": "--", "dotted": ":"}


def plot_sets(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    methods: Sequence[str],
    out_dir: str,
) -> None:
    """One column per target: calibration residuals and prediction sets of the first run.

    Always plots the first 2 dimensions, regardless of output dimensionality.
    """
    targets = list(runs_by_target)
    fig, axes = plt.subplots(1, len(targets), figsize=(4.8 * len(targets), 4.8), squeeze=False)

    for col, target in enumerate(targets):
        ax = axes[0, col]
        runs = runs_by_target[target]
        if not runs:
            ax.set_title(plain_label(target, targets))
            ax.axis("off")
            continue
        run = runs[0]
        E_cal = np.asarray(run.E_cal, dtype=float)[:, :2]
        ax.scatter(
            E_cal[:, 0],
            E_cal[:, 1],
            color="black",
            s=10,
            alpha=0.22,
            zorder=2,
            label="cal. residuals",
        )

        drawn = []
        for m in methods:
            color = METHODS[m].color
            linestyle = _LINESTYLES[METHODS[m].tikz_style]
            result = run.methods[m]
            labeled = False
            for Q_full, c_full in result.ellipsoids:
                Q_eff = np.asarray(Q_full, dtype=float)[:2, :2]
                c = np.asarray(c_full, dtype=float)[:2]
                evals, evecs = np.linalg.eigh(Q_eff)
                if np.any(evals <= 0):
                    continue
                drawn.append((Q_eff, c))
                sa = 1.0 / np.sqrt(evals)
                ang = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
                ax.add_patch(
                    mpatches.Ellipse(
                        (float(c[0]), float(c[1])),
                        2 * sa[0],
                        2 * sa[1],
                        angle=ang,
                        facecolor=color,
                        alpha=0.10,
                        zorder=1,
                    )
                )
                ax.add_patch(
                    mpatches.Ellipse(
                        (float(c[0]), float(c[1])),
                        2 * sa[0],
                        2 * sa[1],
                        angle=ang,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.6,
                        linestyle=linestyle,
                        zorder=3,
                        label=f"{m} ({result.volume:.2f})" if not labeled else None,
                    )
                )
                labeled = True

        ax.axhline(0.0, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axvline(0.0, color="gray", lw=0.7, ls="--", zorder=0)
        ax.set_title(f"{plain_label(target, targets)} (run 1)")
        ax.set_xlabel("$e_1$")
        if col == 0:
            ax.set_ylabel("$e_2$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        xlim, ylim = _axis_limits(E_cal, drawn)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.suptitle(f"Prediction sets for {exp_name}", fontsize=13, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, os.path.join(out_dir, f"fig_{exp_name}_sets.png"))


_HATCHES = {"": None, "dashed": "////", "dotted": "...."}


def _style_boxes(ax, boxes, methods: Sequence[str]) -> None:
    """Fill boxes as in the paper: solid colors, hatched B variants (white face)."""
    for patch, m in zip(boxes, methods, strict=True):
        spec = METHODS[m]
        hatch = _HATCHES[spec.tikz_style]
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        if hatch is None:
            patch.set_facecolor(spec.color)
            continue
        patch.set_facecolor("white")
        ax.add_patch(
            mpatches.PathPatch(
                patch.get_path(),
                facecolor="none",
                edgecolor=spec.color,
                hatch=hatch,
                lw=0,
                zorder=patch.get_zorder() + 0.1,  # above the white face
            )
        )


def _legend_handles(methods: Sequence[str]) -> list:
    handles = []
    for m in methods:
        spec = METHODS[m]
        hatch = _HATCHES[spec.tikz_style]
        if hatch is None:
            handles.append(mpatches.Patch(facecolor=spec.color, edgecolor="black", label=m))
        else:
            handles.append(
                mpatches.Patch(facecolor="white", edgecolor=spec.color, hatch=hatch, label=m)
            )
    return handles


VIOLIN_HALF_WIDTH = 0.32  # maximum half-width of a violin in x units
INNER_HALF_WIDTH = 0.08  # half-width of the box inside a violin


def _draw_violin(ax, x: float, values: np.ndarray, method: str) -> None:
    """Violin with an inner box (quartiles, median, whiskers) and the mean."""
    if values.size == 0:
        return
    spec = METHODS[method]
    hatch = _HATCHES[spec.tikz_style]
    outline = violin_outline(values)
    if outline is not None:
        y, half = outline[0], VIOLIN_HALF_WIDTH * outline[1]
        face = spec.color if hatch is None else "white"
        ax.fill_betweenx(y, x - half, x + half, facecolor=face, edgecolor="black", lw=0.6, zorder=2)
        if hatch is not None:
            ax.fill_betweenx(
                y,
                x - half,
                x + half,
                facecolor="none",
                edgecolor=spec.color,
                hatch=hatch,
                lw=0,
                zorder=2.1,
            )
    _, mean, _, q1, med, q3, wl, wh = boxplot_stats(values)
    w = INNER_HALF_WIDTH
    ax.vlines(x, wl, wh, color="black", lw=0.8, zorder=3)
    ax.add_patch(
        mpatches.Rectangle(
            (x - w, q1), 2 * w, q3 - q1, facecolor="white", edgecolor="black", lw=0.8, zorder=3
        )
    )
    ax.hlines(med, x - w, x + w, color="black", lw=1.2, zorder=4)
    ax.plot(x, mean, marker="D", color="black", markersize=4, zorder=5)


def plot_boxplot(
    runs_by_target: dict[Target, list[RunResult]],
    exp_name: str,
    methods: Sequence[str],
    out_dir: str,
) -> None:
    """Violins of test coverage (top) and boxplots of set volume (bottom) per target.

    The coverage violins (see stats.violin_outline) contain a small box with
    the quartiles, median, and whiskers.  Colors and hatching follow the
    paper; the black diamond marks the mean.
    Axis ranges follow MCP.reporting.axes; a mean above the volume axis is
    marked by a triangle at its upper edge, with its value.
    """
    targets = list(runs_by_target)
    positions = np.arange(1, len(methods) + 1)
    width = max(3.8 * len(targets), 7.5)

    fig, axes = plt.subplots(2, len(targets), figsize=(width, 7.6), squeeze=False)
    for col, target in enumerate(targets):
        runs = runs_by_target[target]
        for row, (attr, ylabel) in enumerate(
            [("test_cov", "Test coverage"), ("volume", "Set volume")]
        ):
            ax = axes[row, col]
            data = [method_values(runs, m, attr) for m in methods]
            if row == 0:
                for x, values, m in zip(positions, data, methods, strict=True):
                    _draw_violin(ax, float(x), values, m)
                ax.set_xlim(0.5, len(methods) + 0.5)
                ax.set_ylim(*coverage_range(data, target))
                ax.axhline(1.0 - target.epsilon, color="#993333", linestyle="--", lw=1.2, zorder=1)
                ax.set_title(plain_label(target, targets))
                ax.set_xticks(positions)
                ax.set_xticklabels(list(methods), rotation=30, ha="right")
                ax.grid(True, axis="y", alpha=0.3)
                if col == 0:
                    ax.set_ylabel(ylabel)
                continue
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "black",
                    "markeredgecolor": "black",
                    "markersize": 4,
                },
                flierprops={
                    "marker": "o",
                    "markersize": 3,
                    "markerfacecolor": "none",
                    "markeredgecolor": "black",
                    "alpha": 0.6,
                },
            )
            _style_boxes(ax, bp["boxes"], methods)
            lo, hi = volume_range([boxplot_stats(v) for v in data])
            ax.set_ylim(lo, hi)
            for x, v in zip(positions, data, strict=True):
                if v.size and v.mean() > hi:  # mean above the axis: triangle at the edge
                    ax.plot(x, hi, marker="^", color="black", markersize=6, clip_on=False)
                    ax.annotate(
                        f"{v.mean():.4g}",
                        xy=(x, hi),
                        xytext=(5, -1),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=6,
                    )
            ax.set_xticks(positions)
            ax.set_xticklabels(list(methods), rotation=30, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel)

    fig.suptitle(f"{exp_name}: test coverage and set volume", fontsize=12, y=0.99)
    handles = _legend_handles(methods) + [
        plt.Line2D([0], [0], color="#993333", linestyle="--", lw=1.2, label="Target 1-epsilon"),
        plt.Line2D([0], [0], marker="D", color="black", lw=0, markersize=4, label="Mean"),
        plt.Line2D(
            [0], [0], marker="^", color="black", lw=0, markersize=6, label="Mean above axis"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 6),
        bbox_to_anchor=(0.5, 0.96),
        fontsize=8,
        frameon=False,
    )
    rows = int(np.ceil(len(handles) / min(len(handles), 6)))
    plt.tight_layout(rect=(0, 0, 1, 0.955 - 0.03 * rows))
    _save(fig, os.path.join(out_dir, f"fig_{exp_name}_boxplot.png"))


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"Figure saved to {path}")
    plt.close(fig)
