"""Registry of the prediction-set methods compared in the experiments (Sec. 4).

Each method is described by a MethodSpec: display name, short key (used in
pickles and file names), plotting styles, and a builder that turns the
calibration residuals of one run into a configured, uncalibrated predictor.

  SCP-1          split conformal prediction with the 2-norm score
  SCP-kX         SCP with a data split: a fraction of the calibration data is
                 reserved for the threshold, the rest fits the set shape;
                 k in {2, 3, 4, 5} selects the fraction, X the shape fit
                 (A: one ellipsoid, B: k-means union, C: CMA-ES union)
  RemMCP         MCP via constraint removal, single ellipsoid
  H-RelMCP-A/B     MCP via constraint relaxation, single ellipsoid / union

All methods of a run share one coverage Target: either the expected guarantee
E[eta] >= 1 - epsilon (alpha = epsilon) or the high-confidence guarantee
P(eta >= 1 - epsilon) >= 1 - beta.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

from MCP.CP import (
    CP,
    EXPECTED,
    HIGH_CONFIDENCE,
    Array,
    _check_unit_interval,
    _require,
    confidence_bound,
    outlier_budget_expected,
    outlier_budget_high_confidence,
)
from MCP.fitting import fit_ellipsoid_kmeans, fit_ellipsoid_tumu, init_q0_from_fit_ellipsoid
from MCP.HRelMCP import HRelMCP, tradeoff_epsilon
from MCP.RemMCP import RemMCP
from MCP.SCP import SCP
from MCP.shapes import Ellipsoid, Ellipsoids, FittedEllipsoids, UnionOfEllipsoids, score_2norm


class TargetKwargs(TypedDict, total=False):
    """Target keyword arguments of CP subclasses: alpha, or epsilon and beta."""

    alpha: float
    epsilon: float
    beta: float


@dataclass(frozen=True)
class Target:
    """Coverage target shared by all methods of a run.

    With beta=None, SCP and RemMCP target the expected guarantee
    E[eta] >= 1 - alpha with alpha = epsilon (the paper's convention), and
    H-RelMCP, which only provides the high-confidence guarantee, targets the
    confidence of SCP-1 (see hrelmcp_target_confidence).  Otherwise all methods
    target P(eta >= 1 - epsilon) >= 1 - beta.

    Attributes:
        epsilon: Miscoverage level in (0, 1).
        beta: Misconfidence level in (0, 1), or None for the expected guarantee.
        variant: LaTeX label that distinguishes experiments with the same
            levels in a combined figure, e.g. "$n_y = 4$" (see
            MCP.reporting.regenerate.combine); not used for calibration.
    """

    epsilon: float
    beta: float | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        _check_unit_interval("epsilon", self.epsilon)
        if self.beta is not None:
            _check_unit_interval("beta", self.beta)

    @property
    def guarantee(self) -> str:
        """EXPECTED or HIGH_CONFIDENCE, as in CP.guarantee."""
        return EXPECTED if self.beta is None else HIGH_CONFIDENCE

    def predictor_kwargs(self) -> TargetKwargs:
        """Target keyword arguments of SCP and RemMCP."""
        if self.beta is None:
            return {"alpha": self.epsilon}
        return {"epsilon": self.epsilon, "beta": self.beta}

    def hrelmcp_beta(self, n_cal: int) -> float:
        """Misconfidence target of H-RelMCP calibrated on n_cal points."""
        if self.beta is not None:
            return self.beta
        return max(1e-10, 1.0 - hrelmcp_target_confidence(n_cal, self.epsilon))

    @property
    def plain_variant(self) -> str | None:
        """The variant without LaTeX markup, e.g. "n_y = 4"."""
        if self.variant is None:
            return None
        return re.sub(r"[$\\{}]", "", self.variant)

    @property
    def label(self) -> str:
        """Plain-text label, e.g. "eps=0.05" or "eps=0.05, beta=1e-06 [n_y = 4]"."""
        label = f"eps={self.epsilon:g}"
        if self.beta is not None:
            label += f", beta={self.beta:.0e}"
        if self.variant is not None:
            label += f" [{self.plain_variant}]"
        return label

    @property
    def slug(self) -> str:
        """File-name fragment, e.g. "eps0.05" or "eps0.05_beta1e-06"."""
        if self.beta is None:
            return f"eps{self.epsilon:g}"
        return f"eps{self.epsilon:g}_beta{self.beta:.0e}"


@dataclass(frozen=True)
class CalibrationTask:
    """Inputs shared by all methods within one run.

    Attributes:
        E_cal: Calibration residuals of shape (n_cal, d).
        target: Coverage target of all methods.
        n_shapes: Number of ellipsoids for union-of-ellipsoids methods.
        seed: Run seed (CMA-ES and H-RelMCP restarts).
    """

    E_cal: Array
    target: Target
    n_shapes: int
    seed: int

    @property
    def epsilon(self) -> float:
        """Miscoverage level, also the level of the split-SCP shape fits."""
        return self.target.epsilon

    @property
    def hrelmcp_beta(self) -> float:
        """Misconfidence target of H-RelMCP on the full calibration set."""
        return self.target.hrelmcp_beta(self.E_cal.shape[0])


@dataclass
class Setup:
    """A configured predictor and the data it is calibrated on.

    Attributes:
        predictor: Uncalibrated predictor.
        Y_cal: Residuals used for calibration (a subset for split SCP).
        geometry: Returns the calibrated set as [(Q, center)] for plotting.
        fitted_shape: Fitted shape of split-SCP methods, else None.
    """

    predictor: CP
    Y_cal: Array
    geometry: Callable[[], Ellipsoids]
    fitted_shape: FittedEllipsoids | None = None


# Colors of the paper (\definecolor in the paper preamble), shared by the TikZ and PNG figures.
PAPER_COLORS: dict[str, str] = {
    "scp1color": "#FFE101",  # RGB 255, 225, 1
    "scp2Ccolor": "#916BA4",
    "scp3color": "#F2AA84",  # RGB 242, 170, 132
    "scp3Ccolor": "#916BA4",
    "scp4color": "#916BA4",
    "scp4Bcolor": "#AE6264",  # RGB 174, 98, 100
    "scp4Ccolor": "#916BA4",
    "scp5color": "#F2AA84",
    "cmcp1color": "#4472C4",  # RGB 68, 114, 196
    "nmcp1color": "#548235",  # RGB 84, 130, 53
}


def _scalar_n_q(dim: int, n_shapes: int) -> int:
    return 1


def _n_threshold(split_ratio: float, n_cal: int) -> int:
    """Number of calibration points reserved for the threshold of split SCP."""
    return int(np.floor(split_ratio * n_cal))


@dataclass(frozen=True)
class MethodSpec:
    """Description of one method in the comparison.

    Attributes:
        name: Display name, e.g. "H-RelMCP-B".
        key: Short identifier used in pickles, e.g. "hrelmcpb".
        tikz_color: Name of the LaTeX color defined in the paper preamble
            (PAPER_COLORS); the PNG figures use the same color.
        tikz_style: Fill style: "" (solid), "dashed" (hatched, the B variants
            and H-RelMCP-B in the paper), or "dotted".
        build: Creates the predictor setup for a calibration task.
        split_ratio: Fraction of calibration data reserved for the threshold
            (split-SCP methods only).
        n_q: Parameter dimension as a function of (output dim, n_shapes).
        relaxation: True for H-RelMCP, whose guarantee follows from the
            confidence-coverage tradeoff instead of an outlier budget.
    """

    name: str
    key: str
    tikz_color: str
    tikz_style: str
    build: Callable[[CalibrationTask], Setup] = field(repr=False)
    split_ratio: float | None = None
    n_q: Callable[[int, int], int] = field(default=_scalar_n_q, repr=False)
    relaxation: bool = False

    @property
    def color(self) -> str:
        """Hex color of the method, as in the paper."""
        return PAPER_COLORS[self.tikz_color]

    def n_threshold(self, n_cal: int) -> int:
        """Number of points the guarantee is certified on (the split for split SCP)."""
        return n_cal if self.split_ratio is None else _n_threshold(self.split_ratio, n_cal)


def hrelmcp_target_confidence(n_cal: int, epsilon: float) -> float:
    """Confidence of SCP-1 with n_cal points at expected level alpha = epsilon.

    Under the expected guarantee, H-RelMCP targets at least this confidence, so
    that all methods are compared at a comparable high-confidence level.
    """
    rho = outlier_budget_expected(n_cal, 1, epsilon)
    return 1.0 - confidence_bound(rho, n_cal, 1, epsilon)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_scp1(task: CalibrationTask) -> Setup:
    predictor = SCP(score_fn=score_2norm, **task.target.predictor_kwargs())
    dim = task.E_cal.shape[1]

    def geometry() -> Ellipsoids:
        radius = _require(predictor.q_opt, "q_opt")
        return [(np.eye(dim) / (radius * radius), np.zeros(dim))]

    return Setup(predictor, task.E_cal, geometry)


_SHAPE_FITS = {
    "A": lambda E_fit, task: fit_ellipsoid_kmeans(E_fit, alpha=task.epsilon, n_shapes=1),
    "B": lambda E_fit, task: fit_ellipsoid_kmeans(
        E_fit, alpha=task.epsilon, n_shapes=task.n_shapes
    ),
    "C": lambda E_fit, task: fit_ellipsoid_tumu(
        E_fit, alpha=task.epsilon, n_shapes=task.n_shapes, seed=task.seed
    ),
}


def _split_scp_builder(split_ratio: float, fit: str) -> Callable[[CalibrationTask], Setup]:
    """SCP on a fitted union of ellipsoids with a calibration-data split."""

    def build(task: CalibrationTask) -> Setup:
        n_cal = task.E_cal.shape[0]
        n_threshold = _n_threshold(split_ratio, n_cal)
        E_fit, E_threshold = task.E_cal[: n_cal - n_threshold], task.E_cal[n_cal - n_threshold :]
        fitted = _SHAPE_FITS[fit](E_fit, task)
        predictor = SCP(score_fn=fitted.score, **task.target.predictor_kwargs())
        return Setup(
            predictor,
            E_threshold,
            lambda: fitted.ellipsoids(_require(predictor.q_opt, "q_opt")),
            fitted_shape=fitted,
        )

    return build


def _build_remmcp(task: CalibrationTask) -> Setup:
    shape = Ellipsoid(task.E_cal.shape[1])
    predictor = RemMCP(
        score_fn=shape.score,
        cost_fn=shape.cost,
        n_q=shape.n_q,
        **task.target.predictor_kwargs(),
        q0=shape.initial_q(),
        cost_grad_fn=shape.cost_grad,
        bounds=shape.bounds,
    )
    return Setup(predictor, task.E_cal, lambda: shape.ellipsoids(predictor.q_opt))


def _build_hrelmcp_a(task: CalibrationTask) -> Setup:
    shape = Ellipsoid(task.E_cal.shape[1])
    predictor = HRelMCP(
        score_fn=shape.score,
        cost_fn=shape.cost,
        n_q=shape.n_q,
        epsilon=task.epsilon,
        beta=task.hrelmcp_beta,
        q0=shape.initial_q(),
        bounds=shape.bounds,
        score_grad_fn=shape.score_grad,
        cost_grad_fn=shape.cost_grad,
        n_restarts=3,
        max_outer_iter=100,
        max_total_iter=200,
        random_state=task.seed,
    )
    return Setup(predictor, task.E_cal, lambda: shape.ellipsoids(predictor.q_opt))


def _build_hrelmcp_b(task: CalibrationTask) -> Setup:
    shape = UnionOfEllipsoids(task.E_cal.shape[1], task.n_shapes)
    predictor = HRelMCP(
        score_fn=shape.score,
        cost_fn=shape.cost,
        n_q=shape.n_q,
        epsilon=task.epsilon,
        beta=task.hrelmcp_beta,
        q0=init_q0_from_fit_ellipsoid(task.E_cal, alpha=task.epsilon, n_shapes=task.n_shapes),
        bounds=shape.bounds,
        score_grad_fn=shape.score_grad,
        cost_grad_fn=shape.cost_grad,
        n_restarts=10,
        restart_scale=0.5,
        max_outer_iter=100,
        max_total_iter=200,
        max_iter=1000,
        random_state=task.seed,
    )
    return Setup(predictor, task.E_cal, lambda: shape.ellipsoids(predictor.q_opt))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _split_scp_specs() -> list[MethodSpec]:
    ratios = {2: 0.25, 3: 0.50, 4: 0.75, 5: 0.90}
    styles = {"A": "", "B": "dashed", "C": "dotted"}
    # As in the paper legend: SCP-2A/B in scp3color, SCP-3A/B in scp4color (B hatched).
    tikz_colors = {
        "SCP-2A": "scp3color",
        "SCP-2B": "scp3color",
        "SCP-2C": "scp2Ccolor",
        "SCP-3A": "scp4color",
        "SCP-3B": "scp4color",
        "SCP-3C": "scp3Ccolor",
        "SCP-4A": "scp4Bcolor",
        "SCP-4B": "scp4Bcolor",
        "SCP-4C": "scp4Ccolor",
        "SCP-5A": "scp5color",
        "SCP-5B": "scp5color",
        "SCP-5C": "scp2Ccolor",
    }
    return [
        MethodSpec(
            name=f"SCP-{k}{fit}",
            key=f"scp{k}{fit.lower()}",
            tikz_color=tikz_colors[f"SCP-{k}{fit}"],
            tikz_style=styles[fit],
            build=_split_scp_builder(ratio, fit),
            split_ratio=ratio,
        )
        for k, ratio in ratios.items()
        for fit in ("A", "B", "C")
    ]


METHODS: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in [
        MethodSpec("SCP-1", "scp1", "scp1color", "", _build_scp1),
        *_split_scp_specs(),
        MethodSpec(
            "RemMCP",
            "remmcp",
            "cmcp1color",
            "",
            _build_remmcp,
            n_q=lambda dim, n_shapes: Ellipsoid(dim).n_q,
        ),
        MethodSpec(
            "H-RelMCP-A",
            "hrelmcp",
            "nmcp1color",
            "",
            _build_hrelmcp_a,
            n_q=lambda dim, n_shapes: Ellipsoid(dim).n_q,
            relaxation=True,
        ),
        MethodSpec(
            "H-RelMCP-B",
            "hrelmcpb",
            "nmcp1color",
            "dashed",
            _build_hrelmcp_b,
            n_q=lambda dim, n_shapes: UnionOfEllipsoids(dim, n_shapes).n_q,
            relaxation=True,
        ),
    ]
}

PAPER_METHODS = (
    "SCP-1",
    "SCP-2A",
    "SCP-2B",
    "SCP-3A",
    "SCP-3B",
    "RemMCP",
    "H-RelMCP-A",
    "H-RelMCP-B",
)
UNIMODAL_METHODS = ("SCP-1", "SCP-2A", "SCP-3A", "RemMCP", "H-RelMCP-A")


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def _certifiable(spec: MethodSpec, target: Target, n_cal: int, dim: int, n_shapes: int) -> bool:
    """Whether the method can certify the target at all with n_cal points.

    SCP and RemMCP need an outlier budget rho >= 0 (exact).  H-RelMCP needs the
    tradeoff to certify epsilon for a solution without violations at the
    first iteration (d = n_q); otherwise every iteration is invalid, since
    epsilon_i only grows with the iteration i and the complexity d_i.  For
    H-RelMCP this condition is necessary but not sufficient.
    """
    n_q = spec.n_q(dim, n_shapes)
    n = spec.n_threshold(n_cal)
    if spec.relaxation:
        d = min(n_q, n - 1)  # HRelMCP._estimate_d_i for zero violations
        return tradeoff_epsilon(target.hrelmcp_beta(n_cal), 1, d, n) <= target.epsilon
    if target.beta is None:
        return outlier_budget_expected(n, n_q, target.epsilon) >= 0
    return outlier_budget_high_confidence(n, n_q, target.epsilon, target.beta) >= 0


def min_ncal(name: str, target: Target, dim: int, n_shapes: int = 3) -> int:
    """Smallest n_cal with which the method can certify the target (see feasibility_issue).

    Found by bisection, which is exact when certifiability is monotone in
    n_cal.  This holds for the outlier budgets and for H-RelMCP with a fixed
    beta; under the expected guarantee, H-RelMCP's beta (the confidence of
    SCP-1) varies slightly with n_cal.
    """
    spec = METHODS[name]
    hi = 2
    while not _certifiable(spec, target, hi, dim, n_shapes):
        hi *= 2
    lo = hi // 2  # not certifiable (or 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _certifiable(spec, target, mid, dim, n_shapes):
            hi = mid
        else:
            lo = mid
    return hi


def feasibility_issue(
    name: str, target: Target, n_cal: int, dim: int, n_shapes: int = 3
) -> str | None:
    """Reason why the method cannot certify the target with n_cal points, or None.

    Calibrating such a method would raise CalibrationError on every data
    draw, so experiments can skip it up front.  None does not guarantee that
    H-RelMCP finds a certified solution (see _certifiable).

    Args:
        name: Method name from METHODS.
        target: Coverage target.
        n_cal: Number of calibration points (before any split).
        dim: Output dimension.
        n_shapes: Number of ellipsoids in unions.
    """
    spec = METHODS[name]
    if _certifiable(spec, target, n_cal, dim, n_shapes):
        return None
    n_q = spec.n_q(dim, n_shapes)
    n = spec.n_threshold(n_cal)
    if spec.relaxation:
        reason = (
            f"the tradeoff cannot certify eps={target.epsilon:g}, "
            f"beta={target.hrelmcp_beta(n_cal):.1e} even without violations (n_q={n_q}, n_cal={n})"
        )
    else:
        reason = f"no valid outlier budget for {target.label} (n_q={n_q}, n_cal={n})"
    return f"{name} infeasible: {reason}; needs n_cal >= {min_ncal(name, target, dim, n_shapes)}"
