"""Result containers for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from MCP.CP import CP, Array
from MCP.methods import Target
from MCP.shapes import Ellipsoids, FittedEllipsoids


@dataclass
class MethodResult:
    """Outcome of calibrating and evaluating one method in one run.

    Attributes:
        method: Method name, e.g. "RemMCP".
        predictor: Calibrated predictor (None if calibration failed).
        error: Error message if calibration failed.
        infeasible: True if the method was skipped because it cannot certify
            the target with this number of calibration points.
        uncertified: True if calibration found no solution that certifies the
            target (CalibrationError, e.g. H-RelMCP within its iteration limit).
        n_cal: Number of points used for calibration (the threshold split for
            split-SCP methods).
        cal_cov: Coverage on the full calibration set.
        test_cov: Coverage on the test set.
        cal_time: Wall-clock time of shape fitting and calibration [s].
        volume: Monte Carlo estimate of the prediction-set volume.
        ellipsoids: Calibrated set as [(Q, center)].
        fitted_shape: Fitted shape of split-SCP methods.
    """

    method: str
    predictor: CP | None = None
    error: str | None = None
    infeasible: bool = False
    uncertified: bool = False
    n_cal: int | None = None
    cal_cov: float = np.nan
    test_cov: float = np.nan
    cal_time: float = np.nan
    volume: float = np.nan
    ellipsoids: Ellipsoids = field(default_factory=list)
    fitted_shape: FittedEllipsoids | None = None

    @property
    def ok(self) -> bool:
        return self.predictor is not None


@dataclass
class RunResult:
    """All methods calibrated on one data draw.

    Attributes:
        seed: Data seed.
        target: Coverage target of all methods.
        E_cal: Calibration residuals of shape (n_cal, d).
        E_test: Test residuals of shape (n_test, d).
        hrelmcp_target_conf: Confidence H-RelMCP targets under the expected
            guarantee (that of SCP-1); NaN under the high-confidence guarantee.
        hrelmcp_beta: Misconfidence target of H-RelMCP.
        methods: MethodResult per method name, in calibration order.
    """

    seed: int
    target: Target
    E_cal: Array
    E_test: Array
    hrelmcp_target_conf: float = np.nan
    hrelmcp_beta: float = np.nan
    methods: dict[str, MethodResult] = field(default_factory=dict)

    @property
    def epsilon(self) -> float:
        return self.target.epsilon

    @property
    def beta(self) -> float | None:
        return self.target.beta
