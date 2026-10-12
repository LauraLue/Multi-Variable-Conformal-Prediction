"""Constant turn-rate and velocity predictor for the vehicle benchmark [Tumu et al.].

Given the observed states (x, y, theta, v) of a vehicle, the predictor
extrapolates the trajectory with the last velocity and the average turn rate
over the observation window.  Computations use NumPy in float64.

Adapted from conformal_region_designer (Tumu & Cleaveland, MIT License); see
LICENSE_conformal_region_designer in this directory.
"""

import numpy as np


class TrajectoryPredictor:
    """Constant turn-rate, constant velocity trajectory predictor.

    Args:
        dt: Time-step size [s].
        predict_steps: Number of predicted time steps.
    """

    def __init__(self, dt: float, predict_steps: int = 50):
        self.dt = dt
        self.steps = predict_steps

    def predict_batched_trajectories(self, states: np.ndarray) -> np.ndarray:
        """Predict trajectories for a batch of observed state sequences.

        Args:
            states: Observed states of shape (n, n_obs, 4) with columns
                (x, y, theta, v).

        Returns:
            Predicted states of shape (n, predict_steps, 4).
        """
        states = np.asarray(states, dtype=float)
        v = states[:, -1, 3]
        dtheta = (states[:, -1, 2] - states[:, 0, 2]) / (states.shape[1] - 1)
        x, y, theta = states[:, -1, 0], states[:, -1, 1], states[:, -1, 2]

        trajectory = np.empty((states.shape[0], self.steps, 4))
        for k in range(self.steps):
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            theta = theta + dtheta
            trajectory[:, k] = np.stack([x, y, theta, v], axis=-1)
        return trajectory

    def predict_trajectory(self, states: np.ndarray) -> np.ndarray:
        """Predict the trajectory for one observed state sequence of shape (n_obs, 4)."""
        return self.predict_batched_trajectories(np.asarray(states)[np.newaxis])[0]
