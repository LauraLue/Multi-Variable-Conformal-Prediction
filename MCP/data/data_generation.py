# Adapted from conformal_region_designer (Tumu & Cleaveland, MIT License);
# see LICENSE_conformal_region_designer in this directory.
import random

import numpy as np

from .car_env import CarEnvironment

# Constants / Building Blocks
TURN_VALUE = 0.3
ACCEL_VALUE = 0.15
INITIAL_CONDITIONS = np.array([0.0, 0.0, 0.0, 0.5])
COAST = np.array([0.0, 0.0])
LEFT = np.array([TURN_VALUE, 0.0])
RIGHT = np.array([-TURN_VALUE, 0.0])
ACCELERATE = np.array([0.0, ACCEL_VALUE])
DECELERATE = np.array([0.0, -ACCEL_VALUE])

PREFIX_SIZE = 5
SUFFIX_SIZE = 50

# Reference Policies
STRAIGHT_ACCEL = [ACCELERATE] * PREFIX_SIZE + [COAST] * SUFFIX_SIZE
STRAIGHT_ACCEL_LEFT = [ACCELERATE] * PREFIX_SIZE + [ACCELERATE * 1.2 + LEFT] * SUFFIX_SIZE
STRAIGHT_ACCEL_RIGHT = [ACCELERATE] * PREFIX_SIZE + [ACCELERATE * 1.2 + RIGHT] * SUFFIX_SIZE
STRAIGHT_ACCEL_FORWARD = [ACCELERATE] * PREFIX_SIZE + [ACCELERATE * 1.5] * SUFFIX_SIZE

# Randomness
SEED = 42
STD_DEVS = [TURN_VALUE / 1, ACCEL_VALUE / 2.5]


class ActionPolicy:
    def __init__(self, controls: list[np.ndarray], std_dev: list[float]):
        self.controls = controls
        self.std_dev = std_dev

    def generate_noisy_control(self, control: np.ndarray) -> np.ndarray:
        noisy_control = np.array(
            [random.gauss(control[0], self.std_dev[0]), random.gauss(control[1], self.std_dev[1])]
        )
        return noisy_control

    def generate_controls(self) -> list[np.ndarray]:
        controls = [self.generate_noisy_control(control) for control in self.controls]
        return controls


def _rollout(
    env: CarEnvironment, action_policy: ActionPolicy, initial_conditions: np.ndarray
) -> np.ndarray:
    """One rollout of the noisy policy, shape (len(controls), 4)."""
    controls = action_policy.generate_controls()
    env.reset(initial_conditions)
    return np.array([env.step(control) for control in controls])


def create_dataset(
    action_policy: ActionPolicy, initial_conditions: np.ndarray, num_rollouts: int
) -> np.ndarray:
    """Roll out the noisy policy, drawing the noise from Python's global RNG.

    The caller seeds the RNG once per dataset, so that the rollouts of
    different policies draw independent noise.
    """
    env = CarEnvironment(initial_conditions)
    dataset = np.zeros((num_rollouts, len(action_policy.controls), 4))
    for i in range(num_rollouts):
        dataset[i] = _rollout(env, action_policy, initial_conditions)
    return dataset


CALIBRATION_POLICIES = (STRAIGHT_ACCEL_LEFT, STRAIGHT_ACCEL_RIGHT, STRAIGHT_ACCEL)


def sample_rollouts(num_rollouts: int, seed=SEED) -> tuple[np.ndarray, np.ndarray]:
    """I.i.d. rollouts of the three calibration policies.

    Each rollout follows one of CALIBRATION_POLICIES, drawn uniformly at
    random, so that the rollouts are i.i.d. samples of the mixture.  (Tumu et
    al. generate exactly a third of the rollouts with each policy, which
    makes the samples exchangeable but not independent.)

    Returns:
        Tuple of (dataset, policy) with the states of shape
        (num_rollouts, PREFIX_SIZE + SUFFIX_SIZE, 4) and the policy index of
        each rollout.
    """
    # Seed once: reseeding per policy would give rollout i of every policy the
    # same noise sequence, so the samples would not be independent.
    random.seed(seed)
    policies = [ActionPolicy(controls, STD_DEVS) for controls in CALIBRATION_POLICIES]
    env = CarEnvironment(INITIAL_CONDITIONS)
    policy = np.array([random.randrange(len(policies)) for _ in range(num_rollouts)], dtype=int)
    dataset = np.zeros((num_rollouts, PREFIX_SIZE + SUFFIX_SIZE, 4))
    for i, k in enumerate(policy):
        dataset[i] = _rollout(env, policies[k], INITIAL_CONDITIONS)
    return dataset, policy


def generate_calibration_dataset(num_rollouts: int, seed=SEED) -> tuple[np.ndarray, np.ndarray]:
    """I.i.d. calibration rollouts (see sample_rollouts), split into (x, y).

    x holds the first PREFIX_SIZE states and y the remaining SUFFIX_SIZE states.
    """
    dataset, _ = sample_rollouts(num_rollouts, seed=seed)
    return dataset[:, :PREFIX_SIZE, :], dataset[:, PREFIX_SIZE:, :]


def generate_train_dataset(num_rollouts: int, seed=SEED) -> tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    action_policy = ActionPolicy(STRAIGHT_ACCEL, STD_DEVS)
    dataset = create_dataset(action_policy, INITIAL_CONDITIONS, num_rollouts)
    x = dataset[:, :5, :]
    y = dataset[:, 5:, :]
    return (x, y)
