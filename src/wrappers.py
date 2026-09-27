"""Observation-only history and Bayesian baselines; no rewards or labels as inputs."""

from collections import deque

import gymnasium as gym
import numpy as np

from .bayes_filter import ExactBayes


class HistoryWrapper(gym.Wrapper):
    """Flatten the latest ``window`` observations, oldest first, zero-padded.

    Each original observation contains its previous bid, ask and abstention
    flag, so quote actions paired with the observed outcomes are retained too.
    At reset only the last row contains the episode-start observation.
    """

    def __init__(self, env, window=8):
        super().__init__(env)
        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window
        self.history = deque(maxlen=window)
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low, window),
            high=np.tile(env.observation_space.high, window), dtype=np.float32,
        )

    def _observation(self):
        return np.concatenate(self.history).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        for _ in range(self.window - 1):
            self.history.append(np.zeros_like(obs))
        self.history.append(obs.copy())
        return self._observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs.copy())
        return self._observation(), reward, terminated, truncated, info


class BeliefWrapper(gym.Wrapper):
    """Exact joint posterior plus normalized inventory and remaining time.

    The filter receives only the actual submitted action and the outcome read
    from the public observation. ``info`` and reward never enter the features.
    Support and prior are explicit constructor inputs, not hidden env labels.
    """

    def __init__(self, env, regimes=None, prior=None):
        super().__init__(env)
        self.regimes = regimes
        self.prior = prior
        self.belief = ExactBayes(regimes=regimes, prior=prior)
        count = len(self.belief.posterior)
        self.observation_space = gym.spaces.Box(
            low=np.r_[np.zeros(count), -1.0, 0.0].astype(np.float32),
            high=np.ones(count + 2, dtype=np.float32), dtype=np.float32,
        )

    def _observation(self, public_obs):
        return np.r_[self.belief.posterior, public_obs[-2:]].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.belief = ExactBayes(regimes=self.regimes, prior=self.prior)
        return self._observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        outcome = int(np.argmax(obs[3:7]))
        self.belief.update(int(action), outcome)
        return self._observation(obs), reward, terminated, truncated, info


def wrap_observations(env, kind, history_window=8, regimes=None, prior=None):
    if kind == "history":
        return HistoryWrapper(env, window=history_window)
    if kind == "belief":
        return BeliefWrapper(env, regimes=regimes, prior=prior)
    if kind in {"recurrent", "feedforward"}:
        return env
    raise ValueError(f"Unknown learned policy: {kind}")
