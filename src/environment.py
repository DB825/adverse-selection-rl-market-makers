"""Finite-horizon dealer POMDP with private fixed episode regimes."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .accounting import dense_reward, execution_deltas, inventory_cost, terminal_profit
from .customer_model import (ABSTAIN, N_ACTIONS, NO_TRADE, QUOTES, START,
                             simulate_customer, validate_action, validate_prior,
                             validate_regimes)


class MarketMakingEnv(gym.Env):
    """Observation: [bid, ask, abstain, outcome_onehot(4), q/T, remaining/T].

    Quotes are naturally normalized to [0,1]. An abstention (including reset)
    has bid=ask=0 and abstain=1; reset uses outcome START. Neither diagnostic
    info nor reward enters this feature vector. Inventory is post-arrival.
    """

    metadata = {"render_modes": []}

    def __init__(self, horizon=64, inventory_penalty=0.001, fee=0.0,
                 regimes=None, prior=None):
        super().__init__()
        if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
        if not np.isfinite(inventory_penalty) or inventory_penalty < 0:
            raise ValueError("Inventory penalty must be finite and nonnegative.")
        if not np.isfinite(fee) or fee < 0:
            raise ValueError("Fee must be finite and nonnegative.")
        self.horizon = int(horizon)
        self.inventory_penalty = float(inventory_penalty)
        self.fee = float(fee)
        self.regimes = validate_regimes(regimes)
        self.prior = validate_prior(prior, len(self.regimes))
        self.action_space = spaces.Discrete(N_ACTIONS)
        low = np.zeros(9, dtype=np.float32)
        low[7] = -1.0
        self.observation_space = spaces.Box(low, np.ones(9, dtype=np.float32),
                                            dtype=np.float32)
        self._has_reset = False

    def _observation(self) -> np.ndarray:
        """Construct solely from public history, inventory, and time."""
        obs = np.zeros(9, dtype=np.float32)
        if self._previous_action == ABSTAIN:
            obs[2] = 1.0
        else:
            obs[:2] = QUOTES[self._previous_action]
        obs[3 + self._previous_outcome] = 1.0
        obs[7] = self.inventory / self.horizon
        obs[8] = 1 - self.t / self.horizon
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        chosen = options.get("regime")
        if chosen is None:
            self.regime_index = int(self.np_random.choice(len(self.regimes), p=self.prior))
            self.regime = self.regimes[self.regime_index].copy()
        elif isinstance(chosen, (int, np.integer)):
            if not 0 <= chosen < len(self.regimes):
                raise ValueError("Regime index is outside support.")
            self.regime_index = int(chosen)
            self.regime = self.regimes[self.regime_index].copy()
        else:
            self.regime = validate_regimes([chosen])[0]
            matches = np.flatnonzero(np.all(self.regimes == self.regime, axis=1))
            self.regime_index = int(matches[0]) if len(matches) else -1
        supplied_tape = options.get("tape")
        if supplied_tape is None:
            self._tape = self.np_random.random((self.horizon, 3))
        else:
            tape = np.asarray(supplied_tape, dtype=np.float64)
            if (tape.shape != (self.horizon, 3) or not np.isfinite(tape).all()
                    or np.any((tape < 0) | (tape >= 1))):
                raise ValueError("Tape must have shape (horizon, 3) and values in [0,1).")
            self._tape = tape.copy()
        self.inventory = 0
        self.cash = 0.0
        self.t = 0
        self.inventory_penalty_total = 0.0
        self._previous_action = ABSTAIN
        self._previous_outcome = START
        self._has_reset = True
        return self._observation(), {}

    def step(self, action):
        if not self._has_reset or self.t >= self.horizon:
            raise RuntimeError("Call reset before stepping a new or terminated episode.")
        action = validate_action(action)
        outcome, informed = simulate_customer(self.regime, action, self._tape[self.t])
        cash_delta, inventory_delta = execution_deltas(action, outcome, self.fee)
        self.cash += cash_delta
        self.inventory += inventory_delta
        self.t += 1
        self._previous_action = action
        self._previous_outcome = outcome
        self.inventory_penalty_total += inventory_cost(
            self.inventory, self.inventory_penalty, self.horizon)
        terminated = self.t == self.horizon
        reward = dense_reward(cash_delta, inventory_delta, self.inventory,
                              self.inventory_penalty, self.horizon,
                              terminal=terminated, value=self.regime[0])
        info = {"action": action, "outcome": outcome,
                "customer_informed": informed,
                "execution_profit": float(cash_delta + inventory_delta * self.regime[0])}
        if terminated:
            profit = terminal_profit(self.cash, self.inventory, self.regime[0])
            info.update(profit=profit,
                        inventory_penalty_total=self.inventory_penalty_total,
                        objective=profit - self.inventory_penalty_total)
        return self._observation(), reward, terminated, False, info
