"""Known customer mechanism; outcomes are always from the customer's perspective."""

from itertools import product
from numbers import Integral

import numpy as np

BUY, SELL, NO_TRADE, START = range(4)
OUTCOME_NAMES = ("buy", "sell", "no_trade", "start")
QUOTES = np.array([(c - h, c + h) for c in (0.35, 0.50, 0.65)
                   for h in (0.10, 0.20, 0.30)], dtype=np.float64)
ABSTAIN = len(QUOTES)
N_ACTIONS = ABSTAIN + 1
DEFAULT_REGIMES = np.array(list(product((0.0, 1.0), (0.05, 0.30),
                                        (0.25, 0.50, 0.75))), dtype=np.float64)
QUOTES.setflags(write=False)
DEFAULT_REGIMES.setflags(write=False)


def validate_action(action: int) -> int:
    if not isinstance(action, Integral) or not 0 <= action < N_ACTIONS:
        raise ValueError(f"Action must be an integer in [0, {N_ACTIONS - 1}].")
    return int(action)


def validate_regimes(regimes=None) -> np.ndarray:
    result = np.array(DEFAULT_REGIMES if regimes is None else regimes,
                      dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError("Regimes must have nonempty shape (n, 3): value, alpha, p.")
    if not np.isfinite(result).all() or not np.isin(result[:, 0], (0, 1)).all():
        raise ValueError("Regime values must be finite and terminal values binary.")
    if np.any((result[:, 1:] < 0) | (result[:, 1:] > 1)):
        raise ValueError("Alpha and p must be probabilities.")
    return result


def validate_prior(prior, n_regimes: int) -> np.ndarray:
    weights = np.ones(n_regimes) if prior is None else np.asarray(prior, dtype=float)
    if (weights.shape != (n_regimes,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.max(weights) <= 0):
        raise ValueError("Prior must be finite, nonnegative, and have positive total mass.")
    scaled = weights / np.max(weights)
    return scaled / scaled.sum()


def likelihoods(regimes, action: int) -> np.ndarray:
    """Return exact probabilities in BUY, SELL, NO_TRADE order, shape (n, 3)."""
    action = validate_action(action)
    states = np.asarray(regimes, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("Regimes must have shape (n, 3).")
    if action == ABSTAIN:
        return np.tile((0.0, 0.0, 1.0), (len(states), 1))
    bid, ask = QUOTES[action]
    value, alpha, preference = states.T
    buy = alpha * value + (1 - alpha) * preference * (1 - ask)
    sell = alpha * (1 - value) + (1 - alpha) * (1 - preference) * bid
    return np.column_stack((buy, sell, 1 - buy - sell))


def simulate_customer(regime, action: int, uniforms) -> tuple[int, bool]:
    """Simulate one customer using independent type, side, reservation uniforms.

    The tape is indexed by arrival, including abstention; it is never consumed
    conditionally on the quote or customer outcome. Trader type is diagnostic only.
    """
    action = validate_action(action)
    value, alpha, preference = regime
    type_u, side_u, reservation = uniforms
    informed = bool(type_u < alpha)
    if action == ABSTAIN:
        return NO_TRADE, informed
    bid, ask = QUOTES[action]
    if informed:
        if ask < value:
            return BUY, informed
        if bid > value:
            return SELL, informed
        return NO_TRADE, informed
    if side_u < preference:
        return (BUY if reservation >= ask else NO_TRADE), informed
    return (SELL if reservation <= bid else NO_TRADE), informed


def simulate_customers(regime, action: int, uniforms) -> np.ndarray:
    """Vectorized customer-level mechanism, useful for independent frequency tests."""
    action = validate_action(action)
    tape = np.asarray(uniforms)
    if tape.ndim != 2 or tape.shape[1] != 3:
        raise ValueError("Uniform tape must have shape (n, 3).")
    outcomes = np.full(len(tape), NO_TRADE, dtype=np.int64)
    if action == ABSTAIN:
        return outcomes
    value, alpha, preference = regime
    bid, ask = QUOTES[action]
    informed = tape[:, 0] < alpha
    buys = tape[:, 1] < preference
    outcomes[informed & (ask < value)] = BUY
    outcomes[informed & (bid > value)] = SELL
    outcomes[~informed & buys & (tape[:, 2] >= ask)] = BUY
    outcomes[~informed & ~buys & (tape[:, 2] <= bid)] = SELL
    return outcomes
