"""Dealer accounting in unit size; all rewards use the public mark 0.5."""

from .customer_model import ABSTAIN, BUY, NO_TRADE, QUOTES, SELL, validate_action

PUBLIC_MARK = 0.5


def execution_deltas(action: int, outcome: int, fee: float = 0.0) -> tuple[float, int]:
    """Return (cash change, inventory change), fee charged once per execution."""
    action = validate_action(action)
    if fee < 0:
        raise ValueError("Execution fee must be nonnegative.")
    if outcome not in (BUY, SELL, NO_TRADE):
        raise ValueError("Execution outcome must be buy, sell, or no trade.")
    if action == ABSTAIN:
        if outcome != NO_TRADE:
            raise ValueError("An abstaining dealer cannot execute.")
        return 0.0, 0
    if outcome == BUY:
        return float(QUOTES[action, 1] - fee), -1
    if outcome == SELL:
        return float(-QUOTES[action, 0] - fee), 1
    return 0.0, 0


def inventory_cost(inventory: int, penalty: float, horizon: int) -> float:
    """Cost for post-arrival inventory, including arrivals with no execution."""
    return float(penalty * inventory**2 / horizon)


def dense_reward(cash_delta: float, inventory_delta: int, inventory: int,
                 penalty: float, horizon: int, *, terminal: bool = False,
                 value: float | None = None) -> float:
    """At gamma=1, summing these rewards telescopes to the economic objective."""
    reward = cash_delta + PUBLIC_MARK * inventory_delta - inventory_cost(
        inventory, penalty, horizon)
    if terminal:
        if value is None:
            raise ValueError("Terminal settlement requires a value.")
        reward += inventory * (value - PUBLIC_MARK)
    return float(reward)


def terminal_profit(cash: float, inventory: int, value: float) -> float:
    return float(cash + inventory * value)
