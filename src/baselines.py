"""Predeclared fixed quotes and an exact-Bayes myopic decision rule."""

import numpy as np


# Menu is center-major, half-spread-minor. No choices are fitted on evaluation.
FIXED_ACTIONS = {"fixed_narrow": 3, "fixed_medium": 4,
                 "fixed_wide": 5, "abstain": 9}


def myopic_scores(belief, inventory, inventory_penalty=0.001, horizon=64, fee=0.0):
    """One-step profit minus expected *incremental* inventory carrying cost.

    The common current-inventory carrying charge cancels across actions. This
    is a greedy reference, not the solution of the finite-horizon POMDP.
    A customer buy changes dealer q by -1, a customer sell by +1.
    """
    if horizon <= 0 or inventory_penalty < 0:
        raise ValueError("horizon must be positive and penalty nonnegative")
    scores = np.zeros(10, dtype=float)
    for action in range(9):
        stats = belief.action_statistics(action, fee=fee)
        expected_delta_q2 = (
            stats["p_buy"] * (1 - 2 * inventory)
            + stats["p_sell"] * (1 + 2 * inventory)
        )
        scores[action] = (stats["expected_profit"]
                          - inventory_penalty / horizon * expected_delta_q2)
    return scores


def myopic_action(belief, inventory, inventory_penalty=0.001, horizon=64, fee=0.0):
    """Deterministic lowest-index tie breaking (including abstention)."""
    return int(np.argmax(myopic_scores(belief, inventory, inventory_penalty, horizon, fee)))


class FixedPolicy:
    def __init__(self, action):
        if int(action) != action or not 0 <= action < 10:
            raise ValueError("action must be an integer in [0,9]")
        self.action = int(action)

    def probabilities(self):
        result = np.zeros(10)
        result[self.action] = 1.0
        return result

    def predict(self, observation=None, **kwargs):
        return self.action, None
