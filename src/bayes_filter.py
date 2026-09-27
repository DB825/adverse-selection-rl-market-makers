"""Exact joint finite-state inference; diagnostic truth is never used."""

import numpy as np

from .customer_model import (ABSTAIN, BUY, N_ACTIONS, NO_TRADE, QUOTES, SELL, likelihoods,
                             validate_action, validate_prior, validate_regimes)


class ExactBayes:
    def __init__(self, regimes=None, prior=None, conditioning_epsilon=1e-12):
        self.regimes = validate_regimes(regimes)
        self.prior = validate_prior(prior, len(self.regimes))
        if not 0 <= conditioning_epsilon < 1:
            raise ValueError("Conditioning epsilon must lie in [0, 1).")
        self.conditioning_epsilon = conditioning_epsilon
        self._likelihoods = np.stack([likelihoods(self.regimes, action)
                                      for action in range(N_ACTIONS)])
        with np.errstate(divide="ignore"):
            self._log_likelihoods = np.log(self._likelihoods)
        self.reset()

    @property
    def posterior(self):
        return np.exp(self._log_posterior)

    def reset(self):
        with np.errstate(divide="ignore"):
            self._log_posterior = np.log(self.prior)
        return self.posterior

    def update(self, action: int, outcome: int):
        """Update on actual quote and observed outcome; impossible events raise.

        Abstention with NO_TRADE is explicitly a no-op. A contradictory buy/sell
        or event impossible under all positive-prior states leaves state unchanged
        and raises ValueError rather than silently replacing the prior.
        """
        action = validate_action(action)
        if outcome not in (BUY, SELL, NO_TRADE):
            raise ValueError("Only buy, sell, or no-trade observations can update beliefs.")
        if action == ABSTAIN:
            if outcome != NO_TRADE:
                raise ValueError("Buy/sell is impossible under abstention.")
            return self.posterior
        updated = self._log_posterior + self._log_likelihoods[action, :, outcome]
        maximum = updated.max()
        if not np.isfinite(maximum):
            raise ValueError("Observation has zero probability under the current support.")
        log_normalizer = maximum + np.log(np.exp(updated - maximum).sum())
        self._log_posterior = updated - log_normalizer
        return self.posterior

    def summary(self) -> dict[str, float]:
        weights = self.posterior
        moments = weights @ self.regimes
        positive = weights > 0
        entropy = -np.sum(weights[positive] * self._log_posterior[positive])
        return dict(value=float(moments[0]), alpha=float(moments[1]),
                    p=float(moments[2]), entropy=float(entropy))

    def action_statistics(self, action: int, fee=0.0) -> dict[str, float]:
        """Conditional value shifts and quote-specific net dealer profits.

        value/as/profit conditioned on negligible events are NaN. Unconditional
        expected profit is evaluated directly from probability-weighted masses,
        so it remains finite even when a conditioning event is impossible.
        """
        action = validate_action(action)
        if not np.isfinite(fee) or fee < 0:
            raise ValueError("Fee must be finite and nonnegative.")
        weights = self.posterior
        probs = self._likelihoods[action]
        predictive = weights @ probs
        value_masses = (weights * self.regimes[:, 0]) @ probs
        fair = float(weights @ self.regimes[:, 0])
        value_buy = (float(value_masses[BUY] / predictive[BUY])
                     if predictive[BUY] > self.conditioning_epsilon else float("nan"))
        value_sell = (float(value_masses[SELL] / predictive[SELL])
                      if predictive[SELL] > self.conditioning_epsilon else float("nan"))
        bid, ask = (0.0, 0.0) if action == ABSTAIN else QUOTES[action]
        expected_profit = ((ask - fee) * predictive[BUY] - value_masses[BUY]
                           + value_masses[SELL] - (bid + fee) * predictive[SELL])
        return dict(p_buy=float(predictive[BUY]), p_sell=float(predictive[SELL]),
                    p_no_trade=float(predictive[NO_TRADE]), value_buy=value_buy,
                    value_sell=value_sell, as_buy=value_buy - fair,
                    as_sell=fair - value_sell, profit_buy=float(ask - value_buy - fee),
                    profit_sell=float(value_sell - bid - fee),
                    expected_profit=float(expected_profit))
