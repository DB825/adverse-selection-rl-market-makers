"""Exact joint inference, rare events, and action-conditioned economics."""

import numpy as np
import pytest

from src.bayes_filter import ExactBayes
from src.customer_model import ABSTAIN, BUY, DEFAULT_REGIMES, NO_TRADE, QUOTES, SELL, likelihoods


def test_hand_computed_one_step_update():
    bayes = ExactBayes(regimes=[[0, 0.3, 0.5], [1, 0.3, 0.5]])
    # At b=.4,a=.6 the buy likelihoods are .14 and .44.
    posterior = bayes.update(3, BUY)
    np.testing.assert_allclose(posterior, [0.14 / 0.58, 0.44 / 0.58])
    assert bayes.summary()["value"] == pytest.approx(0.44 / 0.58)
    # Sell probabilities are .44 and .14, restoring symmetry after buy then sell.
    np.testing.assert_allclose(bayes.update(3, SELL), [0.5, 0.5])


def test_sequential_updates_match_full_history_enumeration():
    prior = np.arange(1, 13, dtype=float)
    prior /= prior.sum()
    bayes = ExactBayes(prior=prior)
    histories = [(0, BUY), (8, NO_TRADE), (5, SELL), (2, BUY),
                 (ABSTAIN, NO_TRADE), (6, NO_TRADE), (4, BUY)]
    full = prior.copy()
    for action, outcome in histories:
        full *= likelihoods(DEFAULT_REGIMES, action)[:, outcome]
        bayes.update(action, outcome)
    full /= full.sum()
    np.testing.assert_allclose(bayes.posterior, full, atol=1e-14)
    assert bayes.posterior.sum() == pytest.approx(1)
    # The joint posterior develops dependence; a product of marginals loses it.
    joint = full.reshape(2, 2, 3)
    independent = (joint.sum(axis=(1, 2))[:, None, None]
                   * joint.sum(axis=(0, 2))[None, :, None]
                   * joint.sum(axis=(0, 1))[None, None, :])
    assert np.linalg.norm(joint - independent) > 0.01


def test_abstention_is_exact_noop_and_no_trade_is_informative():
    bayes = ExactBayes()
    original = bayes.posterior
    logs = bayes._log_posterior.copy()
    for _ in range(30):
        bayes.update(ABSTAIN, NO_TRADE)
    np.testing.assert_array_equal(bayes.posterior, original)
    np.testing.assert_array_equal(bayes._log_posterior, logs)
    old_alpha = bayes.summary()["alpha"]
    bayes.update(3, NO_TRADE)
    assert bayes.summary()["alpha"] < old_alpha
    assert not np.allclose(bayes.posterior, original)


def test_actual_quote_changes_inference_from_identical_buy_flow():
    low_ask, high_ask = ExactBayes(), ExactBayes()
    for _ in range(5):
        low_ask.update(0, BUY)
        high_ask.update(8, BUY)
    assert high_ask.summary()["value"] > low_ask.summary()["value"]
    assert high_ask.summary()["alpha"] > low_ask.summary()["alpha"]


def test_log_updates_are_stable_for_long_extreme_history():
    bayes = ExactBayes(prior=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    for _ in range(3000):
        bayes.update(8, BUY)
    posterior = bayes.posterior
    assert np.isfinite(posterior).all()
    assert posterior.sum() == pytest.approx(1)
    assert bayes.summary()["value"] == pytest.approx(1)
    assert np.isfinite(bayes.summary()["entropy"])
    assert np.all(posterior[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]] == 0)


def test_impossible_events_raise_without_corrupting_filter():
    bayes = ExactBayes(regimes=[[0, 1, 0.5]])
    original = bayes.posterior
    for action, outcome in [(3, BUY), (3, NO_TRADE), (ABSTAIN, BUY), (3, 3)]:
        with pytest.raises(ValueError):
            bayes.update(action, outcome)
        np.testing.assert_array_equal(bayes.posterior, original)
    assert bayes.update(3, SELL)[0] == 1


def test_negligible_conditioning_is_explicit_but_expectation_is_finite():
    bayes = ExactBayes(regimes=[[0, 1, 0.5]])
    stats = bayes.action_statistics(3)
    assert stats["p_buy"] == 0
    assert np.isnan(stats["value_buy"])
    assert np.isnan(stats["as_buy"])
    assert np.isnan(stats["profit_buy"])
    assert stats["expected_profit"] == pytest.approx(-0.4)
    abstain = bayes.action_statistics(ABSTAIN)
    assert abstain["expected_profit"] == 0
    assert abstain["p_no_trade"] == 1
    assert np.isnan(abstain["value_sell"])
    tiny = ExactBayes(regimes=[[0, 0, 1e-14]])
    assert np.isnan(tiny.action_statistics(3)["value_buy"])
    assert np.isfinite(tiny.action_statistics(3)["expected_profit"])


def test_action_statistics_match_direct_economic_enumeration():
    bayes = ExactBayes()
    bayes.update(0, BUY)
    bayes.update(8, NO_TRADE)
    fee = 0.007
    for action, (bid, ask) in enumerate(QUOTES):
        stats = bayes.action_statistics(action, fee=fee)
        probs = likelihoods(DEFAULT_REGIMES, action)
        values = DEFAULT_REGIMES[:, 0]
        direct_profit = np.sum(bayes.posterior * (probs[:, BUY] * (ask - values - fee)
                                                  + probs[:, SELL] * (values - bid - fee)))
        assert stats["expected_profit"] == pytest.approx(direct_profit)
        assert stats["expected_profit"] == pytest.approx(
            stats["p_buy"] * stats["profit_buy"] + stats["p_sell"] * stats["profit_sell"])
        assert stats["as_buy"] == pytest.approx(stats["value_buy"] - bayes.summary()["value"])
        assert stats["as_sell"] == pytest.approx(bayes.summary()["value"] - stats["value_sell"])
        # Paying a fee affects profits but cannot affect customer inference.
        free = bayes.action_statistics(action)
        assert free["value_buy"] == stats["value_buy"]
        assert free["expected_profit"] - stats["expected_profit"] == pytest.approx(
            fee * (stats["p_buy"] + stats["p_sell"]))


def test_posterior_and_adverse_selection_mirror_symmetry():
    bayes = ExactBayes()
    mirror = ExactBayes()
    history = [(0, BUY), (7, SELL), (2, NO_TRADE), (6, BUY)]
    for action, outcome in history:
        mirrored_action = (2 - action // 3) * 3 + action % 3
        mirrored_outcome = {BUY: SELL, SELL: BUY, NO_TRADE: NO_TRADE}[outcome]
        bayes.update(action, outcome)
        mirror.update(mirrored_action, mirrored_outcome)
    np.testing.assert_allclose(bayes.posterior.reshape(2, 2, 3),
                               mirror.posterior.reshape(2, 2, 3)[::-1, :, ::-1])
    assert bayes.summary()["value"] == pytest.approx(1 - mirror.summary()["value"])
    for action in range(9):
        a = bayes.action_statistics(action)
        b = mirror.action_statistics((2 - action // 3) * 3 + action % 3)
        assert a["as_buy"] == pytest.approx(b["as_sell"])
        assert a["profit_buy"] == pytest.approx(b["profit_sell"])
        assert a["expected_profit"] == pytest.approx(b["expected_profit"])


def test_custom_shift_support_and_reset():
    states = [[0, 0.05, 0.9], [1, 0.05, 0.9]]
    bayes = ExactBayes(regimes=states, prior=[3, 1])
    bayes.update(3, BUY)
    assert bayes.summary()["p"] == pytest.approx(0.9)
    np.testing.assert_allclose(bayes.reset(), [0.75, 0.25])
    assert bayes.summary()["entropy"] == pytest.approx(-0.75 * np.log(0.75) - 0.25 * np.log(0.25))


def test_large_unnormalized_prior_does_not_overflow():
    bayes = ExactBayes(prior=np.full(12, 1e308))
    np.testing.assert_allclose(bayes.posterior, np.full(12, 1 / 12))


@pytest.mark.parametrize("prior", [[0] * 12, [-1] * 12, [1, 1], [np.nan] * 12])
def test_invalid_priors(prior):
    with pytest.raises(ValueError):
        ExactBayes(prior=prior)
