"""Simulator, cash-flow, filtration, and common-random-number checks."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.accounting import dense_reward, execution_deltas
from src.customer_model import (ABSTAIN, BUY, DEFAULT_REGIMES, NO_TRADE, QUOTES,
                                SELL, START, likelihoods, simulate_customer,
                                simulate_customers)
from src.environment import MarketMakingEnv


def test_gym_contract_and_episode_boundaries():
    env = MarketMakingEnv(horizon=3)
    with pytest.raises(RuntimeError):
        env.step(ABSTAIN)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=11)
    assert info == {}
    np.testing.assert_array_equal(obs, [0, 0, 1, 0, 0, 0, 1, 0, 1])
    for t in range(3):
        obs, reward, terminated, truncated, info = env.step(ABSTAIN)
        assert terminated == (t == 2)
        assert truncated is False
        assert reward == 0
        assert env.observation_space.contains(obs)
        assert info["outcome"] == NO_TRADE
        assert ("objective" in info) == terminated
    assert obs[-1] == 0
    with pytest.raises(RuntimeError):
        env.step(ABSTAIN)
    obs, _ = env.reset(seed=11)
    assert env.t == env.inventory == env.cash == 0
    assert obs[3 + START] == 1


@pytest.mark.parametrize("action", range(10))
def test_probabilities_and_independent_customer_frequencies(action):
    probabilities = likelihoods(DEFAULT_REGIMES, action)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)
    n = 60_000
    tape = np.random.default_rng(8137 + action).random((n, 3))
    for regime, expected in zip(DEFAULT_REGIMES, probabilities):
        outcomes = simulate_customers(regime, action, tape)
        actual = np.bincount(outcomes, minlength=3) / n
        tolerance = 5 * np.sqrt(expected * (1 - expected) / n) + 0.0005
        assert np.all(np.abs(actual - expected) <= tolerance), (regime, actual, expected)
        # Check the vectorized frequency engine against the scalar environment engine.
        scalar = [simulate_customer(regime, action, u)[0] for u in tape[:100]]
        np.testing.assert_array_equal(outcomes[:100], scalar)


def test_wider_quotes_reduce_uninformed_execution():
    for center_group in range(3):
        rates = np.stack([likelihoods(DEFAULT_REGIMES, center_group * 3 + width)
                          for width in range(3)])
        assert np.all(np.diff(rates[:, :, BUY], axis=0) < 0)
        assert np.all(np.diff(rates[:, :, SELL], axis=0) < 0)
        assert np.all(np.diff(rates[:, :, NO_TRADE], axis=0) > 0)


@pytest.mark.parametrize("fee", [0.0, 0.017])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_cash_inventory_settlement_and_reward_telescoping(fee, value):
    # Three uninformed customers: buys, sells, then declines the quote.
    env = MarketMakingEnv(horizon=3, inventory_penalty=0.12, fee=fee)
    tape = np.array([[0.9, 0.1, 0.9], [0.9, 0.9, 0.1], [0.9, 0.1, 0.1]])
    env.reset(seed=1, options={"regime": [value, 0.05, 0.5], "tape": tape})
    reward_sum = 0
    # Finish with q=-1 so that terminal settlement is nonzero when V=1.
    for action, expected_outcome in zip([3, ABSTAIN, 3], [BUY, NO_TRADE, NO_TRADE]):
        _, reward, _, _, info = env.step(action)
        assert info["outcome"] == expected_outcome
        reward_sum += reward
    expected_cash = 0.6 - fee
    assert env.cash == pytest.approx(expected_cash)
    assert env.inventory == -1
    assert env.inventory_penalty_total == pytest.approx(0.12)
    assert info["profit"] == pytest.approx(expected_cash - value)
    assert info["objective"] == pytest.approx(expected_cash - value - 0.12)
    assert reward_sum == pytest.approx(info["objective"], abs=1e-14)


def test_both_execution_directions_and_one_fee_per_execution():
    env = MarketMakingEnv(horizon=2, fee=0.01, inventory_penalty=0)
    tape = np.array([[0.9, 0.1, 0.99], [0.9, 0.9, 0.01]])
    env.reset(options={"regime": [1, 0, 0.5], "tape": tape})
    _, buy_reward, _, _, buy_info = env.step(3)
    assert env.cash == pytest.approx(0.59)
    assert env.inventory == -1
    assert buy_info["execution_profit"] == pytest.approx(-0.41)
    _, sell_reward, done, _, sell_info = env.step(3)
    assert done
    assert env.cash == pytest.approx(0.18)
    assert env.inventory == 0
    assert sell_info["execution_profit"] == pytest.approx(0.59)
    assert buy_reward + sell_reward == pytest.approx(0.18)
    assert sell_info["objective"] == pytest.approx(0.18)


def test_random_actions_telescope_for_nonzero_inventory_paths():
    rng = np.random.default_rng(723)
    for seed in range(20):
        env = MarketMakingEnv(horizon=64, inventory_penalty=0.013, fee=0.002)
        env.reset(seed=seed)
        rewards = 0.0
        inventory_squares = 0
        execution_profits = 0.0
        for action in rng.integers(0, 10, 64):
            _, reward, _, _, info = env.step(action)
            rewards += reward
            inventory_squares += env.inventory**2
            execution_profits += info["execution_profit"]
        assert rewards == pytest.approx(env.cash + env.inventory * env.regime[0]
                                        - 0.013 / 64 * inventory_squares, abs=1e-12)
        assert rewards == pytest.approx(info["objective"], abs=1e-12)
        assert execution_profits == pytest.approx(info["profit"], abs=1e-12)


def test_likelihood_and_accounting_mirror_symmetry():
    mirror_outcome = {BUY: SELL, SELL: BUY, NO_TRADE: NO_TRADE}
    mirrored_states = DEFAULT_REGIMES.copy()
    mirrored_states[:, 0] = 1 - mirrored_states[:, 0]
    mirrored_states[:, 2] = 1 - mirrored_states[:, 2]
    for action in range(10):
        mirrored_action = ABSTAIN if action == ABSTAIN else (2 - action // 3) * 3 + action % 3
        probs = likelihoods(DEFAULT_REGIMES, action)
        reflected = likelihoods(mirrored_states, mirrored_action)
        np.testing.assert_allclose(probs, reflected[:, [SELL, BUY, NO_TRADE]], atol=1e-15)
        outcomes = (NO_TRADE,) if action == ABSTAIN else (BUY, SELL, NO_TRADE)
        for outcome in outcomes:
            dc, dq = execution_deltas(action, outcome, fee=0.01)
            mdc, mdq = execution_deltas(mirrored_action, mirror_outcome[outcome], fee=0.01)
            assert mdq == -dq
            assert mdc == pytest.approx(dc + dq)
            for q_before in (-3, 0, 2):
                for value in (0, 1):
                    q = q_before + dq
                    reward = dense_reward(dc, dq, q, 0.001, 64, terminal=True, value=value)
                    mirror_reward = dense_reward(mdc, mdq, -q, 0.001, 64,
                                                 terminal=True, value=1 - value)
                    assert reward == pytest.approx(mirror_reward)


def test_seed_reproducibility_and_action_independent_tape():
    first, second = MarketMakingEnv(), MarketMakingEnv()
    a, _ = first.reset(seed=7731)
    b, _ = second.reset(seed=7731)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.regime, second.regime)
    np.testing.assert_array_equal(first._tape, second._tape)
    for action in np.random.default_rng(994).integers(0, 10, 64):
        a, *rest_a = first.step(action)
        b, *rest_b = second.step(action)
        np.testing.assert_array_equal(a, b)
        assert rest_a == rest_b
    first.reset(seed=82)
    second.reset(seed=82)
    first.step(ABSTAIN)
    second.step(0)
    assert first.t == second.t == 1
    np.testing.assert_array_equal(first._tape, second._tape)
    first.step(3)
    second.step(3)
    assert first._previous_outcome == second._previous_outcome


def test_observations_do_not_access_private_regime_or_diagnostics():
    env = MarketMakingEnv()
    env.reset(seed=17)
    env.step(3)
    expected = env._observation()
    env.regime = object()  # An accidental latent read here would fail or change the observation.
    env.regime_index = object()
    env._tape = object()
    env.cash = object()
    env.inventory_penalty_total = object()
    np.testing.assert_array_equal(env._observation(), expected)


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"horizon": 2.5},
                                    {"fee": -1}, {"fee": np.nan},
                                    {"inventory_penalty": -0.01}])
def test_invalid_environment_configuration(kwargs):
    with pytest.raises(ValueError):
        MarketMakingEnv(**kwargs)


def test_impossible_accounting_and_bad_tapes_are_rejected():
    with pytest.raises(ValueError):
        execution_deltas(ABSTAIN, BUY)
    env = MarketMakingEnv(horizon=2)
    with pytest.raises(ValueError):
        env.reset(options={"tape": np.ones((2, 3))})
    with pytest.raises(ValueError):
        env.reset(options={"regime": [0.5, 0.3, 0.5]})
