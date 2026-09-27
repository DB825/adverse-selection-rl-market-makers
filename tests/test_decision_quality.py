import numpy as np
import pandas as pd
import pytest

from src.bayes_filter import ExactBayes
from src.baselines import myopic_scores
from src.customer_model import DEFAULT_REGIMES, QUOTES, likelihoods
from src.environment import MarketMakingEnv
from scripts.analyze_decisions import score_actions, decompose_gap, conditional_mean, audit_trajectory


def test_vectorized_scores_match_scalar_reference_with_fees_and_inventory():
    rng = np.random.default_rng(418)
    posterior = rng.dirichlet(np.ones(12), size=24)
    posterior[0] = np.eye(12)[3]  # Include a degenerate posterior.
    inventory = rng.integers(-20, 21, len(posterior))
    for penalty, horizon, fee in [(.001, 64, 0.), (.05, 16, .02), (0., 1, .1)]:
        actual = score_actions(posterior, inventory, penalty, horizon, fee)
        expected = [myopic_scores(ExactBayes(prior=w), q, penalty, horizon, fee)
                    for w, q in zip(posterior, inventory)]
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


def test_scores_equal_enumerated_one_step_objective_difference():
    # Enumerate latent states and outcomes, including the common carrying cost.
    w = np.arange(1, 13, dtype=float); w /= w.sum()
    q, penalty, horizon, fee = -7, .02, 64, .003
    expected = []
    for action in range(9):
        value = 0.
        bid, ask = QUOTES[action]
        for weight, state, chances in zip(w, DEFAULT_REGIMES, likelihoods(DEFAULT_REGIMES, action)):
            for chance, cash, dq in zip(chances, [ask-fee, -bid-fee, 0.], [-1, 1, 0]):
                value += weight*chance*(cash+dq*state[0]-penalty/horizon*((q+dq)**2-q**2))
        expected.append(value)
    np.testing.assert_allclose(score_actions(w[None], np.array([q]), penalty, horizon, fee)[0],
                               expected+[0.], atol=1e-14)


def test_gap_decomposition_is_additive_and_symmetric_in_quote_coordinates():
    rng = np.random.default_rng(83)
    scores = np.column_stack((rng.normal(size=(32, 9)), np.zeros(32)))
    p = rng.dirichlet(np.ones(10), size=32)
    gap, parts = decompose_gap(scores, p)
    np.testing.assert_allclose(parts.sum(1), scores.max(1)-(scores*p).sum(1), atol=1e-14)
    assert np.all(parts >= 0)
    transposed_scores = np.column_stack((scores[:, :9].reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9), scores[:, 9]))
    transposed_p = np.column_stack((p[:, :9].reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9), p[:, 9]))
    other_gap, other = decompose_gap(transposed_scores, transposed_p)
    np.testing.assert_allclose(other_gap, gap)
    np.testing.assert_allclose(other[:, [0, 1, 3, 2]], parts)


def test_myopic_actions_have_zero_gap_and_bad_probability_rows_fail():
    scores = np.array([[-1.]*9+[0.], [1.]*9+[0.]])
    gap, parts = decompose_gap(scores, np.eye(10)[scores.argmax(1)])
    np.testing.assert_array_equal(gap, [0., 0.])
    np.testing.assert_array_equal(parts, np.zeros((2, 4)))
    with pytest.raises(ValueError, match='Unnormalized action probabilities'):
        decompose_gap(scores, np.full((2, 10), .11))
    with pytest.raises(ValueError, match='posterior'):
        score_actions(np.full((2, 12), 1/10), np.zeros(2))


def make_trajectories(ids):
    records, economics = [], []
    for episode in ids:
        env, belief = MarketMakingEnv(), ExactBayes()
        obs, _ = env.reset(seed=int(episode))
        for t in range(64):
            action = (t+int(episode)) % 10
            s, ref = belief.summary(), belief.action_statistics(5)
            records.append({'episode': episode, 't': t, 'obs': obs.copy(), 'action': action,
                'posterior': belief.posterior, 'targets': [s['value'], s['alpha'], s['p'], s['entropy'], ref['as_buy'], ref['as_sell']]})
            obs, _, _, _, info = env.step(action)
            belief.update(action, info['outcome'])
        economics.append({'episode': episode, 'value': env.regime[0], 'terminal_inventory': env.inventory,
                          'profit': info['profit'], 'penalty': info['inventory_penalty_total'], 'objective': info['objective']})
    return {k: np.array([r[k] for r in records]) for k in records[0]}, pd.DataFrame(economics).set_index('episode')


def test_audit_replays_public_history_and_rejects_misaligned_posterior(monkeypatch):
    ids = np.array([700, 701, 702])
    monkeypatch.setattr('scripts.analyze_decisions.IDS', ids)
    data, economic = make_trajectories(ids)
    order = np.random.default_rng(13).permutation(len(data['t']))
    shuffled = {k: v[order] for k, v in data.items()}
    audited = audit_trajectory(shuffled, economic)
    np.testing.assert_array_equal(audited['t'], np.tile(np.arange(64), (3, 1)))
    data['posterior'][8] = data['posterior'][7]
    with pytest.raises(ValueError, match='Posterior replay mismatch'):
        audit_trajectory(data, economic)


def test_audit_rejects_inventory_sign_error(monkeypatch):
    ids = np.array([700, 701])
    monkeypatch.setattr('scripts.analyze_decisions.IDS', ids)
    data, economic = make_trajectories(ids)
    data['obs'][:, 7] *= -1
    with pytest.raises(ValueError, match='Inventory/outcome mismatch'):
        audit_trajectory(data, economic)


def test_conditional_mean_weights_decisions_but_resamples_episodes():
    record = conditional_mean(np.array([1., 18.]), np.array([1, 9]))
    assert record['mean'] == 1.9  # Not the unweighted mean of episode means, 1.5.
    assert record['episode_ci'] == [1., 2.]
    assert record['decisions'] == 10 and record['episodes'] == 2
    assert conditional_mean(np.zeros(3), np.zeros(3))['mean'] is None
