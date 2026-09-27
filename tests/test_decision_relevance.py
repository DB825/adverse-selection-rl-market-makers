"""Independent formula, accounting, and observable-history checks for Gate 1."""

import itertools

import numpy as np
import pytest

from scripts.decision_relevance import (
    ABSTAIN, QUOTES, REGIMES, fixed_policy_expectations, likelihood, search,
)
from src.accounting import execution_deltas
from src.baselines import myopic_action, myopic_scores
from src.bayes_filter import ExactBayes
from src.customer_model import DEFAULT_REGIMES, likelihoods
from src.environment import MarketMakingEnv
from src.wrappers import HistoryWrapper


@pytest.fixture(scope="module")
def constructed_pair():
    return search()


def test_independent_likelihood_enumeration():
    np.testing.assert_allclose(DEFAULT_REGIMES, REGIMES, rtol=0, atol=0)
    for action in range(10):
        expected = [[likelihood(action, outcome, state) for outcome in range(3)] for state in REGIMES]
        np.testing.assert_allclose(likelihoods(DEFAULT_REGIMES, action), expected, rtol=1e-14, atol=1e-15)


def test_constructed_histories_match_exact_joint_filter_and_reference(constructed_pair):
    pair = constructed_pair['pair']
    assert pair[0]['best_action'] != pair[1]['best_action']
    assert abs(pair[0]['value'] - pair[1]['value']) < .02
    assert abs(pair[0]['p'] - pair[1]['p']) < .005
    assert abs(pair[0]['alpha'] - pair[1]['alpha']) > .07
    for row in pair:
        belief = ExactBayes()
        for action, outcome in row['history']:
            belief.update(action, outcome)
        np.testing.assert_allclose(belief.posterior, row['posterior'], atol=1e-14)
        for key in ('value', 'alpha', 'p'):
            assert belief.summary()[key] == pytest.approx(row[key], abs=1e-14)
        for expected in row['action_statistics'][:9]:
            actual = belief.action_statistics(expected['action'])
            for key in ('p_buy', 'p_sell', 'value_buy', 'value_sell', 'as_buy', 'as_sell', 'expected_profit'):
                assert actual[key] == pytest.approx(expected[key], abs=1e-14)
        np.testing.assert_allclose(myopic_scores(belief, row['inventory']),
                                   [x['myopic_score'] for x in row['action_statistics']], atol=1e-14)
        assert myopic_action(belief, row['inventory']) == row['best_action']
        assert row['decision_margin'] > .001
        assert row['minimum_event_likelihood_across_regimes'] > 0
        assert row['minimum_predictive_event_probability'] > .02


def test_pair_is_identical_to_actual_eight_observation_wrapper(constructed_pair):
    observations = []
    histories = [row['history'] for row in constructed_pair['pair']]
    assert [o for _, o in histories[0]] == [o for _, o in histories[1]]
    assert histories[0][-8:] == histories[1][-8:]
    for history in histories:
        # All events can be realized by an uninformed customer under a supported
        # fixed regime. Midpoints lie inside positive-measure event intervals.
        tape = np.full((64, 3), .99)
        for t, (action, outcome) in enumerate(history):
            bid, ask = QUOTES[action]
            if outcome == 0:
                tape[t] = (.99, .25, (1 + ask) / 2)
            elif outcome == 1:
                tape[t] = (.99, .75, bid / 2)
            else:
                tape[t] = (.99, .25, ask / 2)
        wrapped = HistoryWrapper(MarketMakingEnv(), window=8)
        wrapped.reset(options={'regime': (1, .05, .5), 'tape': tape})
        for action, outcome in history:
            obs, _, _, _, info = wrapped.step(action)
            assert info['outcome'] == outcome
        assert wrapped.unwrapped.inventory == -2
        assert wrapped.unwrapped.t == 24
        observations.append(obs)
    np.testing.assert_array_equal(*observations)


def test_fixed_policy_closed_form_against_full_trajectory_enumeration():
    horizon, fee, penalty = 3, .003, .01
    expected = fixed_policy_expectations(horizon, (penalty,), fee)
    for action in range(10):
        objectives, profits, mean_q2s = [], [], []
        for regime in DEFAULT_REGIMES:
            probabilities = likelihoods([regime], action)[0]
            for outcomes in itertools.product(range(3), repeat=horizon):
                weight = np.prod([probabilities[o] for o in outcomes]) / len(REGIMES)
                if weight == 0:
                    continue
                cash, q, sum_q2 = 0., 0, 0.
                for outcome in outcomes:
                    dcash, dq = execution_deltas(action, outcome, fee)
                    cash += dcash
                    q += dq
                    sum_q2 += q * q
                profit = cash + q * regime[0]
                profits.append(weight * profit)
                mean_q2s.append(weight * sum_q2 / horizon)
                objectives.append(weight * (profit - penalty * sum_q2 / horizon))
        assert sum(profits) == pytest.approx(expected[action]['terminal_profit'], abs=1e-13)
        assert sum(mean_q2s) == pytest.approx(expected[action]['mean_inventory_squared'], abs=1e-13)
        assert sum(objectives) == pytest.approx(expected[action]['objective_by_lambda'][str(penalty)], abs=1e-13)


def test_prior_fixed_quote_sanity_and_penalty_scale():
    rows = fixed_policy_expectations()
    assert rows[5]['terminal_profit'] == pytest.approx(.928)
    assert rows[5]['objective_by_lambda']['0.001'] == pytest.approx(.847482875)
    assert rows[ABSTAIN]['objective_by_lambda']['0.001'] == 0
    assert rows[3]['objective_by_lambda']['0.001'] < 0
    assert rows[4]['objective_by_lambda']['0.001'] < 0
