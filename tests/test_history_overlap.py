import numpy as np
import pytest

from scripts.analyze_history_overlap import match_history, recent_histories


def test_windows_preserve_actions_inventory_and_time_without_crossing_episodes():
    obs = np.arange(2*12*9).reshape(2, 12, 9)
    history = recent_histories(obs)
    assert history.shape == (2, 3, 72)
    for episode in range(2):
        for j, t in enumerate(range(9, 12)):
            np.testing.assert_array_equal(history[episode, j], obs[episode, t-7:t+1].ravel())
    # First eligible row omits obs[1], which encodes the first action/outcome.
    assert history[0, 0, 0] == obs[0, 2, 0]


def test_peer_loss_matches_explicit_enumeration_with_unequal_group_sizes():
    rng = np.random.default_rng(581)
    history = np.zeros((7, 72)); history[3:5] = 1; history[5] = 2; history[6] = 3
    scores = rng.normal(size=(7, 10))
    risk = rng.normal(size=(7, 2))
    summary, episodes, groups = match_history(history, np.arange(7), scores, risk)
    expected = []
    for ix in ([0, 1, 2], [3, 4]):
        for i in ix:
            expected.append(np.mean([scores[i].max()-scores[i, scores[j].argmax()]
                                     for j in ix if j != i]))
    assert summary['matched_decisions'] == 5
    assert summary['matched_episodes'] == 5
    assert summary['mean_peer_action_gap'] == pytest.approx(np.mean(expected))
    assert episodes.peer_gap_sum.sum() == pytest.approx(sum(expected))
    assert sorted(groups.decisions) == [2, 3]
    permutation = rng.permutation(7)
    shuffled, _, _ = match_history(history[permutation], np.arange(7)[permutation],
                                    scores[permutation], risk[permutation])
    assert shuffled == summary


def test_ties_do_not_create_decisive_conflicts_and_self_pairs_are_excluded():
    scores = np.zeros((3, 10))
    scores[0, 0] = 1.; scores[1, 1] = 2.
    # Third row's numerical near tie must not become a decisive choice.
    scores[2, 2] = 1e-12
    summary, episodes, _ = match_history(np.zeros((3, 72)), np.arange(3), scores, np.zeros((3, 2)))
    assert summary['conflicting_decisions'] == 2
    assert summary['decisive_matched_decisions'] == 2
    assert summary['conflict_fraction_matched'] == 2/3
    np.testing.assert_allclose(episodes.peer_gap_sum, [1., 2., 1e-12], atol=1e-15)
    scores[:] = 0.
    summary, _, _ = match_history(np.zeros((3, 72)), np.arange(3), scores, np.zeros((3, 2)))
    assert summary['conflicting_decisions'] == 0
    assert summary['mean_peer_action_gap'] == 0.


def test_no_overlap_is_missing_support_not_zero_conditional_effect():
    # A difference in any input must prevent a match, including tiny differences.
    history = np.zeros((4, 72))
    history[1, 0] = .2  # Oldest retained quote.
    history[2, -2] = 1/64  # Inventory.
    history[3, -1] = 1e-15  # Exact time, not approximate binning.
    summary, episodes, groups = match_history(history, np.arange(4), np.zeros((4, 10)), np.zeros((4, 2)))
    assert summary['matched_decisions'] == 0 and summary['matched_groups'] == 0
    assert summary['mean_peer_action_gap'] is None
    assert summary['conflict_fraction_matched'] is None
    assert episodes.matched.sum() == 0 and groups.empty


def test_repeated_episode_and_nonfinite_inputs_fail_closed():
    history, scores, risk = np.zeros((2, 72)), np.zeros((2, 10)), np.zeros((2, 2))
    with pytest.raises(ValueError, match='Repeated episode'):
        match_history(history, np.zeros(2), scores, risk)
    history[0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        match_history(history, np.arange(2), scores, risk)
