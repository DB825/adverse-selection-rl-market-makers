import numpy as np
import pytest

from scripts.analyze_iteration import (aggregate_episode_losses, replication_gate,
                                       validate_trajectories, validate_probe_metrics, validate_probe_design, TARGETS)


def statistic(mean, lower):
    return {'mean': mean, 'training_seed': {'mean_t95_ci': [lower, mean+1]}}


def test_replication_gate_requires_seed_uncertainty_and_useful_behavior():
    assert replication_gate(statistic(.5,.1),statistic(.6,.2),[21,22,23])
    assert not replication_gate(statistic(.5,-.1),statistic(.6,.2),[21,22,23])
    assert not replication_gate(statistic(.1,.01),statistic(.6,.2),[21,22,23])
    assert not replication_gate(statistic(.5,.1),statistic(.6,-.2),[21,22,23])
    assert not replication_gate(statistic(.5,.1),statistic(.6,.2),[21,22])


def test_probe_loss_preserves_episode_grouping():
    ids = np.array([14_000_000,14_000_001])
    episode = np.tile(ids,56)
    truth = np.zeros((112,6))
    prediction = np.tile([[1.]*6,[2.]*6],(56,1))
    np.testing.assert_array_equal(aggregate_episode_losses(episode,truth,prediction,ids),[[1.]*6,[4.]*6])
    with pytest.raises(ValueError,match='episode IDs'):
        aggregate_episode_losses(episode,truth,prediction,ids+1)


def test_grouped_loss_matches_reference_on_shuffled_real_valued_rows():
    rng = np.random.default_rng(81)
    ids = np.array([13, 27, 91])
    episode = rng.permutation(np.repeat(ids, 56))
    truth, prediction = rng.normal(size=(2, len(episode), 6))
    expected = np.stack([((truth[episode == e]-prediction[episode == e])**2).mean(0) for e in ids])
    np.testing.assert_array_equal(aggregate_episode_losses(episode, truth, prediction, ids), expected)
    with pytest.raises(ValueError, match='Missing probe decisions'):
        aggregate_episode_losses(episode[:-1], truth[:-1], prediction[:-1], ids)


def test_trajectory_check_rejects_duplicate_timestep_even_with_correct_counts():
    ids = np.array([3, 9])
    episode = np.tile(ids, 64)
    time = np.repeat(np.arange(64), 2)
    validate_trajectories(episode[::-1], time[::-1], ids)
    time[0] = 1  # Same row count and IDs, but time zero is missing for episode 3.
    with pytest.raises(ValueError, match='Incomplete activation trajectories'):
        validate_trajectories(episode, time, ids)


def test_probe_metrics_reject_stale_r2_with_valid_mse():
    from sklearn.metrics import r2_score, mean_squared_error
    rng = np.random.default_rng(32)
    truth = rng.normal(size=(112, 6))
    prediction = truth + rng.normal(scale=.2, size=truth.shape)
    record = {'test_r2': dict(zip(TARGETS, r2_score(truth, prediction, multioutput='raw_values'))),
              'test_mse': dict(zip(TARGETS, mean_squared_error(truth, prediction, multioutput='raw_values')))}
    validate_probe_metrics(truth, prediction, record)
    record['test_r2'][TARGETS[0]] += .01
    with pytest.raises(ValueError, match='Probe R2 mismatch'):
        validate_probe_metrics(truth, prediction, record)


def test_probe_design_rejects_omitted_controls_and_training_split_changes():
    import copy
    import json
    from pathlib import Path
    # Use the archived design: valid metrics or hashes alone cannot establish
    # that a comparator used the required fitting split and control panels.
    folder = Path('published-results/iteration-v1/nonlinear/seed_11')
    ridge = json.loads((folder/'probes.json').read_text(encoding='utf-8'))
    nonlinear = json.loads((folder/'nonlinear.json').read_text(encoding='utf-8'))
    from src.probes import episode_split
    splits, _ = episode_split(np.arange(14_000_000, 14_001_024))
    validate_probe_design(ridge, nonlinear, splits)
    omitted = copy.deepcopy(ridge)
    del omitted['results']['basic']
    with pytest.raises(ValueError, match='ridge controls'):
        validate_probe_design(omitted, nonlinear, splits)
    changed = copy.deepcopy(ridge)
    a, b = changed['episode_splits']['train'], changed['episode_splits']['validation']
    a[0], b[0] = b[0], a[0]  # Test rows and all split sizes remain unchanged.
    with pytest.raises(ValueError, match='Changed episode split'):
        validate_probe_design(changed, nonlinear, splits)
    changed = copy.deepcopy(ridge)
    changed['reference_quote']['bid'] = .3
    with pytest.raises(ValueError, match='reference quote'):
        validate_probe_design(changed, nonlinear, splits)
