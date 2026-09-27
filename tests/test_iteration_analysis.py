import numpy as np
import pytest

from scripts.analyze_iteration import aggregate_episode_losses, replication_gate


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
