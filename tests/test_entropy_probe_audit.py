"""Validate episode-level loss grouping and reject contaminated probe inputs."""
import numpy as np
import pytest

from scripts.analyze_entropy_probes import episode_losses


def fixture():
    # Interleaved rows exercise grouping rather than assuming episode blocks.
    episode = np.tile([12_000_000, 12_000_001], 56)
    truth = np.zeros((112, 6))
    prediction = np.tile(np.repeat([[1.], [3.]], 6, axis=1), (56, 1))
    return episode, truth, prediction, np.array([12_000_000, 12_000_001])


def test_loss_uses_whole_episodes_and_all_targets():
    np.testing.assert_array_equal(episode_losses(*fixture()), np.array([[1.] * 6, [9.] * 6]))


def test_wrong_probe_cohort_rejected():
    episode, truth, prediction, expected = fixture()
    with pytest.raises(ValueError, match='episode IDs'):
        episode_losses(episode, truth, prediction, expected + 1)


def test_missing_decision_rejected():
    episode, truth, prediction, expected = fixture()
    with pytest.raises(ValueError, match='56 decisions'):
        episode_losses(episode[:-1], truth[:-1], prediction[:-1], expected)


def test_nonfinite_probe_predictions_rejected():
    episode, truth, prediction, expected = fixture()
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        episode_losses(episode, truth, prediction, expected)
