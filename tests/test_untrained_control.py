"""Same-trajectory random-memory control, including shuffled archive rows."""
import hashlib

import numpy as np
import pytest

from src.interventions import RecurrentActorAdapter
from src.train import make_model
from src.untrained_control import collect_untrained_states, encode_observations


TEST_CONFIG = {
    "environment": {"horizon": 8},
    "training": {"n_envs": 1, "n_steps": 8, "batch_size": 8, "torch_threads": 1,
                 "fused_lstm_reset": True, "gamma": 1.0, "device": "cpu"},
}


def recorded_rows():
    rng = np.random.default_rng(43)
    # Nonconsecutive episode IDs and different lengths exercise slot mapping.
    episode = np.array([31] * 4 + [97] * 3)
    times = np.array([0, 1, 2, 3, 0, 1, 2])
    obs = rng.uniform(0, 1, (7, 9)).astype(np.float32)
    obs[:, 7] -= .5
    return episode, times, obs


def test_shuffled_rows_preserve_temporal_processing_and_original_row_alignment():
    episode, times, obs = recorded_rows()
    model = make_model(TEST_CONFIG, "recurrent", seed=11)
    try:
        adapter = RecurrentActorAdapter(model)
        expected_hidden, expected_cell = encode_observations(episode, times, obs, adapter)
        permutation = np.array([6, 2, 4, 0, 5, 3, 1])
        hidden, cell = encode_observations(episode[permutation], times[permutation], obs[permutation], adapter)
        np.testing.assert_allclose(hidden, expected_hidden[permutation], atol=1e-7)
        np.testing.assert_allclose(cell, expected_cell[permutation], atol=1e-7)
        assert model.num_timesteps == 0
    finally:
        model.get_env().close()


def test_batched_episodes_equal_independent_reset_replay():
    episode, times, obs = recorded_rows()
    model = make_model(TEST_CONFIG, "recurrent", seed=12)
    try:
        adapter = RecurrentActorAdapter(model)
        hidden, cell = encode_observations(episode, times, obs, adapter)
        for eid in np.unique(episode):
            rows = np.flatnonzero(episode == eid)
            single_hidden, single_cell = encode_observations(episode[rows], times[rows], obs[rows], adapter)
            np.testing.assert_allclose(hidden[rows], single_hidden, atol=1e-7)
            np.testing.assert_allclose(cell[rows], single_cell, atol=1e-7)
            state = None
            for row in rows:
                step = adapter.process(obs[row], state=state, episode_start=times[row] == 0)
                state = step.state
                np.testing.assert_allclose(hidden[row], step.hidden, atol=1e-7)
                np.testing.assert_allclose(cell[row], step.cell, atol=1e-7)
    finally:
        model.get_env().close()


def test_rejects_missing_duplicate_or_noninitial_times():
    episode, times, obs = recorded_rows()
    model = make_model(TEST_CONFIG, "recurrent", seed=11)
    try:
        adapter = RecurrentActorAdapter(model)
        for malformed in (np.array([0, 1, 1, 3, 0, 1, 2]), times + 1):
            with pytest.raises(ValueError, match="exactly one row"):
                encode_observations(episode, malformed, obs, adapter)
    finally:
        model.get_env().close()


def test_control_artifact_keeps_source_unchanged_and_contains_alignment_keys(tmp_path):
    episode, times, obs = recorded_rows()
    source = tmp_path / "activations.npz"
    output = tmp_path / "untrained_states.npz"
    # Object diagnostic arrays would raise under allow_pickle=False if loaded;
    # the encoder must select only the three public/indexing arrays.
    np.savez_compressed(source, episode=episode, t=times, obs=obs,
                        diagnostic=np.array([{"private": True}], dtype=object))
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = collect_untrained_states(source, TEST_CONFIG, seed=11, output=output)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert metadata['training_timesteps'] == 0
    assert metadata['dataset_sha256'] == original_hash
    assert metadata['source_row_order_preserved']
    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == {'episode', 't', 'untrained_hidden', 'untrained_cell'}
        np.testing.assert_array_equal(data['episode'], episode)
        np.testing.assert_array_equal(data['t'], times)
        assert data['untrained_hidden'].shape == (7, 64)
        assert data['untrained_cell'].shape == (7, 64)
    with pytest.raises(ValueError, match="must differ"):
        collect_untrained_states(source, TEST_CONFIG, seed=11, output=source)
