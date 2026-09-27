import copy
from pathlib import Path
import shutil

import pytest

from src.managed_runs import managed_train, managed_evaluate
from src.run_store import verify_run
from src.train import load_config


@pytest.fixture
def fitted(tmp_path):
    cfg = load_config('configs/smoke.yaml')
    cfg['training'].update(total_timesteps=128, n_envs=2, n_steps=64, n_epochs=1, fused_lstm_reset=True)
    path = tmp_path / 'train'
    result = managed_train(cfg, 'recurrent', 31, path)
    return cfg, path, result


def test_real_train_reuse_move_and_checkpoint_provenance(fitted, tmp_path):
    cfg, path, result = fitted
    assert result['actual_steps'] == 128 and result['checkpoint'] == 'model.zip'
    assert managed_train(cfg, 'recurrent', 31, path) == result
    relocated = tmp_path / 'relocated'
    shutil.move(path, relocated)
    assert verify_run(relocated)['run_id'] == result['run_id']
    evaluation = managed_evaluate('recurrent', relocated / 'model.zip', tmp_path / 'eval', episodes=4, seed_start=19_000_000)
    assert evaluation['checkpoint_sha256'] == result['checkpoint_sha256']
    assert evaluation['training_run_id'] == result['run_id']
    assert managed_evaluate('recurrent', relocated / 'model.zip', tmp_path / 'eval', episodes=4, seed_start=19_000_000) == evaluation
    with pytest.raises(ValueError, match='different inputs'):
        managed_evaluate('recurrent', relocated / 'model.zip', tmp_path / 'eval', episodes=4, seed_start=19_000_100)


def test_training_configuration_and_evaluation_environment_mismatch(fitted, tmp_path):
    cfg, path, _ = fitted
    other = copy.deepcopy(cfg)
    other['training']['ent_coef'] = .01
    with pytest.raises(ValueError, match='different inputs'):
        managed_train(other, 'recurrent', 31, path)
    with pytest.raises(ValueError, match='horizon differs'):
        managed_evaluate('recurrent', path / 'model.zip', tmp_path / 'eval', episodes=2, horizon=32)


def test_modified_checkpoint_cannot_reuse_evaluation(fitted, tmp_path):
    _, path, _ = fitted
    managed_evaluate('recurrent', path / 'model.zip', tmp_path / 'eval', episodes=2)
    with (path / 'model.zip').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        managed_evaluate('recurrent', path / 'model.zip', tmp_path / 'eval', episodes=2)


def test_managed_persistence_does_not_change_training_weights(fitted, tmp_path):
    from sb3_contrib import RecurrentPPO
    from src.train import train
    from scripts.audit_followup import parameter_digest
    cfg, path, _ = fitted
    direct, _ = train(cfg, 'recurrent', 31, tmp_path / 'direct')
    direct.logger.close()
    managed = RecurrentPPO.load(path / 'model.zip',device='cpu')
    assert parameter_digest(direct) == parameter_digest(managed)
