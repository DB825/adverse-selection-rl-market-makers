"""Protect the single-factor comparison and its complete-episode interpretation."""
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.buffers import RolloutBuffer
import torch
import yaml

from scripts import analyze_gae as analysis


def configs():
    return [yaml.safe_load(Path(p).read_text()) for p in
            ('configs/followup_budget.yaml','configs/gae_lambda_one.yaml')]


def test_only_gae_changes_and_complete_episodes_fit_rollouts():
    old,new=configs()
    analysis.validate_configs(old,new)
    assert new['training']['n_steps'] % new['environment']['horizon'] == 0


@pytest.mark.parametrize('key,value',[('ent_coef',.01),('gamma',.99),('batch_size',128)])
def test_extra_training_change_rejected(key,value):
    old,new=configs();new['training'][key]=value
    with pytest.raises(ValueError,match='only training change'):
        analysis.validate_configs(old,new)


def test_overlapping_reserved_cohorts_rejected():
    old,new=configs();new['analysis']['seed']=new['evaluation']['seed']
    with pytest.raises(ValueError,match='overlap'):
        analysis.validate_configs(old,new)


def test_lambda_one_equals_whole_episode_returns_with_arbitrary_critic():
    rng=np.random.default_rng(901)
    rewards=rng.normal(size=(128,2)).astype(np.float32)
    values=rng.normal(scale=5,size=(128,2)).astype(np.float32)
    expected=np.concatenate([np.cumsum(r[::-1],axis=0)[::-1] for r in np.split(rewards,2)])
    buffers=[]
    for lam in (1.,.95):
        buffer=RolloutBuffer(128,spaces.Box(-1,1,(9,)),spaces.Discrete(10),gamma=1.,gae_lambda=lam,n_envs=2)
        buffer.rewards[:]=rewards;buffer.values[:]=values
        buffer.episode_starts[[0,64]]=1
        # A huge terminal bootstrap must be masked; neither episode may leak into the other.
        buffer.compute_returns_and_advantage(torch.tensor([1e6,-1e6]),np.ones(2,dtype=bool))
        buffers.append(buffer)
    np.testing.assert_allclose(buffers[0].returns,expected,rtol=1e-5,atol=2e-5)
    np.testing.assert_allclose(buffers[0].advantages,expected-values,rtol=1e-5,atol=2e-5)
    assert not np.allclose(buffers[1].returns,expected)
    np.testing.assert_array_equal(buffers[0].rewards,buffers[1].rewards)


def evaluation_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'EPISODES',2)
    checkpoint=tmp_path/'model.zip';checkpoint.write_bytes(b'fixture')
    folder=tmp_path/'evaluation';folder.mkdir()
    ids=pd.Index([8_000_000,8_000_001],name='episode')
    pd.DataFrame({'episode':ids,'objective':[1.,3.],'profit':[1.,3.],'penalty':[0.,0.]}).to_csv(folder/'episodes.csv',index=False)
    summary=dict(policy='recurrent',checkpoint=str(checkpoint),episodes=2,seed_start=8_000_000,
                 seed_stop_exclusive=8_000_002,objective_mean=2.,behavior={'checked':True},**analysis.EVALUATION_SETTINGS)
    (folder/'summary.json').write_text(json.dumps(summary))
    return folder,checkpoint,ids,summary


@pytest.mark.parametrize('change',[{'checkpoint':'wrong.zip'},{'deterministic':True},
                                  {'objective_mean':99.},{'seed_start':5_000_000}])
def test_wrong_evaluation_identity_or_metrics_rejected(tmp_path,monkeypatch,change):
    folder,checkpoint,ids,summary=evaluation_fixture(tmp_path,monkeypatch)
    analysis.verify_evaluation(folder,checkpoint,'recurrent',ids,dict(analysis.EVALUATION_SETTINGS))
    summary.update(change);(folder/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        analysis.verify_evaluation(folder,checkpoint,'recurrent',ids,dict(analysis.EVALUATION_SETTINGS))


def test_missing_runs_refuse_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):
        analysis.run()
    assert not (tmp_path/'gae_summary.json').exists()


def test_undefined_explained_variance_is_counted_without_hiding_it():
    raw=pd.DataFrame({'train/explained_variance':['','nan','.5'], 'train/loss':['',1.,2.]})
    result=analysis.check_logged_values(raw)
    assert result['undefined_explained_variance_rows']==1
    assert result['other_nonfinite_logged_values']==0


@pytest.mark.parametrize('column,value',[('train/loss','nan'),('train/explained_variance','inf'),
                                        ('economics/objective_mean','-inf')])
def test_nonfinite_losses_or_infinite_diagnostics_still_fail(column,value):
    with pytest.raises(ValueError,match='Unexpected nonfinite'):
        analysis.check_logged_values(pd.DataFrame({column:[value]}))
