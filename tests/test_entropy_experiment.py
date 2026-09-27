"""Keep the entropy treatment a single-factor, fresh-cohort comparison."""
from pathlib import Path
import json

import pandas as pd
import pytest
import yaml

from scripts import analyze_entropy as analysis


def configs():
    return [yaml.safe_load(Path(p).read_text()) for p in
            ('configs/followup_budget.yaml','configs/entropy_001.yaml')]


def test_single_factor_and_unused_cohorts():
    old,new=configs();analysis.validate_configs(old,new)
    assert old['training']['gae_lambda']==new['training']['gae_lambda']==.95
    assert new['development']['seed']==10_000_000
    assert new['analysis']['seed']==12_000_000


@pytest.mark.parametrize('key,value',[('gae_lambda',1.),('gamma',.99),('batch_size',128)])
def test_extra_training_change_fails(key,value):
    old,new=configs();new['training'][key]=value
    with pytest.raises(ValueError,match='only training change'):
        analysis.validate_configs(old,new)


def test_wrong_entropy_coefficient_fails():
    old,new=configs();new['training']['ent_coef']=.001
    with pytest.raises(ValueError,match='Wrong entropy contrast'):
        analysis.validate_configs(old,new)


def test_previous_test_cohort_cannot_be_reused():
    old,new=configs();new['evaluation']['seed']=8_000_000
    with pytest.raises(ValueError,match='Wrong economic cohort'):
        analysis.validate_configs(old,new)


def test_evaluation_checkpoint_identity_and_objective(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'EPISODES',2)
    checkpoint=tmp_path/'model.zip';checkpoint.write_bytes(b'fixture')
    folder=tmp_path/'evaluation';folder.mkdir()
    ids=pd.Index([11_000_000,11_000_001],name='episode')
    pd.DataFrame({'episode':ids,'objective':[1.,3.],'profit':[1.,3.],'penalty':[0.,0.]}).to_csv(folder/'episodes.csv',index=False)
    s=dict(policy='recurrent',checkpoint=str(checkpoint),episodes=2,seed_start=11_000_000,
           seed_stop_exclusive=11_000_002,objective_mean=2.,behavior={'checked':True},**analysis.EVALUATION_SETTINGS)
    (folder/'summary.json').write_text(json.dumps(s))
    analysis.verify_evaluation(folder,checkpoint,'recurrent',ids,dict(analysis.EVALUATION_SETTINGS))
    s['checkpoint']='wrong.zip';(folder/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Wrong evaluated checkpoint'):
        analysis.verify_evaluation(folder,checkpoint,'recurrent',ids,dict(analysis.EVALUATION_SETTINGS))


def test_incomplete_experiment_cannot_write_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):analysis.run()
    assert not (tmp_path/'entropy_summary.json').exists()
