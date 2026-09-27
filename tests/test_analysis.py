import numpy as np
import pytest

from src.probes import episode_split, bootstrap_mean_ci_matrix
from src.evaluate import evaluate, myopic_batch
from src.bayes_filter import ExactBayes
from src.baselines import myopic_action


def test_episode_split_is_disjoint_complete_and_reproducible():
    episode=np.repeat(np.arange(30),64)
    groups,masks=episode_split(episode)
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["validation"]) & set(groups["test"]))
    assert not (set(groups["train"]) & set(groups["validation"]))
    assert np.all(sum(m.astype(int) for m in masks.values())==1)
    for key, mask in masks.items():
        for eid in np.unique(episode):
            assert len(np.unique(mask[episode==eid]))==1
        np.testing.assert_array_equal(groups[key],episode_split(episode)[0][key])


def test_myopic_evaluation_uses_same_reference():
    b=ExactBayes()
    for a,o in [(5,0),(8,2),(2,1),(5,0)]:
        b.update(a,o)
    for q in [-20,0,20]:
        assert myopic_batch([b],[q],.001,64,0)[0]==myopic_action(b,q)


def test_paired_evaluation_reproduces_and_abstains(tmp_path):
    import pandas as pd
    a=evaluate("fixed_9",episodes=16,output=tmp_path/"a")
    b=evaluate("fixed_9",episodes=16,output=tmp_path/"b")
    assert a["objective_mean"]==b["objective_mean"]==0
    assert a["abstain_rate"]==1
    x=pd.read_csv(tmp_path/"a"/"episodes.csv")
    y=pd.read_csv(tmp_path/"b"/"episodes.csv")
    pd.testing.assert_frame_equal(x,y)
    evaluate("fixed_5",episodes=16,output=tmp_path/"c")
    z=pd.read_csv(tmp_path/"c"/"episodes.csv")
    pd.testing.assert_frame_equal(x[["episode","value","alpha","p"]],z[["episode","value","alpha","p"]])


def test_shift_correct_support_has_shift_alpha(tmp_path):
    a=evaluate("fixed_5",episodes=16,output=tmp_path,shift=True)
    import pandas as pd
    rows=pd.read_csv(tmp_path/"episodes.csv")
    assert np.allclose(rows.correct_alpha,.05)
    assert set(rows.p)=={.9}
    assert (rows.inferred_alpha>.05).any()


def test_cluster_bootstrap_constant_difference():
    lo,hi=bootstrap_mean_ci_matrix(np.ones((30,6)))
    np.testing.assert_allclose(lo,1)
    np.testing.assert_allclose(hi,1)


def test_shift_control_changes_only_p_under_paired_randomness(tmp_path):
    from src.environment import MarketMakingEnv
    from src.evaluate import SHIFT_REGIMES, SHIFT_CONTROL_REGIMES
    a=MarketMakingEnv(regimes=SHIFT_REGIMES)
    b=MarketMakingEnv(regimes=SHIFT_CONTROL_REGIMES)
    for seed in range(20):
        a.reset(seed=seed);b.reset(seed=seed)
        np.testing.assert_array_equal(a.regime[:2],b.regime[:2])
        np.testing.assert_array_equal(a._tape,b._tape)
        assert a.regime[2]==.9 and b.regime[2]==.75
