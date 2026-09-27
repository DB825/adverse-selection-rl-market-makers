import numpy as np
import pytest
from src.behavior import PolicyBehavior


def test_time_only_policy_is_not_mistaken_for_conditional_behavior():
    a=PolicyBehavior(2)
    a.update(0,np.tile([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],(20,1)))
    a.update(1,np.tile([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],(20,1)))
    s=a.summary()
    assert s["mean_within_time_probability_variance"]==0
    assert s["mean_action_entropy_nats"]==0
    assert s["greedy_action_fractions"][:2]==[.5,.5]


def test_variation_is_computed_across_batches_at_same_time():
    a=PolicyBehavior(1,2)
    a.update(0,[[1.,0.]])
    a.update(0,[[0.,1.]])
    s=a.summary()
    assert s["mean_within_time_probability_variance"]==.5
    assert s["mean_action_entropy_nats"]==0
    b=PolicyBehavior(1,2)
    b.update(0,[[.5,.5],[.5,.5]])
    assert b.summary()["mean_within_time_probability_variance"]==0
    assert b.summary()["mean_action_entropy_nats"]==pytest.approx(np.log(2))


def test_batch_partition_does_not_change_summary():
    rng=np.random.default_rng(7)
    p=rng.dirichlet(np.ones(10),size=25)
    a,b=PolicyBehavior(1),PolicyBehavior(1)
    a.update(0,p)
    b.update(0,p[:11]);b.update(0,p[11:])
    assert a.summary()["mean_within_time_probability_variance"]==pytest.approx(b.summary()["mean_within_time_probability_variance"])


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        PolicyBehavior(1,2).update(0,[[.9,.9]])
