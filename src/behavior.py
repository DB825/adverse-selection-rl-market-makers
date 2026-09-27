"""Descriptive action-distribution checks, without interpreting neural states."""
import numpy as np


class PolicyBehavior:
    def __init__(self, horizon, actions=10):
        self.count=np.zeros(horizon,dtype=int)
        self.prob_sum=np.zeros((horizon,actions),dtype=np.float64)
        self.prob_square_sum=np.zeros_like(self.prob_sum)
        self.entropy_sum=np.zeros(horizon,dtype=np.float64)
        self.argmax_counts=np.zeros(actions,dtype=int)

    def update(self, time, probabilities):
        p=np.asarray(probabilities,dtype=np.float64)
        if p.ndim!=2 or p.shape[1]!=self.prob_sum.shape[1] or not np.isfinite(p).all():
            raise ValueError("Expected finite batch-by-action probabilities")
        if (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6):
            raise ValueError("Action probabilities must be nonnegative and normalized")
        self.count[time]+=len(p)
        self.prob_sum[time]+=p.sum(0)
        self.prob_square_sum[time]+=(p*p).sum(0)
        self.entropy_sum[time]+=(-p*np.log(np.maximum(p,1e-300))).sum()
        self.argmax_counts+=np.bincount(p.argmax(1),minlength=p.shape[1])

    def summary(self):
        valid=self.count>0
        if not valid.any():
            raise ValueError("No action distributions recorded")
        means=self.prob_sum[valid]/self.count[valid,None]
        variance=np.maximum(self.prob_square_sum[valid]/self.count[valid,None]-means**2,0)
        total=int(self.count.sum())
        return {"mean_action_entropy_nats":float(self.entropy_sum.sum()/total),
                "greedy_action_fractions":(self.argmax_counts/total).tolist(),
                "mean_action_probabilities":(self.prob_sum.sum(0)/total).tolist(),
                "mean_within_time_probability_variance":float(variance.sum(1).mean()),
                "per_time_probability_variance":variance.sum(1).tolist(),
                "description":"Descriptive across-episode variance of action probabilities at each fixed time, then averaged over time. Excludes pure time-only variation and action sampling noise. Nonzero variation can reflect inventory/current observations and does not establish recurrent memory, adverse-selection inference, or causal use."}
