"""Episode-held-out ridge probes; decoding is not evidence of causal policy use."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from .artifacts import write_json
from .evaluate import TARGET_NAMES


def episode_split(episode, seed=73):
    ids=np.unique(episode)
    if len(ids)<15:
        raise ValueError("Use at least 15 whole episodes for three-way probing")
    ids=np.random.default_rng(seed).permutation(ids)
    a,b=int(.6*len(ids)),int(.8*len(ids))
    groups={"train":ids[:a],"validation":ids[a:b],"test":ids[b:]}
    return groups,{name:np.isin(episode,values) for name,values in groups.items()}


def bootstrap_mean_ci(values, seed=19, draws=2000):
    values=np.asarray(values)
    rng=np.random.default_rng(seed)
    boot=values[rng.integers(0,len(values),(draws,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot,[.025,.975],axis=0)]


def run_probes(dataset, output="results/probes", split_seed=73, min_time=8):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with np.load(dataset,allow_pickle=False) as raw:
        keep=raw["t"]>=min_time
        data={key:raw[key][keep] for key in raw.files}
    ids,masks=episode_split(data["episode"],split_seed)
    train,val,test=[masks[k] for k in ("train","validation","test")]
    targets=data["targets"].astype(float)
    y_mean=targets[train].mean(0); y_scale=targets[train].std(0)
    y_scale=np.maximum(y_scale,1e-8)
    y=(targets-y_mean)/y_scale
    state=np.concatenate([data["hidden"],data["cell"]],axis=1)
    features={"basic":data["basic"],"history8":data["history"],
              "hidden_cell":state,"basic_hidden_cell":np.c_[data["basic"],state],
              "history8_hidden_cell":np.c_[data["history"],state]}
    control_path=Path(dataset).with_name("untrained_states.npz")
    has_control=control_path.exists()
    if has_control:
        with np.load(control_path,allow_pickle=False) as control:
            for key in ("episode","t"):
                if not np.array_equal(control[key][keep],data[key]):
                    raise ValueError("Untrained-state control rows must match the activation archive")
            untrained=np.c_[control["untrained_hidden"][keep],control["untrained_cell"][keep]]
        features["untrained_hidden_cell"]=untrained
        features["history8_untrained_hidden_cell"]=np.c_[data["history"],untrained]
    results={}; predictions={};losses={}
    with threadpool_limits(limits=1):
        for name,x in features.items():
            # LSTM coordinates are highly correlated; float64 prevents the
            # avoidable conditioning warnings observed in the float32 smoke fit.
            x=x.astype(np.float64)
            scaler=StandardScaler().fit(x[train])
            xtrain=scaler.transform(x[train]);xval=scaler.transform(x[val]);xtest=scaler.transform(x[test])
            trials=[]
            for alpha in (.01,.1,1.,10.,100.,1000.):
                reg=Ridge(alpha=alpha).fit(xtrain,y[train])
                score=float(np.mean((reg.predict(xval)-y[val])**2))
                trials.append((score,alpha,reg))
            score,alpha,reg=min(trials,key=lambda entry:entry[0])
            # Keep the train-only fit after choosing alpha on validation episodes.
            pred=reg.predict(xtest)*y_scale+y_mean
            predictions[name]=pred
            loss=(pred-targets[test])**2
            losses[name]=loss
            results[name]={"features":x.shape[1],"ridge_alpha":alpha,"validation_scaled_mse":score,
                           "test_r2":dict(zip(TARGET_NAMES,r2_score(targets[test],pred,multioutput="raw_values"))),
                           "test_mse":dict(zip(TARGET_NAMES,loss.mean(0)))}
    test_ids=data["episode"][test]
    comparisons={}
    comparator_names=["basic","history8"]
    if has_control:
        comparator_names.append("history8_untrained_hidden_cell")
    for comparator in comparator_names:
        enhanced=("history8_hidden_cell" if comparator=="history8_untrained_hidden_cell"
                  else comparator+"_hidden_cell")
        delta=losses[comparator]-losses[enhanced]
        per_ep=np.stack([delta[test_ids==eid].mean(0) for eid in ids["test"]])
        lo,hi=bootstrap_mean_ci_matrix(per_ep)
        comparisons[comparator]={target:{"mse_reduction":float(per_ep[:,j].mean()),
                                  "episode_bootstrap_95_ci":[float(lo[j]),float(hi[j])],
                                  "relative_mse_reduction":float(delta[:,j].mean()/max(losses[comparator][:,j].mean(),1e-20))}
                                 for j,target in enumerate(TARGET_NAMES)}
    payload={"dataset":str(dataset),"rows":len(targets),"min_time":min_time,
             "reference_quote":{"action":5,"bid":.2,"ask":.8},
             "untrained_control":str(control_path) if has_control else None,
             "split_seed":split_seed,"episode_splits":ids,"split_counts":{k:len(v) for k,v in ids.items()},
             "results":results,"added_state_vs":comparisons,
             "interpretation":"Held-out predictive association only; not a causal-use test. Intervals resample entire test episodes for a fixed fitted probe and policy, and exclude probe-fitting/training-seed uncertainty."}
    write_json(out/"probes.json",payload)
    np.savez_compressed(out/"predictions.npz",episode=test_ids,truth=targets[test],**predictions)
    return payload


def bootstrap_mean_ci_matrix(values, seed=19, draws=2000):
    rng=np.random.default_rng(seed)
    means=np.stack([values[rng.integers(0,len(values),len(values))].mean(0) for _ in range(draws)])
    return np.quantile(means,[.025,.975],axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",default="results/probes")
    p.add_argument("--split-seed",type=int,default=73)
    p.add_argument("--min-time",type=int,default=8)
    result=run_probes(**vars(p.parse_args()))
    print({k:v["test_r2"] for k,v in result["results"].items()})


if __name__=="__main__":
    main()
