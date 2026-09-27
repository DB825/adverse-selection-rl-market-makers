"""Paired episode evaluation. Customer and action randomness are separate streams."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from .artifacts import write_json
from .bayes_filter import ExactBayes
from .behavior import PolicyBehavior
from .customer_model import ABSTAIN, DEFAULT_REGIMES, QUOTES
from .environment import MarketMakingEnv


SHIFT_REGIMES = np.array([[0., .05, .9], [1., .05, .9]])
SHIFT_CONTROL_REGIMES = np.array([[0., .05, .75], [1., .05, .75]])
TARGET_NAMES = ["value", "alpha", "p", "entropy", "as_buy_ref", "as_sell_ref"]


def myopic_batch(filters, inventories, penalty, horizon, fee):
    scores = []
    for belief, q in zip(filters, inventories):
        row = []
        for a in range(10):
            s = belief.action_statistics(a, fee=fee)
            incremental = s["p_buy"] * (-2*q + 1) + s["p_sell"] * (2*q + 1)
            row.append(s["expected_profit"] - penalty / horizon * incremental)
        scores.append(row)
    return np.argmax(scores, axis=1)


def evaluate(policy="myopic", checkpoint=None, episodes=512, seed_start=1_000_000,
             output="results/evaluation", deterministic=False, shift=False,
             correct_filter=False, collect=False, batch_size=128,
             inventory_penalty=.001, horizon=64, fee=0., shift_control=False):
    """Evaluate frozen policy, never select hyperparameters here.

    Common seeds regenerate the same hidden regime and full customer tape for
    every policy. Quotes still determine distinct outcomes. Stochastic learned
    policies use one independent uniform per episode/arrival, common across
    checkpoints. Fixed/myopic policies are deterministic. Whole episodes, not
    transitions, are statistical sampling units.
    """
    torch.set_num_threads(1)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    recurrent = policy == "recurrent"
    model = None
    if checkpoint:
        model = (RecurrentPPO if recurrent else PPO).load(checkpoint, device="cpu")
        model.policy.set_training_mode(False)
    elif policy not in {"myopic", "fixed", "random"} and not policy.startswith("fixed_"):
        raise ValueError("Learned policies require a checkpoint")
    if collect and not recurrent:
        raise ValueError("Activation collection requires a recurrent checkpoint")
    if recurrent:
        from .interventions import RecurrentActorAdapter
        adapter = RecurrentActorAdapter(model)
    if shift and shift_control:
        raise ValueError("Choose either the shift or its matched control")
    is_shift = shift or shift_control
    regimes = SHIFT_REGIMES if shift else (SHIFT_CONTROL_REGIMES if shift_control else DEFAULT_REGIMES)
    belief_support = regimes if correct_filter else DEFAULT_REGIMES
    if correct_filter and model is not None:
        raise ValueError("Correct shift support is a reference-filter comparison only")
    history_window = (int(model.observation_space.shape[0]) // 9
                      if policy == "history" and model is not None else 8)
    episode_rows, activation = [], {}
    all_actions = np.zeros(10, dtype=np.int64)
    behavior=PolicyBehavior(horizon)
    def save_array(name, value):
        activation.setdefault(name, []).append(np.asarray(value).copy())
    for offset in range(0, episodes, batch_size):
        n = min(batch_size, episodes-offset)
        ids = np.arange(seed_start+offset, seed_start+offset+n)
        envs = [MarketMakingEnv(horizon=horizon, inventory_penalty=inventory_penalty,
                               fee=fee, regimes=regimes) for _ in range(n)]
        obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, ids)])
        beliefs = [ExactBayes(regimes=belief_support) for _ in envs]
        correct = [ExactBayes(regimes=regimes) for _ in envs] if is_shift else beliefs
        uniforms = np.stack([np.random.default_rng(int(s)+40_000_000).random(horizon) for s in ids])
        history = np.zeros((n,history_window,9), np.float32)
        history[:,-1] = obs
        state = None
        totals = {k: np.zeros(n) for k in ["reward", "q2", "abs_q", "buy", "sell", "abstain",
                  "informed_profit", "uninformed_profit", "informed_exec", "uninformed_exec",
                  "spread", "center", "active"]}
        action_counts = np.zeros((n,10), dtype=int)
        for t in range(horizon):
            probs = None
            if recurrent:
                step = adapter.process(obs, state=state, episode_start=np.full(n,t == 0))
                probs = step.probabilities
                state = step.state
            elif model is not None:
                features = obs
                if policy == "history":
                    features = history.reshape(n,-1)
                elif policy == "belief":
                    features = np.stack([np.r_[b.posterior, o[7:9]] for b,o in zip(beliefs,obs)]).astype(np.float32)
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(features)
                    probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
            if probs is not None:
                actions = (probs.argmax(1) if deterministic else
                           (uniforms[:,t,None] > np.cumsum(probs,axis=1)).sum(1).clip(0,9))
            elif policy == "myopic":
                actions = myopic_batch(beliefs,[e.inventory for e in envs], inventory_penalty,horizon,fee)
            elif policy == "random":
                actions = (uniforms[:,t]*10).astype(int)
            else:
                actions = np.full(n, int(policy.split("_")[-1]) if "_" in policy else 5)
            diagnostic_probs=(probs if probs is not None else
                              np.full((n,10),.1) if policy=="random" else np.eye(10)[actions])
            behavior.update(t,diagnostic_probs)
            if collect:
                sums = [b.summary() for b in beliefs]
                refs = [b.action_statistics(5,fee=fee) for b in beliefs]  # (.2,.8) for every row
                targets = [[s["value"],s["alpha"],s["p"],s["entropy"],r["as_buy"],r["as_sell"]]
                           for s,r in zip(sums,refs)]
                # Valid slots determined by START or outcomes; zero padding never counts as no trade.
                valid = history[:,:,3:7].sum(2) > 0
                count = np.maximum(valid.sum(1),1)
                active = valid & (history[:,:,2] == 0)
                rejection_rate = (history[:,:,5]*active).sum(1)/np.maximum(active.sum(1),1)
                basic = np.column_stack([(history[:,:,3]-history[:,:,4]).sum(1)/count,
                                         rejection_rate, (history[:,:,2]*valid).sum(1)/count,
                                         obs[:,:3],obs[:,7:9]])
                for key,val in dict(episode=ids,t=np.full(n,t),hidden=step.hidden,cell=step.cell,
                                    obs=obs,history=history.reshape(n,-1),basic=basic,action=actions,
                                    probabilities=probs,posterior=np.stack([b.posterior for b in beliefs]),
                                    targets=targets,regime=np.stack([e.regime for e in envs])).items():
                    save_array(key,val)
            next_obs=[]
            for j,(env,b,a) in enumerate(zip(envs,beliefs,actions)):
                o,r,done,trunc,info = env.step(int(a))
                b.update(int(a),info["outcome"])
                if is_shift:
                    correct[j].update(int(a),info["outcome"])
                next_obs.append(o)
                totals["reward"][j] += r
                totals["q2"][j] += env.inventory**2
                totals["abs_q"][j] += abs(env.inventory)
                totals["buy"][j] += info["outcome"] == 0
                totals["sell"][j] += info["outcome"] == 1
                totals["abstain"][j] += a == ABSTAIN
                action_counts[j,a] += 1
                if a != ABSTAIN:
                    bid,ask=QUOTES[a]
                    totals["spread"][j] += ask-bid
                    totals["center"][j] += (ask+bid)/2
                    totals["active"][j] += 1
                if info["outcome"] != 2:
                    label = "informed" if info["customer_informed"] else "uninformed"
                    totals[label+"_profit"][j] += info["execution_profit"]
                    totals[label+"_exec"][j] += 1
                if done:
                    if trunc or not np.isclose(totals["reward"][j],info["objective"],atol=1e-9):
                        raise AssertionError("Evaluation accounting mismatch")
                    s=b.summary(); c=correct[j].summary()
                    row=dict(episode=int(ids[j]),value=env.regime[0],alpha=env.regime[1],p=env.regime[2],
                             objective=info["objective"],profit=info["profit"],
                             penalty=info["inventory_penalty_total"],terminal_inventory=env.inventory,
                             mean_q2=totals["q2"][j]/horizon,mean_abs_inventory=totals["abs_q"][j]/horizon,
                             buy_rate=totals["buy"][j]/horizon,sell_rate=totals["sell"][j]/horizon,
                             abstain_rate=totals["abstain"][j]/horizon,
                             mean_spread=totals["spread"][j]/max(totals["active"][j],1),
                             mean_center=totals["center"][j]/max(totals["active"][j],1),
                             inferred_value=s["value"],inferred_alpha=s["alpha"],
                             correct_value=c["value"],correct_alpha=c["alpha"])
                    row.update({k:totals[k][j] for k in ["informed_profit","uninformed_profit","informed_exec","uninformed_exec"]})
                    row.update({f"action_{a}":int(v) for a,v in enumerate(action_counts[j])})
                    episode_rows.append(row)
            obs=np.stack(next_obs)
            history=np.roll(history,-1,axis=1); history[:,-1]=obs
        all_actions += action_counts.sum(0)
    frame=pd.DataFrame(episode_rows)
    frame.to_csv(out/"episodes.csv",index=False)
    quantile=float(frame.objective.quantile(.05))
    metrics={"policy":policy,"checkpoint":str(checkpoint) if checkpoint else None,
             "episodes":episodes,"seed_start":seed_start,"seed_stop_exclusive":seed_start+episodes,
             "deterministic":deterministic,"shift":shift,"shift_control":shift_control,"correct_filter":correct_filter,
             "horizon":horizon,"inventory_penalty":inventory_penalty,"fee":fee,
             "objective_mean":float(frame.objective.mean()),"objective_episode_se":float(frame.objective.std(ddof=1)/np.sqrt(episodes)),
             "objective_p05":quantile,"objective_lower5_mean":float(frame.loc[frame.objective<=quantile,"objective"].mean()),
             "profit_mean":float(frame.profit.mean()),"inventory_penalty_mean":float(frame.penalty.mean()),
             "action_fractions":all_actions/(episodes*horizon),
             "elapsed_seconds":time.perf_counter()-started}
    metrics["behavior"]=behavior.summary()
    for key in ["mean_abs_inventory","mean_q2","buy_rate","sell_rate","abstain_rate","mean_spread","mean_center"]:
        metrics[key]=float(frame[key].mean())
    for side in ["informed","uninformed"]:
        count=frame[side+"_exec"].sum()
        metrics[side+"_profit_per_execution"]=float(frame[side+"_profit"].sum()/count) if count else None
    metrics["regimes"]=frame.groupby(["value","alpha","p"])[["objective","profit","abstain_rate","mean_abs_inventory"]].mean().reset_index().to_dict("records")
    write_json(out/"summary.json",metrics)
    if collect:
        arrays={k:np.concatenate(v,axis=0) for k,v in activation.items()}
        np.savez_compressed(out/"activations.npz",**arrays)
        write_json(out/"activation_schema.json",dict(target_names=TARGET_NAMES,reference_quote_action=5,
                    timing="after current observation, before actor MLP/action head",shapes={k:list(v.shape) for k,v in arrays.items()},
                    diagnostic_only=["regime"],format="NumPy compressed NPZ, no pickled objects"))
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",default="myopic")
    p.add_argument("--checkpoint")
    p.add_argument("--episodes",type=int,default=512)
    p.add_argument("--seed-start",type=int,default=1_000_000)
    p.add_argument("--output",default="results/evaluation")
    p.add_argument("--deterministic",action="store_true")
    p.add_argument("--shift",action="store_true")
    p.add_argument("--shift-control",action="store_true")
    p.add_argument("--correct-filter",action="store_true")
    p.add_argument("--collect",action="store_true")
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--inventory-penalty",type=float,default=.001)
    p.add_argument("--horizon",type=int,default=64)
    p.add_argument("--fee",type=float,default=0.)
    args=vars(p.parse_args())
    result=evaluate(**args)
    print({k:result[k] for k in ["policy","objective_mean","objective_episode_se","elapsed_seconds"]},flush=True)


if __name__ == "__main__":
    main()
