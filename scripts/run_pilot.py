"""Bounded local runner. Stages remain separate so scientific gates stay explicit."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time


def train_one(job):
    policy,seed,config,root=job
    folder=Path(root)/policy/f"seed_{seed}"
    if (folder/"metadata.json").exists() and (folder/"model.zip").exists():
        return json.loads((folder/"metadata.json").read_text())
    from src.train import train
    _,metadata=train(config,policy,seed,folder)
    return metadata


def evaluate_one(job):
    policy,seed,root,output,episodes,seed_start,shift,control=job
    from src.evaluate import evaluate
    folder=Path(output)/policy/f"seed_{seed}"
    checkpoint=Path(root)/policy/f"seed_{seed}"/"model.zip"
    if (folder/"summary.json").exists():
        saved=json.loads((folder/"summary.json").read_text())
        if (saved["episodes"]!=episodes or saved["seed_start"]!=seed_start or saved["shift"]!=shift
                or saved.get("shift_control",False)!=control):
            raise ValueError(f"Existing evaluation settings differ: {folder}")
        return saved
    return evaluate(policy,checkpoint,episodes,seed_start,folder,shift=shift,shift_control=control)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["train","evaluate","shift","shift_control","probes"])
    p.add_argument("--config",default="configs/pilot.yaml")
    p.add_argument("--policies",nargs="+",default=["recurrent","feedforward","history","belief"])
    p.add_argument("--seeds",nargs="+",type=int)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--root",default="results/pilot")
    p.add_argument("--output")
    p.add_argument("--episodes",type=int)
    p.add_argument("--seed-start",type=int)
    args=p.parse_args()
    from src.train import load_config
    config=load_config(args.config)
    args.seeds=args.seeds or config.get("training",{}).get("seeds",[11,12,13,14,15])
    evaluation=config.get("analysis" if args.stage=="probes" else "evaluation",{})
    args.episodes=args.episodes or evaluation.get("episodes",1024)
    if args.seed_start is None:
        args.seed_start=(2_000_000 if args.stage in {"shift","shift_control"} else
                         evaluation.get("seed",3_000_000 if args.stage=="probes" else 1_000_000))
    started=time.perf_counter()
    if args.stage=="train":
        jobs=[(policy,seed,args.config,args.root) for seed in args.seeds for policy in args.policies]
        fn=train_one
    elif args.stage in {"evaluate","shift","shift_control"}:
        shift=args.stage=="shift"
        control=args.stage=="shift_control"
        output=args.output or ("results/shift" if shift else ("results/shift_control" if control else "results/evaluation"))
        seed_start=args.seed_start
        jobs=[(policy,seed,args.root,output,args.episodes,seed_start,shift,control) for policy in args.policies for seed in args.seeds]
        fn=evaluate_one
    else:
        from src.evaluate import evaluate
        from src.probes import run_probes
        from src.untrained_control import collect_untrained_states
        output=Path(args.output or "results/probes")
        for seed in args.seeds:
            folder=output/f"seed_{seed}"
            if not (folder/"activations.npz").exists():
                evaluate("recurrent",Path(args.root)/"recurrent"/f"seed_{seed}"/"model.zip",
                         args.episodes,args.seed_start or 3_000_000,folder,collect=True)
            if not (folder/"untrained_states.npz").exists():
                collect_untrained_states(folder/"activations.npz",args.config,seed,folder/"untrained_states.npz")
            result=run_probes(folder/"activations.npz",folder)
            print(json.dumps({"seed":seed,"probe_rows":result["rows"]}),flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fn,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result()
            keys=["policy","seed","actual_steps","steps_per_second","objective_mean","objective_episode_se"]
            print(json.dumps({k:result[k] for k in keys if k in result}),flush=True)
    if args.stage in {"evaluate","shift","shift_control"}:
        from src.evaluate import evaluate
        references=(["myopic","fixed_5","fixed_9"] if shift or control else
                    ["fixed_2","fixed_4","fixed_5","fixed_8","fixed_9","myopic"])
        for policy in references:
            folder=Path(output)/policy
            if not (folder/"summary.json").exists():
                result=evaluate(policy,episodes=args.episodes,seed_start=seed_start,output=folder,shift=shift,shift_control=control)
                print(json.dumps({k:result[k] for k in ["policy","objective_mean","objective_episode_se"]}),flush=True)
        if shift or control:
            folder=Path(output)/"myopic_correct"
            if not (folder/"summary.json").exists():
                result=evaluate("myopic",episodes=args.episodes,seed_start=seed_start,output=folder,shift=shift,shift_control=control,correct_filter=True)
                print(json.dumps({"policy":"myopic_correct","objective_mean":result["objective_mean"]}),flush=True)
    print(f"{args.stage} stage elapsed {time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":
    main()
