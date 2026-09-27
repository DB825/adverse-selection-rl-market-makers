"""Portable, verified training/evaluation jobs; interrupted work restarts safely."""
import argparse
import json
from pathlib import Path

from src.managed_runs import managed_train, managed_evaluate
from src.run_store import verify_run
from src.train import load_config, POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage',required=True)
    train = commands.add_parser('train')
    train.add_argument('--config',required=True)
    train.add_argument('--policy',choices=POLICIES,default='recurrent')
    train.add_argument('--seed',type=int,required=True)
    train.add_argument('--output',required=True)
    train.add_argument('--steps',type=int)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--policy',required=True)
    evaluation.add_argument('--checkpoint')
    evaluation.add_argument('--output',required=True)
    evaluation.add_argument('--episodes',type=int,default=128)
    evaluation.add_argument('--seed-start',type=int,required=True)
    evaluation.add_argument('--collect',action='store_true')
    evaluation.add_argument('--horizon',type=int,default=64)
    evaluation.add_argument('--inventory-penalty',type=float,default=.001)
    evaluation.add_argument('--fee',type=float,default=0.)
    verify = commands.add_parser('verify')
    verify.add_argument('output')
    args = parser.parse_args()
    if args.stage == 'verify':
        record = verify_run(args.output)
        print(json.dumps({'run_id':record['run_id'],'verified_artifacts':len(record['artifacts'])})); return
    if args.stage == 'train':
        config = load_config(args.config)
        if args.steps is not None:
            config['training']['total_timesteps'] = args.steps
        result = managed_train(config,args.policy,args.seed,args.output)
    else:
        if args.checkpoint and not (Path(args.checkpoint).parent/'run_manifest.json').exists():
            parser.error('This CLI requires managed training provenance. Use the historical evaluator for legacy checkpoints with a separately documented protocol.')
        result = managed_evaluate(args.policy,args.checkpoint,args.output,episodes=args.episodes,
                                  seed_start=args.seed_start,collect=args.collect,horizon=args.horizon,
                                  inventory_penalty=args.inventory_penalty,fee=args.fee)
    print(json.dumps({k:result[k] for k in ('run_id','actual_steps','objective_mean') if k in result}))


if __name__ == '__main__':
    main()
