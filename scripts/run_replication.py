"""Predeclare, execute and audit the five-pair independent entropy replication."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

from src.managed_runs import managed_train, managed_evaluate, provenance, TRAIN_SOURCES
from src.run_store import atomic_json, file_hash
from src.train import load_config, make_model
from scripts.audit_followup import parameter_digest


ROOT = Path('results/replication_v1')
SEEDS = (21, 22, 23, 24, 25)
CONFIGS = {arm: f'configs/replication_{arm}.yaml' for arm in ('control', 'treatment')}


def validate_configs():
    configs = {arm: load_config(p) for arm, p in CONFIGS.items()}
    a, b = [json.loads(json.dumps(configs[k])) for k in ('control', 'treatment')]
    if a['training'].pop('ent_coef') != 0 or b['training'].pop('ent_coef') != .01 or a != b:
        raise ValueError('Entropy must be the sole between-arm difference')
    if a['training']['seeds'] != list(SEEDS) or a['training']['total_timesteps'] != 300032:
        raise ValueError('Replication seeds/budget differ from protocol')
    old = load_config('configs/followup_budget.yaml')['training']
    new = configs['control']['training'].copy()
    old.pop('seeds'); new.pop('seeds')
    if old != new:
        raise ValueError('Scientific training settings differ from original control')
    if configs['control']['evaluation'] != {'episodes': 2048, 'seed': 15_000_000}:
        raise ValueError('Wrong replication test cohort')
    domains = [(s * 10000, s * 10000 + a['training']['n_envs']) for s in SEEDS]
    domains += [(a[k]['seed'], a[k]['seed'] + a[k]['episodes']) for k in ('development', 'evaluation', 'analysis')]
    domains += [(14_000_000, 14_001_024)]
    for i, (start, stop) in enumerate(domains):
        if any(start < other_stop and stop > other_start for other_start, other_stop in domains[i + 1:]):
            raise ValueError('Reserved seed domains overlap')
    return configs


def preflight():
    configs = validate_configs()
    pairs = []
    for seed in SEEDS:
        a, b = [make_model(configs[arm], 'recurrent', seed) for arm in ('control', 'treatment')]
        left, right = parameter_digest(a), parameter_digest(b)
        if left != right or a.market_environment_seeds != b.market_environment_seeds:
            raise ValueError('Initialization pair differs')
        pairs.append({'seed': seed, 'initial_parameters_sha256': left,
                      'environment_seeds': a.market_environment_seeds})
        a.get_env().close(); b.get_env().close()
    payload = {'pairs': pairs, 'training_provenance': provenance(TRAIN_SOURCES),
               'config_sha256': {arm: file_hash(p) for arm, p in CONFIGS.items()},
               'protocol_sha256': file_hash('reports/replication_protocol.md')}
    path = ROOT / 'preflight.json'
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('Preflight changed; preserve the previous experiment and choose a new version')
    else:
        atomic_json(path, payload)
    return payload


def check_frozen():
    validate_configs()
    p = json.loads((ROOT / 'preflight.json').read_text())
    if p['training_provenance'] != provenance(TRAIN_SOURCES):
        raise ValueError('Training implementation/dependencies changed after preflight')
    if p['config_sha256'] != {arm: file_hash(c) for arm, c in CONFIGS.items()}:
        raise ValueError('Replication configuration changed')
    if p['protocol_sha256'] != file_hash('reports/replication_protocol.md'):
        raise ValueError('Protocol changed after preflight')


def train_one(job):
    arm, seed = job
    return managed_train(CONFIGS[arm], 'recurrent', seed, ROOT / arm / f'seed_{seed}')


def evaluate_one(job):
    arm, seed = job
    return managed_evaluate('recurrent', ROOT / arm / f'seed_{seed}' / 'model.zip',
                            ROOT / 'evaluation' / arm / f'seed_{seed}', episodes=2048, seed_start=15_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'train', 'evaluate'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.stage == 'preflight':
        preflight(); print('Five fresh paired initializations and frozen protocol verified'); return
    check_frozen()
    started = time.perf_counter()
    function = train_one if args.stage == 'train' else evaluate_one
    jobs = [(arm, seed) for seed in SEEDS for arm in CONFIGS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            arm, seed = futures[future]
            print(json.dumps({'arm': arm, 'seed': seed, 'run_id': result['run_id'],
                              'steps': result.get('actual_steps'), 'objective': result.get('objective_mean')}), flush=True)
    if args.stage == 'evaluate':
        for policy in ('fixed_5', 'myopic'):
            managed_evaluate(policy, output=ROOT / 'evaluation' / policy, episodes=2048, seed_start=15_000_000)
    elapsed = time.perf_counter() - started
    atomic_json(ROOT / f'{args.stage}_stage.json', {'wall_seconds': elapsed, 'workers': args.workers,
                'scope': 'Invocation wall time, including validation; completed jobs may be reused'})
    print(f'{args.stage} finished in {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
