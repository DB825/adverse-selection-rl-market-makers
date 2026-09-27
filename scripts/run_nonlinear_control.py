"""Run all five frozen-policy nonlinear controls on the fresh 14M cohort."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import r2_score
import torch

from src.evaluate import evaluate, TARGET_NAMES
from src.managed_runs import EVAL_SOURCES, provenance
from src.run_store import atomic_json, file_hash, run_job
from src.probes import episode_split, run_probes
from src.untrained_control import collect_untrained_states
from src.nonlinear_probe import fit_history_decoder


ROOT = Path('results/nonlinear_v1')
SEEDS = (11, 12, 13, 14, 15)
SOURCES = EVAL_SOURCES + ('src/probes.py', 'src/untrained_control.py', 'src/nonlinear_probe.py',
                           'scripts/run_nonlinear_control.py')


def run_seed(seed):
    checkpoint = Path('results/entropy_001/recurrent') / f'seed_{seed}' / 'model.zip'
    frozen = json.loads(Path('published-results/entropy_manifest.json').read_text())['artifacts_sha256']
    expected = next(h for p, h in frozen.items() if p.replace('\\', '/') == checkpoint.as_posix())
    if file_hash(checkpoint) != expected:
        raise ValueError('Original treatment checkpoint changed')
    spec = {'stage': 'nonlinear_probe_panel', 'schema_version': 1, 'policy_seed': seed,
            'checkpoint_sha256': expected, 'episodes': 1024, 'seed_start': 14_000_000,
            'protocol_sha256': file_hash('reports/nonlinear_protocol.md'), **provenance(SOURCES)}
    def compute(work):
        started = time.perf_counter()
        summary = evaluate('recurrent', checkpoint, episodes=1024, seed_start=14_000_000, output=work, collect=True)
        if file_hash(checkpoint) != expected:
            raise ValueError('Checkpoint changed during collection')
        summary.update(checkpoint='sha256:'+expected, checkpoint_sha256=expected)
        atomic_json(work/'summary.json', summary)
        control = collect_untrained_states(work/'activations.npz', 'configs/entropy_001.yaml', seed)
        control.update(dataset='activations.npz', output='untrained_states.npz')
        atomic_json(work/'untrained_states.json', control)
        ridge = run_probes(work/'activations.npz', work)
        ridge.update(dataset='activations.npz', untrained_control='untrained_states.npz')
        atomic_json(work/'probes.json', ridge)
        with np.load(work/'activations.npz', allow_pickle=False) as archive:
            keep = archive['t'] >= 8
            episode = archive['episode'][keep]
            x, y = archive['history'][keep], archive['targets'][keep]
        ids, masks = episode_split(episode)
        train, val, test = [masks[k] for k in ('train', 'validation', 'test')]
        variance = y[test].var(0)
        if not np.isfinite(y).all() or np.any(variance <= 0):
            raise ValueError('Nonfinite or constant target; inspect before reporting R2')
        prediction, fit, state = fit_history_decoder(x[train], y[train], x[val], y[val], x[test], 2026+seed)
        np.savez_compressed(work/'nonlinear_predictions.npz', episode=episode[test], truth=y[test], prediction=prediction)
        torch.save(state, work/'decoder.pt')
        return {'seed': seed, 'episodes': 1024, 'seed_start': 14_000_000,
                'checkpoint_sha256': expected, 'rows': len(y), 'min_time': 8,
                'episode_splits': ids, 'split_counts': {k: len(v) for k, v in ids.items()},
                'fit': fit, 'test_target_variance': dict(zip(TARGET_NAMES, variance)),
                'test_r2': dict(zip(TARGET_NAMES, r2_score(y[test], prediction, multioutput='raw_values'))),
                'test_mse': dict(zip(TARGET_NAMES, ((prediction-y[test])**2).mean(0))),
                'elapsed_seconds': time.perf_counter()-started,
                'interpretation': 'Exploratory finite-capacity recent-history control. No test-based selection, older-memory identification or causal-use conclusion.'}
    return run_job(ROOT/f'seed_{seed}', spec, compute, 'nonlinear.json',
                   ('activations.npz', 'untrained_states.npz', 'probes.json', 'predictions.npz', 'nonlinear_predictions.npz', 'decoder.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    if not set(args.seeds).issubset(SEEDS):
        raise ValueError('Unplanned policy seed')
    for seed in args.seeds:
        result = run_seed(seed)
        print(json.dumps({'seed': seed, 'seconds': result['elapsed_seconds'], 'selected_epoch': result['fit']['epoch']}), flush=True)


if __name__ == '__main__':
    main()
