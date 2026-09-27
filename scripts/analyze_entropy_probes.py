"""Validate and summarize all five conditional entropy-treatment probe panels."""
from pathlib import Path

import numpy as np

from src.artifacts import write_json, source_hashes
from src.probes import episode_split
from scripts.analyze_entropy import read, digest, require
from scripts.summarize_results import SEEDS, TARGETS, load_probe_summaries, summarize_matrix, probe_figure

ROOT = Path('results/entropy_analysis')
COMPARISONS = {'basic': 'basic_hidden_cell', 'history8': 'history8_hidden_cell',
               'history8_untrained_hidden_cell': 'history8_hidden_cell'}


def episode_losses(episode, truth, prediction, expected):
    require(truth.shape == prediction.shape and truth.shape == (len(episode), 6), 'Prediction shape differs')
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(), 'Nonfinite targets or predictions')
    require(np.array_equal(np.unique(episode), expected), 'Wrong held-out episode IDs')
    require(all(np.count_nonzero(episode == e) == 56 for e in expected), 'Expected 56 decisions per test episode')
    return np.stack([((truth[episode == e] - prediction[episode == e]) ** 2).mean(0) for e in expected])


def run():
    gates = read('results/entropy_decision_gates.json')
    require(gates['gate_2']['decision'] == 'continue to exploratory decoding', 'Behavior gate has not passed')
    economic_manifest = read('results/entropy_manifest.json')
    ids = np.arange(12_000_000, 12_001_024)
    splits, _ = episode_split(ids)
    held_out = np.sort(splits['test'])
    contrasts = {name: [] for name in COMPARISONS}
    audits, hashes = [], {}
    for seed in SEEDS:
        folder = ROOT / 'probes' / f'seed_{seed}'
        p = read(folder / 'probes.json')
        checkpoint = Path('results/entropy_001/recurrent') / f'seed_{seed}' / 'model.zip'
        require(digest(checkpoint) == economic_manifest['artifacts_sha256'][str(checkpoint)], 'Frozen checkpoint changed')
        evaluation = read(folder / 'summary.json')
        require(Path(evaluation['checkpoint']).resolve() == checkpoint.resolve() and
                evaluation['episodes'] == 1024 and evaluation['seed_start'] == 12_000_000,
                'Wrong policy or collection cohort')
        require(p['min_time'] == 8 and p['rows'] == 1024 * 56 and p['split_seed'] == 73, 'Wrong probe rows or split')
        require(p['reference_quote'] == {'action': 5, 'bid': .2, 'ask': .8}, 'Target quote changed')
        require(all(np.array_equal(p['episode_splits'][k], v) for k, v in splits.items()), 'Probe split changed')
        control = read(folder / 'untrained_states.json')
        require(control['training_timesteps'] == 0 and control['initialization_seed'] == seed and
                control['observation_inputs'] == ['obs'] and control['dataset_sha256'] == digest(folder / 'activations.npz'),
                'Invalid untrained control')
        with np.load(folder / 'activations.npz', allow_pickle=False) as a:
            require(np.array_equal(np.unique(a['episode']), ids), 'Wrong activation cohort')
            require(all(np.array_equal(np.sort(a['t'][a['episode'] == e]), np.arange(64)) for e in ids), 'Incomplete trajectories')
            mask = (a['t'] >= 8) & np.isin(a['episode'], held_out)
            true_targets = a['targets'][mask]
            true_episodes = a['episode'][mask]
            variances = dict(zip(TARGETS, true_targets.var(0).tolist()))
            require(np.isfinite(true_targets).all() and all(v > 0 for v in variances.values()), 'Degenerate target; inspect before reporting R2')
        with np.load(folder / 'predictions.npz', allow_pickle=False) as pred:
            require(np.array_equal(pred['episode'], true_episodes) and np.array_equal(pred['truth'], true_targets), 'Predictions are not the held-out activation targets')
            losses = {name: episode_losses(pred['episode'], pred['truth'], pred[name], held_out) for name in p['results']}
            for name, values in losses.items():
                require(np.allclose(values.mean(0), [p['results'][name]['test_mse'][t] for t in TARGETS]), 'Reported MSE differs from predictions')
            for base, enhanced in COMPARISONS.items():
                delta = losses[base] - losses[enhanced]
                require(np.allclose(delta.mean(0), [p['added_state_vs'][base][t]['mse_reduction'] for t in TARGETS]), 'Reported MSE contrast differs')
                contrasts[base].append(delta)
        for name in ('probes.json', 'summary.json', 'activation_schema.json', 'activations.npz',
                     'untrained_states.json', 'untrained_states.npz', 'predictions.npz'):
            path = folder / name
            hashes[str(path)] = digest(path)
        hashes[str(checkpoint)] = digest(checkpoint)
        audits.append({'seed': seed, 'test_target_variance': variances,
                       'checkpoint_unchanged': True, 'untrained_control_matches_trajectory': True})
    summary = load_probe_summaries(ROOT)
    require(summary['training_seeds'] == list(SEEDS), 'Missing planned probe seed')
    summary['paired_absolute_mse_reduction'] = {
        base: {target: summarize_matrix(np.stack(values)[:, :, j]) for j, target in enumerate(TARGETS)}
        for base, values in contrasts.items()}
    summary.update(cohort_start=12_000_000, episodes=1024,
                   split_counts={k: len(v) for k, v in splits.items()}, audits=audits,
                   interpretation='Exploratory linear accessibility on policy-dependent histories. Seed t4 and whole-episode bootstrap intervals are separate, conditional and unadjusted across targets/comparators; they exclude joint and probe-fit uncertainty. No nonlinear history control or causal-use intervention has run.')
    summary['figure'] = probe_figure(summary, Path('reports/entropy_figures'))
    write_json('results/entropy_probe_summary.json', summary)
    hashes['results/entropy_probe_summary.json'] = digest('results/entropy_probe_summary.json')
    write_json('results/entropy_probe_manifest.json', {'artifacts_sha256': hashes,
               'source_hashes': source_hashes(), 'audits': audits,
               'economic_manifest_sha256': digest('results/entropy_manifest.json')})
    print('Validated all five probe panels; results/entropy_probe_summary.json')
    return summary


if __name__ == '__main__':
    run()
