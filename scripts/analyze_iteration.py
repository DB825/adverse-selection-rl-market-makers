"""Audit complete fresh-cohort experiments and report separate uncertainty sources."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import torch

from src.run_store import atomic_json, file_hash, verify_run
from scripts.run_replication import ROOT as REPLICATION, SEEDS as NEW_SEEDS, CONFIGS, check_frozen
from scripts.run_nonlinear_control import ROOT as NONLINEAR, SEEDS as OLD_SEEDS
from scripts.summarize_results import TARGETS, summarize_matrix, summarize_family, _save_figure


OUT = Path('results/iteration_v1')
FIGURES = Path('reports/iteration_figures')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replication_gate(effect, fixed, qualifying):
    return (effect['mean'] >= .25 and effect['training_seed']['mean_t95_ci'][0] > 0
            and fixed['training_seed']['mean_t95_ci'][0] > 0 and len(qualifying) >= 3)


def read_evaluation(folder, expected_ids):
    manifest = verify_run(folder)
    require(manifest['spec']['stage'] == 'evaluate', 'Wrong evaluation stage')
    settings = manifest['spec']['settings']
    require(settings == dict(episodes=2048, seed_start=15_000_000, deterministic=False,
                            shift=False, correct_filter=False, collect=False, batch_size=128,
                            inventory_penalty=.001, horizon=64, fee=0., shift_control=False), 'Evaluation settings differ')
    frame = pd.read_csv(folder/'episodes.csv').sort_values('episode').set_index('episode')
    require(np.array_equal(frame.index, expected_ids), 'Wrong economic episode IDs')
    require(np.isfinite(frame[['objective', 'profit', 'penalty']]).all().all(), 'Nonfinite economics')
    require(np.allclose(frame.profit-frame.penalty, frame.objective), 'Accounting identity failed')
    summary = read(folder/'summary.json')
    require(np.isclose(summary['objective_mean'], frame.objective.mean()), 'Summary differs from episode data')
    return frame, summary, manifest


def analyze_replication():
    from src.managed_runs import training_spec
    from scripts.analyze_entropy import check_logged_values
    check_frozen()
    frames, summaries, hashes, audits = {}, {}, {}, []
    ids = np.arange(15_000_000, 15_002_048)
    for arm, coefficient in [('control', 0.), ('treatment', .01)]:
        frames[arm], summaries[arm] = {}, {}
        for seed in NEW_SEEDS:
            training = REPLICATION/arm/f'seed_{seed}'
            trained_manifest = verify_run(training, training_spec(CONFIGS[arm], 'recurrent', seed))
            metadata = read(training/'metadata.json')
            log = pd.read_csv(training/'progress.csv')
            require(not np.isinf(log.select_dtypes(include=[np.number]).to_numpy()).any(), 'Infinite training diagnostic')
            require((log['economics/episodes'].dropna() == 16).all()
                    and (log['rollout/ep_len_mean'].dropna() == 64).all(), 'Incomplete rollout episodes')
            diagnostics = check_logged_values(pd.read_csv(training/'progress.csv',keep_default_na=False))
            require(metadata['actual_steps'] == metadata['requested_steps'] == 300032, 'Incomplete training budget')
            require(trained_manifest['spec']['seed'] == seed and trained_manifest['spec']['policy'] == 'recurrent', 'Wrong training identity')
            model = RecurrentPPO.load(training/'model.zip', device='cpu')
            require(model.ent_coef == coefficient and model.gae_lambda == .95 and model.gamma == 1
                    and model.num_timesteps == 300032, 'Serialized settings differ')
            require(all(torch.isfinite(v).all().item() for v in model.policy.state_dict().values()), 'Nonfinite weights')
            del model
            evaluation = REPLICATION/'evaluation'/arm/f'seed_{seed}'
            frame, summary, evaluated_manifest = read_evaluation(evaluation, ids)
            require(evaluated_manifest['spec']['checkpoint_sha256'] == file_hash(training/'model.zip')
                    and evaluated_manifest['spec']['training_run_id'] == trained_manifest['run_id'], 'Evaluation provenance mismatch')
            frames[arm][seed], summaries[arm][seed] = frame, summary
            for path in (training/'run_manifest.json', evaluation/'run_manifest.json'):
                hashes[str(path)] = file_hash(path)
            audits.append({'arm': arm, 'seed': seed, 'run_id': trained_manifest['run_id'],
                           'training_seconds': metadata['elapsed_seconds'], 'serialized_parameters_finite': True,
                           'log_diagnostics': diagnostics})
    for seed in NEW_SEEDS:
        a, b = [pd.read_csv(REPLICATION/arm/f'seed_{seed}'/'progress.csv').iloc[0] for arm in ('control', 'treatment')]
        fields = ['economics/objective_mean', 'economics/profit_mean', 'economics/inventory_penalty_total_mean']
        require(all(a[k] == b[k] for k in fields), 'Paired first rollouts differ')
    refs = {name: read_evaluation(REPLICATION/'evaluation'/name, ids)[0] for name in ('fixed_5', 'myopic')}
    anchor = frames['control'][NEW_SEEDS[0]][['value','alpha','p']].to_numpy()
    for frame in [*frames['control'].values(), *frames['treatment'].values(), *refs.values()]:
        require(np.array_equal(frame[['value','alpha','p']], anchor), 'Unpaired exogenous regimes')
    values = {arm: np.stack([f.objective.to_numpy() for f in by_seed.values()]) for arm, by_seed in frames.items()}
    effect = summarize_matrix(values['treatment']-values['control'])
    fixed = summarize_matrix(values['treatment']-refs['fixed_5'].objective.to_numpy()[None,:])
    qualifying = [seed for seed in NEW_SEEDS if frames['treatment'][seed].objective.mean() > refs['fixed_5'].objective.mean()
                  and summaries['treatment'][seed]['behavior']['mean_within_time_probability_variance'] > .01]
    payload = {'seeds': list(NEW_SEEDS), 'episodes': 2048, 'seed_start': 15_000_000,
               'families': {arm: summarize_family(f) for arm, f in frames.items()},
               'references': {name: summarize_family({'reference': f}, trained=False) for name, f in refs.items()},
               'effect': effect, 'treatment_minus_fixed': fixed,
               'qualifying_behavior_seeds': qualifying, 'predeclared_replication_criterion_met': replication_gate(effect, fixed, qualifying),
               'behavior': {arm: {str(seed): s['behavior'] for seed, s in by_seed.items()} for arm, by_seed in summaries.items()},
               'audits': audits, 'run_manifest_sha256': hashes,
               'uncertainty': 'Separate approximate t4 seed intervals and whole-episode bootstrap intervals conditional on fitted policies; neither is a joint interval.'}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for i, seed in enumerate(NEW_SEEDS):
        axes[0].plot([0,1], [values[arm][i].mean() for arm in ('control','treatment')], 'o-', label=f'Seed {seed}')
    axes[0].set_xticks([0,1], ['Entropy 0', 'Entropy .01']); axes[0].set_ylabel('Mean episode objective')
    axes[0].legend(frameon=False, fontsize=8); axes[0].set_title('Five independent seed pairs')
    for i, ci in enumerate([effect['training_seed']['mean_t95_ci'], effect['episode_bootstrap_95_ci']]):
        axes[1].errorbar(effect['mean'], i, xerr=np.abs(np.array(ci)-effect['mean']).reshape(2,1), fmt='o', capsize=4)
    axes[1].set_yticks([0,1], ['Training seeds', 'Test episodes']); axes[1].set_ylim(-.7,1.7)
    axes[1].axvline(0,color='gray',lw=.8); axes[1].set_xlabel('Paired objective difference'); axes[1].set_title('Separate 95% intervals')
    fig.tight_layout(); payload['figure'] = _save_figure(fig, FIGURES, 'independent_replication')
    atomic_json(OUT/'replication_summary.json', payload)
    print(json.dumps({'effect': effect, 'criterion_met': payload['predeclared_replication_criterion_met']}), flush=True)
    return payload


def aggregate_episode_losses(episode, truth, prediction, expected_ids):
    require(truth.shape == prediction.shape and truth.shape == (len(episode),6), 'Wrong prediction shape')
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(), 'Nonfinite probe output')
    require(np.array_equal(np.unique(episode), expected_ids), 'Probe episode IDs differ')
    require(all(np.count_nonzero(episode == e) == 56 for e in expected_ids), 'Missing probe decisions')
    return np.stack([((truth[episode == e]-prediction[episode == e])**2).mean(0) for e in expected_ids])


def analyze_nonlinear():
    from src.probes import episode_split
    from scripts.summarize_results import training_seed_statistics
    from scripts.run_nonlinear_control import SOURCES
    from src.managed_runs import provenance
    splits, _ = episode_split(np.arange(14_000_000, 14_001_024))
    test_ids = np.sort(splits['test'])
    losses, r2, audits = {}, {}, []
    for seed in OLD_SEEDS:
        folder = NONLINEAR/f'seed_{seed}'
        manifest = verify_run(folder)
        spec = manifest['spec']
        require(spec['stage'] == 'nonlinear_probe_panel' and spec['policy_seed'] == seed
                and spec['episodes'] == 1024 and spec['seed_start'] == 14_000_000, 'Wrong nonlinear panel')
        require(spec['protocol_sha256'] == file_hash('reports/nonlinear_protocol.md'), 'Nonlinear protocol changed')
        require(all(spec[k] == v for k,v in provenance(SOURCES).items()), 'Nonlinear implementation changed')
        record, ridge = read(folder/'nonlinear.json'), read(folder/'probes.json')
        require(all(np.array_equal(record['episode_splits'][k], v) for k,v in splits.items()), 'Changed episode split')
        require(record['fit']['test_used_for_selection'] is False, 'Test-based selection')
        require(record['fit']['architecture'] == [72,64,64,6] and record['fit']['seed'] == 2026+seed
                and record['fit']['epochs_per_candidate'] == 40 and len(record['fit']['trials']) == 16,
                'Decoder tuning budget changed')
        with np.load(folder/'activations.npz', allow_pickle=False) as a:
            require(np.array_equal(np.unique(a['episode']), np.arange(14_000_000,14_001_024)), 'Wrong activation cohort')
            require(all(np.array_equal(np.sort(a['t'][a['episode'] == e]),np.arange(64))
                        for e in np.unique(a['episode'])), 'Incomplete activation trajectories')
            selected = (a['t'] >= 8) & np.isin(a['episode'],test_ids)
            expected_truth, expected_episode = a['targets'][selected], a['episode'][selected]
            require(np.all(expected_truth.var(0) > 0), 'Degenerate held-out target')
        with np.load(folder/'predictions.npz', allow_pickle=False) as predictions:
            require(np.array_equal(predictions['truth'],expected_truth)
                    and np.array_equal(predictions['episode'],expected_episode), 'Prediction truth differs from collection')
            with np.load(folder/'nonlinear_predictions.npz', allow_pickle=False) as nonlinear:
                require(np.array_equal(predictions['episode'], nonlinear['episode'])
                        and np.array_equal(predictions['truth'], nonlinear['truth']), 'Decoder targets are unpaired')
                for name in ridge['results']:
                    values = aggregate_episode_losses(predictions['episode'], predictions['truth'], predictions[name], test_ids)
                    losses.setdefault(name, []).append(values)
                    r2.setdefault(name, []).append([ridge['results'][name]['test_r2'][t] for t in TARGETS])
                    require(np.allclose(values.mean(0), [ridge['results'][name]['test_mse'][t] for t in TARGETS]), 'Ridge MSE mismatch')
                values = aggregate_episode_losses(nonlinear['episode'], nonlinear['truth'], nonlinear['prediction'], test_ids)
                losses.setdefault('nonlinear_history8', []).append(values)
                r2.setdefault('nonlinear_history8', []).append([record['test_r2'][t] for t in TARGETS])
                require(np.allclose(values.mean(0), [record['test_mse'][t] for t in TARGETS]), 'MLP MSE mismatch')
        audits.append({'seed': seed, 'run_id': manifest['run_id'], 'manifest_sha256': file_hash(folder/'run_manifest.json'),
                       'fit': record['fit'], 'test_target_variance': record['test_target_variance']})
    losses = {k: np.stack(v) for k,v in losses.items()}
    comparisons = {}
    pairs = {'nonlinear_over_linear_history': ('history8','nonlinear_history8'),
             'trained_state_over_nonlinear_history': ('nonlinear_history8','history8_hidden_cell'),
             'trained_over_untrained_state': ('history8_untrained_hidden_cell','history8_hidden_cell')}
    for label,(base,enhanced) in pairs.items():
        comparisons[label] = {}
        for j,target in enumerate(TARGETS):
            result = summarize_matrix((losses[base]-losses[enhanced])[:,:,j])
            relative = ((losses[base]-losses[enhanced]).mean(1) / losses[base].mean(1))[:,j]
            result['relative_mse_reduction_across_seeds'] = training_seed_statistics(relative)
            comparisons[label][target] = result
    summary = {'seeds': list(OLD_SEEDS), 'episodes': 1024, 'seed_start': 14_000_000,
               'split_counts': {k: len(v) for k,v in splits.items()},
               'r2': {name: {t: dict(training_seed_statistics(np.array(values)[:,j]), per_seed=np.array(values)[:,j].tolist())
                             for j,t in enumerate(TARGETS)} for name,values in r2.items()},
               'comparisons': comparisons, 'audits': audits,
               'interpretation': 'All targets/seed panels included. Separate conditional seed and episode uncertainty; exploratory unadjusted comparisons. Finite decoder capacity does not identify older information or causal use.'}
    names = ['history8','untrained_hidden_cell','history8_untrained_hidden_cell','hidden_cell','history8_hidden_cell','nonlinear_history8']
    labels = ['Linear history-8','Untrained state','History + untrained state','Trained state','History + trained state','Nonlinear history-8']
    matrix = np.array([[summary['r2'][name][t]['mean'] for t in TARGETS] for name in names])
    fig, ax = plt.subplots(figsize=(10.5,5.2))
    im = ax.imshow(matrix, vmin=min(0,matrix.min()), vmax=1, cmap='RdYlBu', aspect='auto')
    ax.set_xticks(range(6), ['Value','Alpha','p','Entropy','Buy AS','Sell AS']); ax.set_yticks(range(len(names)),labels)
    for i in range(len(names)):
        for j in range(6): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center')
    ax.set_title('Fresh-cohort held-out R²: all five frozen policy seeds'); fig.colorbar(im,ax=ax)
    fig.tight_layout(); summary['figure'] = _save_figure(fig, FIGURES, 'nonlinear_history_control')
    atomic_json(OUT/'nonlinear_summary.json', summary)
    print('All five nonlinear panels audited; summary written',flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['replication','nonlinear'])
    args = parser.parse_args()
    (analyze_replication if args.stage == 'replication' else analyze_nonlinear)()
