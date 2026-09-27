"""Myopic decision gaps on the five retained 14M-cohort policy trajectories.

Exploratory: these episodes have already been inspected. No new policies,
decoder fitting, state matching, or counterfactual trajectories are generated.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from src.customer_model import DEFAULT_REGIMES, QUOTES, likelihoods
from src.run_store import atomic_json, file_hash, run_job, verify_run
from src.managed_runs import provenance
from src.probes import episode_split
from scripts.analyze_iteration import (require, validate_trajectories, validate_probe_design,
                                        validate_probe_metrics)
from scripts.run_nonlinear_control import SOURCES as COLLECTION_SOURCES
from scripts.summarize_results import summarize_matrix, bootstrap_episode_mean, training_seed_statistics


SEEDS = (11, 12, 13, 14, 15)
IDS = np.arange(14_000_000, 14_001_024)
COMPONENTS = ('abstention', 'active_when_abstention_better', 'center', 'spread')
PLAN = {
    'cohort': 'Previously inspected nonlinear-control trajectories; all five original entropy-.01 seeds.',
    'horizon': 64, 'inventory_penalty': .001, 'fee': 0.,
    'time_edges': [0, 16, 32, 48, 64],
    'inventory_bins': ['q <= -4', '-3 <= q <= -1', 'q = 0', '1 <= q <= 3', 'q >= 4'],
    'risk_bins': ['r < .05', '.05 <= r < .15', '.15 <= r < .30', 'r >= .30'],
    'risk_definition': 'Mean of buy/sell posterior adverse value revisions at quote (.2,.8).',
    'decomposition': 'Participation plus symmetric average of both center/spread correction orders.',
    'decoder_comparison': 'Trained-state ridge AS R2/MSE versus mean gap on the same 205 test episodes, t=8..63.',
    'uncertainty': 'Separate t4 seed intervals and 4000-draw whole-episode bootstrap; no joint interval.',
    'interpretation': 'On-policy one-step reference gaps, not counterfactual returns or optimal-policy regret.',
}
SOURCES = tuple(dict.fromkeys((*COLLECTION_SOURCES, 'src/baselines.py',
            'scripts/analyze_iteration.py', 'scripts/summarize_results.py', 'scripts/analyze_decisions.py')))


def normalized_rows(values, columns, tolerance, label):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == columns, f'Wrong {label} shape')
    require(np.isfinite(values).all() and np.all(values >= 0), f'Invalid {label}')
    mass = values.sum(axis=1, keepdims=True)
    require(np.all(np.abs(mass-1) <= tolerance), f'Unnormalized {label}')
    return values / mass


def score_actions(posterior, inventory, penalty=.001, horizon=64, fee=0.):
    """Vectorized equivalent of baselines.myopic_scores, default 12-state support."""
    w = normalized_rows(posterior, 12, 1e-10, 'posterior')
    q = np.asarray(inventory, dtype=np.float64)
    require(q.shape == (len(w),) and np.isfinite(q).all(), 'Invalid inventory')
    require(np.isfinite([penalty, horizon, fee]).all() and penalty >= 0 and horizon > 0 and fee >= 0,
            'Invalid economic parameters')
    probs = np.stack([likelihoods(DEFAULT_REGIMES, a) for a in range(9)])
    buy, sell = w @ probs[:, :, 0].T, w @ probs[:, :, 1].T
    value_mass = w * DEFAULT_REGIMES[:, 0]
    profit = (QUOTES[:, 1]-fee)*buy - value_mass @ probs[:, :, 0].T
    profit += value_mass @ probs[:, :, 1].T - (QUOTES[:, 0]+fee)*sell
    incremental_q2 = buy*(1-2*q[:, None]) + sell*(1+2*q[:, None])
    return np.column_stack((profit-penalty/horizon*incremental_q2, np.zeros(len(w))))


def decompose_gap(scores, probabilities):
    """Nonnegative additive attribution; center/spread interactions split symmetrically."""
    s = np.asarray(scores, dtype=np.float64)
    p = normalized_rows(probabilities, 10, 2e-6, 'action probabilities')
    require(s.shape == p.shape and np.isfinite(s).all() and np.all(s[:, 9] == 0), 'Invalid scores')
    active = s[:, :9].reshape(-1, 3, 3)
    mass = p[:, :9].reshape(-1, 3, 3)
    best_active = active.max(axis=(1, 2))
    best = np.maximum(best_active, 0)
    at_center = active.max(axis=2, keepdims=True)
    at_spread = active.max(axis=1, keepdims=True)
    center_loss = .5*(at_spread-active + best_active[:, None, None]-at_center)
    spread_loss = .5*(at_center-active + best_active[:, None, None]-at_spread)
    components = np.column_stack((p[:, 9]*best, (1-p[:, 9])*(best-best_active),
                                  (mass*center_loss).sum(axis=(1, 2)),
                                  (mass*spread_loss).sum(axis=(1, 2))))
    gap = best - (p*s).sum(axis=1)
    require(np.all(components >= -1e-12) and np.allclose(components.sum(1), gap, atol=1e-12, rtol=1e-12),
            'Gap decomposition failed')
    return gap, components


def audit_trajectory(data, economic):
    """Replay beliefs and accounting using public actions/outcomes; no model reload."""
    validate_trajectories(data['episode'], data['t'], IDS)
    order = np.lexsort((data['t'], data['episode']))
    d = {k: v[order].reshape((len(IDS), 64)+v.shape[1:]) for k, v in data.items()}
    obs, actions, w = d['obs'], d['action'], d['posterior']
    require(np.isfinite(obs).all() and np.all((actions >= 0) & (actions < 10)), 'Invalid observation/action')
    normalized_rows(w.reshape(-1, 12), 12, 1e-10, 'posterior')
    require(np.array_equal(economic.index, IDS), 'Economic episodes differ')
    q = obs[:, :, 7]*64
    require(np.array_equal(q, np.rint(q)), 'Noninteger inventory')
    require(np.array_equal(obs[:, :, 8], np.broadcast_to(1-np.arange(64)/64, (len(IDS), 64))),
            'Incorrect decision timing')
    sentinel = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1])
    require(np.all(obs[:, 0] == sentinel), 'Incorrect reset observation')
    outcomes = obs[:, 1:, 3:7].argmax(axis=2)
    require(np.array_equal(obs[:, 1:, 3:7], np.eye(4)[outcomes]) and np.all(outcomes < 3), 'Invalid outcome')
    menu = np.vstack((QUOTES, [0., 0.]))
    require(np.allclose(obs[:, 1:, :2], menu[actions[:, :-1]], atol=1e-7, rtol=0)
            and np.array_equal(obs[:, 1:, 2], actions[:, :-1] == 9), 'Quote/action timing mismatch')
    require(np.all((actions[:, :-1] != 9) | (outcomes == 2)), 'Execution during abstention')
    require(np.array_equal(q[:, 1:]-q[:, :-1], (outcomes == 1).astype(int)-(outcomes == 0)),
            'Inventory/outcome mismatch')
    table = np.stack([likelihoods(DEFAULT_REGIMES, a) for a in range(10)])
    replay = np.full((len(IDS), 12), 1/12)
    for t in range(64):
        require(np.allclose(replay, w[:, t], atol=1e-12, rtol=1e-10), 'Posterior replay mismatch')
        if t < 63:
            replay *= table[actions[:, t], :, outcomes[:, t]]
            replay /= replay.sum(axis=1, keepdims=True)
    targets = d['targets'].reshape(-1, 6)
    flat = w.reshape(-1, 12)
    moments = flat @ DEFAULT_REGIMES
    entropy = -(flat*np.log(np.maximum(flat, np.finfo(float).tiny))).sum(1)
    ref = table[5]
    execution_value = (flat*DEFAULT_REGIMES[:, 0]) @ ref[:, :2] / (flat @ ref[:, :2])
    expected = np.column_stack((moments, entropy, execution_value[:, 0]-moments[:, 0],
                                moments[:, 0]-execution_value[:, 1]))
    require(np.allclose(targets, expected, atol=1e-12, rtol=1e-10), 'Posterior target mismatch')
    final_delta = economic.terminal_inventory.to_numpy()-q[:, -1]
    require(np.isin(final_delta, [-1, 0, 1]).all(), 'Invalid final inventory')
    final_outcome = np.where(final_delta == -1, 0, np.where(final_delta == 1, 1, 2))
    outcomes = np.column_stack((outcomes, final_outcome))
    require(np.all((actions != 9) | (outcomes == 2)), 'Final execution during abstention')
    delta = (outcomes == 1).astype(int)-(outcomes == 0)
    cash = np.where(outcomes == 0, menu[actions, 1], np.where(outcomes == 1, -menu[actions, 0], 0.)).sum(1)
    profit = cash + economic.terminal_inventory.to_numpy()*economic.value.to_numpy()
    cost = .001/64*((q+delta)**2).sum(1)
    require(np.allclose(profit, economic.profit, atol=1e-10, rtol=0)
            and np.allclose(cost, economic.penalty, atol=1e-10, rtol=0)
            and np.allclose(profit-cost, economic.objective, atol=1e-10, rtol=0), 'Accounting replay mismatch')
    return d


def conditional_mean(sums, counts):
    """Ratio of episode totals; bootstrap complete episodes, including zero-count ones."""
    require(sums.shape == counts.shape and sums.ndim == 1, 'Wrong conditional aggregate shape')
    total = int(counts.sum())
    if not total:
        return {'mean': None, 'episode_ci': None, 'decisions': 0, 'episodes': 0}
    rng = np.random.default_rng(90821)
    estimates = []
    for start in range(0, 4000, 128):
        ix = rng.integers(0, len(sums), (min(128, 4000-start), len(sums)))
        denominator = counts[ix].sum(1)
        require(np.all(denominator > 0), 'Insufficient episode coverage for conditional interval')
        estimates.extend(sums[ix].sum(1)/denominator)
    return {'mean': float(sums.sum()/total), 'episode_ci': np.quantile(estimates, [.025, .975]).tolist(),
            'decisions': total, 'episodes': int(np.count_nonzero(counts))}


def analyze(root, work):
    publication = json.loads(Path('published-results/iteration-v1/publication_manifest.json').read_text())
    splits, _ = episode_split(IDS)
    test_ids = np.sort(splits['test'])
    test_mask = np.isin(IDS, test_ids)
    rows, per_seed, bin_results, inputs = [], [], [], {}
    for seed in SEEDS:
        folder = root/f'seed_{seed}'
        manifest = verify_run(folder)
        digest = file_hash(folder/'run_manifest.json')
        require(digest == publication['files'][f'nonlinear/seed_{seed}/run_manifest.json']['original_sha256'],
                'Input differs from published original record')
        require(manifest['spec']['policy_seed'] == seed and manifest['spec']['seed_start'] == IDS[0]
                and manifest['spec']['episodes'] == len(IDS), 'Wrong input cohort')
        require(all(manifest['spec'][k] == v for k, v in provenance(COLLECTION_SOURCES).items()), 'Collection code changed')
        inputs[str(seed)] = {'run_id': manifest['run_id'], 'manifest_sha256': digest}
        economic = pd.read_csv(folder/'episodes.csv').sort_values('episode').set_index('episode')
        with np.load(folder/'activations.npz', allow_pickle=False) as archive:
            data = {k: archive[k] for k in ('episode', 't', 'obs', 'action', 'posterior', 'probabilities', 'targets')}
        d = audit_trajectory(data, economic)
        scores = score_actions(d['posterior'].reshape(-1, 12), (d['obs'][:, :, 7]*64).ravel())
        probabilities = d['probabilities'].reshape(-1, 10)
        gap, components = decompose_gap(scores, probabilities)
        gap, components = gap.reshape(len(IDS), 64), components.reshape(len(IDS), 64, 4)
        frame = pd.DataFrame({'seed': seed, 'episode': IDS, 'objective': economic.objective.to_numpy(),
                              'gap_sum': gap.sum(1), 'late_gap_mean': gap[:, 8:].mean(1), 'probe_test': test_mask})
        for j, name in enumerate(COMPONENTS):
            frame[name] = components[:, :, j].sum(1)
        ridge = json.loads((folder/'probes.json').read_text())
        neural = json.loads((folder/'nonlinear.json').read_text())
        validate_probe_design(ridge, neural, splits)
        with np.load(folder/'predictions.npz', allow_pickle=False) as archive:
            ep, truth, pred = archive['episode'], archive['truth'], archive['hidden_cell']
        validate_probe_metrics(truth, pred, ridge['results']['hidden_cell'])
        order = np.argsort(ep, kind='stable')
        require(np.array_equal(ep[order], np.repeat(test_ids, 56)), 'Wrong decoder test episodes')
        # Probe rows follow collection order; stable episode grouping preserves time order.
        require(np.array_equal(truth[order], d['targets'][test_mask, 8:].reshape(-1, 6)), 'Unpaired decoder targets')
        as_mse = ((truth[order, 4:]-pred[order, 4:])**2).reshape(len(test_ids), 56, 2).mean(axis=(1, 2))
        frame['test_as_mse'] = np.nan
        frame.loc[test_mask, 'test_as_mse'] = as_mse
        test_gap = frame.loc[test_mask, 'late_gap_mean'].to_numpy()
        panel = ridge['results']['hidden_cell']
        record = {'seed': seed, 'objective': float(frame.objective.mean()), 'gap_sum': float(frame.gap_sum.mean()),
                  'gap_sum_episode_ci': bootstrap_episode_mean(frame.gap_sum.to_numpy()),
                  'components': {k: float(frame[k].mean()) for k in COMPONENTS},
                  'test_gap_mean': float(test_gap.mean()), 'test_gap_episode_ci': bootstrap_episode_mean(test_gap),
                  'test_as_r2': float(np.mean([panel['test_r2'][k] for k in ('as_buy_ref', 'as_sell_ref')])),
                  'test_as_mse': float(as_mse.mean()),
                  'within_seed_episode_mse_gap_spearman': float(spearmanr(as_mse, test_gap).statistic),
                  'probability_mass_max_error': float(np.abs(probabilities.astype(float).sum(1)-1).max())}
        per_seed.append(record); rows.append(frame)
        q = d['obs'][:, :, 7]*64
        risk = d['targets'][:, :, 4:].mean(2)
        groups = {'time': (d['t']//16, ['0–15', '16–31', '32–47', '48–63']),
                  'inventory': (np.digitize(q, [-3, 0, 1, 4]), PLAN['inventory_bins']),
                  'risk': (np.digitize(risk, [.05, .15, .30]), PLAN['risk_bins'])}
        for name, (group, labels) in groups.items():
            for k, label in enumerate(labels):
                mask = group == k
                bin_results.append({'seed': seed, 'group': name, 'bin': label,
                                    **conditional_mean((gap*mask).sum(1), mask.sum(1))})
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(work/'episodes.csv', index=False)
    matrix = lambda key: np.stack([r[key].to_numpy() for r in rows])
    gap_stats = summarize_matrix(matrix('gap_sum'))
    association = {'n_policies': 5,
        'r2_vs_gap_spearman': float(spearmanr([r['test_as_r2'] for r in per_seed], [r['test_gap_mean'] for r in per_seed]).statistic),
        'mse_vs_gap_spearman': float(spearmanr([r['test_as_mse'] for r in per_seed], [r['test_gap_mean'] for r in per_seed]).statistic),
        'interpretation': 'Descriptive five-policy association on policy-specific visited histories; no p-values or causal interpretation. R2 uses a different target variance for each policy.'}
    summary = {'plan': PLAN, 'input_runs': inputs, 'seeds': per_seed, 'gap_sum': gap_stats,
               'components': {k: summarize_matrix(matrix(k)) for k in COMPONENTS}, 'bins': bin_results,
               'association': association, 'audit': 'Artifact hashes, original publication identity, complete trajectories, public-history posterior replay, targets, accounting, decoder design and metrics verified.'}
    plot_results(summary, work)
    print(json.dumps({'gap': gap_stats, 'seeds': per_seed, 'association': association}), flush=True)
    return summary


def plot_results(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    seeds = summary['seeds']; x = np.arange(5); bottom = np.zeros(5)
    colors = ['#888888', '#c57d38', '#216779', '#8571a5']
    for key, label, color in zip(COMPONENTS, ['Abstention', 'Active vs abstain', 'Center', 'Spread'], colors):
        values = np.array([r['components'][key] for r in seeds])
        axes[0, 0].bar(x, values, bottom=bottom, label=label, color=color); bottom += values
    cis = np.array([r['gap_sum_episode_ci'] for r in seeds])
    axes[0, 0].errorbar(x, bottom, yerr=np.abs(cis-bottom[:, None]).T, fmt='none', color='black', capsize=3)
    axes[0, 0].set_xticks(x, [str(s) for s in SEEDS]); axes[0, 0].set_xlabel('Policy seed')
    axes[0, 0].set_ylabel('Mean sum of 64 visited-state gaps')
    axes[0, 0].set_title('Additive gap attribution'); axes[0, 0].legend(fontsize=8, frameon=False)
    for ax, group in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ['time', 'inventory', 'risk']):
        labels = list(dict.fromkeys(r['bin'] for r in summary['bins'] if r['group'] == group))
        for seed in SEEDS:
            points = [r for r in summary['bins'] if r['group'] == group and r['seed'] == seed]
            y = np.array([np.nan if r['mean'] is None else r['mean'] for r in points])
            ax.plot(range(len(y)), y, 'o-', label=str(seed), ms=4)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylabel('Mean gap per eligible decision'); ax.set_title(f'Conditional on {group}')
        ax.legend(title='Seed', fontsize=7, ncol=5, frameon=False)
    fig.suptitle('Myopic reference gaps on retained trajectories; conditional bins are descriptive')
    fig.tight_layout(); fig.savefig(output/'decision_gaps.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key, label in zip(axes, ['test_as_r2', 'test_as_mse'], ['Mean buy/sell AS R²', 'Mean buy/sell AS MSE']):
        for r in seeds:
            lo, hi = r['test_gap_episode_ci']; y = r['test_gap_mean']
            ax.errorbar(r[key], y, yerr=[[y-lo], [hi-y]], fmt='o', capsize=3)
            ax.annotate(str(r['seed']), (r[key], y), xytext=(5, 4), textcoords='offset points')
        ax.set_xlabel(label); ax.set_ylabel('Mean myopic gap per decision')
    fig.suptitle('Trained-state ridge decoding vs quoting; same 205 test episodes, t=8–63')
    fig.tight_layout(); fig.savefig(output/'decoding_and_decisions.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/nonlinear_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/decision_quality_v1'))
    args = parser.parse_args()
    spec = {'stage': 'retained_trajectory_decision_audit', 'plan': PLAN, 'seeds': SEEDS,
            'input_manifest_sha256': {str(s): file_hash(args.root/f'seed_{s}'/'run_manifest.json') for s in SEEDS},
            **provenance(SOURCES)}
    with threadpool_limits(limits=1):
        run_job(args.output, spec, lambda work: analyze(args.root, work), 'summary.json',
                ('episodes.csv', 'decision_gaps.png', 'decoding_and_decisions.png'))


if __name__ == '__main__':
    main()
