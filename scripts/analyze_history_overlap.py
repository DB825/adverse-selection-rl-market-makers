"""Exact history-8 overlap on retained trajectories; no matching tolerance search."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.analyze_decisions import IDS, SEEDS, SOURCES as AUDIT_SOURCES, audit_trajectory, score_actions
from scripts.analyze_iteration import require
from scripts.run_nonlinear_control import SOURCES as COLLECTION_SOURCES
from src.managed_runs import provenance
from src.run_store import file_hash, run_job, verify_run


PROTOCOL = 'reports/history_overlap_protocol.md'
SOURCES = (*AUDIT_SOURCES, 'scripts/analyze_history_overlap.py', PROTOCOL)
MIN_TIME = 9
MARGIN_TOLERANCE = 1e-10


def recent_histories(observations):
    """Oldest-first history-8, excluding times with no omitted outcome evidence."""
    obs = np.asarray(observations)
    require(obs.ndim == 3 and obs.shape[2] == 9 and obs.shape[1] > MIN_TIME
            and np.isfinite(obs).all(), 'Invalid observation trajectories')
    windows = np.lib.stride_tricks.sliding_window_view(obs, 8, axis=1)
    return windows[:, MIN_TIME-7:].swapaxes(-1, -2).reshape(obs.shape[0], -1, 72)


def match_history(history, episode, scores, risk):
    """Group exact public inputs; compute peer losses in O(rows * actions).

    Within each group, action counts suffice to average all ordered peer pairs.
    No dense pairwise distance or loss matrix is materialized.
    """
    h, ep, s, r = map(np.asarray, (history, episode, scores, risk))
    n = len(h)
    require(n > 0 and h.shape == (n, 72) and ep.shape == (n,)
            and s.shape == (n, 10) and r.shape == (n, 2), 'Invalid matching shapes')
    require(all(np.isfinite(x).all() for x in (h, ep, s, r)), 'Nonfinite matching input')
    _, inverse, counts = np.unique(h, axis=0, return_inverse=True, return_counts=True)
    require(len(np.unique(np.column_stack((inverse, ep)), axis=0)) == n,
            'Repeated episode within a matched group')
    best_action = s.argmax(1)
    top = np.partition(s, -2, axis=1)[:, -2:]
    margin = top[:, 1]-top[:, 0]
    decisive = margin > MARGIN_TOLERANCE
    matched = counts[inverse] > 1
    conflict = np.zeros(n, dtype=bool)
    peer_gap = np.zeros(n)
    groups = []
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)]
    for group in np.flatnonzero(counts > 1):
        ix = order[starts[group]:starts[group+1]]
        action_counts = np.bincount(best_action[ix], minlength=10)
        decisive_counts = np.bincount(best_action[ix][decisive[ix]], minlength=10)
        conflict[ix] = decisive[ix] & (decisive_counts.sum()-decisive_counts[best_action[ix]] > 0)
        best = s[ix, best_action[ix]]
        # Each row's own action contributes zero loss; denominator excludes it.
        loss = (counts[group]*best-s[ix] @ action_counts)/(counts[group]-1)
        require(np.all(loss >= -1e-12), 'Negative peer reference loss')
        peer_gap[ix] = np.maximum(loss, 0.)
        groups.append({'group': int(group), 'decisions': len(ix),
                       'decisive_decisions': int(decisive[ix].sum()),
                       'conflicting_decisions': int(conflict[ix].sum()),
                       'reference_actions': int(np.count_nonzero(action_counts)),
                       'decisive_reference_actions': int(np.count_nonzero(decisive_counts)),
                       'max_as_range': float(np.ptp(r[ix], axis=0).max()),
                       'max_score_range': float(np.ptp(s[ix], axis=0).max()),
                       'min_margin': float(margin[ix].min()),
                       'max_margin': float(margin[ix].max()),
                       'mean_peer_action_gap': float(peer_gap[ix].mean())})
    group_columns = ['group', 'decisions', 'decisive_decisions', 'conflicting_decisions',
                     'reference_actions', 'decisive_reference_actions', 'max_as_range',
                     'max_score_range', 'min_margin', 'max_margin', 'mean_peer_action_gap']
    rows = pd.DataFrame({'episode': ep, 'eligible': np.ones(n, dtype=int),
                         'matched': matched.astype(int), 'conflict': conflict.astype(int),
                         'peer_gap_sum': peer_gap})
    episodes = rows.groupby('episode', sort=True).sum().reset_index()
    count = int(matched.sum())
    summary = {'eligible_decisions': n, 'matched_decisions': count,
               'matched_fraction': count/n, 'matched_episodes': int(np.unique(ep[matched]).size),
               'matched_groups': len(groups),
               'largest_group': max((g['decisions'] for g in groups), default=0),
               'conflicting_decisions': int(conflict.sum()),
               'conflict_fraction_eligible': float(conflict.mean()),
               'conflict_fraction_matched': float(conflict.sum()/count) if count else None,
               'decisive_matched_decisions': int((decisive & matched).sum()),
               'matched_margin_median': float(np.median(margin[matched])) if count else None,
               'mean_peer_action_gap': float(peer_gap[matched].mean()) if count else None,
               'max_as_range': max((g['max_as_range'] for g in groups), default=None),
               'max_score_range': max((g['max_score_range'] for g in groups), default=None)}
    return summary, episodes, pd.DataFrame(groups, columns=group_columns)


def analyze(root, work):
    publication = json.loads(Path('published-results/iteration-v1/publication_manifest.json').read_text())
    records, episode_frames, group_frames, inputs = [], [], [], {}
    for seed in SEEDS:
        folder = root/f'seed_{seed}'
        manifest = verify_run(folder)
        digest = file_hash(folder/'run_manifest.json')
        require(digest == publication['files'][f'nonlinear/seed_{seed}/run_manifest.json']['original_sha256'],
                'Input differs from published original record')
        require(manifest['spec']['policy_seed'] == seed and manifest['spec']['seed_start'] == IDS[0]
                and manifest['spec']['episodes'] == len(IDS), 'Wrong input cohort')
        require(all(manifest['spec'][k] == v for k, v in provenance(COLLECTION_SOURCES).items()),
                'Collection code changed')
        inputs[str(seed)] = {'run_id': manifest['run_id'], 'manifest_sha256': digest}
        economic = pd.read_csv(folder/'episodes.csv').sort_values('episode').set_index('episode')
        with np.load(folder/'activations.npz', allow_pickle=False) as archive:
            data = {k: archive[k] for k in ('episode', 't', 'obs', 'history', 'action', 'posterior', 'targets')}
        d = audit_trajectory(data, economic)
        history = recent_histories(d['obs'])
        require(np.array_equal(history, d['history'][:, MIN_TIME:]), 'Saved history mismatch')
        scores = score_actions(d['posterior'][:, MIN_TIME:].reshape(-1, 12),
                               (d['obs'][:, MIN_TIME:, 7]*64).ravel())
        summary, episodes, groups = match_history(history.reshape(-1, 72),
                d['episode'][:, MIN_TIME:].ravel(), scores, d['targets'][:, MIN_TIME:, 4:].reshape(-1, 2))
        summary['seed'] = seed
        episodes.insert(0, 'seed', seed)
        groups.insert(0, 'seed', seed)
        records.append(summary); episode_frames.append(episodes); group_frames.append(groups)
        print(json.dumps(summary), flush=True)
    pd.concat(episode_frames, ignore_index=True).to_csv(work/'episodes.csv', index=False)
    pd.concat(group_frames, ignore_index=True).to_csv(work/'groups.csv', index=False)
    return {'protocol_sha256': file_hash(PROTOCOL), 'min_time': MIN_TIME, 'history_window': 8,
            'margin_tolerance': MARGIN_TOLERANCE, 'seeds': records, 'input_runs': inputs,
            'interpretation': 'Descriptive exact-match census of inspected trajectories. No-match decisions have unidentified older-history dependence. No causal-use or recoverable-return claim.',
            'audit': 'Original collection identity, full artifact hashes, public-history posterior and accounting replay, and saved recent-history inputs verified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/nonlinear_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/history_overlap_v1'))
    args = parser.parse_args()
    spec = {'stage': 'exact_recent_history_overlap', 'seeds': SEEDS,
            'input_manifest_sha256': {str(s): file_hash(args.root/f'seed_{s}'/'run_manifest.json') for s in SEEDS},
            **provenance(SOURCES)}
    with threadpool_limits(limits=1):
        run_job(args.output, spec, lambda work: analyze(args.root, work), 'summary.json',
                ('episodes.csv', 'groups.csv'))


if __name__ == '__main__':
    main()
