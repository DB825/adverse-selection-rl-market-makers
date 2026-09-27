"""Construct supported histories that an eight-arrival memory cannot distinguish.

This module deliberately evaluates likelihoods independently using only the
standard library; the accompanying tests cross-check the production filter.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

REGIMES = tuple(itertools.product((0, 1), (0.05, 0.30), (0.25, 0.50, 0.75)))
QUOTES = tuple((c - h, c + h) for c in (0.35, 0.50, 0.65) for h in (0.10, 0.20, 0.30))
ABSTAIN = 9


def likelihood(action: int, outcome: int, regime: tuple) -> float:
    """Outcome 0/1/2 denotes customer buy/sell/no trade."""
    if action == ABSTAIN:
        return float(outcome == 2)
    value, alpha, preference = regime
    bid, ask = QUOTES[action]
    buy = alpha * value + (1 - alpha) * preference * (1 - ask)
    sell = alpha * (1 - value) + (1 - alpha) * (1 - preference) * bid
    return (buy, sell, 1 - buy - sell)[outcome]


def posterior_from_history(history: list) -> tuple[list[float], float]:
    logs = [-math.log(len(REGIMES)) + sum(math.log(likelihood(a, o, s)) for a, o in history) for s in REGIMES]
    shift = max(logs)
    unnorm = [math.exp(x - shift) for x in logs]
    normalizer = sum(unnorm)
    return [x / normalizer for x in unnorm], shift + math.log(normalizer)


def summarize(posterior: list[float], inventory: int, risk_lambda: float = 0.001, horizon: int = 64) -> dict:
    value, alpha, preference = [sum(w * s[j] for w, s in zip(posterior, REGIMES)) for j in range(3)]
    actions = []
    for a, (bid, ask) in enumerate(QUOTES):
        buys = [w * likelihood(a, 0, s) for w, s in zip(posterior, REGIMES)]
        sells = [w * likelihood(a, 1, s) for w, s in zip(posterior, REGIMES)]
        pb, ps = sum(buys), sum(sells)
        vb = sum(w * s[0] for w, s in zip(buys, REGIMES)) / pb
        vs = sum(w * s[0] for w, s in zip(sells, REGIMES)) / ps
        profit = pb * (ask - vb) + ps * (vs - bid)
        risk = risk_lambda / horizon * (pb * (-2 * inventory + 1) + ps * (2 * inventory + 1))
        actions.append(dict(action=a, bid=bid, ask=ask, p_buy=pb, p_sell=ps, value_buy=vb, value_sell=vs,
                            as_buy=vb - value, as_sell=value - vs, expected_profit=profit, myopic_score=profit - risk))
    actions.append(dict(action=ABSTAIN, expected_profit=0., myopic_score=0.))
    best = max(actions, key=lambda x: x['myopic_score'])
    return dict(value=value, alpha=alpha, p=preference, inventory=inventory, best_action=best['action'],
                posterior=posterior, reference_quote_as_buy=actions[5]['as_buy'],
                reference_quote_as_sell=actions[5]['as_sell'], action_statistics=actions)


def search(seed: int = 23817, candidates: int = 12000, p_tolerance: float | None = .005,
           minimum_decision_margin: float = .001) -> dict:
    rng = random.Random(seed)
    prefix_outcomes = [0, 2, 0, 2, 1, 2, 0, 2, 1, 2, 2, 2, 0, 2, 1, 2]
    suffix = [(5, x) for x in [2, 0, 2, 1, 2, 0, 2, 2]]
    inventory = sum(1 if x == 1 else -1 if x == 0 else 0 for x in prefix_outcomes + [o for _, o in suffix])
    buckets = {}
    best_pair = None
    best_contrast = -1.
    for _ in range(candidates):
        history = [(rng.randrange(9), o) for o in prefix_outcomes] + suffix
        posterior, log_support = posterior_from_history(history)
        summary = summarize(posterior, inventory)
        ordered_scores = sorted((x['myopic_score'] for x in summary['action_statistics']), reverse=True)
        summary['decision_margin'] = ordered_scores[0] - ordered_scores[1]
        if summary['decision_margin'] < minimum_decision_margin:
            continue
        summary.update(history=history, log_marginal_likelihood=log_support)
        bucket = round(summary['value'] / .01)
        for b in range(bucket - 1, bucket + 2):
            for other in buckets.get(b, []):
                if summary['best_action'] == other['best_action'] or abs(summary['value'] - other['value']) > .02:
                    continue
                if p_tolerance is not None and abs(summary['p'] - other['p']) > p_tolerance:
                    continue
                contrast = abs(summary['alpha'] - other['alpha'])
                if contrast > best_contrast:
                    best_contrast, best_pair = contrast, [summary, other]
        existing = buckets.setdefault(bucket, [])
        # Keep extrema by informedness and decision, bounding search cost.
        existing.append(summary)
        if len(existing) > 24:
            ordered = sorted(existing, key=lambda x: x['alpha'])
            buckets[bucket] = ordered[:12] + ordered[-12:]
    if best_pair is None:
        raise RuntimeError('No pair found; report this failure rather than inventing one.')
    for member in best_pair:
        history = member['history']
        log_predictives = [posterior_from_history(history[:n])[1] for n in range(len(history) + 1)]
        member['minimum_predictive_event_probability'] = math.exp(min(b - a for a, b in zip(log_predictives, log_predictives[1:])))
        member['marginal_history_probability_given_actions'] = math.exp(member['log_marginal_likelihood'])
        member['minimum_event_likelihood_across_regimes'] = min(likelihood(a, o, s) for a, o in history for s in REGIMES)
    return dict(seed=seed, candidates=candidates, horizon=64, lambda_inventory=.001,
                history_length=len(best_pair[0]['history']), common_suffix_length=len(suffix),
                same_entire_outcome_sequence=True, value_tolerance=.02, p_tolerance=p_tolerance,
                minimum_decision_margin=minimum_decision_margin,
                regimes=REGIMES, quotes=QUOTES, pair=best_pair)


def fixed_policy_expectations(horizon: int = 64, risk_lambdas=(0., .001, .01), fee: float = 0.) -> list[dict]:
    """Exact iid-increments calculation conditional on each fixed regime.

    For q_0=0 and X=delta inventory, E[q_t^2]=t E[X^2]+t(t-1) E[X]^2.
    Average these conditional moments over the joint prior, rather than using
    unconditional iid increments (the hidden regime couples episode outcomes).
    """
    rows = []
    for action in range(10):
        per_regime = []
        bid, ask = (0., 0.) if action == ABSTAIN else QUOTES[action]
        for regime in REGIMES:
            pb, ps = likelihood(action, 0, regime), likelihood(action, 1, regime)
            mean_increment, second_increment = ps - pb, ps + pb
            summed_q2 = (second_increment * horizon * (horizon + 1) / 2
                         + mean_increment ** 2 * horizon * (horizon + 1) * (horizon - 1) / 3)
            profit = horizon * (pb * (ask - regime[0] - fee) + ps * (regime[0] - bid - fee))
            per_regime.append((profit, summed_q2))
        profit = sum(x[0] for x in per_regime) / len(REGIMES)
        mean_q2 = sum(x[1] for x in per_regime) / (horizon * len(REGIMES))
        rows.append(dict(action=action, terminal_profit=profit, mean_inventory_squared=mean_q2,
                         objective_by_lambda={str(lam): profit - lam * mean_q2 for lam in risk_lambdas}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=23817)
    parser.add_argument('--candidates', type=int, default=12000)
    parser.add_argument('--p-tolerance', type=float, default=.005)
    parser.add_argument('--minimum-decision-margin', type=float, default=.001)
    parser.add_argument('--output', type=Path, default=Path('results/gate1/decision_pairs.json'))
    args = parser.parse_args()
    result = search(args.seed, args.candidates, args.p_tolerance, args.minimum_decision_margin)
    result['fixed_policy_expectations'] = fixed_policy_expectations()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'pair': [{k: s[k] for k in ('value', 'alpha', 'p', 'best_action', 'inventory')} for s in result['pair']]}, indent=2))


if __name__ == '__main__':
    main()
