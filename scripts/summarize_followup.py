"""Summarize the predeclared larger-budget follow-up on its fresh test cohort.

python -m scripts.summarize_followup

Original pilot summaries and figures are never modified. All five seeds of
each long-budget policy and the fresh fixed-5 reference must be complete.
Short-checkpoint comparisons use only pilot_on_followup_evaluation, never the
original pilot's older evaluation cohort.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.summarize_results import (COLORS, LABELS, POLICIES, REFERENCES, SEEDS,
                                      _save_figure, objective_figure, paired_matrix,
                                      read_episodes, summarize_family, summarize_matrix)
from src.artifacts import source_hashes, write_json


SETTINGS = ("horizon", "inventory_penalty", "fee", "shift", "shift_control",
            "deterministic", "correct_filter")


def assert_paired(left, right, context="comparison"):
    """Require identical episodes and, when logged, identical exogenous regimes."""
    if not left.index.equals(right.index):
        raise ValueError(f"Episode IDs differ in {context}; cannot pair different test cohorts.")
    for column in ("value", "alpha", "p"):
        if column in left and column in right and not np.array_equal(left[column], right[column]):
            raise ValueError(f"Hidden regimes differ in {context}: {column}.")


def paired_difference(left, right, *, label, draws=4000, reference=False):
    """Subtract each seed's episode outcomes BEFORE either uncertainty calculation.

    A single fixed-policy reference is reused across training seeds. Averaging
    seed differences first keeps its episode noise from being counted five times.
    """
    if set(left) != set(SEEDS):
        raise ValueError(f"{label} requires all five left-hand training seeds.")
    if not reference and set(right) != set(SEEDS):
        raise ValueError(f"{label} requires all five right-hand training seeds.")
    differences = {}
    for seed in SEEDS:
        a, b = left[seed], right if reference else right[seed]
        assert_paired(a, b, f"{label}, seed {seed}")
        differences[seed] = pd.DataFrame({"objective": a.objective - b.objective}, index=a.index)
    result = summarize_matrix(paired_matrix(differences, "objective"), draws=draws)
    result["per_seed_difference_means"] = dict(zip(map(str, SEEDS), result.pop("per_seed_means")))
    result.update(paired_training_seeds=list(SEEDS), episodes_per_seed=len(left[SEEDS[0]]),
                  fixed_reference_reused_across_seeds=reference)
    return result


def _read_checked(path, expected_ids, settings):
    frame = read_episodes(path)
    if not frame.index.equals(expected_ids):
        raise ValueError(f"Wrong fresh test cohort in {path}; expected episode IDs "
                         f"{expected_ids[0]} through {expected_ids[-1]}.")
    summary_path = path.with_name("summary.json")
    if summary_path.exists():
        metadata = json.loads(summary_path.read_text(encoding="utf-8"))
        signature = {key: metadata.get(key, False if key == "shift_control" else None) for key in SETTINGS}
        if settings and signature != settings:
            raise ValueError(f"Evaluation settings differ in {summary_path}; a budget-only comparison is invalid.")
        settings.update(signature)
    return frame


def load_learned(folder, expected_ids, settings):
    """Require a complete balanced 4-family x 5-seed evaluation grid."""
    paths = {policy: {seed: folder / policy / f"seed_{seed}" / "episodes.csv" for seed in SEEDS}
             for policy in POLICIES}
    missing = [str(path) for by_seed in paths.values() for path in by_seed.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete five-seed evaluation grid: " + ", ".join(missing))
    frames = {policy: {seed: _read_checked(path, expected_ids, settings) for seed, path in by_seed.items()}
              for policy, by_seed in paths.items()}
    anchor = frames[POLICIES[0]][SEEDS[0]]
    for policy, by_seed in frames.items():
        for seed, frame in by_seed.items():
            assert_paired(anchor, frame, f"{folder.name}/{policy}/seed_{seed}")
    return frames, [str(path) for by_seed in paths.values() for path in by_seed.values()]


def training_metadata(folder):
    records = {}
    for policy in POLICIES:
        records[policy] = {}
        for seed in SEEDS:
            path = folder / policy / f"seed_{seed}" / "metadata.json"
            if path.exists():
                metadata = json.loads(path.read_text(encoding="utf-8"))
                records[policy][seed] = {key: metadata[key] for key in
                                        ("actual_steps", "requested_steps", "elapsed_seconds", "checkpoint")
                                        if key in metadata}
    return records


def behavior_summaries(folder):
    """Optional descriptive summaries; older checkpoint evaluations may omit them."""
    result = {}
    for policy in POLICIES:
        by_seed = {}
        for seed in SEEDS:
            path = folder / policy / f"seed_{seed}" / "summary.json"
            if path.exists():
                behavior = json.loads(path.read_text(encoding="utf-8")).get("behavior")
                if behavior:
                    by_seed[str(seed)] = behavior
        if by_seed:
            means = {}
            for name in ("mean_action_entropy_nats", "mean_within_time_probability_variance",
                         "greedy_action_fractions", "mean_action_probabilities", "per_time_probability_variance"):
                if all(name in value for value in by_seed.values()):
                    values = np.asarray([value[name] for value in by_seed.values()], dtype=float)
                    mean = values.mean(axis=0)
                    means[name] = float(mean) if mean.ndim == 0 else mean.tolist()
            result[policy] = {"training_seeds": [int(seed) for seed in by_seed],
                              "per_seed": by_seed, "equal_seed_mean": means,
                              "interpretation": "Descriptive only. Variation at fixed time excludes a pure time schedule or action-sampling noise, but can reflect inventory/current observations; it is not evidence of memory or causal use."}
    return result


def _budget_labels(metadata, fallback):
    counts = {int(record["actual_steps"]) for by_seed in metadata.values() for record in by_seed.values()
              if "actual_steps" in record}
    if len(counts) == 1:
        return f"{counts.pop():,} steps"
    if counts:
        return "Recorded budgets: " + ", ".join(f"{x:,}" for x in sorted(counts))
    return fallback


def budget_figure(long, short, long_metadata, short_metadata, folder):
    """Each line pairs the same training seed on exactly the same test episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), sharey=True)
    colors = ["#126A79", "#CB7540", "#7766A5", "#54884B", "#A85778"]
    short_label = _budget_labels(short_metadata, "Shorter-budget checkpoint")
    long_label = _budget_labels(long_metadata, "Larger-budget checkpoint")
    handles = []
    for ax, policy in zip(axes.ravel(), POLICIES):
        deltas = []
        for color, seed in zip(colors, SEEDS):
            a, b = short[policy][seed], long[policy][seed]
            assert_paired(a, b, f"budget figure {policy}/{seed}")
            means = [float(a.objective.mean()), float(b.objective.mean())]
            line, = ax.plot([0, 1], means, marker="o", color=color, lw=1.5,
                            alpha=.85, label=f"Seed {seed}")
            if policy == POLICIES[0]:
                handles.append(line)
            deltas.append(means[1] - means[0])
        ax.set_xticks([0, 1], [short_label, long_label])
        ax.set_xlim(-.14, 1.14)
        ax.set_title(f"{LABELS[policy]}\nMean paired gain: {np.mean(deltas):+.3f}", fontsize=11)
        ax.axhline(0, color="#69757D", lw=.8)
        ax.grid(axis="y", alpha=.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean objective on fresh test episodes")
    fig.suptitle("Training budget: matched seeds and the same fresh test cohort", fontsize=14)
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .04))
    fig.text(.5, .01, "Lines show seedwise paired means; uncertainty intervals are reported separately in followup_summary.json.",
             ha="center", fontsize=9, color="#515B63")
    fig.tight_layout(rect=(0, .09, 1, .95))
    return _save_figure(fig, folder, "paired_budget_gains")


def learning_figure(root, metadata, folder):
    curves = {}
    counts = []
    for policy in POLICIES:
        series = {}
        for seed in SEEDS:
            path = root / policy / f"seed_{seed}" / "progress.csv"
            record = metadata[policy].get(seed)
            if not path.exists() or not record or "actual_steps" not in record:
                continue
            frame = pd.read_csv(path)
            metric = "economics/objective_mean" if "economics/objective_mean" in frame else "rollout/ep_rew_mean"
            if metric not in frame or "time/total_timesteps" not in frame:
                continue
            values = frame[["time/total_timesteps", metric]].dropna().groupby("time/total_timesteps")[metric].last()
            if len(values) and values.index.max() > record["actual_steps"]:
                raise ValueError(f"Learning-curve steps exceed recorded completed budget in {path}.")
            if len(values):
                series[seed] = values
                counts.append(int(record["actual_steps"]))
        if series:
            curves[policy] = pd.DataFrame(series).sort_index()
    if not curves:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7), sharex=True, sharey=True)
    for ax, policy in zip(axes.ravel(), POLICIES):
        if policy in curves:
            values = curves[policy]
            for seed in values:
                ax.plot(values.index / 1000, values[seed].rolling(5, min_periods=1).mean(),
                        color=COLORS[policy], alpha=.20, lw=1)
            mean = values.mean(axis=1).rolling(5, min_periods=1).mean()
            ax.plot(values.index / 1000, mean, color=COLORS[policy], lw=2.2,
                    label=f"Mean of {len(values.columns)} seeds")
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(LABELS[policy], fontsize=11)
        ax.set_xlim(0, max(counts) / 1000 * 1.015)
        ax.axhline(0, color="#69757D", lw=.8)
        ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel("Environment transitions (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Training episode objective")
    fig.suptitle("Larger-budget learning curves: five-rollout moving means", fontsize=14)
    fig.text(.5, .01, "Light lines: individual seeds; dark lines: seed average. Horizontal scale uses recorded transitions and completed-run metadata.",
             ha="center", fontsize=9, color="#515B63")
    fig.tight_layout(rect=(0, .045, 1, .95))
    return _save_figure(fig, folder, "learning_curves")


def _csv_row(kind, name, metric, **extra):
    seed = metric["training_seed"]
    interval = seed["mean_t95_ci"] if seed else None
    episode_interval = metric["episode_bootstrap_95_ci"]
    return dict(kind=kind, name=name, mean=metric["mean"], episode_only_se=metric["episode_only_se"],
                episode_ci_low=episode_interval[0], episode_ci_high=episode_interval[1],
                training_seed_sd=seed["sample_sd"] if seed else None,
                training_seed_ci_low=interval[0] if interval else None,
                training_seed_ci_high=interval[1] if interval else None, **extra)


def run(results="results", figures="reports/followup_figures", draws=4000,
        seed_start=5_000_000, episodes=2048):
    if episodes < 2 or draws < 1:
        raise ValueError("Use at least two episodes and one bootstrap draw.")
    root, folder = Path(results), Path(figures)
    expected_ids = pd.Index(np.arange(seed_start, seed_start + episodes), name="episode")
    settings = {}
    long_folder = root / "followup_budget_evaluation"
    long, sources = load_learned(long_folder, expected_ids, settings)
    fixed_path = long_folder / "fixed_5" / "episodes.csv"
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fresh-cohort fixed-5 reference is required: {fixed_path}")
    policies = {policy: summarize_family(frames, draws=draws) for policy, frames in long.items()}
    references = {}
    for reference in REFERENCES:
        path = long_folder / reference / "episodes.csv"
        if path.exists():
            frame = _read_checked(path, expected_ids, settings)
            assert_paired(long["recurrent"][SEEDS[0]], frame, f"fresh reference {reference}")
            references[reference] = frame
            policies[reference] = summarize_family({"reference": frame}, trained=False, draws=draws)
            sources.append(str(path))
    comparisons = {f"recurrent_minus_{other}": paired_difference(long["recurrent"], long[other],
                   label=f"recurrent versus {other}", draws=draws)
                   for other in ("feedforward", "history", "belief")}
    comparisons["recurrent_minus_fixed_5"] = paired_difference(long["recurrent"], references["fixed_5"],
                                                              label="recurrent versus fixed-5", draws=draws,
                                                              reference=True)
    short_folder = root / "pilot_on_followup_evaluation"
    short, gains, short_summaries = {}, {}, {}
    if short_folder.exists():
        short, short_sources = load_learned(short_folder, expected_ids, settings)
        sources.extend(short_sources)
        for policy in POLICIES:
            gains[policy] = paired_difference(long[policy], short[policy], label=f"budget gain {policy}", draws=draws)
            short_summaries[policy] = summarize_family(short[policy], draws=draws)
    long_metadata = training_metadata(root / "followup_budget")
    short_metadata = training_metadata(root / "pilot")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#233641", "text.color": "#233641"})
    figure_paths = [objective_figure(policies, folder), learning_figure(root / "followup_budget", long_metadata, folder)]
    if short:
        figure_paths.append(budget_figure(long, short, long_metadata, short_metadata, folder))
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
               "source_hashes": source_hashes(), "bootstrap_draws": draws, "bootstrap_seed": 90821,
               "cohort": {"episodes_per_run": episodes, "seed_start": seed_start,
                          "seed_stop_exclusive": seed_start + episodes, "evaluation_settings": settings},
               "training_seeds": list(SEEDS), "long_training_metadata": long_metadata,
               "short_training_metadata": short_metadata, "policies": policies,
               "paired_comparisons": comparisons, "short_budget_on_fresh_cohort": short_summaries,
               "paired_budget_gains": gains,
               "behavior": behavior_summaries(long_folder),
               "short_budget_behavior": behavior_summaries(short_folder) if short else {},
               "budget_comparison_status": "complete_same_cohort" if short else "optional_short_budget_reevaluation_not_available",
               "uncertainty": {
                   "episode": "Average seed differences/values within each common episode, then estimate SE or percentile-bootstrap complete episodes. Conditions on fitted policies.",
                   "training_seed": "Sample SD and Student t4 95% CI across five independent training-seed means, conditional on this fixed evaluation cohort.",
                   "fixed_reference": "Reuse the identical fixed-5 episode outcomes for each training seed, and average seed differences before episode resampling.",
                   "budget": "Pair long and short checkpoints by training seed and exact fresh episode ID; original pilot evaluation results are not read.",
                   "limitations": "Separate conditional intervals are not a joint interval. Five-seed t intervals are approximate. No timestep or seed-by-episode pseudoreplication."},
               "figures": [path for path in figure_paths if path]}
    rows = [_csv_row("policy", policy, record["metrics"]["objective"], episodes_per_run=episodes,
                     training_seed_count=len(record["training_seeds"])) for policy, record in policies.items()]
    rows.extend(_csv_row("paired_policy_difference", name, metric, episodes_per_run=episodes,
                         training_seed_count=5) for name, metric in comparisons.items())
    rows.extend(_csv_row("paired_budget_gain", name, metric, episodes_per_run=episodes,
                         training_seed_count=5) for name, metric in gains.items())
    write_json(root / "followup_summary.json", summary)
    pd.DataFrame(rows).to_csv(root / "followup_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="reports/followup_figures")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed-start", type=int, default=5_000_000)
    parser.add_argument("--episodes", type=int, default=2048)
    summary = run(**vars(parser.parse_args()))
    print(json.dumps({"cohort": summary["cohort"], "policies": list(summary["policies"]),
                      "budget_comparison_status": summary["budget_comparison_status"],
                      "figures": summary["figures"]}, indent=2))


if __name__ == "__main__":
    main()
