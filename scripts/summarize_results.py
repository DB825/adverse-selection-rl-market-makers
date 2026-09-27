"""Aggregate a completed pilot without treating timesteps or seeds as iid episodes.

Run from the repository root: python -m scripts.summarize_results
Missing runs are recorded explicitly. Inconsistent episode pairing raises an
error instead of silently discarding evaluation cases. Figures use only files
that exist; the script never generates synthetic research results.
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
from scipy.stats import t as student_t

from src.artifacts import source_hashes, write_json
from src.customer_model import QUOTES


POLICIES = ("recurrent", "feedforward", "history", "belief")
REFERENCES = ("fixed_2", "fixed_4", "fixed_5", "fixed_8", "fixed_9", "myopic")
SEEDS = (11, 12, 13, 14, 15)
LABELS = {"recurrent": "Recurrent PPO", "feedforward": "Current-observation PPO",
          "history": "History-8 PPO", "belief": "Exact-belief PPO",
          "myopic": "Bayesian myopic", "fixed_2": "Fixed (0.05, 0.65)",
          "fixed_4": "Fixed (0.30, 0.70)", "fixed_5": "Fixed (0.20, 0.80)",
          "fixed_8": "Fixed (0.35, 0.95)", "fixed_9": "Always abstain"}
COLORS = {"recurrent": "#126A79", "feedforward": "#CB7540", "history": "#7766A5",
          "belief": "#54884B"}
METRICS = ("objective", "profit", "penalty", "mean_q2", "mean_abs_inventory",
           "terminal_inventory", "buy_rate", "sell_rate", "abstain_rate",
           "mean_spread", "mean_center", "inferred_value", "inferred_alpha",
           "correct_value", "correct_alpha")
TARGETS = ("value", "alpha", "p", "entropy", "as_buy_ref", "as_sell_ref")


def bootstrap_episode_mean(values, draws=4000, seed=90821):
    """Percentile CI resampling complete episode averages, not individual seeds."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires at least two finite episode values.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 128):
        stop = min(draws, start + 128)
        idx = rng.integers(0, len(values), (stop - start, len(values)))
        estimates[start:stop] = values[idx].mean(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def training_seed_statistics(means):
    means = np.asarray(means, dtype=float)
    n = len(means)
    result = {"n": n, "mean": float(means.mean()), "sample_sd": None,
              "mean_t95_ci": None, "degrees_of_freedom": None}
    if n >= 2:
        sd = float(means.std(ddof=1))
        margin = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n))
        result.update(sample_sd=sd, mean_t95_ci=[result["mean"] - margin,
                                              result["mean"] + margin],
                      degrees_of_freedom=n - 1)
    return result


def summarize_matrix(matrix, *, trained=True, draws=4000):
    """Rows are training seeds, columns are the SAME held-out episode IDs."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite seed-by-episode matrix with >=2 episodes.")
    per_episode = matrix.mean(axis=0)
    result = {"mean": float(per_episode.mean()),
              "episode_only_se": float(per_episode.std(ddof=1) / np.sqrt(len(per_episode))),
              "episode_bootstrap_95_ci": bootstrap_episode_mean(per_episode, draws=draws),
              "per_seed_means": matrix.mean(axis=1).tolist() if trained else None,
              "training_seed": training_seed_statistics(matrix.mean(axis=1)) if trained else None}
    return result


def read_episodes(path):
    frame = pd.read_csv(path).sort_values("episode").set_index("episode", drop=False)
    if len(frame) < 2 or not frame.index.is_unique:
        raise ValueError(f"Need >=2 distinct whole episodes: {path}")
    for column in ["objective", "profit", "penalty"]:
        if column not in frame or not np.isfinite(frame[column]).all():
            raise ValueError(f"Missing/nonfinite required metric {column}: {path}")
    return frame


def paired_matrix(frames, metric):
    first = next(iter(frames.values())).index
    for seed, frame in frames.items():
        if not first.equals(frame.index):
            raise ValueError(f"Episode IDs differ for seed {seed}; pairing would be invalid.")
    return np.stack([frame[metric].to_numpy(float) for frame in frames.values()])


def summarize_family(frames, *, trained=True, draws=4000):
    seeds = list(frames)
    first = next(iter(frames.values()))
    metrics = {name: summarize_matrix(paired_matrix(frames, name), trained=trained, draws=draws)
               for name in METRICS if all(name in frame for frame in frames.values())}
    counts = [f"action_{i}" for i in range(10)]
    fractions = None
    if all(set(counts).issubset(frame.columns) for frame in frames.values()):
        fractions = np.mean([frame[counts].sum().to_numpy(float) / frame[counts].sum().sum()
                             for frame in frames.values()], axis=0).tolist()
    pooled = paired_matrix(frames, "objective").ravel()
    p05 = float(np.quantile(pooled, .05))
    result = {"training_seeds": seeds if trained else [], "episodes_per_run": len(first),
              "episode_seed_min": int(first.index.min()), "episode_seed_max": int(first.index.max()),
              "metrics": metrics, "action_fractions": fractions,
              "descriptive_pooled_policy_episode_p05": p05,
              "descriptive_pooled_policy_episode_lower5_mean": float(pooled[pooled <= p05].mean())}
    for side in ("informed", "uninformed"):
        columns = [side + "_profit", side + "_exec"]
        if all(set(columns).issubset(frame.columns) for frame in frames.values()):
            total_profit = sum(frame[columns[0]].sum() for frame in frames.values())
            executions = sum(frame[columns[1]].sum() for frame in frames.values())
            result[side + "_profit_per_execution"] = float(total_profit / executions) if executions else None
    # Each regime mean is averaged across seeds after grouping whole episodes.
    if all(set(["value", "alpha", "p"]).issubset(frame.columns) for frame in frames.values()):
        grouped = [frame.groupby(["value", "alpha", "p"])[["objective", "profit", "mean_abs_inventory"]].mean()
                   for frame in frames.values()]
        result["by_regime"] = pd.concat(grouped).groupby(level=[0, 1, 2]).mean().reset_index().to_dict("records")
    return result


def compare_recurrent(learned, draws=4000):
    comparisons = {}
    if "recurrent" not in learned:
        return comparisons
    for other in ("feedforward", "history", "belief"):
        if other not in learned:
            continue
        seeds = sorted(set(learned["recurrent"]) & set(learned[other]))
        differences = {}
        for seed in seeds:
            rec, baseline = learned["recurrent"][seed], learned[other][seed]
            if not rec.index.equals(baseline.index):
                raise ValueError(f"Unpaired episode IDs for recurrent versus {other}, seed {seed}.")
            differences[seed] = pd.DataFrame({"objective": rec.objective - baseline.objective}, index=rec.index)
        if differences:
            result = summarize_matrix(paired_matrix(differences, "objective"), draws=draws)
            result["paired_training_seeds"] = seeds
            result["episodes_per_seed"] = len(next(iter(differences.values())))
            result["per_seed_difference_means"] = dict(zip(map(str, seeds), result.pop("per_seed_means")))
            comparisons[f"recurrent_minus_{other}"] = result
    return comparisons


def load_probe_summaries(root):
    payloads = {}
    for seed in SEEDS:
        path = root / "probes" / f"seed_{seed}" / "probes.json"
        if path.exists():
            payloads[seed] = json.loads(path.read_text(encoding="utf-8"))
    if not payloads:
        return {}
    first = next(iter(payloads.values()))
    result = {"training_seeds": list(payloads), "sources": [str(root / "probes" / f"seed_{s}" / "probes.json") for s in payloads],
              "r2": {}, "added_state": {},
              "note": "Seed-level summaries of fixed fitted probes; do not combine their episode CIs as independent estimates."}
    for feature in first["results"]:
        result["r2"][feature] = {}
        for target in TARGETS:
            values = [p["results"][feature]["test_r2"][target] for p in payloads.values()]
            result["r2"][feature][target] = {**training_seed_statistics(values), "per_seed": values}
    for comparator in first["added_state_vs"]:
        result["added_state"][comparator] = {}
        for target in TARGETS:
            records = [p["added_state_vs"][comparator][target] for p in payloads.values()]
            result["added_state"][comparator][target] = {
                "relative_mse_reduction": training_seed_statistics([x["relative_mse_reduction"] for x in records]),
                "per_seed_mse_reduction": [x["mse_reduction"] for x in records],
                "per_seed_episode_bootstrap_95_ci": [x["episode_bootstrap_95_ci"] for x in records]}
    return result


def _save_figure(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(folder / f"{name}.png")


def objective_figure(policies, folder):
    names = [name for name in (*POLICIES, *REFERENCES) if name in policies]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.3, len(names) * .48)), sharex=True, sharey=True)
    for i, name in enumerate(names):
        metric = policies[name]["metrics"]["objective"]
        mean = metric["mean"]
        seed = metric["training_seed"]
        for ax, interval, color in [(axes[0], seed["mean_t95_ci"] if seed else None, "#126A79"),
                                    (axes[1], metric["episode_bootstrap_95_ci"], "#CB7540")]:
            if interval:
                error = np.abs(np.array(interval) - mean).reshape(2, 1)
                ax.errorbar([mean], [i], xerr=error, fmt="o", capsize=4, color=color, zorder=3)
            else:
                ax.plot(mean, i, "o", color="#7E8790", markersize=5)
        if seed:
            axes[0].scatter(metric["per_seed_means"], np.full(seed["n"], i),
                            color="#126A79", alpha=.30, s=20, zorder=2)
    axes[0].set_yticks(range(len(names)), [LABELS.get(n, n) for n in names])
    axes[0].invert_yaxis()
    axes[0].set_title("95% t interval across training seeds\nConditional on the evaluation episodes", fontsize=11)
    axes[1].set_title("95% bootstrap interval across episodes\nConditional on the fitted policies", fontsize=11)
    for ax in axes:
        ax.axvline(0, color="#69757D", lw=.8)
        ax.grid(axis="x", alpha=.20)
        ax.set_xlabel("Mean episode objective")
    fig.suptitle("Economic performance: two distinct sources of uncertainty", fontsize=15, y=1.02)
    fig.text(.50, -.01, "Gray dots: reference policies with no training-seed uncertainty. Faint dots: individual training-seed means.\nIntervals are separate conditional summaries; neither is a joint uncertainty interval.",
             ha="center", va="top", fontsize=9, color="#515B63")
    fig.tight_layout()
    return _save_figure(fig, folder, "objective_uncertainty")


def learning_figure(root, folder):
    curves = {}
    for policy in POLICIES:
        values = {}
        for seed in SEEDS:
            path = root / "pilot" / policy / f"seed_{seed}" / "progress.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            metric = "economics/objective_mean" if "economics/objective_mean" in frame else "rollout/ep_rew_mean"
            if metric not in frame or "time/total_timesteps" not in frame:
                continue
            usable = frame[["time/total_timesteps", metric]].dropna().groupby("time/total_timesteps")[metric].last()
            if len(usable):
                values[seed] = usable
        if values:
            curves[policy] = pd.DataFrame(values).sort_index()
    if not curves:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, policy in zip(axes.ravel(), POLICIES):
        if policy not in curves:
            ax.text(.5, .5, "No completed rollout logs", transform=ax.transAxes, ha="center")
        else:
            frame = curves[policy]
            for seed in frame:
                ax.plot(frame.index / 1000, frame[seed].rolling(5, min_periods=1).mean(),
                        color=COLORS[policy], alpha=.23, lw=1)
            mean = frame.mean(axis=1).rolling(5, min_periods=1).mean()
            ax.plot(frame.index / 1000, mean, color=COLORS[policy], lw=2.3,
                    label=f"Mean of {len(frame.columns)} seeds")
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(LABELS[policy], fontsize=11)
        ax.axhline(0, color="#69757D", lw=.8)
        ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel("Environment transitions (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Training episode objective")
    fig.suptitle("Learning curves: five-rollout moving means", fontsize=15)
    fig.text(.5, .01, "Light lines are individual training seeds; dark lines average available seeds. Training data, not held-out test results.",
             ha="center", fontsize=9, color="#515B63")
    fig.tight_layout(rect=(0, .04, 1, .96))
    return _save_figure(fig, folder, "learning_curves")


def action_figure(policies, folder):
    names = [n for n in (*POLICIES, "myopic") if n in policies and policies[n]["action_fractions"] is not None]
    if not names:
        return None
    values = np.array([policies[n]["action_fractions"] for n in names])
    fig, ax = plt.subplots(figsize=(12, max(3.4, .65 * len(names))))
    im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=max(.1, float(values.max())))
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    labels = [f"{i}\n({b:.2f}, {a:.2f})" for i, (b, a) in enumerate(QUOTES)] + ["9\nAbstain"]
    ax.set_xticks(range(10), labels, fontsize=8)
    for i in range(len(names)):
        for j in range(10):
            ax.text(j, i, f"{100 * values[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > .57 * values.max() else "#162C3A")
    ax.set_xlabel("Action: (bid, ask); cells show percent of arrivals")
    ax.set_title("Quote choices on held-out episodes", fontsize=15, pad=15)
    fig.colorbar(im, ax=ax, shrink=.8, label="Fraction of arrivals")
    fig.text(.5, -.01, "Each episode and each available training seed has equal weight. Action 5 is the widest centered quote.",
             ha="center", fontsize=9, color="#515B63")
    fig.tight_layout()
    return _save_figure(fig, folder, "action_distribution")


def probe_figure(probes, folder):
    if not probes:
        return None
    selected = ["basic", "history8", "hidden_cell", "history8_hidden_cell"]
    selected = [x for x in selected if x in probes["r2"]]
    labels = {"basic": "Basic observables", "history8": "History-8", "hidden_cell": "Hidden + cell", "history8_hidden_cell": "History-8 + state"}
    target_labels = ["Expected V", "Expected alpha", "Expected p", "Entropy", "Buy AS\n(ref. quote)", "Sell AS\n(ref. quote)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    matrix = np.array([[probes["r2"][feature][t]["mean"] for t in TARGETS] for feature in selected])
    im = axes[0].imshow(matrix, aspect="auto", cmap="RdYlBu", vmin=min(0, matrix.min()), vmax=1)
    axes[0].set_xticks(range(6), target_labels, fontsize=8)
    axes[0].set_yticks(range(len(selected)), [labels[f] for f in selected], fontsize=9)
    for i in range(len(selected)):
        for j in range(6):
            axes[0].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    axes[0].set_title("Mean held-out R² across policy seeds", fontsize=11)
    fig.colorbar(im, ax=axes[0], fraction=.035, pad=.03)
    comparator = probes["added_state"].get("history8", {})
    for j, target in enumerate(TARGETS):
        if target not in comparator:
            continue
        stats = comparator[target]["relative_mse_reduction"]
        ci = stats["mean_t95_ci"]
        errors = np.abs(np.array(ci) - stats["mean"]).reshape(2, 1) * 100 if ci else None
        axes[1].errorbar([stats["mean"] * 100], [j], xerr=errors, fmt="o", capsize=4, color="#126A79")
    axes[1].set_yticks(range(6), target_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#69757D", lw=.8)
    axes[1].grid(axis="x", alpha=.15)
    axes[1].set_xlabel("MSE reduction from adding state (%)")
    axes[1].set_title("Added state versus History-8\n95% t interval across policy seeds", fontsize=11)
    fig.suptitle("Linear decoding on entirely held-out episodes", fontsize=15, y=1.03)
    fig.text(.5, -.035, "AS uses the fixed (0.20, 0.80) quote. Conditional seed intervals omit joint episode/probe-fit uncertainty.\nDecoding is predictive association; it establishes neither causal use nor information beyond nonlinear recent-history features.",
             ha="center", fontsize=9, color="#515B63")
    fig.tight_layout()
    return _save_figure(fig, folder, "belief_probes")


def gate1_figure(root, folder):
    """Plot a supported inference/decision contrast, never a learned-policy claim."""
    path = root / "gate1" / "decision_pairs.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    pair = sorted(data["pair"], key=lambda member: member["alpha"], reverse=True)
    if len(pair) != 2:
        raise ValueError("Gate 1 figure requires exactly two histories.")
    suffix = data["common_suffix_length"]
    histories = [member["history"] for member in pair]
    outcomes = [[outcome for _, outcome in history] for history in histories]
    if (len(histories[0]) != len(histories[1]) or histories[0][-suffix:] != histories[1][-suffix:]
            or outcomes[0] != outcomes[1] or pair[0]["inventory"] != pair[1]["inventory"]):
        raise ValueError("Gate 1 file does not support the matched-observation caption.")
    names = ["History A: higher informedness", "History B: lower informedness"]
    colors = ["#126A79", "#CB7540"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.05, 1, 1.6]})
    x = np.arange(3)
    for j, (member, name, color) in enumerate(zip(pair, names, colors)):
        means = [member[key] for key in ("value", "alpha", "p")]
        positions = x + (j - .5) * .35
        axes[0].bar(positions, means, width=.32, color=color, alpha=.9, label=name)
        for position, mean in zip(positions, means):
            axes[0].text(position, mean + .014, f"{mean:.3f}", ha="center", fontsize=8)
        action = member["best_action"]
        bid, ask = data["quotes"][action]
        axes[1].plot([bid, ask], [1-j, 1-j], color=color, lw=4, solid_capstyle="round")
        axes[1].scatter([bid, ask], [1-j, 1-j], color=color, s=36, zorder=3)
        axes[1].scatter(member["value"], 1-j, color=color, s=70, marker="|", zorder=4)
        axes[1].text((bid + ask)/2, 1-j + .18, f"Action {action}: ({bid:.2f}, {ask:.2f})",
                     ha="center", color=color, fontsize=9)
        stats = member["action_statistics"]
        scores = np.array([entry["myopic_score"] for entry in stats])
        axes[2].plot(range(10), scores, marker="o", markersize=3.5, color=color, lw=1.5)
        axes[2].scatter(action, scores[action], marker="*", color=color, s=150, zorder=5)
        axes[2].annotate(f"Best: {scores[action]:.4f}", (action, scores[action]),
                         xytext=(-6 if j == 0 else -20, 13 if j == 0 else 17),
                         textcoords="offset points", ha="right" if j == 0 else "center",
                         fontsize=9, color=color)
    axes[0].set_xticks(x, ["E[V]", "E[alpha]", "E[p]"])
    axes[0].set_ylim(0, .73)
    axes[0].set_ylabel("Exact posterior mean")
    axes[0].set_title("Nearly matched value and preference", fontsize=10)
    axes[0].grid(axis="y", alpha=.15)
    axes[1].set_yticks([1, 0], ["A", "B"])
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-.35, 1.55)
    axes[1].set_xlabel("Price; tick marks posterior E[V]")
    axes[1].set_title("Selected myopic quotes", fontsize=10)
    axes[1].grid(axis="x", alpha=.15)
    axes[2].set_xticks(range(10), [str(i) for i in range(9)] + ["9\nAbstain"])
    axes[2].axhline(0, color="#69757D", lw=.8)
    axes[2].set_xlabel("Quote action (star = best score)")
    axes[2].set_ylabel("One-step myopic score")
    axes[2].set_title("Different best actions under the full posterior", fontsize=10)
    lower, upper = axes[2].get_ylim()
    axes[2].set_ylim(lower, upper + .012)
    axes[2].grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .87),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Earlier quote evidence changes the Bayesian decision", fontsize=15, y=1.05)
    fig.text(.5, .96,
             f"Identical last {suffix} observations, inventory {pair[0]['inventory']}, and time {len(histories[0])}/{data['horizon']}; "
             "the entire outcome sequence matches, but earlier quotes differ.",
             ha="center", fontsize=10)
    fig.text(.5, -.035,
             f"Score = expected profit minus incremental inventory cost. Supported example from {data['candidates']:,} searched candidates.\n"
             "Constructed histories do not establish frequent occurrence or learned inference.\n"
             "Joint posterior dependencies also differ, so this comparison does not isolate the causal effect of informedness.",
             ha="center", fontsize=9, color="#515B63")
    fig.tight_layout(rect=(0, .025, 1, .84), w_pad=2)
    return _save_figure(fig, folder, "gate1_decision_relevance")


def run(results="results", figures="reports/figures", draws=4000):
    root, folder = Path(results), Path(figures)
    policies, learned, sources, missing = {}, {}, [], []
    for policy in POLICIES:
        frames = {}
        for seed in SEEDS:
            path = root / "evaluation" / policy / f"seed_{seed}" / "episodes.csv"
            if path.exists():
                frames[seed] = read_episodes(path)
                sources.append(str(path))
            else:
                missing.append(str(path))
        if frames:
            learned[policy] = frames
            policies[policy] = summarize_family(frames, draws=draws)
    for policy in REFERENCES:
        path = root / "evaluation" / policy / "episodes.csv"
        if path.exists():
            policies[policy] = summarize_family({"reference": read_episodes(path)}, trained=False, draws=draws)
            sources.append(str(path))
        else:
            missing.append(str(path))
    if not policies:
        raise FileNotFoundError("No completed evaluation episodes.csv files; run evaluation before aggregation.")
    shift_groups = {}
    for path in sorted((root / "shift").glob("**/episodes.csv")):
        relative = path.relative_to(root / "shift")
        if path.parent.name.startswith("seed_"):
            name = str(relative.parent.parent).replace("\\", "/")
            seed = int(path.parent.name.split("_")[-1])
        else:
            name, seed = str(relative.parent).replace("\\", "/"), "reference"
        shift_groups.setdefault(name, {})[seed] = read_episodes(path)
        sources.append(str(path))
    shift = {name: summarize_family(frames, trained="reference" not in frames, draws=draws)
             for name, frames in shift_groups.items()}
    probes = load_probe_summaries(root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#233641", "text.color": "#233641"})
    figure_paths = [objective_figure(policies, folder), learning_figure(root, folder),
                    action_figure(policies, folder),
                    probe_figure(probes, folder) if probes else gate1_figure(root, folder)]
    comparisons = compare_recurrent(learned, draws=draws)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "sources": sources, "missing_expected_evaluation_files": missing,
               "source_hashes": source_hashes(), "bootstrap_draws": draws, "bootstrap_seed": 90821,
               "uncertainty": {
                   "point_estimate": "Mean over common evaluation episodes after averaging equally across available training seeds.",
                   "episode": "SE/bootstrap resamples whole episode averages across fixed fitted policies; common episode randomness is retained across seeds.",
                   "training_seed": "Sample SD and Student t 95% CI across independent training-seed means conditional on the same evaluation episode set; df=4 for five seeds.",
                   "paired_comparisons": "Subtract policies at the same training seed and episode before either seed or episode aggregation.",
                   "limits": "Separate conditional intervals are not a joint interval. With only five training seeds, t intervals are approximate. No timestep independence or n_seeds*n_episodes iid assumption."},
               "policies": policies, "paired_comparisons": comparisons, "probes": probes,
               "probe_status": "completed_pilot_probes" if probes else
                               "not_run_optional_gated_analysis; smoke-only pipeline checks excluded",
               "shift": shift, "figures": [p for p in figure_paths if p]}
    write_json(root / "pilot_summary.json", summary)
    rows = []
    for policy, result in policies.items():
        metric = result["metrics"]["objective"]
        seed = metric["training_seed"]
        rows.append({"policy": policy, "training_seed_count": len(result["training_seeds"]),
                     "episodes_per_run": result["episodes_per_run"], "objective_mean": metric["mean"],
                     "episode_only_se": metric["episode_only_se"],
                     "episode_ci_low": metric["episode_bootstrap_95_ci"][0],
                     "episode_ci_high": metric["episode_bootstrap_95_ci"][1],
                     "training_seed_sd": seed["sample_sd"] if seed else None,
                     "training_seed_ci_low": seed["mean_t95_ci"][0] if seed and seed["mean_t95_ci"] else None,
                     "training_seed_ci_high": seed["mean_t95_ci"][1] if seed and seed["mean_t95_ci"] else None,
                     **{m + "_mean": result["metrics"][m]["mean"] for m in
                        ("profit", "penalty", "mean_abs_inventory", "abstain_rate") if m in result["metrics"]}})
    pd.DataFrame(rows).to_csv(root / "pilot_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--draws", type=int, default=4000)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"policies": list(result["policies"]), "figures": result["figures"],
                      "missing_evaluations": len(result["missing_expected_evaluation_files"])}, indent=2))


if __name__ == "__main__":
    main()
