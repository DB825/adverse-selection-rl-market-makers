"""Read-only comparison of the fixed-budget follow-up and original pilot.

Writes only the requested audit JSON. It never trains, changes checkpoints,
selects hyperparameters or chooses successful seeds. Rerun after all jobs finish
to replace a provisional snapshot with a complete audit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.artifacts import write_json


POLICIES = ("recurrent", "feedforward", "history", "belief")
CORE_SOURCES = ("src/train.py", "src/environment.py", "src/customer_model.py",
                "src/accounting.py", "src/bayes_filter.py", "src/wrappers.py",
                "src/fast_recurrent.py")
EXPECTED_DIFFERENCES = {"training.total_timesteps", "evaluation.episodes",
                        "evaluation.seed", "analysis.seed"}


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def differences(first, second):
    a, b = flatten(first), flatten(second)
    return {key: {"initial": a.get(key), "followup": b.get(key)}
            for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_progress(path):
    # A running logger may have an unfinished final line. The snapshot is
    # explicitly provisional and can be recomputed after training completes.
    raw = path.read_text(encoding="utf-8")
    frame = pd.read_csv(StringIO(raw), keep_default_na=False, on_bad_lines="skip")
    nonfinite = {}
    for column in frame:
        values = frame[column].astype(str).str.strip().str.lower()
        count = int(values.isin(["nan", "inf", "+inf", "-inf", "infinity", "-infinity"]).sum())
        if count:
            nonfinite[column] = count
    return frame.apply(pd.to_numeric, errors="coerce"), nonfinite


def compare_metrics(initial, followup, key, columns, maximum):
    first = initial.loc[initial[key].notna()].set_index(key)
    second = followup.loc[followup[key].notna()].set_index(key)
    first = first.loc[~first.index.duplicated(keep="last")]
    second = second.loc[~second.index.duplicated(keep="last")]
    common = first.index.intersection(second.index)
    common = common[common <= maximum]
    result = {}
    for column in columns:
        if column not in first or column not in second:
            continue
        a = first.loc[common, column].to_numpy(float)
        b = second.loc[common, column].to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        difference = np.abs(a[valid] - b[valid])
        result[column] = {
            "comparisons": int(valid.sum()),
            "finite_mask_mismatches": int((np.isfinite(a) != np.isfinite(b)).sum()),
            "max_absolute_difference": float(difference.max()) if len(difference) else None,
            "exactly_equal": bool(np.array_equal(a, b, equal_nan=True)),
        }
    return {"matched_rows": len(common),
            "largest_matched_index": int(common.max()) if len(common) else None,
            "columns": result,
            "all_compared_values_exactly_equal": all(v["exactly_equal"] for v in result.values())}


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        data = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(data.shape).encode())
        digest.update(str(data.dtype).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def initialization_audit(initial, followup, seeds):
    """Recreate initialized models only; no reset or learning is performed."""
    from src.train import make_model
    rows = []
    for seed in seeds:
        for policy in POLICIES:
            a = make_model(initial, policy, seed)
            initial_digest = parameter_digest(a)
            initial_streams = a.market_environment_seeds
            a.get_env().close()
            b = make_model(followup, policy, seed)
            followup_digest = parameter_digest(b)
            followup_streams = b.market_environment_seeds
            b.get_env().close()
            rows.append(dict(policy=policy, seed=seed, initial_parameters_sha256=initial_digest,
                             followup_parameters_sha256=followup_digest,
                             parameters_identical=initial_digest == followup_digest,
                             environment_streams_identical=initial_streams == followup_streams))
    return rows


def audit(initial_config="configs/pilot.yaml", followup_config="configs/followup_budget.yaml",
          initial_root="results/pilot", followup_root="results/followup_budget",
          output="results/followup_audit.json", check_initialization=False):
    initial = yaml.safe_load(Path(initial_config).read_text(encoding="utf-8"))
    followup = yaml.safe_load(Path(followup_config).read_text(encoding="utf-8"))
    changed = differences(initial, followup)
    seeds = followup["training"]["seeds"]
    n_envs = followup["training"]["n_envs"]
    streams = {str(seed): list(range(seed * 10000, seed * 10000 + n_envs)) for seed in seeds}
    flat_streams = sum(streams.values(), [])
    reserved = {
        f"{name}_{domain}": [conf[domain]["seed"], conf[domain]["seed"] + conf[domain]["episodes"]]
        for name, conf in (("initial", initial), ("followup", followup))
        for domain in ("evaluation", "analysis")
    }
    reserved_overlap = []
    for i, (name, (start, stop)) in enumerate(reserved.items()):
        for other, (a, b) in list(reserved.items())[i + 1:]:
            if start < b and stop > a:
                reserved_overlap.append([name, other])
    initial_steps = initial["training"]["total_timesteps"]
    rollout_size = initial["training"]["n_envs"] * initial["training"]["n_steps"]
    max_updates = (initial_steps // rollout_size) * initial["training"]["n_epochs"]
    rows = []
    for seed in seeds:
        for policy in POLICIES:
            a = Path(initial_root) / policy / f"seed_{seed}"
            b = Path(followup_root) / policy / f"seed_{seed}"
            row = dict(policy=policy, seed=seed, followup_started=(b / "progress.csv").exists(),
                       followup_completed=(b / "metadata.json").exists())
            for name, folder in (("initial", a), ("followup", b)):
                if (folder / "model.zip").exists():
                    row[f"{name}_checkpoint_sha256"] = sha256(folder / "model.zip")
                if (folder / "metadata.json").exists():
                    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                    row[f"{name}_recorded_environment_seeds"] = meta["environment_seeds"]
                    row[f"{name}_actual_steps"] = meta["actual_steps"]
            if (a / "metadata.json").exists() and (b / "metadata.json").exists():
                am = json.loads((a / "metadata.json").read_text(encoding="utf-8"))
                bm = json.loads((b / "metadata.json").read_text(encoding="utf-8"))
                row["core_source_hash_matches"] = {key: am["source_hashes"].get(key) == bm["source_hashes"].get(key)
                                                   for key in CORE_SOURCES}
            if (b / "config.json").exists():
                effective = json.loads((b / "config.json").read_text(encoding="utf-8"))
                row["effective_followup_config_matches_requested"] = not differences(followup, effective)
            if row["followup_started"]:
                old, nonfinite_old = _read_progress(a / "progress.csv")
                new, nonfinite_new = _read_progress(b / "progress.csv")
                row["initial_nonfinite_logged_values"] = nonfinite_old
                row["followup_nonfinite_logged_values"] = nonfinite_new
                available = new["time/total_timesteps"].dropna()
                row["followup_logged_steps"] = int(available.max()) if len(available) else 0
                economics = [c for c in old if c.startswith(("rollout/", "economics/"))]
                optimization = [c for c in old if c.startswith("train/") and c != "train/n_updates"]
                row["rollout_prefix"] = compare_metrics(old, new, "time/total_timesteps", economics, initial_steps)
                if "train/n_updates" in new:
                    row["optimizer_prefix"] = compare_metrics(old, new, "train/n_updates", optimization, max_updates)
                row["latest_training_metrics"] = {
                    c: (float(new[c].dropna().iloc[-1]) if np.isfinite(new[c].dropna().iloc[-1]) else None)
                    for c in optimization
                    if c in new and len(new[c].dropna())
                }
            rows.append(row)
    payload = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": sum(row["followup_completed"] for row in rows),
        "expected_runs": len(rows),
        "config_differences": changed,
        "unexpected_config_difference_keys": sorted(set(changed) - EXPECTED_DIFFERENCES),
        "training_streams": streams,
        "training_seeds_disjoint_across_seed_replicates": len(flat_streams) == len(set(flat_streams)),
        "training_streams_do_not_overlap_evaluation_or_analysis": not any(
            start <= seed < stop for seed in flat_streams for start, stop in reserved.values()),
        "reserved_seed_domains_start_inclusive_stop_exclusive": reserved,
        "reserved_seed_domain_overlaps": reserved_overlap,
        "runs": rows,
        "interpretation": (
            "This is a paired training-budget extension from the same initialization and simulator seeds, "
            "not an independent experiment. It replays the initial training prefix. Equal logged economics "
            "and optimizer diagnostics support reproducibility but do not directly prove every unlogged "
            "observation/action or checkpoint parameter at 100352 steps was identical. Fresh evaluation "
            "episodes assess the fixed final checkpoint; they do not erase the exploratory choice to extend "
            "training after the initial pilot. No success-based seed or checkpoint selection is performed."
        ),
    }
    if check_initialization:
        payload["recreated_initialization_comparison"] = initialization_audit(initial, followup, seeds)
    elif Path(output).exists():
        previous = json.loads(Path(output).read_text(encoding="utf-8"))
        if previous.get("config_differences") == changed and "recreated_initialization_comparison" in previous:
            payload["recreated_initialization_comparison"] = previous["recreated_initialization_comparison"]
            payload["initialization_comparison_preserved_from_previous_audit"] = True
    write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-config", default="configs/pilot.yaml")
    parser.add_argument("--followup-config", default="configs/followup_budget.yaml")
    parser.add_argument("--initial-root", default="results/pilot")
    parser.add_argument("--followup-root", default="results/followup_budget")
    parser.add_argument("--output", default="results/followup_audit.json")
    parser.add_argument("--check-initialization", action="store_true")
    result = audit(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in
                      ("completed_runs", "expected_runs", "config_differences",
                       "unexpected_config_difference_keys", "training_seeds_disjoint_across_seed_replicates",
                       "training_streams_do_not_overlap_evaluation_or_analysis")}))


if __name__ == "__main__":
    main()
