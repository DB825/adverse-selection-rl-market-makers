"""Verify complete fixed-budget evidence before writing its archival manifest."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from src.artifacts import write_json


POLICIES = ("recurrent", "feedforward", "history", "belief")
SEEDS = (11, 12, 13, 14, 15)
INITIAL_STEPS, FOLLOWUP_STEPS = 100352, 300032
EPISODES, TEST_SEED = 2048, 5_000_000
CORE_SOURCES = ("src/train.py", "src/environment.py", "src/customer_model.py",
                "src/accounting.py", "src/bayes_filter.py", "src/wrappers.py", "src/fast_recurrent.py")
EVALUATION_SETTINGS = dict(horizon=64, inventory_penalty=.001, fee=0., deterministic=False,
                           shift=False, shift_control=False, correct_filter=False)
CONFIG_DIFFERENCES = {
    "analysis.seed": {"initial": 3000000, "followup": 6000000},
    "evaluation.episodes": {"initial": 1024, "followup": 2048},
    "evaluation.seed": {"initial": 1000000, "followup": 5000000},
    "training.total_timesteps": {"initial": INITIAL_STEPS, "followup": FOLLOWUP_STEPS},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(path, project):
    result = Path(path)
    return (result if result.is_absolute() else project / result).resolve()


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        result.update(flatten(value, name) if isinstance(value, dict) else {name: value})
    return result


def keyed_grid(rows, description):
    keys = [(row.get("policy"), row.get("seed")) for row in rows]
    expected = {(policy, seed) for policy in POLICIES for seed in SEEDS}
    require(len(keys) == len(expected) and set(keys) == expected,
            f"{description} requires each of the 20 policy/seed pairs exactly once")
    return dict(zip(keys, rows))


def check_junit(path):
    root = ET.parse(path).getroot()
    cases, suites = list(root.iter("testcase")), list(root.iter("testsuite"))
    require(cases and suites, "JUnit report contains no tests")
    require(all(int(s.get("failures", 0)) == 0 and int(s.get("errors", 0)) == 0 for s in suites),
            "JUnit report records failures or errors")
    require(not any(c.find("failure") is not None or c.find("error") is not None for c in cases),
            "JUnit test cases contain failures or errors")
    require(any(c.get("classname", "").endswith("test_followup_manifest") and c.find("skipped") is None for c in cases),
            "JUnit report is stale: it does not include the follow-up manifest guard tests")
    skipped = sum(c.find("skipped") is not None for c in cases)
    return dict(test_cases=len(cases), passed=len(cases) - skipped, skipped=skipped,
                failures=0, errors=0, sha256=sha256(path))


def check_audit(audit):
    require(audit.get("completed_runs") == 20 and audit.get("expected_runs") == 20,
            "Follow-up audit is incomplete; rerun it after all 20 jobs finish")
    require(audit.get("config_differences") == CONFIG_DIFFERENCES and
            audit.get("unexpected_config_difference_keys") == [], "Audited configuration differences are invalid")
    require(audit.get("training_seeds_disjoint_across_seed_replicates") is True and
            audit.get("training_streams_do_not_overlap_evaluation_or_analysis") is True and
            audit.get("reserved_seed_domain_overlaps") == [], "Audit seed-domain checks failed")
    initializations = keyed_grid(audit.get("recreated_initialization_comparison", []), "Initialization audit")
    for row in initializations.values():
        require(row.get("parameters_identical") is True and row.get("environment_streams_identical") is True
                and row.get("initial_parameters_sha256")
                and row.get("initial_parameters_sha256") == row.get("followup_parameters_sha256"),
                "Initialization audit does not establish a matched training prefix")
    rows = keyed_grid(audit.get("runs", []), "Follow-up audit")
    for (policy, seed), row in rows.items():
        context = f"audit {policy}/{seed}"
        require(row.get("followup_completed") is True and row.get("initial_actual_steps") == INITIAL_STEPS
                and row.get("followup_actual_steps") == FOLLOWUP_STEPS, f"Incomplete budgets in {context}")
        require(row.get("effective_followup_config_matches_requested") is True, f"Config mismatch in {context}")
        require(row.get("initial_nonfinite_logged_values") == {} and row.get("followup_nonfinite_logged_values") == {},
                f"Nonfinite training diagnostics in {context}")
        require(all(row.get("core_source_hash_matches", {}).get(key) is True for key in CORE_SOURCES),
                f"Training core source changed in {context}")
        for key, maximum in (("rollout_prefix", INITIAL_STEPS), ("optimizer_prefix", 980)):
            prefix = row.get(key, {})
            require(prefix.get("matched_rows") == 98 and prefix.get("largest_matched_index") == maximum
                    and prefix.get("all_compared_values_exactly_equal") is True,
                    f"Incomplete or unequal {key} in {context}")
            columns = prefix.get("columns", {})
            require(columns and all(c.get("exactly_equal") is True and c.get("finite_mask_mismatches") == 0
                                    and c.get("comparisons") == 98 for c in columns.values()),
                    f"Incomplete metric comparison for {key} in {context}")
    return rows


def check_episodes(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == EPISODES, f"Wrong episode row count in {path}")
    require([int(row["episode"]) for row in rows] == list(range(TEST_SEED, TEST_SEED + EPISODES)),
            f"Wrong or duplicated episode IDs in {path}")
    values = [float(row["objective"]) for row in rows]
    regimes = [tuple(float(row[key]) for key in ("value", "alpha", "p")) for row in rows]
    require(all(math.isfinite(x) for x in values) and all(math.isfinite(x) for r in regimes for x in r),
            f"Nonfinite episode data in {path}")
    return regimes, math.fsum(values) / len(values)


def check_run(folder, evaluation, policy, seed, budget, config, project):
    metadata, summary = read_json(folder / "metadata.json"), read_json(evaluation / "summary.json")
    checkpoint = (folder / "model.zip").resolve()
    require(metadata.get("policy") == policy and metadata.get("seed") == seed,
            f"Training checkpoint policy/seed identity mismatch: {folder}")
    require(metadata.get("actual_steps") == budget and metadata.get("requested_steps") == budget,
            f"Training budget mismatch: {folder}")
    require(canonical(metadata["checkpoint"], project) == checkpoint, f"Metadata checkpoint identity mismatch: {folder}")
    require(metadata.get("environment_seeds") == list(range(seed * 10000, seed * 10000 + 8)),
            f"Training stream identity mismatch: {folder}")
    require(read_json(folder / "config.json") == config, f"Saved training configuration mismatch: {folder}")
    require(summary.get("policy") == policy and canonical(summary["checkpoint"], project) == checkpoint,
            f"Evaluation checkpoint identity mismatch: {evaluation}")
    require(summary.get("episodes") == EPISODES and summary.get("seed_start") == TEST_SEED
            and summary.get("seed_stop_exclusive") == TEST_SEED + EPISODES,
            f"Evaluation cohort mismatch: {evaluation}")
    require(all(summary.get(key) == value for key, value in EVALUATION_SETTINGS.items()),
            f"Evaluation protocol mismatch: {evaluation}")
    regimes, mean = check_episodes(evaluation / "episodes.csv")
    require(math.isclose(mean, summary["objective_mean"], rel_tol=1e-10, abs_tol=1e-10),
            f"Evaluation summary does not match episode CSV: {evaluation}")
    hashes = {"checkpoint": sha256(checkpoint), "metadata": sha256(folder / "metadata.json"),
              "config": sha256(folder / "config.json"), "evaluation": sha256(evaluation / "summary.json"),
              "episodes": sha256(evaluation / "episodes.csv")}
    record = dict(policy=policy, seed=seed, actual_steps=budget,
                  training_seconds=metadata["elapsed_seconds"], environment_seeds=metadata["environment_seeds"],
                  checkpoint=str(checkpoint), checkpoint_sha256=hashes["checkpoint"],
                  metadata=str(folder / "metadata.json"), evaluation=str(evaluation / "summary.json"),
                  episodes=str(evaluation / "episodes.csv"), episodes_sha256=hashes["episodes"], artifact_sha256=hashes)
    return record, regimes, metadata


def build_manifest(results="results", project_root="."):
    project = Path(project_root).resolve()
    root = canonical(results, project)
    configs = {name: yaml.safe_load((project / "configs" / filename).read_text(encoding="utf-8"))
               for name, filename in (("initial", "pilot.yaml"), ("followup", "followup_budget.yaml"))}
    flat = {name: flatten(config) for name, config in configs.items()}
    differences = {key: {"initial": flat["initial"].get(key), "followup": flat["followup"].get(key)}
                   for key in flat["initial"].keys() | flat["followup"].keys()
                   if flat["initial"].get(key) != flat["followup"].get(key)}
    require(differences == CONFIG_DIFFERENCES, "Current configs differ from the fixed budget-only protocol")
    audited = check_audit(read_json(root / "followup_audit.json"))
    junit = check_junit(root / "followup_tests.xml")
    summary = read_json(root / "followup_summary.json")
    require(summary.get("budget_comparison_status") == "complete_same_cohort",
            "Follow-up summary does not include the complete same-cohort short-budget comparison")
    cohort = summary.get("cohort", {})
    require(cohort.get("episodes_per_run") == EPISODES and cohort.get("seed_start") == TEST_SEED
            and cohort.get("seed_stop_exclusive") == TEST_SEED + EPISODES, "Follow-up summary has the wrong fresh cohort")
    require(cohort.get("evaluation_settings") == EVALUATION_SETTINGS, "Follow-up summary has the wrong evaluation protocol")
    require(summary.get("training_seeds") == list(SEEDS) and
            set(summary.get("paired_budget_gains", {})) == set(POLICIES) and
            set(summary.get("short_budget_on_fresh_cohort", {})) == set(POLICIES),
            "Follow-up summary omits training seeds or paired budget comparisons")
    summary_sources = {canonical(path, project) for path in summary.get("sources", [])}
    runs, short_runs, anchor_regimes = [], [], None
    for policy in POLICIES:
        for seed in SEEDS:
            records, metadata = {}, {}
            for label, train_dir, eval_dir, budget in (
                ("initial", "pilot", "pilot_on_followup_evaluation", INITIAL_STEPS),
                ("followup", "followup_budget", "followup_budget_evaluation", FOLLOWUP_STEPS),
            ):
                folder, evaluation = root / train_dir / policy / f"seed_{seed}", root / eval_dir / policy / f"seed_{seed}"
                record, regimes, meta = check_run(folder, evaluation, policy, seed, budget, configs[label], project)
                require((evaluation / "episodes.csv").resolve() in summary_sources, f"Summary omits episode source: {evaluation}")
                if anchor_regimes is None:
                    anchor_regimes = regimes
                require(regimes == anchor_regimes, f"Evaluation exogenous regimes differ: {evaluation}")
                require(audited[policy, seed].get(f"{label}_checkpoint_sha256") == record["checkpoint_sha256"],
                        f"Audit checkpoint hash is stale: {folder}")
                require(audited[policy, seed].get(f"{label}_recorded_environment_seeds") == meta["environment_seeds"],
                        f"Audit training streams are stale: {folder}")
                records[label], metadata[label] = record, meta
            require(metadata["initial"].get("architecture") == metadata["followup"].get("architecture")
                    and metadata["initial"].get("parameter_count") == metadata["followup"].get("parameter_count"),
                    f"Policy architecture changed: {policy}/{seed}")
            require(all(key in metadata["initial"]["source_hashes"] and key in metadata["followup"]["source_hashes"]
                        and metadata["initial"]["source_hashes"][key] == metadata["followup"]["source_hashes"][key]
                        for key in CORE_SOURCES), f"Training core hash changed: {policy}/{seed}")
            runs.append(records["followup"])
            short_runs.append(records["initial"])
    evidence = {str(root / name): sha256(root / name)
                for name in ("followup_summary.json", "followup_audit.json", "followup_tests.xml")}
    sources = {str(path.relative_to(project)).replace("\\", "/"): sha256(path)
               for folder in ("src", "configs", "scripts", "tests") for path in sorted((project / folder).glob("**/*"))
               if path.is_file() and path.suffix in {".py", ".yaml"}}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=sources,
                  dependency_lock_sha256=sha256(project / "requirements-lock.txt"),
                  completed_runs=runs, same_cohort_short_budget_runs=short_runs,
                  completed_followup_transitions=sum(r["actual_steps"] for r in runs),
                  transitions_beyond_repeated_initial_prefix=sum(r["actual_steps"] - INITIAL_STEPS for r in runs),
                  summed_training_process_seconds=sum(r["training_seconds"] for r in runs), evidence_hashes=evidence, junit=junit,
                  interpretation="All final checkpoints at a predeclared 300032-step budget; same training seeds/prefix as initial pilot. Both budgets evaluated on the same fresh 2048 episodes starting at 5m. No independent replication or convergence claim.",
                  provenance_limit="Paths, current checkpoint hashes and paired audit hashes are verified. Evaluation files did not record checkpoint content hashes when generated; the manifest cannot independently prove a same-path checkpoint was never replaced before the final audit.")
    write_json(root / "followup_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--project-root", default=".")
    result = build_manifest(**vars(parser.parse_args()))
    print(f"Manifest saved for {len(result['completed_runs'])} long-budget and "
          f"{len(result['same_cohort_short_budget_runs'])} short-budget evaluation pairs; "
          f"{result['junit']['passed']} tests passed, {result['junit']['skipped']} skipped")


if __name__=="__main__":
    main()
