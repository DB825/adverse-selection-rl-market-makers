"""An archival manifest must refuse partial, mismatched or failing evidence."""
import json
from pathlib import Path

import pytest
import yaml

from scripts import make_followup_manifest as manifest


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def edit(path, mutate):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    save(path, value)


@pytest.fixture
def evidence(tmp_path):
    project, root = tmp_path, tmp_path / "results"
    root.mkdir()
    (project / "configs").mkdir()
    (project / "requirements-lock.txt").write_text("synthetic fixture\n", encoding="utf-8")
    configs = {}
    repository = Path(__file__).resolve().parents[1]
    for label, filename in (("initial", "pilot.yaml"), ("followup", "followup_budget.yaml")):
        text = (repository / "configs" / filename).read_text(encoding="utf-8")
        (project / "configs" / filename).write_text(text, encoding="utf-8")
        configs[label] = yaml.safe_load(text)
    episodes = "episode,value,alpha,p,objective\n" + "".join(
        f"{episode},0,0.05,0.5,1.0\n" for episode in range(manifest.TEST_SEED, manifest.TEST_SEED + manifest.EPISODES))
    sources, audit_rows, initializations = [], [], []
    for policy in manifest.POLICIES:
        for seed in manifest.SEEDS:
            row = dict(policy=policy, seed=seed, followup_completed=True,
                       initial_actual_steps=manifest.INITIAL_STEPS, followup_actual_steps=manifest.FOLLOWUP_STEPS,
                       effective_followup_config_matches_requested=True, initial_nonfinite_logged_values={},
                       followup_nonfinite_logged_values={}, core_source_hash_matches={key: True for key in manifest.CORE_SOURCES})
            for key, maximum in (("rollout_prefix", manifest.INITIAL_STEPS), ("optimizer_prefix", 980)):
                row[key] = dict(matched_rows=98, largest_matched_index=maximum, all_compared_values_exactly_equal=True,
                                columns={"metric": dict(exactly_equal=True, finite_mask_mismatches=0, comparisons=98)})
            for label, train_dir, eval_dir, steps in (
                ("initial", "pilot", "pilot_on_followup_evaluation", manifest.INITIAL_STEPS),
                ("followup", "followup_budget", "followup_budget_evaluation", manifest.FOLLOWUP_STEPS),
            ):
                folder, evaluation = root / train_dir / policy / f"seed_{seed}", root / eval_dir / policy / f"seed_{seed}"
                folder.mkdir(parents=True)
                evaluation.mkdir(parents=True)
                checkpoint = folder / "model.zip"
                checkpoint.write_bytes(f"synthetic checkpoint {label}/{policy}/{seed}".encode())
                streams = list(range(seed * 10000, seed * 10000 + 8))
                metadata = dict(policy=policy, seed=seed, actual_steps=steps, requested_steps=steps,
                                checkpoint=str(checkpoint), environment_seeds=streams, elapsed_seconds=1.,
                                architecture={"name": policy}, parameter_count=50,
                                source_hashes={key: "same-core-hash" for key in manifest.CORE_SOURCES})
                save(folder / "metadata.json", metadata)
                save(folder / "config.json", configs[label])
                summary = dict(policy=policy, checkpoint=str(checkpoint), episodes=manifest.EPISODES,
                               seed_start=manifest.TEST_SEED, seed_stop_exclusive=manifest.TEST_SEED + manifest.EPISODES,
                               objective_mean=1., **manifest.EVALUATION_SETTINGS)
                save(evaluation / "summary.json", summary)
                (evaluation / "episodes.csv").write_text(episodes, encoding="utf-8")
                sources.append(str(evaluation / "episodes.csv"))
                row[f"{label}_checkpoint_sha256"] = manifest.sha256(checkpoint)
                row[f"{label}_recorded_environment_seeds"] = streams
            audit_rows.append(row)
            initializations.append(dict(policy=policy, seed=seed, parameters_identical=True,
                                        environment_streams_identical=True, initial_parameters_sha256="same",
                                        followup_parameters_sha256="same"))
    audit = dict(completed_runs=20, expected_runs=20, config_differences=manifest.CONFIG_DIFFERENCES,
                 unexpected_config_difference_keys=[], training_seeds_disjoint_across_seed_replicates=True,
                 training_streams_do_not_overlap_evaluation_or_analysis=True, reserved_seed_domain_overlaps=[],
                 runs=audit_rows, recreated_initialization_comparison=initializations)
    save(root / "followup_audit.json", audit)
    save(root / "followup_summary.json", dict(
        budget_comparison_status="complete_same_cohort", training_seeds=list(manifest.SEEDS), sources=sources,
        cohort=dict(episodes_per_run=manifest.EPISODES, seed_start=manifest.TEST_SEED,
                    seed_stop_exclusive=manifest.TEST_SEED + manifest.EPISODES,
                    evaluation_settings=manifest.EVALUATION_SETTINGS),
        paired_budget_gains={policy: {} for policy in manifest.POLICIES},
        short_budget_on_fresh_cohort={policy: {} for policy in manifest.POLICIES}))
    (root / "followup_tests.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0">'
        '<testcase classname="tests.test_followup_manifest" name="fixture"/>'
        '</testsuite></testsuites>', encoding="utf-8")
    return project, root


def run(evidence):
    project, root = evidence
    return manifest.build_manifest(results=root, project_root=project)


def test_complete_manifest_binds_both_budgets_and_raw_evidence(evidence):
    result = run(evidence)
    assert len(result["completed_runs"]) == len(result["same_cohort_short_budget_runs"]) == 20
    assert result["completed_followup_transitions"] == 6000640
    assert result["transitions_beyond_repeated_initial_prefix"] == 3993600
    assert result["junit"]["failures"] == result["junit"]["errors"] == 0
    assert "configs/followup_budget.yaml" in result["source_hashes"]
    for row in result["completed_runs"] + result["same_cohort_short_budget_runs"]:
        assert set(row["artifact_sha256"]) == {"checkpoint", "metadata", "config", "evaluation", "episodes"}


def test_manifest_refuses_missing_short_budget_evaluation(evidence):
    _, root = evidence
    (root / "pilot_on_followup_evaluation/recurrent/seed_11/summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        run(evidence)
    assert not (root / "followup_manifest.json").exists()


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(completed_runs=19), "incomplete"),
    (lambda d: d["runs"][0]["rollout_prefix"].update(matched_rows=97), "rollout_prefix"),
    (lambda d: d["runs"][0]["optimizer_prefix"].update(all_compared_values_exactly_equal=False), "optimizer_prefix"),
    (lambda d: d["runs"][0].update(followup_nonfinite_logged_values={"train/loss": 1}), "Nonfinite"),
])
def test_manifest_refuses_incomplete_or_failed_audit(evidence, change, match):
    _, root = evidence
    edit(root / "followup_audit.json", change)
    with pytest.raises(ValueError, match=match):
        run(evidence)


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(checkpoint="wrong-model.zip"), "checkpoint identity"),
    (lambda d: d.update(deterministic=True), "protocol"),
    (lambda d: d.update(shift=True), "protocol"),
    (lambda d: d.update(objective_mean=2.), "episode CSV"),
])
def test_manifest_refuses_stale_or_changed_evaluation(evidence, change, match):
    _, root = evidence
    edit(root / "followup_budget_evaluation/recurrent/seed_11/summary.json", change)
    with pytest.raises(ValueError, match=match):
        run(evidence)


def test_manifest_refuses_changed_checkpoint_content(evidence):
    _, root = evidence
    (root / "followup_budget/recurrent/seed_11/model.zip").write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="Audit checkpoint hash is stale"):
        run(evidence)


def test_manifest_refuses_wrong_training_seed_identity(evidence):
    _, root = evidence
    edit(root / "pilot/recurrent/seed_11/metadata.json", lambda d: d.update(seed=12))
    with pytest.raises(ValueError, match="policy/seed identity"):
        run(evidence)


def test_manifest_refuses_wrong_raw_episode_cohort(evidence):
    _, root = evidence
    path = root / "pilot_on_followup_evaluation/recurrent/seed_11/episodes.csv"
    path.write_text(path.read_text().replace("5000000,", "1000000,", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="episode IDs"):
        run(evidence)


@pytest.mark.parametrize("xml,match", [
    ('<testsuites><testsuite failures="1"><testcase classname="tests.test_followup_manifest"/></testsuite></testsuites>', "failures"),
    ('<testsuites><testsuite><testcase classname="tests.test_followup_manifest"><error/></testcase></testsuite></testsuites>', "failures"),
    ('<testsuites><testsuite><testcase classname="tests.test_old"/></testsuite></testsuites>', "stale"),
])
def test_manifest_refuses_failed_or_stale_junit(evidence, xml, match):
    _, root = evidence
    (root / "followup_tests.xml").write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        run(evidence)


def test_manifest_refuses_summary_without_same_cohort_budget_comparison(evidence):
    _, root = evidence
    edit(root / "followup_summary.json", lambda d: d.update(budget_comparison_status="optional_short_budget_reevaluation_not_available"))
    with pytest.raises(ValueError, match="same-cohort"):
        run(evidence)
