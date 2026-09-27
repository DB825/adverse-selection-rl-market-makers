"""Fresh-cohort pairing and budget effects must preserve the sampling units."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import summarize_followup as followup


def frame(offset=0, ids=None):
    ids = np.arange(5_000_000, 5_000_006) if ids is None else np.asarray(ids)
    values = np.arange(len(ids), dtype=float) * 10 + offset
    return pd.DataFrame({"episode": ids, "objective": values, "profit": values,
                         "penalty": np.zeros(len(ids)), "mean_abs_inventory": np.zeros(len(ids)),
                         "value": np.zeros(len(ids)), "alpha": np.full(len(ids), .05),
                         "p": np.full(len(ids), .5)}, index=pd.Index(ids, name="episode"))


def write_grid(root, folder, extra=0, with_settings=False):
    for policy in followup.POLICIES:
        for seed in followup.SEEDS:
            path = root / folder / policy / f"seed_{seed}"
            path.mkdir(parents=True)
            frame(seed - 10 + extra).to_csv(path / "episodes.csv", index=False)
            if with_settings:
                settings = dict(horizon=64, inventory_penalty=.001, fee=0., shift=False,
                                deterministic=False, correct_filter=False)
                (path / "summary.json").write_text(json.dumps(settings), encoding="utf-8")


def test_fixed_reference_is_paired_before_seed_averaging():
    learned = {seed: frame(seed - 10) for seed in followup.SEEDS}
    result = followup.paired_difference(learned, frame(), label="test", reference=True, draws=100)
    assert result["mean"] == 3
    assert result["episode_only_se"] == 0
    assert result["episode_bootstrap_95_ci"] == [3, 3]
    assert result["training_seed"]["sample_sd"] == pytest.approx(np.std(np.arange(1, 6), ddof=1))
    assert result["training_seed"]["degrees_of_freedom"] == 4
    assert result["fixed_reference_reused_across_seeds"]


def test_budget_pairing_rejects_different_cohorts_and_regimes():
    long = {seed: frame(3) for seed in followup.SEEDS}
    short = {seed: frame() for seed in followup.SEEDS}
    short[15] = frame(ids=np.arange(1_000_000, 1_000_006))
    with pytest.raises(ValueError, match="Episode IDs differ"):
        followup.paired_difference(long, short, label="budget", draws=10)
    short[15] = frame()
    short[15]["p"] = .9
    with pytest.raises(ValueError, match="Hidden regimes differ"):
        followup.paired_difference(long, short, label="budget", draws=10)


def test_all_five_seeds_required_in_each_pair():
    learned = {seed: frame() for seed in followup.SEEDS[:-1]}
    with pytest.raises(ValueError, match="all five"):
        followup.paired_difference(learned, frame(), label="missing", reference=True, draws=10)


def test_missing_followup_cannot_generate_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Incomplete five-seed"):
        followup.run(results=tmp_path, figures=tmp_path / "figures", episodes=6, draws=10)
    assert not (tmp_path / "followup_summary.json").exists()


def test_complete_followup_pairs_short_runs_and_preserves_pilot(tmp_path, monkeypatch):
    write_grid(tmp_path, "followup_budget_evaluation", extra=2)
    write_grid(tmp_path, "pilot_on_followup_evaluation")
    baseline = tmp_path / "followup_budget_evaluation" / "fixed_5"
    baseline.mkdir()
    frame().to_csv(baseline / "episodes.csv", index=False)
    sentinel = tmp_path / "pilot_summary.json"
    sentinel.write_text("initial pilot stays intact", encoding="utf-8")
    monkeypatch.setattr(followup, "objective_figure", lambda *args: None)
    monkeypatch.setattr(followup, "learning_figure", lambda *args: None)
    monkeypatch.setattr(followup, "budget_figure", lambda *args: None)
    result = followup.run(results=tmp_path, figures=tmp_path / "figures", episodes=6, draws=20)
    assert result["budget_comparison_status"] == "complete_same_cohort"
    for policy in followup.POLICIES:
        gain = result["paired_budget_gains"][policy]
        assert gain["mean"] == 2
        assert gain["episode_bootstrap_95_ci"] == [2, 2]
        assert gain["training_seed"]["mean_t95_ci"] == [2, 2]
    assert result["paired_comparisons"]["recurrent_minus_fixed_5"]["mean"] == 5
    assert sentinel.read_text(encoding="utf-8") == "initial pilot stays intact"
    assert (tmp_path / "followup_summary.csv").exists()


def test_short_checkpoint_old_cohort_is_rejected(tmp_path):
    write_grid(tmp_path, "pilot_on_followup_evaluation")
    bad = tmp_path / "pilot_on_followup_evaluation" / "recurrent" / "seed_11" / "episodes.csv"
    frame(ids=np.arange(1_000_000, 1_000_006)).to_csv(bad, index=False)
    expected = pd.Index(np.arange(5_000_000, 5_000_006), name="episode")
    with pytest.raises(ValueError, match="Wrong fresh test cohort"):
        followup.load_learned(tmp_path / "pilot_on_followup_evaluation", expected, {})


def test_changed_evaluation_objective_is_rejected(tmp_path):
    write_grid(tmp_path, "followup_budget_evaluation", with_settings=True)
    path = tmp_path / "followup_budget_evaluation" / "belief" / "seed_15" / "summary.json"
    settings = json.loads(path.read_text())
    settings["inventory_penalty"] = .01
    path.write_text(json.dumps(settings))
    expected = pd.Index(np.arange(5_000_000, 5_000_006), name="episode")
    with pytest.raises(ValueError, match="Evaluation settings differ"):
        followup.load_learned(tmp_path / "followup_budget_evaluation", expected, {})


def test_behavior_metrics_are_optional_descriptive_and_seedwise(tmp_path):
    assert followup.behavior_summaries(tmp_path) == {}
    for seed, entropy in ((11, .1), (12, .3)):
        folder = tmp_path / "recurrent" / f"seed_{seed}"
        folder.mkdir(parents=True)
        behavior = dict(mean_action_entropy_nats=entropy, mean_within_time_probability_variance=.02,
                        mean_action_probabilities=[.1] * 10)
        (folder / "summary.json").write_text(json.dumps({"behavior": behavior}))
    result = followup.behavior_summaries(tmp_path)["recurrent"]
    assert result["training_seeds"] == [11, 12]
    assert result["equal_seed_mean"]["mean_action_entropy_nats"] == pytest.approx(.2)
    assert "not evidence of memory" in result["interpretation"]
