"""Guard the two distinct sampling units used in research reporting."""

import numpy as np
import pandas as pd
import pytest

from scripts.summarize_results import compare_recurrent, paired_matrix, run, summarize_matrix


def test_episode_and_training_seed_uncertainty_are_separate():
    episodes = np.array([-2., -1., 1., 2.])
    seed_offsets = np.arange(-2., 3.)
    matrix = seed_offsets[:, None] + episodes[None, :]
    summary = summarize_matrix(matrix, draws=1000)
    assert summary["mean"] == 0
    assert summary["episode_only_se"] == pytest.approx(episodes.std(ddof=1) / 2)
    assert summary["training_seed"]["sample_sd"] == pytest.approx(seed_offsets.std(ddof=1))
    assert summary["training_seed"]["degrees_of_freedom"] == 4
    margin = 2.7764451051977987 * seed_offsets.std(ddof=1) / np.sqrt(5)
    np.testing.assert_allclose(summary["training_seed"]["mean_t95_ci"], [-margin, margin])
    # Repeating the same fixed policies cannot create more independent episodes.
    repeated = summarize_matrix(np.repeat(matrix, 2, axis=0), draws=1000)
    assert repeated["episode_only_se"] == summary["episode_only_se"]
    assert repeated["episode_bootstrap_95_ci"] == summary["episode_bootstrap_95_ci"]


def test_pair_before_aggregation_retains_common_episode_randomness():
    learned = {"recurrent": {}, "feedforward": {}}
    for seed in range(11, 16):
        common = np.arange(10, dtype=float) * 100
        learned["recurrent"][seed] = pd.DataFrame({"objective": common + seed - 10})
        learned["feedforward"][seed] = pd.DataFrame({"objective": common})
    result = compare_recurrent(learned, draws=100)["recurrent_minus_feedforward"]
    assert result["mean"] == 3
    assert result["episode_only_se"] == 0
    assert result["episode_bootstrap_95_ci"] == [3, 3]
    assert result["per_seed_difference_means"] == {str(s): float(s - 10) for s in range(11, 16)}
    assert result["training_seed"]["sample_sd"] > 0


def test_mismatched_episode_sets_are_rejected():
    frames = {11: pd.DataFrame({"objective": [0, 1]}, index=[1, 2]),
              12: pd.DataFrame({"objective": [0, 1]}, index=[2, 3])}
    with pytest.raises(ValueError, match="Episode IDs differ"):
        paired_matrix(frames, "objective")


def test_no_results_cannot_generate_a_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="No completed evaluation"):
        run(results=tmp_path, figures=tmp_path / "figures", draws=10)
    assert not (tmp_path / "pilot_summary.json").exists()
