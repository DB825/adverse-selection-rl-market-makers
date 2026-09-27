"""One-factor p=.75 -> .90 shift at alpha=.05, using paired customer tapes."""
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.summarize_results import summarize_matrix, read_episodes, POLICIES, SEEDS
from src.artifacts import write_json


def main():
    results={}
    for policy in (*POLICIES,"fixed_5","fixed_9","myopic","myopic_correct"):
        seeds=SEEDS if policy in POLICIES else [None]
        matrices={"shift":[],"control":[]}
        for seed in seeds:
            suffix=Path(policy)/(f"seed_{seed}" if seed is not None else "")/"episodes.csv"
            shift=read_episodes(Path("results/shift")/suffix)
            control=read_episodes(Path("results/shift_control")/suffix)
            if not shift.index.equals(control.index):
                raise ValueError("Shift and control episodes are unpaired")
            for column in ("value","alpha"):
                np.testing.assert_array_equal(shift[column],control[column])
            matrices["shift"].append(shift.objective.to_numpy())
            matrices["control"].append(control.objective.to_numpy())
        trained=policy in POLICIES
        matrices={k:np.stack(v) for k,v in matrices.items()}
        results[policy]={k:summarize_matrix(v,trained=trained) for k,v in matrices.items()}
        results[policy]["paired_p90_minus_p75"]=summarize_matrix(matrices["shift"]-matrices["control"],trained=trained)
    gaps={}
    for cohort in ("shift","shift_control"):
        correct=read_episodes(Path("results")/cohort/"myopic_correct/episodes.csv")
        retaining=read_episodes(Path("results")/cohort/"myopic/episodes.csv")
        gaps[cohort]=correct.objective.to_numpy()-retaining.objective.to_numpy()
    inference={}
    for cohort in ("shift","shift_control"):
        # Fixed policy isolates inference on exactly the same quoted history.
        frame=read_episodes(Path("results")/cohort/"fixed_5/episodes.csv")
        inference[cohort]={"training_model_value_mse":float(np.mean((frame.inferred_value-frame.value)**2)),
                           "correct_model_value_mse":float(np.mean((frame.correct_value-frame.value)**2)),
                           "mean_training_model_alpha":float(frame.inferred_alpha.mean()),
                           "by_true_value":frame.groupby("value")[["inferred_value","correct_value","inferred_alpha","correct_alpha"]].mean().reset_index().to_dict("records")}
    result={"design":{"alpha":.05,"p_control":.75,"p_shift":.9,"episodes":1024,
                       "seed_start":2_000_000,"common_customer_tapes":True,
                       "interpretation":"Correct filters know each cohort's alpha and p but not V. Retaining filters use the original joint prior/support. Learned recurrent policies failed the ID behavior gate, so these results do not establish learned belief generalization."},
            "policies":results,"correct_minus_retaining_myopic":{k:summarize_matrix(v[None],trained=False) for k,v in gaps.items()},
            "increase_in_reference_gap":summarize_matrix((gaps["shift"]-gaps["shift_control"])[None],trained=False),
            "fixed_quote_inference":inference}
    write_json("results/shift_diagnostics.json",result)
    print({p:round(v["paired_p90_minus_p75"]["mean"],4) for p,v in results.items()})


if __name__=="__main__":
    main()
