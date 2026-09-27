"""Record checksums and completed-run facts without scanning the virtual environment."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from src.artifacts import source_hashes, write_json


def main():
    records=[]
    for path in sorted(Path("results/pilot").glob("*/seed_*/metadata.json")):
        metadata=json.loads(path.read_text())
        model=path.with_name("model.zip")
        if not model.exists():
            raise FileNotFoundError(model)
        records.append({key:metadata[key] for key in ["policy","seed","actual_steps","elapsed_seconds","environment_seeds"]}
                       | {"metadata":str(path),"checkpoint":str(model),
                          "checkpoint_sha256":hashlib.sha256(model.read_bytes()).hexdigest()})
    try:
        revision=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError:
        revision=None
    data={"python":platform.python_version(),"platform":platform.platform(),"git_revision":revision,
          "source_hashes":source_hashes(),"dependency_lock_sha256":hashlib.sha256(Path("requirements-lock.txt").read_bytes()).hexdigest(),
          "main_completed_runs":records,"main_total_transitions":sum(r["actual_steps"] for r in records),
          "main_training_process_seconds":sum(r["elapsed_seconds"] for r in records),
          "artifact_status":{"results/pilot":"completed main training only",
                             "results/evaluation":"main fixed-budget held-out evaluation",
                             "results/shift":"p=.90, alpha=.05 focused shift; retained and correctly specified references",
                             "results/shift_control":"paired p=.75, alpha=.05 control changing only p",
                             "results/gate1":"constructive simulator/inference evidence",
                             "results/penalty":"reference-policy penalty sensitivity, no learned-policy tuning",
                             "results/smoke":"pipeline validation only, never a scientific probe claim",
                             "results/benchmark":"CPU throughput tests before training",
                             "results/deprecated_seed_overlap":"interrupted partial runs, excluded",
                             "configs/followup_budget.yaml":"recommended, not executed",
                             "configs/large.yaml":"available, not executed"}}
    write_json("results/manifest.json",data)
    print(f"Recorded {len(records)} completed main runs and {data['main_total_transitions']} transitions")


if __name__=="__main__":
    main()
