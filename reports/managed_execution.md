# Managed experiment execution

The managed runner adds persistence and provenance around the unchanged scientific training and evaluation functions. It is used by the independent replication. The historical `run_pilot` entry point remains available for reproducing the original experiments; it does not acquire these guarantees retroactively.

## New runs

```powershell
python -m scripts.run_managed train --config configs/smoke.yaml --seed 11 --output results/example_train
python -m scripts.run_managed evaluate --policy recurrent --checkpoint results/example_train/model.zip --episodes 128 --seed-start 900000 --output results/example_eval
python -m scripts.run_managed verify results/example_train
python -m scripts.run_managed verify results/example_eval
```

Repeating an identical command verifies and reuses completed artifacts. Changing the training configuration, evaluation cohort, checkpoint bytes, recorded implementation or dependency versions raises an error for that destination. Choose a new output folder for a new experiment. The public CLI requires managed provenance for learned checkpoints; historical raw checkpoints use the legacy evaluator and an explicitly documented protocol.

## Completion and recovery

A job first acquires a nonblocking OS file lock associated with its destination. The lock is released by the OS if the process dies; the persistent lock file is not itself evidence that a job is running. A second process cannot compute the same destination concurrently.

Computation writes to a private sibling `.attempts` directory. Required files and portable JSON results are written before the completed manifest. The manifest records input identity, schema version, artifact sizes and SHA-256 hashes. Only after verification is the whole directory renamed to its final location on the same filesystem. Readers do not mistake an incomplete checkpoint for a completed run.

If computation raises or is interrupted, the final output remains absent and the attempt is retained for diagnosis. Rerunning starts a fresh attempt from the declared initialization. **This is restart safety, not mid-training optimizer/RNG resumption.** Completed jobs in a multi-job invocation can be reused; incomplete ones are recomputed. A killed process may leave an attempt without a failure record, which is still incomplete because it was never published to the final destination.

JSON file replacement is atomic and flushed before replacement. Directory publication assumes a local filesystem supporting same-volume rename and OS advisory locks. This is a single-host design, not a distributed scheduler or a guarantee against every filesystem/power failure. Shared network filesystems need separate validation.

## Identity and provenance

Run identity is a canonical hash of the declared stage, configuration, seed, scientific source hashes and recorded dependencies. Output locations do not enter identity. Training metadata references `model.zip` relatively. Evaluation records the model hash at collection time, links its managed training run ID, and checks the hash again after collection. Moving a completed directory preserves its identity and artifact checks.

The inherited Git revision field identifies the checkout's HEAD, which may precede uncommitted implementation work. The source hashes are authoritative for executed file contents; a Git revision alone does not establish a clean source snapshot. The wrappers and scientific source hashes recorded for this iteration can be checked against the published source files.

Hashes detect accidental corruption and stale results; they are not cryptographic authentication against someone rewriting both a file and its manifest. Git history and release review provide the surrounding trust context. The original experiments predate this contract and remain frozen in the first public snapshot.

Tests cover altered inputs, checkpoint corruption, missing outputs, preserving failed attempts, process-lock exclusion and release after abrupt exit, moving completed directories, and atomic JSON failure. A real short training comparison also checks that the managed wrapper produces exactly the same parameter digest as the underlying trainer with the same seed/configuration.
