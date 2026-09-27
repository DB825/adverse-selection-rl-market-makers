# CPU training benchmark and equivalent recurrent optimization

Measured on this Windows host with Python 3.13.7, PyTorch 2.8.0,
Stable-Baselines3 2.7.0 and sb3-contrib 2.7.0. CUDA was unavailable. All runs
used one PyTorch CPU thread, eight serial DummyVecEnv environments, 128 steps
per environment per rollout, ten PPO epochs, seed 11, gamma=1, and 2,048 total
environment steps. They used the same environment and network architecture.
These historical timing runs preceded the simulator-seed audit and used SB3's
default environment streams 11 through 18. The substantive pilot instead uses
disjoint streams `10000 * training_seed + environment_index`. A training-seed
11 pilot therefore uses 110000 through 110007, with unchanged optimizer seed
11. Partial pilot runs started before this correction were excluded and kept
separately in `results/deprecated_seed_overlap`. They contribute no reported
substantive results.

Elapsed time below covers learning and checkpoint save (not interpreter startup,
imports or model construction). These short timings are throughput estimates,
not precise hardware benchmarks; background processes can affect them. No policy
return was used to select a setting.

| Recurrent implementation | Minibatch | MKLDNN | Seconds | Steps/second |
|---|---:|---|---:|---:|
| Stock SB3 helper | 64 | enabled | 21.401 | 95.7 |
| Stock SB3 helper | 64 | disabled | 18.032 | 113.6 |
| Stock SB3 helper | 256 | enabled | 7.474 | 274.0 |
| Stock SB3 helper | 512 | enabled | 3.854 | 531.3 |
| Equivalent fused reset helper | 64 | enabled | 2.837 | 722.0 |
| Equivalent fused reset helper | 256 | enabled | 1.220 | 1,678.5 |

The selected pilot configuration retains **minibatch 64, ten epochs, MKLDNN
enabled, one CPU thread**. It enables the fused reset helper. The pilot uses
100,352 environment steps per learned policy seed (98 complete rollouts), so
the short-run extrapolation for a recurrent seed is about 139 seconds before
additional contention and evaluation. This is an estimate, not a promised
runtime or evidence of convergence. Actual per-run timings are in each saved
`metadata.json`.

The larger minibatch experiments changed the number of optimizer updates per
rollout and therefore would have changed training dynamics. They were not used
for the substantive pilot. The final choice preserves the original minibatch
and optimizer schedule. The main departure from feedforward PPO defaults is
128 rollout steps per environment instead of 2,048. RecurrentPPO already uses
128 by default, but its default minibatch is 128 rather than this pilot's 64.

## What the optimization changes

The stock recurrent helper processes one time step at a time whenever any
episode-start flag exists in a sequence batch. PPO's sequence padding commonly
places all such flags at the first time step. `src/fast_recurrent.py` checks
this condition: it multiplies incoming hidden and cell state by the same reset
mask once, then makes one fused PyTorch LSTM call for the sequence. An interior
reset or nonzero LSTM dropout uses the unmodified upstream helper.

This changes execution grouping only. It preserves PPO's loss, sampling,
minibatches, sequence boundaries, truncation, padding, optimizer, and gradient
clipping. The actor and critic remain separate one-layer 64-unit LSTMs with
64-unit tanh heads. It is not claimed to be bitwise identical.

Tests cover one and multiple sequences, lengths 1 to 64, one and two layers,
selective initial resets, no resets, and an interior reset fallback. Outputs and
final hidden/cell states agree with the installed upstream implementation with
relative tolerance 2e-5 and absolute tolerance 1e-6. Input, incoming-state and
every LSTM parameter gradient agree at relative tolerance 2e-5 and absolute
tolerance 2e-6. After both stock and fused implementations train for the same
2,048 steps and seed, the largest absolute difference across any checkpoint
parameter is **5.96e-8**. Nineteen training/wrapper/adapter tests passed at this
stage; later complete-suite results are recorded separately.

The adapter independently matches official `predict` states and categorical
probabilities, including selectively reset batched episodes. Post-observation
hidden-state patches change the action head input; cell-state-only patches
change future recurrent state, not the immediate probabilities. These tests
validate the interface, not a causal scientific claim.

## Artifacts and reproduction

The stock 64 benchmark is in `results/smoke/recurrent11`; the remaining runs are
in `results/benchmark/{no_mkldnn,batch256,batch512,fused_batch64,fused_batch256}`.
Checkpoints, CSV diagnostics, TensorBoard events, effective configurations and
metadata are saved for each. `results/benchmark/summary.json` records the timings
and backend choices explicitly; the first backend experiment ran before the
metadata schema gained an explicit MKLDNN flag.

```powershell
.venv/Scripts/python.exe -m src.train --config configs/smoke.yaml --policy recurrent --seed 11 --output results/reproduce_smoke
.venv/Scripts/python.exe -m pytest tests/test_training.py -q
.venv/Scripts/python.exe -c "from src.train import train,load_config; c=load_config('configs/smoke.yaml'); c['training']['fused_lstm_reset']=True; train(c,'recurrent',11,'results/reproduce_fused')"
```

For other benchmark rows, set `c['training']['batch_size']` to 256 or 512, or
set `c['training']['torch_mkldnn']=False` before calling `train`. Use a fresh
output directory: training refuses to overwrite existing checkpoints.
These reproduction commands use the corrected simulator-seed allocation;
their precise trajectories differ from the historical timing rows above.

Official API documentation was checked before implementation:
[RecurrentPPO API](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html),
[recurrent policy source](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/common/recurrent/policies.html).
The currently served documentation was version 2.9.0; the actual installed
2.7.0 signatures and source were inspected separately before adapter use.


## Probe-audit efficiency refinement

The nonlinear-control audit was compared against commit `c656018` on the same Windows CPU environment and the same five saved policy panels. Three before/after pairs alternated execution order within one process. Timing includes artifact hashing, trajectory checks, all decoder metrics, bootstrap summaries and figure rendering; it excludes Python import startup. Each complete output was compared after removing only its destination-dependent figure paths: all numerical values and provenance fields matched exactly.

| Run | Previous audit (s) | Refined audit (s) |
|---|---:|---:|
| 1 | 10.043 | 2.151 |
| 2 | 10.703 | 2.162 |
| 3 | 7.171 | .969 |
| Median | **10.043** | **2.151** |

The observed median ratio is **4.67×**. These are local timings with filesystem caching and ordinary machine-load variability, not a portable performance guarantee or a training-speed result. No models were retrained.

The change removes repeated NPZ member decompression inside episode loops and replaces repeated full-array loss scans with stable episode grouping. Stable grouping preserves each episode's summation order. The audit additionally recomputes R² from the predictions rather than trusting the saved score. Regression tests cover interleaved rows, exact agreement with the previous loss calculation, missing decisions, duplicate timestamps and stale R² with valid MSE. Historical protocols and public evidence bytes remain unchanged.
