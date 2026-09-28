# Adverse-Selection Inference in Recurrent Market Makers

[![Tests](https://github.com/DB825/adverse-selection-rl-market-makers/actions/workflows/tests.yml/badge.svg)](https://github.com/DB825/adverse-selection-rl-market-makers/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A finite-horizon market-making experiment testing whether recurrent PPO represents adverse-selection risk conditional on its own quotes and observed executions. A joint Bayesian filter over 12 latent regimes supplies posterior targets and a myopic quoting reference. The simulator separates informed trading from uninformed directional demand; no market data or exchange connection is used.

Each episode fixes asset value `V`, informed-arrival probability `alpha`, and uninformed buy preference `p`. The dealer observes quote/outcome history, inventory and time. Active no-trade events update the posterior; abstention supplies no evidence. The analysis distinguishes economic performance, posterior decodability, information beyond recent observations, and causal policy use.

[Model and likelihood derivation](reports/model_design.md) · [Information flow and estimands](reports/technical_walkthrough.md) · [Research priorities](reports/engineering_roadmap.md)

## Results

| Experiment | Design | Finding |
|---|---|---|
| [Exact recent-history overlap](reports/history_overlap.md) | Five frozen policies; 281,600 eligible decisions | Exact history-8 matches cover 5.6% of decisions; 1,541 matched decisions have conflicting decisive reference actions. Overall prevalence remains unidentified. |
| [Retained-trajectory decision audit](reports/decision_quality.md) | Five frozen policies; 1,024 retained episodes each | Quote-center choice accounts for 81.5% of the mean myopic score gap. AS decoding R² does not consistently track quoting quality. |
| [Independent entropy replication](reports/replication_memo.md) | Five fresh seed pairs; 300,032 steps/model; 2,048 paired evaluation episodes | Entropy .01 increased objective from .635 to 1.540. Paired gain .905; seed 95% interval [−.162, 1.972]. The predeclared criterion was not met. |
| [Nonlinear history control](reports/nonlinear_memo.md) | All five frozen development policies; 1,024 fresh episodes; whole-episode splits | History-8 MLP buy/sell adverse-selection R² .847/.830; history plus recurrent-state ridge .717/.699. Linear decoding gains do not establish information unique to older recurrent memory. |
| [Development entropy comparison](reports/entropy_memo.md) | Entropy 0 versus .01; seeds 11–15 | Paired objective gain .715; seed interval [−.638, 2.069]. |
| [GAE comparison](reports/gae_memo.md) | GAE .95 versus 1.0; seeds 11–15 | Paired objective gain .018; seed interval [−.434, .470]. |
| [Budget extension](reports/followup_memo.md) | Four policy families; 100,352 versus 300,032 steps | Recurrent gain uncertain; feedforward and history policies improved consistently across the five seeds. |

Economic experiments use separate evaluation cohorts; the decision audit reuses the nonlinear-control trajectories. The [original pilot](reports/research_memo.md) and subsequent protocols retain their configurations, episode ranges and per-seed outcomes. Episode intervals condition on fitted policies; seed intervals condition on the evaluation cohort. Neither combines both sources of uncertainty. Decoder comparisons are exploratory and do not establish causal use.

## Reproduce a small experiment

Commands run from the repository root. CI uses Python 3.13, CPU PyTorch 2.8.0 and the direct pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.verify_publication
python -m scripts.run_managed train --config configs/smoke.yaml --seed 11 --output results/example_train
python -m scripts.run_managed evaluate --policy recurrent --checkpoint results/example_train/model.zip --episodes 128 --seed-start 900000 --output results/example_eval
python -m scripts.run_managed verify results/example_eval
```

On Linux, activate with `source .venv/bin/activate`. If PowerShell activation is disabled, invoke `.\.venv\Scripts\python.exe` directly. `requirements-lock.txt` records the original Windows/Python 3.13.7 environment; it is separate from the portable CI installation above. The smoke run checks execution and accounting, not learning performance.

[Managed execution](reports/managed_execution.md) binds runs to configuration, source and dependency hashes. Identical reruns verify and reuse completed output. Incomplete runs restart from their declared seeds; optimizer/RNG continuation is not implemented. Process locks and atomic directory publication assume a single host and local filesystem.

## Evidence and implementation

`published-results/` contains four immutable evidence snapshots covering 795 files. Their [original](published-results/publication_manifest.json) and [iteration](published-results/iteration-v1/publication_manifest.json) manifests map original hashes to sanitized public bytes. The [decision-audit snapshot](published-results/decision-quality-v1/publication_manifest.json) adds episode aggregates and two figures. The [history-overlap snapshot](published-results/history-overlap-v1/publication_manifest.json) adds exact-match group and episode aggregates. `results/` contains ignored local runs. Checkpoints, activation arrays and event logs are excluded; full audits require regenerating those artifacts. See [artifact provenance](reports/repository_readiness.md).

Tests cover accounting, action-conditioned likelihoods, recurrent resets and gradients, episode grouping, decoder metrics, checkpoint identity and interrupted runs. [CPU benchmarks](reports/compute_benchmark.md) report a fused-LSTM microbenchmark of 95.7 to 722.0 steps/s and a full probe-audit median of 10.04 to 2.15 seconds. These are local measurements with distinct workloads.

## Full recorded pilot

```powershell
python -m scripts.run_pilot train --workers 3
python -m scripts.run_pilot evaluate --workers 3 --episodes 1024
python -m scripts.summarize_results
```

Training runs all four learned families for seeds 11–15, with 100,352 transitions each. A process uses one PyTorch CPU thread. `--workers 1` lowers simultaneous resource use. The runner resumes completed checkpoints and otherwise trains missing jobs. It evaluates fixed quotes and the Bayesian myopic reference as well. Main evaluation uses stochastic categorical policy actions, matching the trained policy distribution; `src.evaluate --deterministic` is an optional, separately labeled diagnostic. No checkpoint is selected using held-out evaluation performance.

## Recorded fixed-budget follow-up

The follow-up restarts each initialization with unchanged learning settings, repeating the original training prefix and extending the fixed budget to 300,032 steps. It is not an independent replication. Train the original pilot first to reproduce the paired budget comparison, then run:

```powershell
python -m scripts.run_pilot train --config configs/followup_budget.yaml --root results/followup_budget --workers 3
python -m scripts.run_pilot evaluate --config configs/followup_budget.yaml --root results/pilot --output results/pilot_on_followup_evaluation --workers 3
python -m scripts.run_pilot evaluate --config configs/followup_budget.yaml --root results/followup_budget --output results/followup_budget_evaluation --workers 3
python -m scripts.summarize_followup
python -m pytest -q --junitxml=results/followup_tests.xml
python -m scripts.audit_followup --check-initialization
python -m scripts.make_followup_manifest
```

The comparison uses seeds 5,000,000–5,002,047 for both budgets. Summary generation requires all five seeds of all four families; the manifest validates both budgets, checkpoint identities, configurations, episode pairing and passing tests. Training took 31.6 minutes with three CPU workers on this machine; evaluation/reporting were additional. Output is separate from the original pilot: `results/followup_summary.{json,csv}`, `results/followup_audit.json`, `results/followup_manifest.json`, and `reports/followup_figures/`.

## Controlled GAE comparison

The single-factor experiment changes only recurrent GAE lambda from .95 to 1.0. It reuses the existing .95 control checkpoints at 300,032 steps and trains five new recurrent models. Read the [fixed protocol](reports/gae_protocol.md). After the budget follow-up checkpoints exist:

```powershell
python -m scripts.gae_preflight
python -m scripts.run_pilot train --config configs/gae_lambda_one.yaml --policies recurrent --root results/gae_lambda_one --workers 3
python -m scripts.run_pilot evaluate --config configs/gae_lambda_one.yaml --root results/followup_budget --output results/gae_control_evaluation --workers 3
python -m scripts.run_pilot evaluate --config configs/gae_lambda_one.yaml --policies recurrent --root results/gae_lambda_one --output results/gae_treatment_evaluation --workers 3
python -m pytest -q --junitxml=results/gae_tests.xml
python -m scripts.analyze_gae
```

Both arms and contextual baselines use 2,048 episodes beginning at 8,000,000. The preflight checks identical initial weights and simulator streams. Final analysis requires every planned checkpoint, validates serialized GAE settings and parameter finiteness, and archives paired results in `results/gae_summary.{json,csv}` and `results/gae_manifest.json`. Undefined explained variance is counted explicitly; nonfinite losses or infinite diagnostics are rejected. Development episodes at 7,000,000 and conditional probe episodes at 9,000,000 are separate from this economic test cohort.

## Controlled entropy comparison

The single-factor experiment changes only recurrent entropy coefficient from zero to .01, retaining GAE .95 and 300,032 steps. It reuses the budget follow-up controls and trains five new paired recurrent models. Read the [fixed protocol](reports/entropy_protocol.md). After the budget follow-up checkpoints exist:

```powershell
python -m scripts.entropy_preflight
python -m scripts.run_pilot train --config configs/entropy_001.yaml --policies recurrent --root results/entropy_001 --workers 3
python -m scripts.run_pilot evaluate --config configs/entropy_001.yaml --root results/followup_budget --output results/entropy_control_evaluation --workers 3
python -m scripts.run_pilot evaluate --config configs/entropy_001.yaml --policies recurrent --root results/entropy_001 --output results/entropy_treatment_evaluation --workers 3
python -m pytest -q --junitxml=results/entropy_tests.xml
python -m scripts.analyze_entropy
```

Both arms and contextual baselines use 2,048 episodes beginning at 11,000,000. The preflight checks matching initialization and streams. Analysis requires all planned runs, passing tests, serialized settings, finite model parameters and matched episodes, and writes `results/entropy_summary.{json,csv}` and `results/entropy_manifest.json`. Development and conditional probe cohorts begin at 10,000,000 and 12,000,000 respectively. Entropy changes the training loss; evaluation retains the economic objective.

For this completed treatment family, the documented behavior decision in `results/entropy_decision_gates.json` permits exploratory probes. All five models, including the weak seeds, are included:

```powershell
python -m scripts.run_pilot probes --config configs/entropy_001.yaml --root results/entropy_001 --output results/entropy_analysis/probes --episodes 1024
python -m scripts.analyze_entropy_probes
```

Probe analysis checks the saved gate decision, complete trajectories, held-out targets and predictions, target variance, frozen checkpoint hashes and untrained controls. It reports all six targets with separate seed and episode uncertainty in `results/entropy_probe_summary.json`. It does not establish information beyond nonlinear recent-history features or causal use.

## Independent replication and nonlinear control

The [replication protocol](reports/replication_protocol.md) freezes seeds 21–25, both entropy arms and 300,032 steps per model. It needs no historical checkpoints:

```powershell
python -m scripts.run_replication preflight
python -m scripts.run_replication train --workers 3
python -m scripts.run_replication evaluate --workers 3
python -m scripts.analyze_iteration replication
```

The separate [nonlinear protocol](reports/nonlinear_protocol.md) uses the original five entropy-.01 checkpoints, so reproduce that development experiment first:

```powershell
python -m scripts.run_nonlinear_control
python -m scripts.analyze_iteration nonlinear
python -m scripts.verify_publication
```

These stages verify completed outputs before reuse. Code, configuration and dependency fingerprints bind local runs to their declared protocol. See [managed execution](reports/managed_execution.md) for portability and restart limits. Public text evidence is in `published-results/iteration-v1/`; omitted checkpoints and arrays must be regenerated for full audits.

## Conditional representation analysis

The original entropy policies have now undergone a [nonlinear recent-history control](reports/nonlinear_memo.md) on fresh episodes. Nonlinear history outperformed the linear trained-state decoder on all six targets on average. Read that result before interpreting the historical linear probes as an older-memory advantage.

Only after inspecting economic behavior, collect representations on a separate set of episodes:

```powershell
python -m scripts.run_pilot probes --episodes 512
```

The runner creates an untrained-state control for every seed. It processes exactly the same public histories as the trained model, and never trains on rewards. Representation analysis uses whole-episode train/validation/test splits. Probe success is predictive association, not a causal-use result.

For a manually collected activation archive that does not yet have an untrained control, use the following alternative. Do not run the first command after the runner has already created that file; it deliberately refuses to overwrite existing controls. When a matching `untrained_states.npz` is next to the activation archive, `src.probes` includes it automatically.

```powershell
python -m src.untrained_control --dataset results/probes/seed_11/activations.npz --seed 11 --output results/probes/seed_11/untrained_states.npz
python -m src.probes --dataset results/probes/seed_11/activations.npz --output results/probes/seed_11
```

After in-distribution behavior works, the focused shift is:

```powershell
python -m scripts.run_pilot shift --workers 3 --episodes 1024
python -m scripts.run_pilot shift_control --workers 3 --episodes 1024
python -m scripts.analyze_shift
python -m scripts.summarize_results
```

This fixes `alpha=.05, p=.9`, samples `V` equally, and preserves all execution mechanics. The ordinary filter keeps the 12-state training support. `myopic_correct` uses the two-state support `{(0,.05,.9),(1,.05,.9)}`; it is explicitly correctly specified for this shift. Learned belief-input PPO retains its original 12-dimensional training posterior input.

The matched control fixes `alpha=.05, p=.75` with the same value draw and full customer tape. Comparing these two cohorts changes only `p`, avoiding confounding the shift with the lower informed fraction. Both correctly specified references know the cohort's alpha/p support but still infer V.

`configs/large.yaml` provides a configurable 1,000,448-step budget. It is a future experiment unless a corresponding completed metadata record exists. A longer budget is not a convergence guarantee.

## Environment and accounting

There are 64 arrivals and 12 equally likely fixed episode regimes `(V, alpha, p)` from `{0,1} × {.05,.30} × {.25,.50,.75}`. Customer type, intended side, and reservation value are generated by independent uniforms. The full customer tape is generated at reset, including arrivals on which the dealer abstains. Policy quotes determine whether each customer executes; paired policies never have their observed outcomes forced to match.

Quote centers are `.35,.50,.65`; half-spreads are `.10,.20,.30`. Actions are center-major, half-spread-minor (`0..8`). Action `9` abstains. Useful fixed choices are `fixed_3` for `(.4,.6)`, `fixed_4` for `(.3,.7)`, `fixed_5` for `(.2,.8)`, and `fixed_9` for abstention. `fixed_2` and `fixed_8` are wide off-center quotes. Directions always refer to the **customer**: customer buy means the dealer sells and inventory decreases.

At each decision, the 9 features are:

| Indices | Meaning | Fixed normalization |
|---|---|---|
| 0–1 | Previous bid and ask | Native `[0,1]` prices |
| 2 | Previous abstention | 0 or 1 |
| 3–6 | Previous buy/sell/no-trade/start | Four one-hot indicators |
| 7 | Current dealer inventory | Divide by horizon |
| 8 | Episode remaining | `1 - arrivals_completed/horizon` |

Reset uses zero quote prices, abstention flag one, and the start sentinel. Abstention uses a no-trade outcome with its flag set, so it is distinguishable from an informative customer rejection. History features are zero-padded and reset each episode. Rewards, cash, posterior, `V`, `alpha`, `p`, type, and reservation value are not inputs to ordinary neural policies. The belief-input baseline is the explicit posterior exception. Inventory already summarizes cumulative executed order imbalance, so the feedforward baseline has access to that historical statistic.

The objective is

`J = E[cash_T + q_T V - (lambda/T) sum(t=1..T) q_t²]`,

with post-arrival inventory, `lambda=.001`, and per-execution fee zero by default. A fee reduces cash on each execution. Dense reward is `delta_cash + .5 delta_inventory - lambda q_t²/T`, plus `q_T(V-.5)` on the final step. Its undiscounted sum equals the economic objective. Gamma is one; the finite horizon returns `terminated=True, truncated=False`. Abstaining still pays the carrying penalty on an existing inventory.

The exact filter keeps the joint 12-state posterior in log space and conditions on actual actions, including no-trade evidence. Abstention leaves beliefs unchanged. Impossible observations raise rather than silently replacing the prior. Conditional values/profits are `NaN` when the conditioning event has negligible probability; unconditional expected profits remain finite. Prior weights and latent support are configurable in the simulator/filter and training environment.

## Baselines and learning

| Policy | Inputs | Actor / critic | Total parameters |
|---|---|---|---:|
| Fixed quotes | None | Predeclared action | 0 |
| Bayesian myopic | Exact posterior, inventory | Expected immediate profit minus incremental inventory charge | 0 |
| Feedforward PPO | Current 9 features | Separate tanh `[64,64]` networks | 10,315 |
| History PPO | Last 8 observations, 72 features | Separate tanh `[64,64]` networks | 18,379 |
| Recurrent PPO | Current 9 features plus recurrent memory | Separate 64-unit actor/critic LSTMs, each followed by tanh `[64]` | 47,435 |
| Belief PPO | Joint posterior plus inventory/time, 14 features | Separate tanh `[64,64]` networks | 10,955 |

Recent actions are already represented by the quote/abstention fields in subsequent observations. The myopic reference ignores future information acquisition and future carrying costs; it is **not** the finite-horizon optimum. Parameter counts differ materially, so this is not a capacity-matched recurrence experiment.

The original and budget-follow-up learned families use learning rate `.0003`, gamma `1`, GAE lambda `.95`, 10 PPO epochs, batch size 64, clip range `.2`, entropy coefficient zero, and gradient clipping `.5`. The GAE and entropy experiments each change their named factor only. Eight serial environments collect 128 steps each per rollout. This uses RecurrentPPO's default rollout length and PPO's default minibatch size. The feedforward PPO rollout length differs from its library default. No normalization wrapper, reward scaling, checkpoint selection, or hyperparameter sweep is used.

The fused recurrent subclass changes only how a sequence with resets at its first timestep is passed through PyTorch's LSTM. Sequences with interior resets or dropout fall back to stock SB3. Output/state/input-gradient/parameter-gradient and short-training equivalence are tested against the pinned library. Optimizer, PPO losses, sequence sampling, padding, and truncation rules are unchanged. Its use is recorded in configuration and metadata.

## Statistical protocol and artifacts

The historical studies use training seeds 11–15; the independent replication uses 21–25 and its own predeclared cohorts. The original pilot uses the following ranges. Vector-environment randomness uses disjoint seed ranges derived from each run seed; actual ranges are recorded in metadata. Development/smoke evaluation starts at 900,000; final in-distribution evaluation at 1,000,000; shift evaluation at 2,000,000; representation collection at 3,000,000. Action-sampling uniforms use a separate offset stream. Common evaluation seeds pair the regime and complete customer tape across policies.

Episode CSVs contain objective, raw profit, lower-tail inputs, inventory exposure, participation, buy/sell rates, spreads/centers, action counts, true regime, and profit conditional on informed/uninformed executions. Hidden labels are diagnostic columns only. Summary JSON includes regime breakdowns. Aggregate analysis separates uncertainty across evaluation episodes from uncertainty across five training seeds; timesteps are never treated as independent replicates.

Training writes `config.json`, `metadata.json`, `progress.csv`, TensorBoard events, and `model.zip`. Metadata includes Python/dependency versions, architecture, timing, source hashes, checkpoint path, and Git revision where available. Historical runs predate the first commit; source hashes preserve their executed code identity. New runs record the available revision. TensorBoard can be opened with `tensorboard --logdir results/pilot`.

Activations use compressed NumPy NPZ with no pickle. `activation_schema.json` records shapes and fixed target definitions. Each row contains episode/time, predecision observation/history, action probabilities, chosen action, posterior, targets, actor hidden/cell states after the observation and before the actor head, and separate diagnostic regime labels. Ridge regularization is chosen on validation episodes; all scalers and fits use training episodes only. The six targets are posterior expected `V,alpha,p`, entropy, and buy/sell adverse information at fixed quote `(.2,.8)`. Probes evaluate arrivals 8–63 to focus on history beyond the shortest initial prefix.

`src/interventions.py` exposes explicit before-observation and after-observation/pre-head state-patching interfaces. No matched causal intervention experiment is asserted by this pilot. Patching a full recurrent state changes several kinds of information and may be off distribution; cell-only and hidden-only patches have different immediate effects.

The supplied evaluation CLI reproduces the default 12-state training family and its documented shift. It supports other horizons, fees, penalties, and trained history-window sizes. For custom supports/priors, extend the evaluator consistently before interpreting results; a changed simulator must have a corresponding filter.

## Repository map

`src/customer_model.py`, `environment.py`, and `accounting.py` separate generative mechanics from economics. `bayes_filter.py` handles inference; `baselines.py` holds reference policies; `wrappers.py`, `train.py`, and `fast_recurrent.py` handle learning. `evaluate.py`, `behavior.py`, `collect_activations.py`, `probes.py`, and `untrained_control.py` provide analysis. `scripts/decision_relevance.py` gives the constructive Gate 1 example, `run_pilot.py` runs bounded jobs, and `summarize_results.py` produces initial aggregate results and scientific figures. `summarize_followup.py`, `audit_followup.py`, and `make_followup_manifest.py` produce and validate the paired budget extension separately.

## API sources checked

Implementation was checked against the official [Gymnasium environment API](https://gymnasium.farama.org/api/env/), [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), and [RecurrentPPO API](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html), then against the installed, pinned 2.7.0 source for recurrent sequence processing and distribution extraction. The [upstream policy source](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/common/recurrent/policies.html) explains the recurrent state and episode-start interface. This is an implementation reference list, not a novelty review.

Probe interfaces were also checked against official [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) documentation; the environment pins scikit-learn 1.7.2.
