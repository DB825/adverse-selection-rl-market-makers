# Does the Agent Know When It's Being Picked Off?

[![Tests](https://github.com/DB825/adverse-selection-rl-market-makers/actions/workflows/tests.yml/badge.svg)](https://github.com/DB825/adverse-selection-rl-market-makers/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A reproducible research pilot on belief formation in a recurrent market-making policy. The question is whether a dealer distinguishes informed flow from ordinary directional demand, and whether its recurrent state represents the observation-supported adverse-selection risk. This simulator is an inference and sequential-decision experiment, with no live trading connection or profitability claim.

**Code and public evidence:** [DB825/adverse-selection-rl-market-makers](https://github.com/DB825/adverse-selection-rl-market-makers), under the MIT license. Start with the [latest research memo](reports/entropy_memo.md), [engineering roadmap and résumé framing](reports/engineering_roadmap.md), and [publication/reproduction notes](reports/repository_readiness.md).

The engineering contribution is a 12-regime partially observed simulator with exact action-conditioned Bayesian inference, accounting invariants, paired experiments across four learned policy families, and tested recurrent reset/gradient behavior. A fused LSTM path improves the measured short CPU benchmark from 95.7 to 722.0 steps/second at unchanged minibatch settings; this is a local microbenchmark, not an end-to-end speed guarantee. The research separates useful behavior, decodable beliefs and causal use rather than treating probe accuracy as proof of learning.

**Public data layout:** `published-results/` is a frozen text-evidence snapshot with local paths sanitized. `results/` is ignored and reserved for your own runs. Historic manifests refer to original run bytes and omitted checkpoints; the [publication manifest](published-results/publication_manifest.json) maps original hashes to public bytes. Run `python -m scripts.verify_publication` to verify the published evidence. Full checkpoint audits require retraining; copying sanitized records into `results/` is not a substitute.

Read the latest [entropy experiment memo](reports/entropy_memo.md), the [GAE experiment memo](reports/gae_memo.md), [budget follow-up memo](reports/followup_memo.md), [initial research memo](reports/research_memo.md), [model and Gate 1 derivation](reports/model_design.md), and [compute benchmark](reports/compute_benchmark.md). Small aggregate results, episode CSVs, configurations, and figures are retained; checkpoint and activation binaries are ignored by Git.

**Latest completed entropy experiment:** 138 passing tests; five new recurrent runs at 300,032 steps, totaling 1,500,160 new transitions. On 2,048 fresh episodes beginning at 11,000,000, recurrent objective was **0.630 with entropy coefficient 0** and **1.345 with .01**, retaining GAE .95. The paired effect **+0.715** has training-seed 95% interval **[−0.638, 2.069]**. Three seeds beat fixed wide **0.940** with input-dependent actions, while two worsened. History-8 earned **2.723** and myopic **3.602**. All five models underwent exploratory representation probes on separate episodes; reliable learning, recurrence benefit and causal use remain unestablished. See the memo for complete outcomes and decoding controls.

**Completed GAE experiment:** 126 tests passed at that stage; five new recurrent runs at 300,032 steps totaled 1,500,160 new transitions. On its separate cohort beginning at 8,000,000, recurrent objective was **0.479 with GAE .95** and **0.497 with GAE 1.0**. The paired effect **+0.018** had training-seed interval **[−0.434, 0.470]**. None beat fixed wide **0.790**; probes were deferred for that family. Those historical results remain in the GAE memo.

**Completed budget follow-up:** 111 tests passed at that stage; all 20 runs used 300,032 steps, totaling 6,000,640 transitions including repeated initial prefixes. Its separate cohort beginning at 5,000,000 gave recurrent **0.657**, feedforward **2.500**, history-8 **2.700**, belief-input **2.871**, fixed-wide **0.946**, and Bayesian myopic **3.513**. Paired reevaluation of shorter checkpoints showed consistent feedforward/history gains but uncertain recurrent gains across training seeds. Those results remain in the budget memo.

**Original pilot:** 82 tests passed at that stage; 20 runs totaled 2,007,040 training transitions. Its separate 1,024-episode cohort gave recurrent **0.486**, feedforward **1.081**, history-8 **1.860**, belief-input **2.612**, fixed-wide **0.884**, and Bayesian myopic **3.460**. Those historical results and shift checks remain unchanged. Smoke probe outputs are pipeline checks only.

## Reproduce a small experiment

The recorded environment is **Windows, Python 3.13.7, CPU PyTorch 2.8.0**. `requirements-lock.txt` pins every installed distribution; `requirements.txt` lists direct dependencies. Python and platform versions are also saved with each checkpoint. Commands below assume the repository root and an activated virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m scripts.decision_relevance
python -m src.train --config configs/smoke.yaml --policy recurrent --seed 11 --output results/my_smoke
python -m src.evaluate --policy recurrent --checkpoint results/my_smoke/model.zip --episodes 128 --seed-start 900000 --output results/my_smoke_eval
```

On systems that prevent PowerShell activation, use `.\.venv\Scripts\python.exe` in place of `python`. Training refuses to overwrite an existing checkpoint. Choose a new output directory for a fresh experiment.

For a fresh environment using the same setup as CI, install `torch==2.8.0` from `https://download.pytorch.org/whl/cpu`, then `python -m pip install -r requirements.txt`. On Linux, activate with `source .venv/bin/activate`. CI tests Windows and Linux functionality using direct pinned requirements. The full lock records the historical research environment; portable installation and exact historical replay are different checks.

The smoke run checks execution and accounting; its learning and probe outputs are not scientific evidence. The default smoke configuration uses stock SB3's recurrent policy. The pilot uses a tested, numerically equivalent fused LSTM reset path to improve CPU throughput.

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

## Conditional representation analysis

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

Training initializations use seeds 11–15. Vector-environment randomness uses disjoint seed ranges derived from each run seed; actual ranges are recorded in metadata. Development/smoke evaluation starts at 900,000; final in-distribution evaluation at 1,000,000; shift evaluation at 2,000,000; representation collection at 3,000,000. Action-sampling uniforms use a separate offset stream. Common evaluation seeds pair the regime and complete customer tape across policies.

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
