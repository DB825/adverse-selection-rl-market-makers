# Engineering value and next steps

The project is strongest as a research-engineering case study: a precisely specified simulator, controlled learning experiments, and explicit boundaries on what the evidence supports. It has no live market connection.

## What is already worth presenting

- Exact action-conditioned inference over 12 latent regimes, including informative no-trade observations and noninformative abstention.
- Accounting invariants connecting dense rewards to terminal economic profit and inventory costs, with tested customer/dealer sign conventions.
- Paired customer tapes and action uniforms, disjoint training streams, frozen budgets, all-seed reporting, and separate episode/training uncertainty.
- A fused recurrent sequence path with forward, hidden/cell reset and gradient equivalence checks against upstream SB3. At fixed minibatch64, the recorded short benchmark improved from 95.7 to 722.0 steps/second (about 7.5x); this is not full-pipeline acceleration.
- Controls showing why linear decoding is not proof of learned inference or causal use: random recurrent features decode much of the target, and a weak economic policy can decode well.
- Public CI, pinned direct dependencies, a train/evaluate smoke path, and checksummed public evidence preserving original provenance while removing local paths.

## Completed in the latest iteration

1. **Independent replication:** ten models, five fresh seed pairs and 3,000,320 transitions under a frozen protocol. The mean entropy gain was +.905, but its seed interval [−.162, 1.972] failed the predeclared criterion. All failures and references are retained. [Results](replication_memo.md).
2. **Nonlinear history control:** a validation-selected history decoder surpassed linear trained-state decoding on all six targets on average. The updated interpretation separates a useful feature transformation from evidence of uniquely older information. [Results](nonlinear_memo.md).
3. **Durable execution and provenance:** input-derived run IDs, collection-time checkpoint hashes, process locks, immutable completed artifacts, atomic directory publication and retained failed attempts. Tests cover interrupted work, corrupt caches, conflicting inputs, relocation and equivalence to direct training. Recovery restarts from the declared seed; optimizer/RNG continuation is not implemented. Guarantees assume a single host and local filesystem. [Execution contract](managed_execution.md).
4. **Technical walkthrough:** information boundaries, an accounting invariant, recurrent reset behavior, common random numbers and seed uncertainty in a concise [case study](technical_walkthrough.md).

The suite now has **157 tests**. CI exercises Windows and Linux, including managed training, evaluation, cache reuse and artifact verification. Two frozen public evidence snapshots preserve earlier conclusions and the new controls independently.

## Highest-value remaining work

First, plan a fixed replication budget around a useful-effect threshold and interval precision, accounting for the uncertainty in the observed seed variability. Second, measure the prevalence of decision-relevant older evidence on actual trajectories before investing in causal state patches. Third, add a compact calibration and distribution-shift regression suite with correctly specified reference controls. The [next-experiment plan](next_experiments.md) defines the boundaries; these follow-up studies have not run.

For engineering depth, the next justified extension is a versioned artifact migration and compatibility policy if schemas actually evolve. Multi-host scheduling and mid-optimizer resumption would need additional storage and RNG guarantees; neither is required for the current bounded CPU workflow. Avoid adding infrastructure without a concrete failure mode or workload.

## Résumé wording supported by current work

Use these only to the extent you can explain, maintain and defend the implementation yourself. Describe your own contribution accurately, including AI-assisted development if asked.

**Research / quantitative role:** “Developed a 12-regime partially observed market-making simulator with exact Bayesian inference; evaluated four PPO policy families using paired simulations, five-seed experiments and held-out representation controls to distinguish economic behavior from decodable beliefs.”

**Research-engineering / software role:** “Built a reproducible RL experiment pipeline with 157 tests, Windows/Linux CI, atomic run completion and checkpoint provenance; implemented an equivalence-tested LSTM path improving local short-benchmark throughput from 96 to 722 steps/s.”

Avoid claiming production trading, market-beating returns, causal adverse-selection reasoning or a statistically reliable entropy improvement. The fresh replication increased mean objective from .635 to 1.540 but retained a seed interval including zero; a percentage uplift headline would hide that uncertainty.

For interviews, be ready to derive a no-trade likelihood, explain why inventory itself is a history channel, distinguish conditional from joint uncertainty, and justify the untrained decoder control. Those details demonstrate understanding beyond running an RL library.

The emphasis on systematic hypothesis testing is consistent with [Two Sigma's description of quantitative research](https://www.twosigma.com/careers/quantitative-research-data-science/). The prioritization above is project-specific judgment, not a hiring promise.
