# Engineering value and next steps

The project is strongest as a research-engineering case study: a precisely specified simulator, controlled learning experiments, and explicit boundaries on what the evidence supports. It has no live market connection.

## What is already worth presenting

- Exact action-conditioned inference over 12 latent regimes, including informative no-trade observations and noninformative abstention.
- Accounting invariants connecting dense rewards to terminal economic profit and inventory costs, with tested customer/dealer sign conventions.
- Paired customer tapes and action uniforms, disjoint training streams, frozen budgets, all-seed reporting, and separate episode/training uncertainty.
- A fused recurrent sequence path with forward, hidden/cell reset and gradient equivalence checks against upstream SB3. At fixed minibatch64, the recorded short benchmark improved from 95.7 to 722.0 steps/second (about 7.5x); this is not full-pipeline acceleration.
- Controls showing why linear decoding is not proof of learned inference or causal use: random recurrent features decode much of the target, and a weak economic policy can decode well.
- Public CI, pinned direct dependencies, a train/evaluate smoke path, and checksummed public evidence preserving original provenance while removing local paths.

## Highest-value next work

1. **Independent replication of the fixed entropy comparison.** Predeclare fresh training seeds, a disjoint test cohort, an economically meaningful effect threshold and fixed compute budget. Keep both arms and every failed seed. Success means a result that survives training variability. This strengthens research credibility more than another untargeted parameter sweep.
2. **A nonlinear recent-history control.** Freeze inputs, architecture choices and validation budget before collecting fresh probe episodes. Compare history-only, trained state and untrained-state controls on all six targets and all policy seeds. This tests whether the linear-probe advantage is feature transformation rather than information from older observations.
3. **Portable checkpoint provenance and durable experiment execution.** Replace absolute checkpoint paths with run IDs, checkpoint hashes recorded at collection time and versioned schemas. Test interruption/restart, failed jobs, stale outputs, and recovery in a bounded run. The present runner skips completed files; it is not a transactional workflow engine. This is particularly valuable for engineering interviews.
4. **A small distribution-shift regression suite.** Fix a compact set of support changes and compare retained versus correctly specified Bayesian references. Separate simulator changes from inference misspecification. Publish failure cases and calibration alongside reward.
5. **A five-minute technical walkthrough.** Explain one accounting invariant, one recurrent reset bug the tests would catch, why common random numbers help, and why seed uncertainty changes the entropy conclusion. Use the existing figures. A concise case study and reproducible command add more value than a dashboard without new analytical capability.

Causal interventions come after stronger representation controls. They need matched donors, unchanged public state, random/same-belief controls and separate hidden/cell treatment; arbitrary state edits can be off distribution. These next experiments have not run.

## Résumé wording supported by current work

Use these only to the extent you can explain, maintain and defend the implementation yourself. Describe your own contribution accurately, including AI-assisted development if asked.

**Research / quantitative role:** “Developed a 12-regime partially observed market-making simulator with exact Bayesian inference; evaluated four PPO policy families using paired simulations, five-seed experiments and held-out representation controls to distinguish economic behavior from decodable beliefs.”

**Research-engineering / software role:** “Built a reproducible RL experiment pipeline with automated tests, Windows/Linux CI and checksummed evidence; implemented an equivalence-tested LSTM execution path that improved local short-benchmark throughput from 96 to 722 steps/s at unchanged optimizer settings.”

Use the second bullet's CI claim after both hosted jobs pass. Avoid claiming production trading, market-beating returns, causal adverse-selection reasoning or a statistically reliable entropy improvement. The observed mean objective increase from 0.630 to 1.345 has a wide seed interval including zero; a percentage uplift headline would hide that uncertainty.

For interviews, be ready to derive a no-trade likelihood, explain why inventory itself is a history channel, distinguish conditional from joint uncertainty, and justify the untrained decoder control. Those details demonstrate understanding beyond running an RL library.

The emphasis on systematic hypothesis testing is consistent with [Two Sigma's description of quantitative research](https://www.twosigma.com/careers/quantitative-research-data-science/). The prioritization above is project-specific judgment, not a hiring promise.
