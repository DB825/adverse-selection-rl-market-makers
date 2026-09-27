# Prioritized next experiments

## Completed: fixed larger budget, unchanged learning setup

The unchanged **300,032-step** experiment has now run for all five initialization seeds and all four learned families using `configs/followup_budget.yaml`. Both budgets were evaluated on the same 2,048 episodes beginning at 5,000,000. Recurrent mean objective changed from **0.570 to 0.657**, current observation **1.143 to 2.500**, history-8 **1.901 to 2.700**, and belief input **2.613 to 2.871**. No recurrent point estimate beat fixed-wide **0.946**. The recurrent budget gain's training-seed interval is **[−0.405, 0.578]**; longer training did not establish useful recurrent adaptation. [Full follow-up memo](followup_memo.md), [reproduction commands](../README.md#recorded-fixed-budget-follow-up).

The restarted runs repeated their original training prefixes, whose logged metrics matched exactly. This was a paired budget follow-up, not independent replication or evidence of convergence. All 20 final checkpoints were retained. The test cohort is now consumed for development decisions; do not reuse it to claim untouched confirmation of a new configuration.

## Completed: change only GAE lambda

The controlled **GAE lambda .95 versus 1.0** comparison has now run for recurrent PPO at **300,032 steps**, seeds 11–15, holding every other training setting fixed. On 2,048 paired fresh episodes beginning at 8,000,000, mean objective changed from **0.479 to 0.497**. The paired effect **+0.018** has training-seed interval **[−0.434, 0.470]** and separate episode interval **[−0.032, 0.068]**. No lambda-one seed beat fixed wide **0.790** in its point estimate. One seed nearly always abstains; three mostly use fixed wide quotes. [GAE experiment memo](gae_memo.md), [reproduction commands](../README.md#controlled-gae-comparison).

The change did not fix recurrent learning in these runs and gives no convincing reason to adopt lambda one as a new default. This does not prove equivalence of estimators, convergence, or the cause of failure. The 8,000,000 test cohort is now consumed for development decisions. The reserved 7,000,000 development and 9,000,000 probe cohorts remain unused.

## Completed: entropy regularization alone

The fixed **entropy coefficient 0 versus .01** experiment has now run with GAE .95, 300,032 steps and seeds 11–15. On 2,048 new episodes beginning at 11,000,000, mean objective increased from **0.630 to 1.345**, paired effect **+0.715**, training-seed interval **[−0.638, 2.069]**. Seeds 11–13 beat fixed wide and showed input-dependent actions; seeds 14–15 worsened. History-8 still earned **2.723**, above every recurrent treatment. [Full entropy memo](entropy_memo.md).

The three useful seeds met the predeclared qualitative gate for exploratory decoding. All five models, including failures, were probed on 1,024 episodes beginning at 12,000,000, split by whole episode. Trained state decoded posterior quantities better than linear history features. However, untrained recurrent controls also decoded well, and trained-over-untrained adverse-selection gains have seed intervals containing zero. A poorly performing policy had particularly high adverse-selection decoding R². Linear accessibility is established on these trajectories; reliable learning, unique older-memory information and causal use are not.

The 11,000,000 economic and 12,000,000 probe cohorts are now consumed. The 10,000,000 development cohort remains unused. Keep the five fitted policies frozen; do not select only the three good seeds for subsequent analysis.

## Completed: independent-seed replication

Both entropy arms ran at 300,032 steps for fresh seeds 21–25, with 2,048 paired evaluation episodes beginning at 15,000,000. Mean objective increased from **.635 to 1.540**, paired gain **+.905**, training-seed interval **[−.162, 1.972]**. Four of five pairs improved, but the predeclared criterion was not met: both the treatment-effect and treatment-minus-fixed seed intervals include zero. The criterion was not relaxed and the development results were not pooled into confirmation. [Replication memo](replication_memo.md).

## Completed: nonlinear recent-history control

All five original entropy-.01 policies were evaluated on fresh probe episodes beginning at 14,000,000. A fixed nonlinear history-8 decoder outperformed the trained-state ridge panels on all six targets on average. Mean buy/sell adverse-selection R² was **.847/.830**, versus **.717/.699** for history plus trained-state ridge. All five adverse-selection contrasts favored nonlinear history. Capacity and tuning budgets differ, so this is a stronger comparator, not a complete test of conditional information. The historical linear-decoding advantage no longer supports a uniquely older-memory interpretation. [Nonlinear-control memo](nonlinear_memo.md).

The 14,000,000 and 15,000,000 cohorts are now consumed. Replication development at 13,000,000 and potential replication probes at 16,000,000 remain unused. Do not select only successful seeds or reuse inspected cohorts as untouched confirmation.

## Completed: retained-trajectory decision audit

The [myopic score-gap analysis](decision_quality.md) used the five existing nonlinear-control panels. Center choice contributes 81.5% of the mean gap, with substantial extra abstention for seed 14. Decoder R² and absolute error give different cross-policy rankings. The result motivates a bounded frozen-state action-score readout comparison; that experiment remains unrun. No new training or data collection was needed for the audit.

## Unresolved: earlier-history prevalence

The constructed Gate 1 example proves existence, not prevalence. On retained trajectories, quantify how often histories with similar current state and recent observations imply materially different posterior risks and preferred reference actions. Predeclare matching tolerances, overlap diagnostics and the unit of replication. Inventory and earlier policy actions already carry history; account for these channels before attributing any difference to recurrent memory. This is an exploratory diagnostic using existing data, not untouched confirmation.

Only if that diagnostic supports a specific learning question, define the smallest useful effect and desired interval width before selecting another fixed training budget. Allow for uncertainty in the seed variability estimated from five pairs. More evaluation episodes alone will not remove initialization uncertainty. A new confirmatory experiment would need fresh training seeds and an untouched cohort. No additional training has started.

Keep the scope narrow. Additional architectures, infrastructure and shift suites are deferred. Causal state interventions would need matched inventory/time/public state, same-belief and random donors, separate hidden/cell patches, and off-distribution diagnostics. Strong decoding alone is insufficient. See the [engineering roadmap](engineering_roadmap.md) for the remaining inference questions and implementation constraints.
