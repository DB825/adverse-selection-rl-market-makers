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

## 1. Replicate the fixed entropy configuration with fresh training seeds

Before another hyperparameter sweep, preregister a new set of training seeds with disjoint environment streams and an untouched economic cohort. Compare entropy 0 versus .01 at the same 300,032-step budget, GAE .95 and architecture, retaining every final model. Determine the replication count and useful-effect criterion before training. The current five-seed result is heterogeneous and informed development; it is not an untouched confirmation that .01 reliably improves learning. This replication has not run.

## 2. Strengthen the representation comparison

Add a nonlinear recent-history decoder using the same public features, with model capacity and tuning budget fixed in advance. Match whole-episode splits, train-only preprocessing and validation-only selection. Include basic summaries and the untrained-state controls, report all targets and all five policy seeds, and preserve separate training-seed and episode uncertainty. Existing probe-test results have now been inspected; use a fresh episode cohort to evaluate a newly chosen comparator.

This control is needed before attributing the linear-probe gain to information beyond the recent window. Inventory and previous policy actions already carry some history. Estimate how often decision-relevant earlier evidence occurs on actual trajectories; the constructed Gate 1 pair only establishes existence. This stronger comparison has not run.

## 3. Design causal interventions only after the stronger controls

Use the existing state-patching adapter only with a predeclared matching design. Match inventory, time, current observation and approximately posterior expected value, while varying estimated adverse-selection risk. Examine immediate action probabilities with same-belief donors and matched random interventions across trained seeds. Treat hidden and cell patches separately and test off-distribution effects; full-state patches change several beliefs simultaneously. Decoding in the poorly performing seed makes these controls especially necessary. No causal intervention has run.

Keep simulator mechanics unchanged unless a diagnosed issue warrants revision. Public repository preparation is a separate task; its current status is documented in [repository readiness](repository_readiness.md).
