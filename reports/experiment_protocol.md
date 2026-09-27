# Recorded pilot protocol and deviations

The scientific comparison is four policy families, five independent initialization seeds (11–15), and exactly 100,352 transitions per policy. The environment, fee, objective, gamma, observation normalization, PPO settings, and final-checkpoint rule are fixed across the comparison. No checkpoint is selected by its test return. All 20 completed runs are to be reported, including collapsed policies.

The main evaluation is 1,024 paired whole episodes with seeds 1,000,000–1,001,023. Customer randomness and action-sampling randomness use distinct streams. Models see outcomes induced by their own quotes. Fixed/myopic policies need no sampled actions. The main neural-policy metric samples the learned categorical distribution; it is not greedy argmax evaluation.

Representation collection is a separate conditional stage. A nearly constant-quote recurrent policy will not be presented as a successful belief-learning result merely because its states can be decoded. If a policy has meaningful conditional behavior, collect fresh episodes beginning at 3,000,000 and split those episodes 60/20/20 with seed 73. Validation selects ridge regularization only; test episodes report accuracy. Late-episode rows (t≥8) use a fixed adverse-selection reference quote `(.2,.8)`. An untrained recurrent control, when used, processes exactly the same recorded observations.

There is no full causal intervention experiment in this pilot. Gate 1 establishes an environment property; economic comparisons evaluate learned behavior; probes, if run, evaluate predictive information. These claims remain separate.

## Engineering decisions before the substantive comparison

1. Installed a pinned CPU Python stack and checked official Gymnasium/SB3/RecurrentPPO APIs plus installed source. The host has no available CUDA device.
2. The initial stock RecurrentPPO smoke measured 95.7 transitions/second. CPU backend and minibatch benchmarks were run using 2,048-transition jobs. A fused first-timestep-reset LSTM path achieved 722 transitions/second at the original batch size. Tests verified forward and backward equivalence; short-run checkpoint parameters agreed within 5.96e-8. The pilot keeps batch64 and ten epochs. No benchmark reward was used to select this optimization.
3. A seed audit caught overlap in SB3's vector-environment seeds when model seeds are consecutive. The first partial training batch was interrupted and moved to `results/deprecated_seed_overlap`; no completed model from it is used in the main comparison. Each corrected run explicitly seeds its environments at `model_seed*10000 + environment_index` after SB3 initialization and before the first reset. All five sets are disjoint; model/optimizer seeds stay 11–15.
4. A probe pipeline smoke fit showed ill-conditioning warnings from float32 ridge matrices. Features are now cast to float64 before fitting. This is numerical stabilization; feature sets and regularization candidates were unchanged. Smoke probe metrics are not treated as research results.

## Conditional follow-up rule

The first inspected recurrent seed at the fixed budget chose the centered wide quote 96.5% of the time and did not beat that fixed baseline. It is recorded as a warning about Gate 2; the remaining four seeds still run to completion. No architecture, reward, environment, or hyperparameter is changed to repair this result during the main comparison.

Any later training change prompted by these results must be labeled exploratory and use fresh validation and test episodes. It must not be pooled into the original equal-budget comparison or presented as a predeclared confirmatory result. An unsuccessful pilot is an acceptable outcome.

## Inventory-penalty diagnostic

In addition to the closed-form fixed-policy calculation, the Bayesian myopic policy is reevaluated at lambda 0 and .01 on paired episodes, with its posterior model/update and one-step decision rule unchanged. Changed actions can change observations and posterior values. This is a cheap reference-policy sensitivity check, not a learned-policy sweep or a selection of a new default. The default stays .001. Report changes in inventory/actions as well as the mechanical objective subtraction: the latter alone is not evidence of behavioral calibration.
