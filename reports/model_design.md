# Model design and independent Gate 1 audit

This note records simulator mathematics and a constructive inference result. It does not report neural learning or establish a finite-horizon optimum; the experiment memo reports those empirical results separately.

## Known data-generating model

An episode has 64 customer arrivals and a fixed latent state `(V, alpha, p)` in the 12-element Cartesian product `{0,1} × {.05,.30} × {.25,.50,.75}`. The initial prior is uniform and independent across coordinates. Outcomes always describe the customer: a buy decreases dealer inventory by one; a sell increases it by one.

For an active quote `(b,a)`, an informed customer buys exactly when `V=1`, and sells exactly when `V=0`. An uninformed buyer has probability `p` and accepts the ask with probability `1-a`; an uninformed seller has probability `1-p` and accepts the bid with probability `b`. Therefore

```
P(buy | s,a)  = alpha V + (1-alpha) p (1-ask)
P(sell | s,a) = alpha (1-V) + (1-alpha) (1-p) bid
P(no trade | s,a) = 1 - P(buy | s,a) - P(sell | s,a).
```

The quotes are indexed by center first, then half-spread. Centers are `.35,.50,.65`, half-spreads `.10,.20,.30`; action 9 abstains. All active quotes are strictly interior. Abstention has outcome likelihood `(0,0,1)` for every latent state and gives no evidence. In contrast, a customer declining an active quote is informative. Conditioning on quotes is essential: a rejected expensive ask and a rejected cheap ask supply different evidence.

The production likelihood is implemented algebraically, and customer simulation draws independent type, intended side, and reservation-value uniforms. The decision-relevance script independently evaluates the formulas with Python scalar arithmetic and direct full-history log likelihoods. Its tests compare those results against the production exact Bayesian filter and action statistics.

## Objective, posterior, and reference decision

Cash starts at zero. A customer buy adds `ask-fee` to cash and subtracts one from inventory; a customer sell subtracts `bid+fee` from cash and adds one to inventory. The objective is terminal cash plus `q_T V`, less `lambda/T` times the sum of **post-arrival** squared inventories. No-trade and abstention arrivals still incur this inventory charge. Dense rewards use the fixed public mark `.5` plus terminal correction `q_T(V-.5)`. At gamma one they telescope exactly to the economic objective.

Inference preserves the joint posterior, not three independent marginals. Each active update multiplies its weights by the actual quote/outcome likelihood, then normalizes in log space. Abstention is a no-op. An observation impossible on the filter's entire positive-prior support raises an exception without replacing the prior. Conditional statistics for negligible execution probabilities are explicitly undefined; expected unconditional profit remains finite.

For a posterior `w`, let `P_B,P_S` denote predictive execution probabilities and `v_B,v_S` the expected terminal values conditional on those executions. Expected immediate dealer profit is

```
P_B (ask-v_B-fee) + P_S (v_S-bid-fee).
```

Buy-side adverse price information is `v_B-E[V]`; sell-side adverse price information is `E[V]-v_S`. These are value changes, not quote-dependent dealer profit. They need not be universal monotonic functions of expected informedness because the posterior has dependencies among `V,alpha,p`.

The Bayesian myopic reference subtracts the expected incremental current-arrival inventory charge:

```
(lambda/T) [P_B (1-2q) + P_S (1+2q)].
```

The common charge `lambda q²/T` cancels when comparing actions, so abstention can have score zero even with nonzero inventory. This one-step comparison ignores information acquisition, subsequent quoting opportunities, and future carrying costs. It is not labeled optimal.

## Gate 1: earlier quote evidence changes the reference response

Run `python scripts/decision_relevance.py` from the repository root. The deterministic search uses seed 23817 and 12,000 candidate histories, with predeclared tolerances `|ΔE[V]|≤.02`, `|ΔE[p]|≤.005`, and a minimum `.001` best-versus-second-best myopic score margin. Search varies the first 16 quotes while holding all 24 customer outcomes fixed. The last eight quote/outcome events are identical. The output stores both full histories, all 12 posterior weights, every action's score, support diagnostics, and search settings in `results/gate1/decision_pairs.json`.

The final pair, evaluated after 24 arrivals, has dealer inventory `-2` and 40 arrivals remaining in both cases. Replay through the actual eight-observation wrapper confirms its entire `8×9` feature vector is exactly identical for the two histories. The following are computed simulator facts, not sample estimates:

| Quantity | History A | History B |
|---|---:|---:|
| Expected asset value | 0.611005 | 0.602257 |
| Expected informed fraction | 0.205845 | 0.117220 |
| Expected uninformed buy preference | 0.516549 | 0.516514 |
| Buy-side adverse information at `(0.2,0.8)` | 0.185593 | 0.135047 |
| Sell-side adverse information at `(0.2,0.8)` | 0.183927 | 0.100611 |
| Myopic selected quote | `(0.35,0.95)` | `(0.2,0.8)` |
| Best-versus-second-best score margin | 0.005162 | 0.001021 |
| Smallest predictive probability of any observed event | 0.024564 | 0.111558 |
| Probability of full outcome sequence given its quotes | 1.506e-13 | 5.576e-11 |

Every selected event has positive probability under **every** one of the 12 regimes. The smallest individual regime-conditioned event likelihood is `.00875` in A and `.035` in B. These history probabilities condition on the prescribed quotes; they do not include a probability for choosing the action sequence. Long exact strings naturally have small probabilities, and this constructed search does not estimate how often comparable situations occur under a trained policy.

**Gate 1 recommendation: continue.** A policy given only the current observation or the latest eight observations cannot distinguish these inputs, while the full-history Bayesian reference chooses different actions. Earlier quote-conditioned evidence matters even when inventory, time, the entire customer outcome sequence, and approximately both `E[V]` and `E[p]` match. This is an existence result; it does not prove that recurrence will improve average return or that the learned policy encounters these examples often. The posterior dependencies also change, so it is not an intervention isolating expected alpha alone. Both selected actions have the same spread and different centers: higher estimated adverse selection does not universally imply a wider spread.

## Independent inventory-penalty scale check

For a fixed action and latent regime, inventory increments are iid with `mu=P_S-P_B` and `nu=P_S+P_B`. Thus `E[q_t²|s]=t nu+t(t-1)mu²`. Summing this expression over arrivals and averaging over regimes gives exact expected carrying cost. It is essential to condition on the episode regime first: unconditional increments are dependent through the common hidden state.

The script reports all ten actions under lambda `0,.001,.01`. The predeclared centered fixed quotes have the following 64-arrival expectations:

| Fixed quote | Profit before penalty | Objective, lambda .001 | Objective, lambda .01 |
|---|---:|---:|---:|
| `(0.4,0.6)` | -2.368000 | -2.472887 | -3.416873 |
| `(0.3,0.7)` | -0.192000 | -0.283118 | -1.103182 |
| `(0.2,0.8)` | 0.928000 | 0.847483 | 0.122829 |
| Abstain | 0 | 0 | 0 |

For the widest centered quote, expected mean squared inventory is `80.517125`. Lambda `.001` subtracts `.080517`, about 8.7% of its expected profit. Lambda `.01` removes about 86.8%. This establishes an objective scale, not a behavioral calibration: policy-specific inventory and action changes must still be evaluated. The closed form is separately checked by exhaustive enumeration of every three-arrival outcome path, with nonzero fees and inventory cost, using the production accounting functions.

## Limits carried into interpretation

Gate 1 supports studying memory, but a constructive pair is not an evaluation of trained recurrence. Later gates require held-out episodes, independent training seeds, and comparisons against fixed, current-observation, recent-history, and belief-input policies. Probe targets should be the observation-supported exact posterior, with adverse-selection labels at the same fixed quote for every row. Successful decoding is only predictive association. It cannot establish that the policy uses this information, nor that a hidden-state intervention changes only adverse-selection beliefs.
