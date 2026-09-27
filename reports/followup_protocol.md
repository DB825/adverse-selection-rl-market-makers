# Fixed-budget follow-up protocol

This protocol records the larger-budget comparison authorized after the negative initial pilot. It was written while the larger training jobs were running, before this audit inspected their held-out results. It supplements the original protocol; it does not rewrite or replace the 100,352-step result.

## Fixed experiment and evaluation cohorts

Use `configs/followup_budget.yaml`: four learned policy families, initialization seeds 11–15, and exactly 300,032 transitions per run. All 20 final checkpoints are included. The architecture, simulator, objective, gamma, GAE lambda, entropy coefficient, learning rate, rollout size, batch size, and epoch count remain unchanged. Do not extend runs, select intermediate checkpoints, or replace unsuccessful seeds after inspecting returns.

Each run restarts from its original initialization and environment seed schedule. Its initial training prefix repeats the original experiment; the follow-up is a comparison of budgets along the same seeded training procedure, not 20 independent replications of the initial result. Seeds are independent within a policy family. Reusing seeds across policy families and budgets deliberately permits paired comparisons. The 20 larger runs require 6,000,640 training transitions, including the repeated prefixes; only 3,993,600 transitions lie beyond the original per-run budget.

The final economic cohort has 2,048 episodes with seeds 5,000,000–5,002,047. Evaluate each frozen policy stochastically, preserving separate customer and action-sampling streams. Reuse customer tapes and action uniforms across policies where applicable, while allowing quotes to cause different executions. Include the predeclared fixed choices, abstention, and Bayesian myopic reference.

For a claim about the effect of additional training, reevaluate the original 100,352-step checkpoints on **these same fresh episodes**. Compare the two budgets within policy, initialization seed, and episode. Comparing larger-budget returns on the new cohort directly with the original reported returns on seeds 1,000,000 onward would mix a budget change with a cohort change. The original report stays intact; the reevaluation is a separate paired-budget result.

Report all five seed means and paired differences. Keep uncertainty across training seeds separate from uncertainty across evaluation episodes. Episode intervals resample whole episodes after averaging over the fixed set of fitted seeds; they do not use 5×2,048 outcomes as independent samples. Seed intervals use five paired seed-level differences and remain approximate. Neither is a joint interval over all uncertainty sources. A positive budget effect supports improvement along this training procedure; it does not establish convergence, a universal sample-complexity boundary, or the cause of the original failure.

## Behavior gate before substantive probes

Assess economic behavior for the recurrent **family**, retaining every seed. The main fixed anchor is the predeclared centered wide quote `(0.2,0.8)`. Report the other fixed choices and abstention too. Do not choose a favorable comparator after observing results.

Proceed to the bounded representation stage when the combined evidence shows an economically meaningful recurrent policy to study: paired improvements over fixed quoting of useful magnitude, a pattern supported by several seeded runs rather than one exceptional result, and evidence that quoting responds conditionally to public observations or history. Consider exposure, tails, participation, and the full seed distribution alongside mean objective. A universal significance threshold or a demand that every seed outperform the fixed policy is not the gate.

Action-frequency diversity alone is insufficient. A pooled distribution can mix several collapsed policies, and even an individual policy can use a constant stochastic mixture of quotes. Inspect each seed separately and, when needed, inspect its frozen action probabilities on observed histories to distinguish conditional responses from such mixtures. These checks justify further study; they do not establish that the policy uses older information or adverse-selection beliefs.

If apparent gains come mainly from one seed while the family remains dominated by fixed quoting or arbitrary directional choices, record the gate as unresolved or failed and defer substantive probes. If evidence is mixed but several policies behave meaningfully, an explicitly labeled exploratory analysis may proceed **for all five seeds**, with weak or collapsed policies retained as controls. State why the gate was considered adequate; do not describe a selected subset as the trained family.

Gate 3 is distinct: a claim that recurrence improves economic decisions requires the matched-budget current-observation and history-8 comparisons. Economically meaningful recurrent behavior can justify a descriptive representation analysis even when recurrence has no advantage over those baselines. Such probes cannot reverse a negative Gate 3 result.

The current-observation comparator is not devoid of historical information: inventory accumulates earlier executions, and the previous quote records an earlier policy decision. In a closed-loop policy, those action fields can potentially carry information from older observations. Likewise, "history-8" describes its explicit input window, not a proof that all information in that window originated during the last eight arrivals. The constructed matched-history checks control inventory and the entire recent quote/outcome window; ordinary performance comparisons do not isolate these channels.

## Conditional representation panel

Only after documenting the behavior decision, use the separate 1,024-episode analysis cohort beginning at seed 6,000,000. Freeze all five final recurrent checkpoints. Use the same collection settings and probe panel for each seed, including unsuccessful seeds. This protocol does not include a full causal intervention experiment.

The implemented panel collects actor hidden and cell states after the current observation and before the action head. Targets are the observation-supported exact joint posterior's expected value, informed fraction, uninformed preference, entropy, and buy/sell adverse price information at the fixed reference quote `(0.2,0.8)`. Rows at `t>=8` enter the probe stage. Entire episodes, rather than timesteps, are split 60/20/20 with split seed 73. Training episodes fit feature scaling and ridge models; validation episodes choose regularization; test episodes report accuracy. These analysis episodes are separate from the economic test cohort.

Use all of the existing feature panels: basic public summaries, full history-8 features, trained hidden+cell states, and their documented combinations. The untrained network must process the **identical recorded observations** from each trained policy, using the corresponding initialization seed. It must not generate an alternative trajectory or observe private diagnostic labels. Compare history plus trained states both with history alone and with history plus untrained states.

Report every target and every seed. A result for one target or seed is exploratory, not grounds to suppress contrary outcomes. The six targets are correlated and the existing intervals are pointwise descriptions, not multiplicity-adjusted discoveries. The episode bootstrap is conditional on the fitted probe, selected regularization, frozen policy, and initialization of the untrained control. It excludes probe-fitting and training-seed uncertainty.

## Interpretation limits specific to these probes

The current probe implementation uses ridge regression. A trained recurrent state is a nonlinear transformation of observation history. Better linear prediction from that state than from raw history-8 can reflect useful feature computation on recent history; it does **not** by itself establish retained information beyond eight arrivals. A nonlinear predictor of the same recent-history input, or supported matched-history comparisons, is required before making that stronger claim. The untrained recurrent control is important but does not remove this limitation: it supplies random nonlinear features and generic memory with a different encoding.

If trained states beat both raw-history ridge and the untrained control on identical observations, the result supports an advantage in linear accessibility of the target under the learned encoding. It does not establish that the information is used in quoting, that alpha is represented independently of value and preference, or that a state patch changes only adverse-selection risk. If untrained states do as well, successful decoding alone is especially weak evidence of a learned inference mechanism.

Observations and posterior targets are generated by the policy's own actions. Different policy seeds can induce different histories and target variances. Compare trained and untrained encodings within the same recorded trajectory and interpret cross-seed differences in R² with that distribution change in mind. Show target MSE alongside R² and keep seed-level results visible.

Conditional quoting plus probe accuracy remains insufficient for causal use. A later intervention needs matched histories, explicit timing, hidden-versus-cell controls, matched random interventions, same-belief controls, and checks for states outside the usual distribution. Gate 1's constructed histories demonstrate existence, not frequency under any learned policy.
