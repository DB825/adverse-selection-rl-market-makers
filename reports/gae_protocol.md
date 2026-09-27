# Single-factor GAE comparison — protocol

Recorded before training or inspecting any result for this ablation, 27 September 2026. The user authorized the proposed GAE comparison after the completed budget follow-up. That earlier test cohort is development evidence for choosing this experiment, not untouched confirmation.

## Fixed intervention

Train recurrent PPO for exactly 300,032 transitions for initialization seeds 11–15 using `configs/gae_lambda_one.yaml`. Relative to the existing recurrent control, change only training GAE lambda from .95 to 1.0. Keep gamma one, entropy coefficient zero, learning rate .0003, eight environments, 128 steps, batch64, ten epochs, the actor/critic architecture, simulator and economic objective unchanged. Reuse the same initialization and simulator streams within each paired seed. Retain every final checkpoint, including failures. No intermediate checkpoint selection, early stopping based on returns, extended budgets, seed replacement or additional hyperparameter sweep.

Existing .95 checkpoints at `results/followup_budget/recurrent/seed_*` are the control. They need no retraining. New files use `results/gae_lambda_one/`; preserve earlier results. This is five paired training-seed comparisons, not ten independent runs. Once learning starts, trajectories and optimizer updates may diverge because the advantage estimator changes.

With these 128-step synchronous rollouts, every 64-arrival episode ends within the rollout. Gamma one and GAE lambda one yield return-minus-baseline advantages for complete episodes, avoiding intermediate critic bootstrapping in that estimate. This can increase variance. The experiment tests sensitivity to the estimator; it does not establish that terminal credit assignment caused the previous failure. Reward accounting itself is unchanged.

## Cohorts and references

- Reserve seeds 7,000,000–7,001,023 (1,024 episodes) for development/validation. No additional model selection is planned, so leave this cohort unused unless a specific implementation issue needs diagnosis.
- Final economic evaluation: 2,048 episodes, seeds 8,000,000–8,002,047. Evaluate both recurrent arms stochastically with identical customer tapes and separate paired action uniforms. Quotes still produce different executions. Evaluate the existing .95 feedforward, history-8 and belief-input checkpoints on the same cohort, alongside all predeclared fixed quotes, abstention and Bayesian myopic. These are contextual comparisons; only the recurrent .95-versus-1 contrast isolates the changed training setting.
- Conditional representation analysis: 1,024 new episodes, seeds 9,000,000–9,001,023. Do not use economic test episodes to fit probes. The original unused analysis cohort at 6,000,000 remains separate.

Report the paired recurrent objective difference as the primary economic comparison, showing each seed and the five-seed mean. Show an approximate Student t4 interval over seed differences, separately from an episode bootstrap that resamples whole episodes after averaging paired differences over the fixed five fitted policies. Neither interval is joint uncertainty. Five potentially heterogeneous seeds limit inference. Retain exposure, tails, participation, regime results and per-seed action distributions; no timestep pseudoreplication.

## Gates

Apply the behavior criteria from `followup_protocol.md` to the complete recurrent lambda-one family: useful improvements over predeclared fixed wide quoting supported by several seeds, plus meaningful input-dependent quoting, rather than one exceptional seed or a pooled mixture of collapsed policies. Probability variation alone is insufficient. Compare against history-8 and current-observation policies separately before claiming a recurrence advantage; only the recurrent family receives this changed hyperparameter, so contextual comparisons do not establish a generally superior architecture.

If behavior fails, document the result and defer substantive probes. If behavior is sufficiently useful across seeds, run the existing frozen-state probe panel for **all five** new recurrent checkpoints, keeping failures as controls. Use whole-episode 60/20/20 splits, train-only scaling, validation-only ridge selection, fixed-quote targets, and untrained networks replaying identical public histories. Report all targets and seeds, target variance, and separate episode/seed uncertainty. Linear accessibility does not establish information beyond an arbitrary nonlinear recent-history predictor or causal use. No causal interventions or second training ablation are included in this authorization.
